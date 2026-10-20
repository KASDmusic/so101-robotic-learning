"""Optional, sequential Optuna searches around the existing PPO/SAC trainers.

Install with ``pip install -e ".[hpo]"``. Importing regular trainers does not
require Optuna. Each trial trains from scratch with the same task and seed.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np


PPO_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "n_steps": [128, 256, 512],
    "batch_size": [64, 128],
    "n_epochs": [5, 10],
    "ent_coef": {"type": "float", "low": 1e-4, "high": .05, "log": True},
}
SAC_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "batch_size": [64, 128, 256],
    "tau": [.005, .01, .02],
    "ent_coef": ["auto_0.1", "auto_0.5", "auto_1.0"],
}
PPO_GRID = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps": [128, 256], "batch_size": [64, 128], "n_epochs": [5, 10],
}
SAC_GRID = {
    "learning_rate": [1e-4, 3e-4], "batch_size": [64, 128],
    "tau": [.005, .01], "ent_coef": ["auto_0.1", "auto_1.0"],
}
_PARAMETERS = {
    "ppo": {"learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"},
    "sac": {"learning_rate", "batch_size", "tau", "gamma", "ent_coef",
            "buffer_size", "learning_starts", "gradient_steps", "train_freq"},
}


def _optuna():
    try:
        import optuna
    except ImportError as error:
        raise ImportError('Install search dependencies with: pip install -e ".[hpo]"') from error
    return optuna


def _describe(value):
    """JSON description of configuration, including reward terms and dataclasses."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_describe(item) for item in value]
    if hasattr(value, "__dict__") and not callable(value):
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}",
                "settings": _describe(vars(value))}
    raise TypeError(f"Search configuration cannot serialize {type(value).__name__}.")


def _write_json(path, value):
    path.write_text(json.dumps(_describe(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _validate_space(optuna, algorithm, method, space):
    if not space:
        raise ValueError("search_space must contain at least one hyperparameter.")
    unknown = set(space) - _PARAMETERS[algorithm]
    if unknown:
        raise ValueError(f"Unsupported search parameters for {algorithm}: {sorted(unknown)}")
    for name, spec in space.items():
        if isinstance(spec, list):
            if not spec or any(value is not None and type(value) not in (str, bool, int, float)
                               for value in spec):
                raise ValueError(f"{name} requires a nonempty list of scalar choices.")
            if any(isinstance(value, float) and not math.isfinite(value) for value in spec):
                raise ValueError(f"{name} choices must be finite.")
            if len({(type(value), value) for value in spec}) != len(spec):
                raise ValueError(f"{name} choices must be unique.")
        elif method == "grid":
            raise ValueError("Grid search requires lists of explicit values.")
        elif isinstance(spec, dict) and spec.get("type") in ("float", "int"):
            constructor = (optuna.distributions.FloatDistribution if spec["type"] == "float"
                           else optuna.distributions.IntDistribution)
            constructor(**{key: value for key, value in spec.items() if key != "type"})
        else:
            raise ValueError(f"Invalid search distribution for {name}.")


def _suggest(trial, name, spec):
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    options = {key: value for key, value in spec.items() if key != "type"}
    if spec["type"] == "float":
        return trial.suggest_float(name, **options)
    return trial.suggest_int(name, **options)


def _training_budget(algorithm, config, space, module):
    """Round all trials to a shared collection boundary, independent of parameters."""
    defaults = inspect.signature(module.train).parameters
    key = "n_steps" if algorithm == "ppo" else "train_freq"
    choices = space.get(key, [config.get(key, defaults[key].default)])
    if not isinstance(choices, list) or any(type(v) is not int or v < 1 for v in choices):
        raise ValueError(f"{key} must use positive integer choices (step-based collection).")
    workers = config.get("nb_vec_env_train", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("nb_vec_env_train must be a positive integer.")
    block = math.lcm(*(value * workers for value in choices))
    requested = config.get("total_timesteps", defaults["total_timesteps"].default)
    if type(requested) is not int or requested < 1:
        raise ValueError("total_timesteps must be a positive integer.")
    return math.ceil(requested / block) * block


def _export(study, directory):
    with (directory / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["number", "state", "value", "params", "model_dir"])
        writer.writeheader()
        for trial in study.trials:
            writer.writerow({"number": trial.number, "state": trial.state.name,
                             "value": trial.value, "params": json.dumps(trial.params),
                             "model_dir": trial.user_attrs.get("model_dir")})
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if completed:
        best = study.best_trial
        _write_json(directory / "best_params.json", best.params)
        _write_json(directory / "best_trial.json", {
            "number": best.number, "value": best.value, "params": best.params,
            "model_dir": best.user_attrs.get("model_dir"),
            "metric": study.user_attrs["search_config"]["metric"],
        })


def optimize(
    algorithm, base_config, *, method="tpe", search_space=None, study_name=None,
    n_trials=None, seed=42, eval_every_timesteps=10_000,
    metric="mean_max_consecutive_bounces", pruning=False,
    pruning_warmup_steps=30_000, n_startup_trials=5, timeout=None,
):
    """Run additional independent trials, or resume an existing SQLite study.

    method: 'grid', 'tpe', 'gp' ('bayesian' aliases 'gp'). Search spaces use
    lists for categories, or dict(type='float'/'int', low=..., high=..., ...).
    Results live in root/models/<study_name>. Logs stay below runs/<algo>_bounce.
    n_trials=None runs the whole remaining grid, or 10 new TPE/GP trials.
    Existing studies require identical task, budget and search settings. Changing
    n_trials or timeout is allowed. A resumed grid only visits remaining points.
    The budget is rounded up to a shared rollout/collection boundary. The final
    updated policy determines the objective. Pruning is opt-in and never applied
    to final evaluations. Existing ordinary training settings are not modified.
    """
    optuna = _optuna()
    from sqlalchemy.pool import NullPool
    from . import train_ppo, train_sac

    algorithm = algorithm.lower()
    method = "gp" if method == "bayesian" else method
    if algorithm not in _PARAMETERS or method not in ("grid", "tpe", "gp"):
        raise ValueError("Choose algorithm='ppo'/'sac' and method='grid'/'tpe'/'gp'.")
    if ((n_trials is not None and (type(n_trials) is not int or n_trials < 1)) or type(seed) is not int
            or type(eval_every_timesteps) is not int or eval_every_timesteps < 1
            or type(n_startup_trials) is not int or n_startup_trials < 1
            or type(pruning_warmup_steps) is not int or pruning_warmup_steps < 0):
        raise ValueError("Invalid trial count, seed, evaluation interval or pruning settings.")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive seconds.")
    module = train_ppo if algorithm == "ppo" else train_sac
    config = copy.deepcopy(base_config)
    if config.get("resume_from") is not None or config.get("replay_buffer_from") is not None:
        raise ValueError("Hyperparameter trials must start fresh. Remove resume_from/replay_buffer_from.")
    if config.get("evaluation_hook") is not None:
        raise ValueError("The search manages evaluation_hook.")
    if config.get("gravity_curriculum") is not None:
        raise ValueError("Hyperparameter comparisons require fixed gravity. Set gravity_curriculum=None.")
    config.pop("gravity_curriculum", None)
    unknown = set(config) - set(inspect.signature(module.train).parameters)
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    root = Path(config["root"]).resolve()
    config["root"] = root
    xml_path = Path(config["xml_path"]).resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    config["xml_path"] = xml_path
    if search_space is None:
        search_space = ({"ppo": PPO_GRID, "sac": SAC_GRID} if method == "grid"
                        else {"ppo": PPO_SPACE, "sac": SAC_SPACE})[algorithm]
    space = copy.deepcopy(search_space)
    _validate_space(optuna, algorithm, method, space)
    budget = _training_budget(algorithm, config, space, module)
    if n_trials is None:
        n_trials = math.prod(len(values) for values in space.values()) if method == "grid" else 10
    config.update(total_timesteps=budget, seed=seed, resume_from=None,
                  save_eval_video=False, train_render_mode=None, eval_render_mode=None,
                  eval_every_timesteps=eval_every_timesteps, evaluate_at_end=True)
    for ignored in ("model_dir_name", "run_subdir", "evaluation_hook", "eval_every_episodes"):
        config.pop(ignored, None)
    if algorithm == "sac":
        config["save_replay_buffer"] = False
    study_name = study_name or f"{algorithm}_{method}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", study_name):
        raise ValueError("study_name must be a simple directory name, without slashes.")
    settings = _describe({
        "algorithm": algorithm, "method": method, "base_config": config,
        "search_space": space, "seed": seed, "eval_every_timesteps": eval_every_timesteps,
        "effective_total_timesteps": budget, "metric": metric, "pruning": pruning,
        "pruning_warmup_steps": pruning_warmup_steps, "n_startup_trials": n_startup_trials,
        "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
    })
    # Normalize tuples, arrays and object settings before comparing stored studies.
    settings = json.loads(json.dumps(settings, allow_nan=False))
    if method == "grid":
        sampler = optuna.samplers.GridSampler(space, seed=seed)
    elif method == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    else:
        sampler = optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_startup_trials,
                                            deterministic_objective=False)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials,
                                        n_warmup_steps=pruning_warmup_steps)
              if pruning else optuna.pruners.NopPruner())
    directory = root / "models" / study_name
    directory.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(directory / 'study.db').as_posix()}",
        # Release SQLite file handles between operations, also on Windows.
        engine_kwargs={"poolclass": NullPool},
    )
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction="maximize", sampler=sampler, pruner=pruner, load_if_exists=True,
    )
    previous = study.user_attrs.get("search_config")
    if previous is not None and previous != settings:
        raise ValueError("This study has different settings. Use a new study_name.")
    if previous is None and study.trials:
        raise ValueError("Existing trials have no compatible search metadata. Use a new study_name.")
    study.set_user_attr("search_config", settings)
    _write_json(directory / "search_config.json", settings)

    def score(result):
        value = result.get("task_metrics", {}).get(metric, result.get(metric))
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Evaluation metric {metric!r} is missing or non-finite.")
        return float(value)

    def objective(trial):
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        current = copy.deepcopy(config)
        current.update(params)
        if algorithm == "ppo":
            defaults = inspect.signature(module.train).parameters
            rollout = current.get("n_steps", defaults["n_steps"].default) * current.get("nb_vec_env_train", 1)
            batch = current.get("batch_size", defaults["batch_size"].default)
            if batch < 2 or batch > rollout or rollout % batch:
                raise optuna.TrialPruned("PPO batch_size must divide n_steps * nb_vec_env_train.")
        model_dir = directory / f"trial_{trial.number:04d}"
        trial.set_user_attr("model_dir", str(model_dir))

        def on_evaluation(result, timesteps):
            # Do not prune a final policy, or a final rollout before its update.
            if result.get("is_final") or timesteps >= budget:
                return
            trial.report(score(result), step=timesteps)
            if trial.should_prune():
                raise optuna.TrialPruned()

        current.update(
            total_timesteps=budget, seed=seed, resume_from=None,
            eval_every_timesteps=eval_every_timesteps, evaluation_hook=on_evaluation,
            evaluate_at_end=True, save_eval_video=False,
            train_render_mode=None, eval_render_mode=None,
            model_dir_name=str(model_dir),
            run_subdir=f"{algorithm}_bounce/{study_name}/trial_{trial.number:04d}",
        )
        if algorithm == "sac":
            current.update(save_replay_buffer=False)
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_json(model_dir / "trial_config.json", {
            key: value for key, value in current.items() if key != "evaluation_hook"
        })
        result = module.train(**current)
        value = score(result)
        trial.set_user_attr("final_timesteps", result["training_timesteps"])
        trial.set_user_attr("final_metrics", _describe(result["task_metrics"]))
        trial.report(value, step=result["training_timesteps"])
        return value

    # An exhausted GridSampler starts a duplicate if optimize() is called again.
    exhausted = method == "grid" and sampler.is_exhausted(study)
    try:
        if not exhausted:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1,
                           gc_after_trial=True, callbacks=[lambda study, trial: _export(study, directory)])
    finally:
        _export(study, directory)
        storage.remove_session()
    return study
