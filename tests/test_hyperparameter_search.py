"""Real short HPO studies, SQLite resumption and evaluation/pruning contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import optuna
except ImportError:
    optuna = None

from so101_robotic_learning.bounce_rl.rl import train_ppo, train_sac
from so101_robotic_learning.bounce_rl.env.gravity_curriculum import GravityCurriculumConfig
from so101_robotic_learning.bounce_rl.rl.hyperparameter_search import optimize


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/mjcf/so101_new_calib copy.xml"


@unittest.skipIf(optuna is None, 'Install optional dependencies: pip install -e ".[hpo]"')
class HyperparameterSearchTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self, root, algo="ppo"):
        config = dict(
            root=Path(root), xml_path=XML, total_timesteps=4,
            env_kwargs={"max_episode_steps": 3, "frame_skip": 1},
            n_eval_episodes=1, eval_max_steps=3, print_spaces=False,
            device="cpu", verbose=0, callback_verbose=0, pi_layers=(16,),
        )
        if algo == "ppo":
            config.update(n_steps=2, batch_size=2, n_epochs=1, vf_layers=(16,))
        else:
            config.update(buffer_size=16, learning_starts=0, batch_size=2, qf_layers=(16,))
        return config

    def test_grid_persists_resumes_and_does_not_repeat_exhausted_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp)
            original = copy.deepcopy(config)
            kwargs = dict(algorithm="ppo", base_config=config, method="grid",
                          study_name="grid_test", search_space={"n_epochs": [1, 2]},
                          eval_every_timesteps=2)
            study = optimize(**kwargs, n_trials=1)
            self.assertEqual(len(study.trials), 1)
            study = optimize(**kwargs)  # all remaining combinations
            self.assertEqual(len(study.trials), 2)
            self.assertEqual({trial.params["n_epochs"] for trial in study.trials}, {1, 2})
            self.assertTrue(all(trial.state.name == "COMPLETE" for trial in study.trials))
            self.assertEqual(len(optimize(**kwargs).trials), 2)
            self.assertEqual(config, original)
            directory = Path(tmp) / "models/grid_test"
            self.assertTrue((directory / "study.db").is_file())
            self.assertEqual(json.loads((directory / "best_params.json").read_text()), study.best_params)
            self.assertEqual(len((directory / "trials.csv").read_text().splitlines()), 3)
            self.assertEqual(sorted(path.name for path in (Path(tmp) / "runs").iterdir()), ["ppo_bounce"])
            for trial in study.trials:
                model_dir = Path(trial.user_attrs["model_dir"])
                self.assertTrue((model_dir / "ppo_bounce_last.zip").is_file())
                self.assertEqual(trial.user_attrs["final_timesteps"], 4)
                report = json.loads(next(model_dir.glob("*_final_step_*.json")).read_text())
                self.assertEqual(report["task_metrics"]["mean_max_consecutive_bounces"], trial.value)
                events = next((Path(tmp) / "runs/ppo_bounce/grid_test" / model_dir.name).rglob("events.out.tfevents.*"))
                accumulator = EventAccumulator(str(events.parent))
                accumulator.Reload()
                self.assertEqual([e.step for e in accumulator.Scalars("eval/mean_reward")], [2, 4, 4])
            self.assertFalse(list(Path(tmp).rglob("*.pkl")))
            changed = self.config(tmp)
            changed["env_kwargs"]["frame_skip"] = 2
            with self.assertRaisesRegex(ValueError, "different settings"):
                optimize(**{**kwargs, "base_config": changed})

    def test_tpe_and_gp_execute_adaptive_trials_for_sac_and_ppo(self):
        for method, algo in (("tpe", "sac"), ("gp", "ppo")):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                study = optimize(
                    algo, self.config(tmp, algo), method=method, study_name="adaptive",
                    search_space={"learning_rate": {"type": "float", "low": 1e-4,
                                                    "high": 5e-4, "log": True}},
                    n_trials=2, n_startup_trials=1, eval_every_timesteps=2,
                )
                self.assertEqual(len(study.trials), 2)
                self.assertTrue(all(trial.state.name == "COMPLETE" for trial in study.trials))
                self.assertNotEqual(study.trials[0].params, study.trials[1].params)
                self.assertFalse(list(Path(tmp).rglob("*.pkl")))

    def test_final_evaluation_observes_updated_ppo_policy_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp)
            config.update(model_dir_name="models/direct", eval_every_timesteps=2,
                          evaluate_at_end=True, save_eval_video=False, eval_render_mode=None)
            evaluations = []
            original_evaluate = train_ppo.EvalVideoSaveBestCallback._evaluate_record_and_save
            def record_updates(callback, *args, **kwargs):
                evaluations.append(callback.model._n_updates)
                return original_evaluate(callback, *args, **kwargs)
            observed = []
            config["evaluation_hook"] = lambda result, step: observed.append((result["is_final"], step))
            with patch.object(train_ppo.EvalVideoSaveBestCallback, "_evaluate_record_and_save", record_updates):
                result = train_ppo.train(**config)
            self.assertEqual(evaluations, [0, 1, 2])
            self.assertEqual(observed, [(False, 2), (False, 4), (True, 4)])
            self.assertTrue(result["is_final"])
            last = PPO.load(Path(tmp) / "models/direct/ppo_bounce_last.zip", device="cpu")
            self.assertEqual(last._n_updates, evaluations[-1])
            self.assertTrue((Path(tmp) / "models/direct/ppo_bounce_best_final_step_000000004.json").is_file())

    def test_pruning_interrupts_trial_and_closes_both_environments(self):
        with tempfile.TemporaryDirectory() as tmp:
            environments = []
            factory = train_ppo.make_env
            def tracked_factory(*args, **kwargs):
                make = factory(*args, **kwargs)
                def create():
                    env = make()
                    environments.append(env)
                    return env
                return create
            with patch.object(train_ppo, "make_env", side_effect=tracked_factory), \
                    patch.object(optuna.trial.Trial, "should_prune", return_value=True):
                study = optimize("ppo", self.config(tmp), method="grid", study_name="prune",
                                 search_space={"n_epochs": [1]}, n_trials=1,
                                 eval_every_timesteps=2, pruning=True, pruning_warmup_steps=0)
            self.assertEqual(study.trials[0].state.name, "PRUNED")
            self.assertTrue(environments)
            self.assertTrue(all(env.unwrapped.renderer is None for env in environments))
            self.assertFalse(list((Path(tmp) / "models/prune").rglob("*_last.zip")))
            self.assertFalse((Path(tmp) / "models/prune/best_params.json").exists())

    def test_shared_budget_rounds_all_ppo_trials_to_same_rollout_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp)
            config["total_timesteps"] = 5
            study = optimize("ppo", config, method="grid", study_name="budget",
                             search_space={"n_steps": [2, 4]}, eval_every_timesteps=4)
            self.assertEqual(study.user_attrs["search_config"]["effective_total_timesteps"], 8)
            self.assertEqual([t.user_attrs["final_timesteps"] for t in study.trials], [8, 8])

    def test_invalid_configurations_fail_before_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(train_ppo, "train", wraps=train_ppo.train) as train:
            # inspect.signature must remain the trainer signature when spying.
            train.__signature__ = __import__("inspect").signature(train_ppo.train._mock_wraps)
            config = self.config(tmp)
            for extra in (
                {"search_space": {"frame_skip": [1, 2]}},
                {"search_space": {"learning_rate": {"type": "float", "low": .1, "high": .01}}},
                {"method": "grid", "search_space": {"learning_rate": {"type": "float", "low": .1, "high": .2}}},
                {"search_space": {"n_steps": {"type": "int", "low": 2, "high": 4}}},
                {"study_name": "../outside"},
                {"base_config": {**config, "resume_from": "checkpoint.zip"}},
                {"base_config": {**config, "gravity_curriculum": GravityCurriculumConfig()}},
            ):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    optimize("ppo", **{"base_config": config, **extra})
            train.assert_not_called()

    def test_new_notebook_executes_a_short_search_without_changing_existing_notebooks(self):
        notebook_path = ROOT / "notebooks/bounce_rl/notebook_hyperparameter_search.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        existing = [ROOT / f"notebooks/bounce_rl/notebook_train_{algo}.ipynb" for algo in ("ppo", "sac")]
        before = [path.read_bytes() for path in existing]
        with tempfile.TemporaryDirectory() as tmp:
            namespace = {"__name__": "__main__"}
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                exec(compile("".join(cell["source"]), str(notebook_path), "exec"), namespace)
                if "configuration" in cell["metadata"].get("tags", []):
                    config = self.config(tmp)
                    namespace["search_config"].update(
                        base_config=config, method="grid", study_name="notebook_test",
                        n_trials=1, search_space={"n_epochs": [1]}, eval_every_timesteps=2,
                    )
            self.assertEqual(len(namespace["study"].trials), 1)
            self.assertEqual(namespace["study"].trials[0].state.name, "COMPLETE")
        self.assertEqual([path.read_bytes() for path in existing], before)


if __name__ == "__main__":
    unittest.main()
