"""Checkpoint paths and fresh logging sessions for continued training."""
from pathlib import Path

from stable_baselines3.common.utils import configure_logger


def checkpoint_path(root, path):
    """Resolve relative paths against the project root, allowing omitted .zip."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file() and Path(str(path) + ".zip").is_file():
        path = Path(str(path) + ".zip")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def replay_buffer_path(model_path):
    path = Path(model_path)
    name = path.stem if path.suffix == ".zip" else path.name
    return path.with_name(name + "_replay_buffer.pkl")


def load_training_model(algorithm, path, env, device, run_dir, tensorboard_log, verbose):
    """Keep checkpoint hyperparameters and optimizer, but create a new log run."""
    model = algorithm.load(str(path), env=env, device=device)
    model.verbose = verbose
    # SB3 otherwise reuses the most recent run when reset_num_timesteps=False.
    model.set_logger(configure_logger(
        verbose=verbose,
        tensorboard_log=str(run_dir) if tensorboard_log else None,
        tb_log_name=algorithm.__name__,
        reset_num_timesteps=True,
    ))
    return model
