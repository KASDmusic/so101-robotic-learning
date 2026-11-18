"""Training vector environments with a process-safe shared curriculum."""
import multiprocessing as mp

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_training_vec_env(env_fns, *, backend, render_mode, curriculum, resources):
    if backend not in ("dummy", "subproc"):
        raise ValueError("vec_env_backend must be 'dummy' or 'subproc'.")
    if not env_fns:
        raise ValueError("nb_vec_env_train must be at least 1.")
    if backend == "dummy":
        env = DummyVecEnv(env_fns)
    else:
        if render_mode == "human":
            raise ValueError("SubprocVecEnv requires a non-interactive train_render_mode. Use the viewer for evaluation.")
        # Do not inherit the parent's MuJoCo/OpenGL or PyTorch runtime.
        context = mp.get_context("spawn")
        if curriculum is not None:
            curriculum.share(context)
        env = SubprocVecEnv(env_fns, start_method=context.get_start_method())
    resources.callback(env.close)
    return env
