import json
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from .checkpoints import checkpoint_path, load_training_model
from .rl_utils import (
    EvalVideoSaveBestCallback,
    ensure_dirs,
    print_env_spaces,
    log_camera_videos,
)

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv


def make_env(
    xml_path,
    render_mode: Optional[str] = None,
    reward=None,
    record_video=False,
    env_kwargs=None,
):
    def _init():
        env = BounceEnv(
            xml_path=str(xml_path),
            render_mode=render_mode,
            record_video=record_video,
            reward=reward,
            **{"render_profile": "train_fast", **(env_kwargs or {})},
        )
        env = Monitor(env)
        return env

    return _init


def train(
    xml_path,
    root,
    *,
    # Environnement
    nb_vec_env_train=1,
    nb_vec_env_test=1,
    env_kwargs=None,
    train_render_mode=None,
    eval_render_mode="rgb_array_list",
    reward=None,
    use_vec_transpose_image=True,
    print_spaces=True,
    # Répertoires
    video_dir_name=None,  # Legacy argument; evaluation GIFs use TensorBoard only.
    model_dir_name="models",
    run_subdir="ppo_bounce",
    # PPO
    policy="MultiInputPolicy",
    learning_rate=3e-4,
    n_steps=15,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="auto",
    verbose=1,
    tensorboard_log=True,
    # Architecture du réseau
    pi_layers=(128, 64),
    vf_layers=(128, 64),
    normalize_images=True,
    # Entraînement
    total_timesteps=20_000,
    resume_from=None,
    seed=None,
    # Évaluation / callback
    best_model_name="ppo_bounce_best",
    last_model_name="ppo_bounce_last",
    eval_every_episodes=10,
    eval_every_timesteps=None,
    evaluation_hook=None,
    evaluate_at_end=False,
    n_eval_episodes=3,
    eval_max_steps=1024,
    deterministic_eval=True,
    save_eval_video=True,
    callback_verbose=1,
):
    """Train a new model or continue a checkpoint for additional timesteps.

    resume_from paths are relative to root. On resume, checkpoint architecture
    and algorithm hyperparameters are retained, while environments, reward,
    device, logging and evaluation settings come from this call.
    TensorBoard starts a new run with cumulative timestep values. Episode
    counters and best-score selection restart for this training session.
    Optional timestep-based evaluations override the episode schedule.
    evaluate_at_end evaluates the final updated policy and returns its metrics.
    seed controls fresh training and uses seed + 10_000 for evaluation.
    """
    root = Path(root)
    resume_path = checkpoint_path(root, resume_from) if resume_from is not None else None

    train_env = DummyVecEnv([
        make_env(
            xml_path=xml_path,
            render_mode=train_render_mode,
            env_kwargs=env_kwargs,
            reward=reward,
        )
        for _ in range(nb_vec_env_train)
    ])
    eval_env = DummyVecEnv([
        make_env(
            xml_path=xml_path,
            render_mode=eval_render_mode,
            env_kwargs=env_kwargs,
            reward=reward,
            record_video=save_eval_video,
        )
        for _ in range(nb_vec_env_test)
    ])

    if use_vec_transpose_image:
        train_env = VecTransposeImage(train_env)
        eval_env = VecTransposeImage(eval_env)

    if print_spaces:
        print_env_spaces(train_env)

    model_dir = root / model_dir_name
    run_dir = root / "runs" / run_subdir
    ensure_dirs(model_dir, run_dir)

    model = None
    try:
        policy_kwargs = dict(
            net_arch=dict(
                pi=list(pi_layers),
                vf=list(vf_layers),
            ),
            normalize_images=normalize_images,
        )

        if resume_path is not None:
            model = load_training_model(
                PPO, resume_path, train_env, device, run_dir, tensorboard_log, verbose,
            )
        else:
            model = PPO(
                policy=policy,
                env=train_env,
                learning_rate=learning_rate,
                seed=seed,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                tensorboard_log=str(run_dir) if tensorboard_log else None,
                device=device,
            )

        eval_callback = EvalVideoSaveBestCallback(
            eval_env=eval_env,
            best_model_path=model_dir / best_model_name,
            eval_every_episodes=eval_every_episodes,
            n_eval_episodes=n_eval_episodes,
            max_steps=eval_max_steps,
            deterministic=deterministic_eval,
            save_video=save_eval_video,
            verbose=callback_verbose,
            eval_every_timesteps=eval_every_timesteps,
            evaluation_hook=evaluation_hook,
            evaluate_at_end=evaluate_at_end,
            eval_seed=None if seed is None else seed + 10_000,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            reset_num_timesteps=resume_path is None,
        )

        model_path = model_dir / last_model_name
        model.save(str(model_path))
        print(f"Modèle final sauvegardé dans : {model_path}.zip")
        return eval_callback.last_result

    finally:
        train_env.close()
        eval_env.close()
        if model is not None and hasattr(model, "_logger"):
            model.logger.close()


def test(
    xml_path,
    root,
    *,
    model_path=None,
    reward=None,
    render_mode="human",
    deterministic=True,
    n_episodes=5,
    env_kwargs=None,
    max_steps=1024,
    video_dir_name="runs/ppo_bounce/test",  # TensorBoard events and JSON reports
    video_prefix="test_episode",
    save_test_video=True,
    fps=None,
    verbose=1,
):
    if n_episodes < 1 or max_steps < 1:
        raise ValueError("n_episodes and max_steps must be positive.")
    root = Path(root)
    model_path = Path(model_path) if model_path is not None else root / "models" / "ppo_bounce_last.zip"

    env = make_env(
        xml_path=xml_path,
        render_mode=render_mode,
        reward=reward,
        record_video=save_test_video,
        env_kwargs=env_kwargs,
    )()

    video_logger = None
    try:
        model = PPO.load(str(model_path))

        video_dir = root / video_dir_name
        ensure_dirs(video_dir)
        if save_test_video:
            video_logger = configure(str(video_dir), ["tensorboard"])

        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            frames = env.unwrapped.pop_render_frames() if save_test_video else None
            gripper_frames = env.unwrapped.pop_observation_frames() if save_test_video else None

            for step_idx in range(max_steps):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if save_test_video:
                    frames.extend(env.unwrapped.pop_render_frames())
                    gripper_frames.extend(env.unwrapped.pop_observation_frames())

                if terminated or truncated:
                    if verbose:
                        print(f"[Test] épisode terminé à step={step_idx + 1}")
                    break

            metrics = dict(info["task_metrics"])
            metrics["evaluation_censored"] = not terminated
            metrics_path = video_dir / f"{video_prefix}_{ep + 1:04d}.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            if verbose:
                print(f"[Task] episode={ep + 1:04d} | {metrics}")

            if save_test_video:
                current_fps = fps if fps is not None else float(getattr(env.unwrapped, "camera_fps", 30))
                log_camera_videos(video_logger, frames, gripper_frames, current_fps,
                                  ep + 1, prefix="test")
            if verbose:
                print(f"[Test] episode={ep + 1:04d} | reward={total_reward:.3f}")
    finally:
        if video_logger is not None:
            video_logger.close()
        env.close()
