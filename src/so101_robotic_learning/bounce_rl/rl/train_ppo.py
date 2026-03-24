from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from .rl_utils import (
    EvalVideoSaveBestCallback,
    ensure_dirs,
    print_env_spaces,
    save_video,
)

from so101_robotic_learning.bounce_rl.rewards.reward_ball_aligned_on_z_and_above_paddle import (
    BallAlignedOnZAndAbovePaddleReward,
)
from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv


def make_env(xml_path, render_mode=None):
    def _init():
        reward = BallAlignedOnZAndAbovePaddleReward()

        env = BounceEnv(
            xml_path=str(xml_path),
            render_mode=render_mode,
            reward=reward,
        )
        env = Monitor(env)
        return env

    return _init


def train(xml_path, root):
    root = Path(root)

    train_env = DummyVecEnv([make_env(xml_path=xml_path, render_mode=None)])
    train_env = VecTransposeImage(train_env)

    eval_env = DummyVecEnv([make_env(xml_path=xml_path, render_mode="rgb_array_list")])
    eval_env = VecTransposeImage(eval_env)

    print_env_spaces(train_env)

    video_dir = root / "videos_sb3"
    model_dir = root / "models"
    run_dir = root / "runs" / "ppo_bounce"
    ensure_dirs(video_dir, model_dir, run_dir)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        normalize_images=True,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
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
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(run_dir),
        device="auto",
    )

    eval_callback = EvalVideoSaveBestCallback(
        eval_env=eval_env,
        best_model_path=model_dir / "ppo_bounce_best",
        video_dir=video_dir,
        eval_every_episodes=10,
        n_eval_episodes=3,
        max_steps=1024,
        deterministic=True,
        save_video=True,
        verbose=1,
    )

    total_timesteps = 20_000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model_path = model_dir / "ppo_bounce_last"
    model.save(str(model_path))
    print(f"Modèle final sauvegardé dans : {model_path}.zip")

    train_env.close()
    eval_env.close()


def test(xml_path, root, model_path=None, n_episodes=5, max_steps=1024):
    root = Path(root)
    model_path = Path(model_path) if model_path is not None else root / "models" / "ppo_bounce_last.zip"

    env = make_env(xml_path=xml_path, render_mode="human")()

    model = PPO.load(str(model_path))

    video_dir = root / "videos_sb3_test"
    ensure_dirs(video_dir)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for step_idx in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                print(f"[Test] épisode terminé à step={step_idx + 1}")
                break

        frames = env.render()
        fps = int(getattr(env.unwrapped, "camera_fps", 30))
        video_path = video_dir / f"test_episode_{ep + 1:04d}.mp4"
        save_video(frames, video_path, fps=fps)

        print(
            f"[Test] episode={ep + 1:04d} | "
            f"reward={total_reward:.3f} | "
            f"video={video_path}"
        )

    env.close()