from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .rl_utils import (
    EvalVideoSaveBestCallback,
    add_src_to_path,
    ensure_dirs,
    get_root_from_cwd,
    print_env_spaces,
    save_video,
)

ROOT = get_root_from_cwd(levels_up=3)
add_src_to_path(ROOT)

from bounce_rl.env.bounce_env import BounceEnv
from bounce_rl.rewards.reward_ball_aligned_on_z_and_above_paddle import (
    BallAlignedOnZAndAbovePaddleReward,
)


def make_env(xml_path, render_mode=None):
    reward = BallAlignedOnZAndAbovePaddleReward()

    env = BounceEnv(
        xml_path=str(xml_path),
        render_mode=render_mode,
        reward=reward,
    )
    env = Monitor(env)
    return env


def train(xml_path=ROOT / "assets" / "mjcf" / "so101_new_calib copy.xml"):
    train_env = make_env(xml_path=xml_path, render_mode=None)
    eval_env = make_env(xml_path=xml_path, render_mode="rgb_array_list")

    print_env_spaces(train_env)

    video_dir = ROOT / "videos_sb3"
    model_dir = ROOT / "models"
    run_dir = ROOT / "runs" / "ppo_bounce"
    ensure_dirs(video_dir, model_dir, run_dir)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        normalize_images=False,
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


def test(xml_path=ROOT / "assets" / "mjcf" / "so101_new_calib copy.xml", model_path=None, n_episodes=5, max_steps=1024):
    env = make_env(xml_path=xml_path, render_mode="human")

    if model_path is None:
        model_path = ROOT / "models" / "ppo_bounce_best.zip"

    model = PPO.load(str(model_path))

    video_dir = ROOT / "videos_sb3_test"
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


if __name__ == "__main__":
    train()
    # test()