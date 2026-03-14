import sys
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path.cwd().resolve().parent.parent.parent
sys.path.append(str(ROOT / "src"))

from bounce_rl.env.bounce_env import BounceEnv


def save_video(frames, output_path, fps=30):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


class EpisodeVideoCallback(BaseCallback):
    """
    Enregistre une vidéo périodiquement pendant l'entraînement.
    Comme PPO gère lui-même la boucle d'entraînement, on fait ici
    un petit rollout d'évaluation avec la policy courante.
    """
    def __init__(self, eval_env, video_dir, video_every=10, max_steps=50, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_dir = Path(video_dir)
        self.video_every = video_every
        self.max_steps = max_steps
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Cette callback est appelée à chaque step du training.
        # On déclenche l'évaluation lorsqu'un épisode se termine.
        dones = self.locals.get("dones")
        if dones is None:
            return True

        if np.any(dones):
            self.episode_count += 1

            if self.episode_count % self.video_every == 0:
                self._record_video()

        return True

    def _record_video(self):
        self.eval_env.render_mode = "rgb_array_list"
        obs, _ = self.eval_env.reset()

        total_reward = 0.0
        start_t = time.time()

        for _ in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        frames = self.eval_env.render()
        video_path = self.video_dir / f"episode_{self.episode_count:04d}.mp4"
        save_video(frames, video_path, fps=int(getattr(self.eval_env, "camera_fps", 30)))

        if self.verbose:
            print(
                f"[Video] episode={self.episode_count:04d} | "
                f"reward={total_reward:.3f} | "
                f"time={time.time() - start_t:.3f}s | "
                f"path={video_path}"
            )

        self.eval_env.render_mode = None


def make_env(render_mode=None):
    env = BounceEnv(
        xml_path=str(ROOT / "assets" / "mjcf" / "so101_new_calib copy.xml"),
        render_mode=render_mode,
    )
    env = Monitor(env)
    return env


def train():
    train_env = make_env(render_mode=None)
    eval_env = make_env(render_mode=None)

    print("Observation space:", train_env.observation_space)
    print("Action space:", train_env.action_space)

    video_dir = ROOT / "videos_sb3"
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Si ton image est uint8 dans [0, 255], laisse normalize_images=True (par défaut).
    # Si elle est déjà float32 normalisée, remplace par normalize_images=False.
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        # normalize_images=False,  # <- à activer seulement si tes images sont déjà normalisées
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-3,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(ROOT / "runs" / "ppo_bounce"),
        device="auto",
    )

    video_callback = EpisodeVideoCallback(
        eval_env=eval_env,
        video_dir=video_dir,
        video_every=2,
        max_steps=1024,
        verbose=1,
    )

    total_timesteps = 20_000
    model.learn(total_timesteps=total_timesteps, callback=video_callback)

    model_path = model_dir / "ppo_bounce"
    model.save(str(model_path))
    print(f"Modèle sauvegardé dans : {model_path}.zip")

    train_env.close()
    eval_env.close()


def test(model_path=None, n_episodes=5, max_steps=50):
    env = make_env(render_mode="rgb_array_list")

    if model_path is None:
        model_path = ROOT / "models" / "ppo_bounce.zip"

    model = PPO.load(str(model_path))

    video_dir = ROOT / "videos_sb3_test"
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        frames = env.render()
        video_path = video_dir / f"test_episode_{ep + 1:04d}.mp4"
        save_video(frames, video_path, fps=int(getattr(env, "camera_fps", 30)))

        print(
            f"[Test] episode={ep + 1:04d} | "
            f"reward={total_reward:.3f} | "
            f"video={video_path}"
        )

    env.close()


if __name__ == "__main__":
    #train()
    test()