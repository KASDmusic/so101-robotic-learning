import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(*paths):
    return [ensure_dir(p) for p in paths]


def print_env_spaces(env, prefix=""):
    if prefix:
        prefix = f"{prefix} "
    print(f"{prefix}Observation space: {env.observation_space}")
    print(f"{prefix}Action space: {env.action_space}")


def validate_continuous_action_space(env):
    if not hasattr(env.action_space, "shape"):
        raise ValueError(
            "Cet algorithme nécessite un espace d'action continu "
            "(gymnasium.spaces.Box)."
        )


def save_video(frames, output_path, fps=30):
    if frames is None or len(frames) == 0:
        print(f"[Video] aucune frame à sauvegarder pour {output_path}")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    safe_frames = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        safe_frames.append(arr)

    imageio.mimsave(output_path, safe_frames, fps=fps)


def unwrap_first_env(env):
    """
    Récupère le premier environnement gym sous-jacent depuis une pile VecEnv/wrappers.
    Compatible avec DummyVecEnv, VecTransposeImage, VecNormalize, etc.
    """
    current = env
    visited = set()

    while True:
        if id(current) in visited:
            break
        visited.add(id(current))

        if hasattr(current, "envs") and len(current.envs) > 0:
            return current.envs[0]

        if hasattr(current, "venv"):
            current = current.venv
            continue

        if hasattr(current, "env"):
            current = current.env
            continue

        break

    return current


def get_camera_fps(env, default=30):
    base_env = unwrap_first_env(env)
    return int(getattr(base_env.unwrapped, "camera_fps", default))


def render_vecenv_frames(env):
    """
    Tente de récupérer les frames depuis le vrai env gym sous-jacent.
    """
    base_env = unwrap_first_env(env)
    return base_env.render()

def render_vecenv_frame(env):
    """
    Récupère une frame RGB depuis le vrai env gym sous-jacent.
    Retourne un ndarray HxWxC ou None.
    """
    base_env = unwrap_first_env(env)
    rendered = base_env.render()

    if rendered is None:
        return None

    # Cas où render() retourne une liste de frames
    if isinstance(rendered, list):
        if len(rendered) == 0:
            return None
        return rendered[-1]

    return rendered

def rollout_policy(model, env, max_steps=1024, deterministic=True, capture_video=False):
    obs = env.reset()
    total_reward = 0.0
    start_t = time.time()

    done = False
    step_idx = 0
    frames = [] if capture_video else None

    for step_idx in range(max_steps):
        if capture_video:
            frame = render_vecenv_frame(env)
            if frame is not None:
                frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)

        total_reward += float(rewards[0])
        done = bool(dones[0])

        if done:
            break

    return {
        "total_reward": total_reward,
        "episode_length": step_idx + 1,
        "frames": frames,
        "done": done,
        "elapsed_time": time.time() - start_t,
    }


def evaluate_policy_on_env(
    model,
    env,
    n_eval_episodes=3,
    max_steps=1024,
    deterministic=True,
    record_video_first_episode=False,
):
    """
    Évalue un modèle sur plusieurs épisodes d'un VecEnv.

    Retourne :
    - mean_reward
    - std_reward
    - rewards
    - video_frames (frames du 1er épisode si demandé)
    - first_episode_length
    - first_episode_time
    """
    rewards = []
    video_frames = None
    first_episode_length = None
    first_episode_time = None

    for ep in range(n_eval_episodes):
        result = rollout_policy(
            model=model,
            env=env,
            max_steps=max_steps,
            deterministic=deterministic,
            capture_video=(record_video_first_episode and ep == 0),
        )

        rewards.append(result["total_reward"])

        if ep == 0:
            first_episode_length = result["episode_length"]
            first_episode_time = result["elapsed_time"]
            if record_video_first_episode:
                video_frames = result["frames"]

    mean_reward = float(np.mean(rewards)) if rewards else float("-inf")
    std_reward = float(np.std(rewards)) if rewards else 0.0

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "rewards": rewards,
        "video_frames": video_frames,
        "first_episode_length": first_episode_length,
        "first_episode_time": first_episode_time,
    }


class EvalVideoSaveBestCallback(BaseCallback):
    """
    Callback unifié compatible VecEnv qui, tous les `eval_every_episodes`
    épisodes terminés :
    - évalue le modèle sur `n_eval_episodes`
    - enregistre une vidéo du 1er épisode d'évaluation (optionnel)
    - sauvegarde le modèle si le score moyen est meilleur que le précédent best
    """

    def __init__(
        self,
        eval_env,
        best_model_path,
        video_dir=None,
        eval_every_episodes=10,
        n_eval_episodes=3,
        max_steps=1024,
        deterministic=True,
        save_video=True,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_path = Path(best_model_path)
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.eval_every_episodes = eval_every_episodes
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.save_video = save_video

        self.episode_count = 0
        self.best_mean_reward = -np.inf

        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True

        n_done = int(np.sum(dones))
        if n_done > 0:
            self.episode_count += n_done

            if self.verbose:
                print(
                    f"[Callback] épisode(s) terminé(s) détecté(s) | "
                    f"+{n_done} | count={self.episode_count}"
                )

            if self.episode_count % self.eval_every_episodes == 0:
                self._evaluate_record_and_save()

        return True

    def _evaluate_record_and_save(self):
        if self.verbose:
            print(
                f"[Eval] lancement évaluation | "
                f"episode={self.episode_count} | "
                f"n_eval_episodes={self.n_eval_episodes}"
            )

        eval_result = evaluate_policy_on_env(
            model=self.model,
            env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            max_steps=self.max_steps,
            deterministic=self.deterministic,
            record_video_first_episode=self.save_video,
        )

        mean_reward = eval_result["mean_reward"]
        std_reward = eval_result["std_reward"]
        rewards = eval_result["rewards"]
        video_frames = eval_result["video_frames"]
        first_episode_length = eval_result["first_episode_length"]
        first_episode_time = eval_result["first_episode_time"]

        if self.verbose:
            print(
                f"[Eval] episode={self.episode_count:04d} | "
                f"mean_reward={mean_reward:.3f} | "
                f"std_reward={std_reward:.3f} | "
                f"rewards={rewards}"
            )

        if self.save_video and self.video_dir is not None:
            if video_frames is None:
                print("[Video] aucune frame retournée pour la vidéo d'évaluation")
            else:
                fps = get_camera_fps(self.eval_env, default=30)
                video_path = self.video_dir / f"episode_{self.episode_count:04d}.mp4"
                save_video(video_frames, video_path, fps=fps)

                if self.verbose:
                    print(
                        f"[Video] episode={self.episode_count:04d} | "
                        f"frames={len(video_frames)} | "
                        f"rollout_steps={first_episode_length} | "
                        f"time={first_episode_time:.3f}s | "
                        f"path={video_path}"
                    )

        if mean_reward > self.best_mean_reward:
            old_best = self.best_mean_reward
            self.best_mean_reward = mean_reward
            self.model.save(str(self.best_model_path))

            if self.verbose:
                print(
                    f"[BestModel] nouveau meilleur score : "
                    f"{old_best:.3f} -> {self.best_mean_reward:.3f} | "
                    f"modèle sauvegardé dans {self.best_model_path}"
                )