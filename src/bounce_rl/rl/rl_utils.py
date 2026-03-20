import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


def get_root_from_cwd(levels_up=3) -> Path:
    """
    Remonte de `levels_up` dossiers depuis le cwd courant.
    """
    root = Path.cwd().resolve()
    for _ in range(levels_up):
        root = root.parent
    return root


def add_src_to_path(root: Path, src_dirname="src") -> Path:
    """
    Ajoute le dossier <root>/<src_dirname> au PYTHONPATH si nécessaire.
    """
    src_path = root / src_dirname
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.append(src_str)
    return src_path


def ensure_dir(path) -> Path:
    """
    Crée le dossier s'il n'existe pas et retourne un objet Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(*paths):
    """
    Crée plusieurs dossiers et retourne la liste des Path.
    """
    return [ensure_dir(p) for p in paths]


def print_env_spaces(env, prefix=""):
    """
    Affiche les espaces d'observation et d'action.
    """
    if prefix:
        prefix = f"{prefix} "
    print(f"{prefix}Observation space: {env.observation_space}")
    print(f"{prefix}Action space: {env.action_space}")


def validate_continuous_action_space(env):
    """
    Vérification simple pour les algos nécessitant un espace d'action continu.
    """
    if not hasattr(env.action_space, "shape"):
        raise ValueError(
            "Cet algorithme nécessite un espace d'action continu "
            "(gymnasium.spaces.Box)."
        )


def save_video(frames, output_path, fps=30):
    """
    Sauvegarde une séquence de frames en vidéo mp4.
    """
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


def evaluate_policy_on_env(model, env, n_eval_episodes=3, max_steps=1024, deterministic=True):
    """
    Évalue un modèle sur `n_eval_episodes` et retourne :
    - mean_reward
    - std_reward
    - rewards_per_episode
    """
    rewards = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                break

        rewards.append(total_reward)

    mean_reward = float(np.mean(rewards)) if rewards else float("-inf")
    std_reward = float(np.std(rewards)) if rewards else 0.0
    return mean_reward, std_reward, rewards


class EpisodeVideoCallback(BaseCallback):
    """
    Enregistre une vidéo de l'agent courant tous les `video_every` épisodes
    terminés pendant l'entraînement.
    """

    def __init__(self, eval_env, video_dir, video_every=10, max_steps=1024, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_dir = Path(video_dir)
        self.video_every = video_every
        self.max_steps = max_steps
        self.episode_count = 0

        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")

        if dones is None:
            return True

        if np.any(dones):
            self.episode_count += 1

            if self.verbose:
                print(f"[Callback] épisode terminé détecté | count={self.episode_count}")

            if self.episode_count % self.video_every == 0:
                self._record_video()

        return True

    def _record_video(self):
        if self.verbose:
            print(f"[Video] enregistrement de la vidéo pour l'épisode {self.episode_count}")

        obs, _ = self.eval_env.reset()
        total_reward = 0.0
        start_t = time.time()

        for step_idx in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                if self.verbose:
                    print(f"[Video] rollout d'éval terminé à step={step_idx + 1}")
                break

        frames = self.eval_env.render()

        if frames is None:
            print("[Video] render() a renvoyé None")
            return

        if self.verbose:
            print(f"[Video] nombre de frames capturées : {len(frames)}")

        fps = int(getattr(self.eval_env.unwrapped, "camera_fps", 30))
        video_path = self.video_dir / f"episode_{self.episode_count:04d}.mp4"
        save_video(frames, video_path, fps=fps)

        if self.verbose:
            print(
                f"[Video] episode={self.episode_count:04d} | "
                f"reward={total_reward:.3f} | "
                f"time={time.time() - start_t:.3f}s | "
                f"path={video_path}"
            )


class SaveBestModelCallback(BaseCallback):
    """
    Évalue périodiquement le modèle et le sauvegarde si le meilleur score
    moyen est battu.

    Le score utilisé est la reward moyenne sur `n_eval_episodes`.
    """

    def __init__(
        self,
        eval_env,
        save_path,
        eval_every_episodes=10,
        n_eval_episodes=3,
        max_steps=1024,
        deterministic=True,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = Path(save_path)
        self.eval_every_episodes = eval_every_episodes
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.deterministic = deterministic

        self.episode_count = 0
        self.best_mean_reward = -np.inf

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")

        if dones is None:
            return True

        if np.any(dones):
            self.episode_count += 1

            if self.episode_count % self.eval_every_episodes == 0:
                self._evaluate_and_maybe_save()

        return True

    def _evaluate_and_maybe_save(self):
        mean_reward, std_reward, rewards = evaluate_policy_on_env(
            model=self.model,
            env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            max_steps=self.max_steps,
            deterministic=self.deterministic,
        )

        if self.verbose:
            print(
                f"[BestModel] episode={self.episode_count} | "
                f"eval_mean_reward={mean_reward:.3f} | "
                f"eval_std={std_reward:.3f} | "
                f"best_so_far={self.best_mean_reward:.3f}"
            )

        if mean_reward > self.best_mean_reward:
            old_best = self.best_mean_reward
            self.best_mean_reward = mean_reward
            self.model.save(str(self.save_path))

            if self.verbose:
                print(
                    f"[BestModel] nouveau meilleur score : "
                    f"{old_best:.3f} -> {self.best_mean_reward:.3f} | "
                    f"modèle sauvegardé dans {self.save_path}"
                )


def build_default_callbacks(*callbacks):
    """
    Petit helper pour composer plusieurs callbacks SB3.
    """
    return CallbackList(list(callbacks))