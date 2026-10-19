import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from tensorboard.compat.proto.summary_pb2 import Summary

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


def combine_camera_frames(overview, gripper):
    """Pair corresponding camera frames into one labeled, aspect-preserving view."""
    if overview is None or gripper is None or len(overview) == 0 or len(gripper) == 0:
        return []
    if len(overview) != len(gripper):
        raise ValueError("Camera streams must contain the same number of frames.")
    frames = []
    for external_frame, gripper_frame in zip(overview, gripper):
        external = Image.fromarray(external_frame)
        policy = Image.fromarray(gripper_frame)
        height = external.height
        policy = policy.resize(
            (max(1, round(policy.width * height / policy.height)), height),
            resample=Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (external.width + policy.width, height + 20))
        canvas.paste(external, (0, 20))
        canvas.paste(policy, (external.width, 20))
        labels = ImageDraw.Draw(canvas)
        labels.text((4, 3), "Overview", fill="white")
        labels.text((external.width + 4, 3), "Gripper", fill="white")
        frames.append(np.asarray(canvas))
    return frames


def log_tensorboard_video(logger, frames, fps, step, tag="eval/video"):
    """Write an animated GIF to TensorBoard's Images tab, entirely in memory.

    The native video summary is an encoded GIF. Encoding it here avoids the
    optional MoviePy dependency and locked temporary files on Windows.
    """
    outputs = [output for output in getattr(logger, "output_formats", ())
               if isinstance(output, TensorBoardOutputFormat)]
    if not outputs or frames is None or len(frames) == 0:
        return False
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video fps must be finite and positive.")
    images = [Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)) for frame in frames]
    # GIF stores centiseconds. Alternate durations to preserve e.g. 30 FPS.
    ticks = np.rint(np.arange(len(images) + 1) * 100 / fps).astype(int)
    durations = np.maximum(np.diff(ticks), 1) * 10
    with BytesIO() as buffer:
        images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:],
                       duration=durations.tolist(), loop=0, disposal=2)
        summary = Summary(value=[Summary.Value(tag=tag, image=Summary.Image(
            height=images[0].height, width=images[0].width, colorspace=3,
            encoded_image_string=buffer.getvalue(),
        ))])
    for output in outputs:
        output.writer.file_writer.add_summary(summary, global_step=step)
        output.writer.flush()
    return True


def log_camera_videos(logger, frames, gripper_frames, fps, step, prefix="eval"):
    """Store synchronized camera GIFs directly in TensorBoard event files."""
    log_tensorboard_video(logger, frames, fps, step, tag=f"{prefix}/video")
    log_tensorboard_video(logger, gripper_frames, fps, step, tag=f"{prefix}/gripper_cam")
    if any(isinstance(output, TensorBoardOutputFormat)
           for output in getattr(logger, "output_formats", ())):
        log_tensorboard_video(logger, combine_camera_frames(frames, gripper_frames),
                              fps, step, tag=f"{prefix}/cameras")


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
    if hasattr(env, "env_method"):
        return float(env.env_method("get_wrapper_attr", "camera_fps", indices=0)[0])
    return float(getattr(env.unwrapped, "camera_fps", default))


def render_vecenv_frames(env):
    """Drain every camera frame from environment zero, including through Vec wrappers."""
    return env.env_method("pop_render_frames", indices=0)[0]


def render_vecenv_observation_frames(env):
    """Optional second video stream; other Gym environments may not provide it."""
    try:
        return env.env_method("pop_observation_frames", indices=0)[0]
    except AttributeError:
        return []


def render_vecenv_frame(env):
    frames = render_vecenv_frames(env)
    return frames[-1] if frames else None


def rollout_policy(model, env, max_steps=1024, deterministic=True, capture_video=False):
    result = evaluate_policy_on_env(
        model, env, n_eval_episodes=1, max_steps=max_steps,
        deterministic=deterministic, record_video_first_episode=capture_video,
    )
    return {
        "total_reward": result["rewards"][0],
        "episode_length": result["first_episode_length"],
        "frames": result["video_frames"],
        "done": result["episode_dones"][0],
        "elapsed_time": result["first_episode_time"],
    }


def evaluate_policy_on_env(
    model,
    env,
    n_eval_episodes=3,
    max_steps=1024,
    deterministic=True,
    record_video_first_episode=False,
):
    """Evaluate a policy, restoring video capture budgets even on failure."""
    try:
        return _evaluate_policy_on_env(
            model, env, n_eval_episodes, max_steps, deterministic,
            record_video_first_episode,
        )
    finally:
        _set_video_capture_limit(env, None)


def _set_video_capture_limit(env, episodes, indices=None):
    try:
        env.env_method("set_video_capture_limit", episodes, indices=indices)
    except AttributeError:
        # Generic Gym environments need not implement optional camera control.
        pass


def _evaluate_policy_on_env(
    model,
    env,
    n_eval_episodes=3,
    max_steps=1024,
    deterministic=True,
    record_video_first_episode=False,
):
    """Evaluate exactly n episodes in batches across all vector environments.

    Each batch uses at most one episode per worker. Faster workers' auto-reset
    episodes are ignored until the batch finishes, avoiding short-episode bias.
    max_steps caps each recorded episode, including environments that never end.
    The video is the first episode of worker zero, with its terminal frames.
    """
    if n_eval_episodes < 1 or max_steps < 1:
        raise ValueError("n_eval_episodes and max_steps must be positive.")
    rewards = []
    episode_dones = []
    episode_metrics = []
    video_frames = [] if record_video_first_episode else None
    gripper_video_frames = [] if record_video_first_episode else None
    first_episode_length = None
    first_episode_time = None

    while len(rewards) < n_eval_episodes:
        recording = record_video_first_episode and not rewards
        _set_video_capture_limit(env, 0)
        if recording:
            _set_video_capture_limit(env, 1, indices=0)
        obs = env.reset()
        batch_size = min(env.num_envs, n_eval_episodes - len(rewards))
        totals = np.zeros(batch_size, dtype=np.float64)
        active = np.ones(batch_size, dtype=bool)
        batch_dones = np.zeros(batch_size, dtype=bool)
        batch_metrics = [None] * batch_size
        start_t = time.perf_counter()
        if recording:
            video_frames.extend(render_vecenv_frames(env))
            gripper_video_frames.extend(render_vecenv_observation_frames(env))

        for step_idx in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, step_rewards, dones, infos = env.step(action)
            totals[active] += np.asarray(step_rewards)[:batch_size][active]
            if recording and active[0]:
                if dones[0]:
                    # The live buffer now belongs to the auto-reset episode.
                    video_frames.extend(infos[0].get("terminal_render_frames", []))
                    gripper_video_frames.extend(infos[0].get("terminal_observation_frames", []))
                else:
                    video_frames.extend(render_vecenv_frames(env))
                    gripper_video_frames.extend(render_vecenv_observation_frames(env))
            for i in range(batch_size):
                if active[i] and (dones[i] or step_idx + 1 == max_steps):
                    active[i] = False
                    batch_dones[i] = dones[i]
                    if "task_metrics" in infos[i]:
                        batch_metrics[i] = dict(infos[i]["task_metrics"])
                        batch_metrics[i]["evaluation_censored"] = (
                            not bool(dones[i]) or bool(infos[i].get("TimeLimit.truncated", False))
                        )
                    if i == 0 and first_episode_length is None:
                        first_episode_length = step_idx + 1
                        first_episode_time = time.perf_counter() - start_t
            if not active.any():
                break

        rewards.extend(totals.tolist())
        episode_dones.extend(batch_dones.tolist())
        episode_metrics.extend(batch_metrics)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "rewards": rewards,
        "episode_dones": episode_dones,
        "episode_metrics": episode_metrics,
        "task_metrics": summarize_task_metrics(episode_metrics),
        "video_frames": video_frames,
        "gripper_video_frames": gripper_video_frames,
        "first_episode_length": first_episode_length,
        "first_episode_time": first_episode_time,
    }


def summarize_task_metrics(episodes):
    """Aggregate task metrics; missing/censored falls are never treated as zero."""
    measured = [episode for episode in episodes if episode is not None]
    if not measured:
        return {}

    def mean(key):
        values = [episode[key] for episode in measured if episode.get(key) is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "success_rate": mean("success"),
        "mean_valid_bounces": mean("valid_bounces"),
        "mean_max_consecutive_bounces": mean("max_consecutive_bounces"),
        "mean_duration_s": mean("elapsed_sim_time_s"),
        "mean_time_to_fall_s": mean("time_to_fall_s"),
        "fall_rate": mean("fell"),
        "paddle_ground_contact_rate": mean("paddle_ground_contact"),
        "ball_too_far_rate": mean("ball_too_far"),
        "mean_mechanical_energy_j": mean("mechanical_energy_j"),
        "mean_energy_penalty": mean("energy_penalty"),
        "mean_ball_segment_distance_m": mean("mean_ball_segment_distance_m"),
        "mean_ball_segment_penalty": mean("ball_segment_penalty"),
        "mean_contact_dispersion_m": mean("contact_dispersion_m"),
    }
    component_names = sorted({name for episode in measured
                              for name in episode.get("reward_components", {})})
    for name in component_names:
        summary[f"reward_components/{name}"] = float(np.mean([
            episode.get("reward_components", {}).get(name, 0.0) for episode in measured
        ]))
    return summary


def evaluation_score(result):
    """Prefer actual juggling success; reward is a fallback for other tasks."""
    metrics = result["task_metrics"]
    if metrics:
        return (
            metrics["success_rate"], metrics["mean_max_consecutive_bounces"],
            metrics["mean_valid_bounces"], metrics["mean_duration_s"],
        )
    return (result["mean_reward"],)


class EvalVideoSaveBestCallback(BaseCallback):
    """
    Callback unifié compatible VecEnv qui, tous les `eval_every_episodes`
    épisodes terminés :
    - évalue le modèle sur `n_eval_episodes`
    - enregistre les GIF du 1er épisode dans TensorBoard (optionnel)
    - sauvegarde le modèle selon les métriques de jonglage (reward en fallback)
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
        # video_dir is accepted for compatibility; GIFs live in TensorBoard.
        if eval_every_episodes < 1:
            raise ValueError("eval_every_episodes must be positive.")
        self.eval_every_episodes = eval_every_episodes
        self.next_eval_episode = eval_every_episodes
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.save_video = save_video

        self.episode_count = 0
        self.best_mean_reward = -np.inf
        self.best_task_score = None

        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

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

            if self.episode_count >= self.next_eval_episode:
                self._evaluate_record_and_save()
                # Evaluate once even when a vector step crosses several thresholds.
                self.next_eval_episode = (
                    self.episode_count // self.eval_every_episodes + 1
                ) * self.eval_every_episodes

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
        metrics = eval_result["task_metrics"]
        for name, value in metrics.items():
            if value is not None:
                self.logger.record(f"eval/{name}", value)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        if self.save_video:
            log_camera_videos(
                self.logger, video_frames, eval_result.get("gripper_video_frames"),
                get_camera_fps(self.eval_env), self.num_timesteps,
            )
        # record() only buffers values. Flush each evaluation before another one
        # overwrites it, independently of PPO/SAC's training log interval.
        self.logger.dump(self.num_timesteps)
        report = {
            "training_episodes": self.episode_count,
            "training_timesteps": self.num_timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "rewards": rewards,
            "task_metrics": metrics,
            "episodes": eval_result["episode_metrics"],
            "selection_score": list(evaluation_score(eval_result)),
        }
        report_path = self.best_model_path.parent / f"{self.best_model_path.stem}_eval_{self.episode_count:06d}.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

        if self.verbose:
            print(
                f"[Eval] episode={self.episode_count:04d} | "
                f"mean_reward={mean_reward:.3f} | "
                f"std_reward={std_reward:.3f} | "
                f"rewards={rewards}"
            )
            if metrics:
                print(f"[Task] {metrics}")

        score = evaluation_score(eval_result)
        if self.best_task_score is None or score > self.best_task_score:
            old_best = self.best_mean_reward
            self.best_task_score = score
            self.best_mean_reward = mean_reward
            self.model.save(str(self.best_model_path))

            if self.verbose:
                print(
                    f"[BestModel] nouveau meilleur score de tâche : {score} | "
                    f"reward {old_best:.3f} -> {self.best_mean_reward:.3f} | "
                    f"modèle sauvegardé dans {self.best_model_path}"
                )
