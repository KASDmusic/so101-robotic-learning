import time
from copy import deepcopy
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

from .bounce_events import BounceConfig, BounceTracker
from ..rewards.reward_utils import distance_to_vertical_segment
from ..rewards.reward_multi_objectif import (
    RewardContext, RewardMultiObjectif, make_default_reward,
)

class BounceEnv(gym.Env):
    """
    Gym env qui renvoie une observation Dict:
      - obs["image"]: concaténation des images RGB uint8 (H, W, 3 * (K+1))
                      avec image courante + images passées demandées
      - obs["state"]: concaténation des états float32 ((K+1) * (nq+nv),)
                      avec état courant + états passés demandés

    reward:
      Objectif complet, copié pour chaque environnement. None crée le composite
      de jonglage par défaut. Aucune pondération ou pénalité implicite n'est
      ajoutée à un objectif fourni. Les poids se règlent dans le composite.

    latent_steps:
      tableau numpy d'entiers positifs.
      Chaque entier k signifie: prendre l'observation du step passé t-k.

      Exemple:
        latent_steps = np.array([1, 2, 10])

      => l'observation contient:
         [présent, t-1, t-2, t-10]

    record_video:
      Collecte les images caméra, y compris avec le viewer human.
      pop_render_frames() récupère et vide le buffer.

    video_camera_name:
      Caméra de visualisation uniquement (render et vidéos), jamais dans obs.
      None réutilise camera_name. Le rendu additionnel est créé à la demande.

    render_mode:
      - None: aucun rendu explicite
      - "rgb_array": render() renvoie la dernière image caméra non concaténée
      - "rgb_array_list": render() renvoie la liste des images caméra accumulées depuis reset
      - "human": ouvre une fenêtre viewer MuJoCo native
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "rgb_array_list"],
        "render_fps": 30,
    }

    def __init__(
        self,
        xml_path: str,
        reward=None,
        max_episode_steps: int = 1024,
        camera_name: str = "gripper_cam",
        width: int = 84,
        height: int = 84,
        mujoco_step_periode: float = 0.002,
        frame_skip: int = 20,
        camera_fps: int = 30,
        paddle_color=(0.9, 0.05, 0.05, 1),
        ball_mass: float = 0.003,
        latent_steps: np.ndarray | None = None,
        render_mode: str | None = None,
        record_video: bool = False,
        bounce_config: BounceConfig | None = None,
        video_camera_name: str | None = None,
        video_width: int | None = None,
        video_height: int | None = None,
        render_profile: str = "standard",
        video_render_profile: str = "standard",
        ball_segment_start_height: float = 0.2,
        ball_segment_length: float = 1.0,
    ):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode={render_mode!r} invalide. "
                f"Modes supportés: {self.metadata['render_modes']}"
            )

        self.xml_path = xml_path
        objective = make_default_reward() if reward is None else deepcopy(reward)
        self.reward = (objective if isinstance(objective, RewardMultiObjectif)
                       else RewardMultiObjectif().add_reward("custom", objective))
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self.record_video = bool(record_video)
        if render_profile not in ("standard", "train_fast") or video_render_profile not in ("standard", "train_fast"):
            raise ValueError("Render profiles must be 'standard' or 'train_fast'.")
        self.render_profile = render_profile
        self.video_render_profile = video_render_profile
        self._capture_episodes_remaining = None
        self._capture_this_episode = True
        self._last_observation_time = None
        self._episode_mechanical_energy_j = 0.0
        if not np.isfinite(ball_segment_start_height) or ball_segment_start_height < 0:
            raise ValueError("ball_segment_start_height must be finite and nonnegative.")
        if not np.isfinite(ball_segment_length) or ball_segment_length <= 0:
            raise ValueError("ball_segment_length must be finite and positive.")
        self.ball_segment_start_height = float(ball_segment_start_height)
        self.ball_segment_length = float(ball_segment_length)
        self._episode_segment_distance_integral_m_s = 0.0
        self._episode_energy_penalty = 0.0
        self._episode_ball_segment_penalty = 0.0
        self._episode_reward_components = {}
        self.camera_name = camera_name
        self.video_camera_name = video_camera_name or camera_name
        self.video_width = int(video_width) if video_width is not None else width
        self.video_height = int(video_height) if video_height is not None else height
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("Video dimensions must be positive.")
        self.width = width
        self.height = height
        self.frame_skip = int(frame_skip)
        self.mujoco_step_periode = float(mujoco_step_periode)
        if self.frame_skip < 1 or not np.isfinite(self.mujoco_step_periode) or self.mujoco_step_periode <= 0:
            raise ValueError("frame_skip and mujoco_step_periode must be positive.")

        self.step_count = 0

        # Steps passés à inclure dans l'observation
        if latent_steps is None:
            latent_steps = np.array([], dtype=np.int32)

        self.latent_steps = np.asarray(latent_steps, dtype=np.int32).reshape(-1)

        if self.latent_steps.size > 0:
            if np.any(self.latent_steps <= 0):
                raise ValueError("latent_steps doit contenir uniquement des entiers strictement positifs.")
            if len(np.unique(self.latent_steps)) != len(self.latent_steps):
                raise ValueError("latent_steps ne doit pas contenir de doublons.")

        # On conserve l'ordre donné par l'utilisateur
        self.num_frames = len(self.latent_steps) + 1
        self.max_history = int(self.latent_steps.max()) if len(self.latent_steps) > 0 else 0

        # Caméra RGB throttlée sur temps simulé
        self.camera_fps = float(camera_fps)
        if not np.isfinite(self.camera_fps) or self.camera_fps <= 0:
            raise ValueError("camera_fps must be finite and positive.")
        if self.render_profile == "train_fast":
            self.camera_fps = min(self.camera_fps, 1.0 / (self.frame_skip * self.mujoco_step_periode))
        self.camera_period = 1.0 / self.camera_fps
        self.next_frame_time = 0.0

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self._standard_offsamples = int(self.model.vis.quality.offsamples)
        self.model.opt.timestep = mujoco_step_periode
        self.data = mujoco.MjData(self.model)
        self.bounce_tracker = BounceTracker(self.model, bounce_config)

        # Vérifie que la caméra existe
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            raise ValueError(f'Camera "{camera_name}" introuvable dans le XML.')
        video_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.video_camera_name)
        if video_cam_id == -1:
            raise ValueError(f'Video camera "{self.video_camera_name}" introuvable dans le XML.')

        # Action space: commandes normalisées [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Dimensions d'état / image concaténés
        self.state_dim = self.model.nq + self.model.nv
        image_channels = 3 * self.num_frames
        stacked_state_dim = self.state_dim * self.num_frames

        # Observation space: Dict(image, state)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, image_channels),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(stacked_state_dim,),
                    dtype=np.float32,
                ),
            }
        )

        # Renderer offscreen pour rgb_array / rgb_array_list / obs["image"]
        self.renderer = self._create_renderer(width, height, self.render_profile)
        self.video_renderer = None

        # Viewer interactif pour human
        self.viewer = None

        # Paramètres dynamiques optionnels
        self._set_paddle_color(paddle_color)
        self._set_ball_mass(ball_mass)

        # Dernière image rendue (non concaténée)
        self._last_image = np.zeros((height, width, 3), dtype=np.uint8)
        self._last_video_image = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)

        # Historiques de snapshots par step env
        # On stocke les observations non concaténées
        self._state_history = deque(maxlen=self.max_history + 1 if self.max_history > 0 else 1)
        self._image_history = deque(maxlen=self.max_history + 1 if self.max_history > 0 else 1)

        # Buffer des frames pour render_mode="rgb_array_list"
        self._rendered_frames = []
        self._observation_frames = []

    # -------------------------
    # Paramètres dynamiques
    # -------------------------
    def _set_paddle_color(self, color):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_col")
        if geom_id != -1:
            self.model.geom_rgba[geom_id] = np.array(color, dtype=np.float32)

    def _set_ball_mass(self, mass):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        if geom_id != -1:
            body_id = self.model.geom_bodyid[geom_id]
            self.model.body_mass[body_id] = float(mass)

    # -------------------------
    # Viewer human
    # -------------------------
    def _ensure_human_viewer(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _sync_human_viewer(self):
        if self.render_mode != "human":
            return

        self._ensure_human_viewer()

        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                self.viewer = None

    # -------------------------
    # Observations / rendu caméra
    # -------------------------
    def _create_renderer(self, width, height, profile):
        # Samples belong to each OpenGL framebuffer. Restore model settings so
        # a standard external renderer can coexist with a fast policy renderer.
        previous = self.model.vis.quality.offsamples
        self.model.vis.quality.offsamples = 0 if profile == "train_fast" else self._standard_offsamples
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
        finally:
            self.model.vis.quality.offsamples = previous
        if profile == "train_fast":
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        return renderer

    def set_video_capture_limit(self, episodes=None):
        """Allow only the next N resets to capture video; None restores normal use.

        Disabling also takes effect immediately for a capped evaluation episode.
        Automatic VecEnv resets consume the budget just like explicit resets.
        """
        if episodes is not None and (not isinstance(episodes, int) or episodes < 0):
            raise ValueError("Capture limit must be a nonnegative integer or None.")
        self._capture_episodes_remaining = episodes
        if episodes == 0:
            self._capture_this_episode = False

    def _store_frame_if_needed(self, frame: np.ndarray):
        if self._capture_this_episode and (self.render_mode == "rgb_array_list" or self.record_video):
            self._rendered_frames.append(frame.copy())
            # Reuse the unstacked policy camera pixels, with exactly the same
            # timestamps as the external view. No extra render is necessary.
            self._observation_frames.append(self._last_image.copy())

    def _maybe_update_camera(self):
        """
        En profil standard, actualise les pixels à camera_fps simulé.
        En profil train_fast, actualise l'observation à chaque décision et
        réutilise ces pixels pour la vidéo, limitée à camera_fps.
        """
        fast = self.render_profile == "train_fast"
        frame_due = self.data.time + 1e-12 >= self.next_frame_time
        if not fast and not frame_due:
            return
        # Fast mode is called only at reset and decision boundaries. Repeated
        # render() calls at the same simulation time reuse the observation.
        if not fast or self._last_observation_time != self.data.time:
            self.renderer.update_scene(self.data, camera=self.camera_name)
            self._last_image = self.renderer.render().copy()
            self._last_observation_time = float(self.data.time)
        if not frame_due:
            return

        # Observation pixels always come from camera_name. Render the external
        # view only when a visualization or recording is requested.
        if self._capture_this_episode and (self.record_video or self.render_mode in ("rgb_array", "rgb_array_list")):
            if (self.video_camera_name == self.camera_name
                    and self.video_width == self.width and self.video_height == self.height
                    and self.render_profile == self.video_render_profile):
                self._last_video_image = self._last_image.copy()
            else:
                if self.video_renderer is None:
                    self.video_renderer = self._create_renderer(
                        self.video_width, self.video_height, self.video_render_profile,
                    )
                self.video_renderer.update_scene(self.data, camera=self.video_camera_name)
                self._last_video_image = self.video_renderer.render().copy()
            self._store_frame_if_needed(self._last_video_image)

        # avancer le prochain créneau caméra
        self.next_frame_time += self.camera_period

        # sécurité si gros saut de temps simulé
        if self.data.time >= self.next_frame_time + 5 * self.camera_period:
            self.next_frame_time = self.data.time + self.camera_period

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _task_metrics(self):
        metrics = self.bounce_tracker.metrics(self.data)
        elapsed = metrics["elapsed_sim_time_s"]
        metrics.update({
            "mechanical_energy_j": self._episode_mechanical_energy_j,
            "energy_penalty": self._episode_energy_penalty,
            "segment_distance_integral_m_s": self._episode_segment_distance_integral_m_s,
            "mean_ball_segment_distance_m": (
                self._episode_segment_distance_integral_m_s / elapsed if elapsed > 0 else None
            ),
            "ball_segment_penalty": self._episode_ball_segment_penalty,
            "reward_components": dict(self._episode_reward_components),
        })
        return metrics

    def _get_past_state(self, lag: int) -> np.ndarray:
        # lag=1 => step précédent, donc élément [-1]
        # lag=2 => élément [-2], etc.
        if lag <= len(self._state_history):
            return self._state_history[-lag].copy()
        return self._state_history[0].copy()

    def _get_past_image(self, lag: int) -> np.ndarray:
        if lag <= len(self._image_history):
            return self._image_history[-lag].copy()
        return self._image_history[0].copy()

    def _build_obs(self, current_state: np.ndarray, current_image: np.ndarray):
        state_parts = [current_state]
        image_parts = [current_image]

        for lag in self.latent_steps:
            state_parts.append(self._get_past_state(int(lag)))
            image_parts.append(self._get_past_image(int(lag)))

        stacked_state = np.concatenate(state_parts, axis=0).astype(np.float32)
        stacked_image = np.concatenate(image_parts, axis=2).astype(np.uint8)

        obs = {
            "image": stacked_image,
            "state": stacked_state,
        }
        return obs

    # -------------------------
    # Reset / Step
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.next_frame_time = 0.0
        self.step_count = 0
        self._rendered_frames = []
        self._episode_mechanical_energy_j = 0.0
        self._episode_segment_distance_integral_m_s = 0.0
        self._episode_energy_penalty = 0.0
        self._episode_ball_segment_penalty = 0.0
        self._episode_reward_components = {}
        self._observation_frames = []
        self._last_observation_time = None
        self._capture_this_episode = (
            self._capture_episodes_remaining is None or self._capture_episodes_remaining > 0
        )
        if self._capture_episodes_remaining is not None and self._capture_episodes_remaining > 0:
            self._capture_episodes_remaining -= 1

        mujoco.mj_resetData(self.model, self.data)

        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.ctrl[act_id] = self.data.qpos[qpos_index]

        mujoco.mj_forward(self.model, self.data)
        self.bounce_tracker.reset(self.data)
        self.reward.reset(self.model, self.data)

        self.next_frame_time = self.data.time
        self._maybe_update_camera()

        current_state = self._get_state()
        current_image = self._last_image.copy()

        # Au reset, on remplit tout l'historique avec l'observation courante
        self._state_history.clear()
        self._image_history.clear()

        history_len = self.max_history + 1 if self.max_history > 0 else 1
        for _ in range(history_len):
            self._state_history.append(current_state.copy())
            self._image_history.append(current_image.copy())

        obs = self._build_obs(current_state, current_image)

        if self.render_mode == "human":
            self._sync_human_viewer()

        return obs, {"task_metrics": self._task_metrics()}

    def step(self, action):
        self.step_count += 1
        t0 = time.time()

        ctrl_range = self.model.actuator_ctrlrange
        actions_rescaled = np.clip(action, -1, 1)
        actions_rescaled = (
            0.5 * (actions_rescaled + 1.0) * (ctrl_range[:, 1] - ctrl_range[:, 0])
            + ctrl_range[:, 0]
        )
        self.data.ctrl[:] = np.asarray(actions_rescaled, dtype=np.float32)

        bounce_events = []
        energy_step_j = 0.0
        segment_distance_integral_m_s = 0.0
        for _ in range(self.frame_skip):
            before = self.bounce_tracker.sample(self.data)
            mujoco.mj_step(self.model, self.data)
            # Synchronize positions, contacts and velocities after integration.
            mujoco.mj_forward(self.model, self.data)
            # Absolute mechanical work, including braking. Sum per actuator
            # before integrating so opposite motor powers cannot cancel out.
            power_w = float(np.sum(np.abs(self.data.actuator_force * self.data.actuator_velocity)))
            energy_step_j += power_w * self.model.opt.timestep
            ball_segment_distance_m, segment_closest_point = distance_to_vertical_segment(
                self.data.geom_xpos[self.bounce_tracker.ball],
                self.data.site_xpos[self.bounce_tracker.site],
                self.ball_segment_start_height, self.ball_segment_length,
            )
            segment_distance_integral_m_s += ball_segment_distance_m * self.model.opt.timestep
            event = self.bounce_tracker.update(before, self.bounce_tracker.sample(self.data))
            if event is not None:
                bounce_events.append(event)
            if self.render_profile != "train_fast":
                self._maybe_update_camera()
            if self.bounce_tracker.termination_reason is not None:
                break

        self._episode_mechanical_energy_j += energy_step_j
        self._episode_segment_distance_integral_m_s += segment_distance_integral_m_s

        if self.render_profile == "train_fast":
            self._maybe_update_camera()

        current_image = self._last_image.copy()
        current_state = self._get_state()

        # Construit obs avec:
        # - présent = current_*
        # - passé = contenu des buffers avant d'ajouter current_*
        obs = self._build_obs(current_state, current_image)

        # Puis met à jour l'historique pour le prochain step
        self._state_history.append(current_state.copy())
        self._image_history.append(current_image.copy())

        if self.render_mode == "human":
            self._sync_human_viewer()

        terminated = self.bounce_tracker.termination_reason is not None
        truncated = False

        if self.step_count >= self.max_episode_steps:
            truncated = True
            self.step_count = 0

        context = RewardContext(
            bounce_count=len(bounce_events), failed=terminated, energy_j=energy_step_j,
            segment_distance_integral_m_s=segment_distance_integral_m_s,
        )
        reward, reward_info = self.reward.compute(self.model, self.data, context)
        components = reward_info["components"]
        for name, component in components.items():
            self._episode_reward_components[name] = (
                self._episode_reward_components.get(name, 0.0) + component["contribution"]
            )
        # Preserve legacy channels for the canonical top-level objective names.
        def contribution(name):
            return components.get(name, {}).get("contribution", 0.0)

        shaping = components.get("shaping", {})
        info = dict(shaping.get("details", {}))
        self._episode_energy_penalty += contribution("energy")
        self._episode_ball_segment_penalty += contribution("ball_segment")
        info.update({
            "reward_shaping": shaping.get("value", 0.0),
            "reward_bounces": contribution("bounces"),
            "penalty_fall": contribution("failure") if self.bounce_tracker.time_to_fall is not None else 0.0,
            "penalty_failure": contribution("failure"),
            "energy_step_j": energy_step_j,
            "penalty_energy": contribution("energy"),
            "ball_segment_distance_m": ball_segment_distance_m,
            "segment_closest_point": segment_closest_point.tolist(),
            "segment_distance_integral_m_s": segment_distance_integral_m_s,
            "penalty_ball_segment": contribution("ball_segment"),
            "reward_components": components,
            "termination_reason": self.bounce_tracker.termination_reason,
            "reward_total": float(reward),
            "bounce_events": bounce_events,
            "task_metrics": self._task_metrics(),
        })
        # VecEnv resets immediately on done; preserve frames before reset clears them.
        if (terminated or truncated) and self._capture_this_episode and (self.record_video or self.render_mode == "rgb_array_list"):
            info["terminal_render_frames"] = self._rendered_frames.copy()
            info["terminal_observation_frames"] = self._observation_frames.copy()

        print(f"[Step] time={time.time() - t0:.3f}s | reward={reward:.3f}")
        return obs, reward, terminated, truncated, info

    def pop_render_frames(self):
        """Drain camera frames independently of the human viewer/render mode."""
        frames = self._rendered_frames
        self._rendered_frames = []
        return frames

    def pop_observation_frames(self):
        """Drain unstacked policy-camera frames for TensorBoard visualization."""
        frames = self._observation_frames
        self._observation_frames = []
        return frames

    def render(self):
        if self.render_mode == "rgb_array":
            self._maybe_update_camera()
            return self._last_video_image

        if self.render_mode == "rgb_array_list":
            self._maybe_update_camera()
            return self.pop_render_frames()

        if self.render_mode == "human":
            self._sync_human_viewer()
            return None

        return None

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.video_renderer is not None:
            self.video_renderer.close()
            self.video_renderer = None
