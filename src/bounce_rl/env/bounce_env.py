import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class BounceEnv(gym.Env):
    """
    Gym env qui renvoie une observation Dict:
      - obs["image"]: image RGB uint8 (H, W, 3) depuis camera_name, calculée à camera_fps
      - obs["state"]: état float32 (qpos||qvel)

    render_mode:
      - None: aucun rendu explicite
      - "rgb_array": render() renvoie la dernière image caméra
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
        width: int = 320,
        height: int = 240,
        mujoco_step_periode: float = 0.002,
        frame_skip: int = 20,
        camera_fps: int = 30,
        paddle_color=(0.9, 0.05, 0.05, 1),
        ball_mass: float = 0.003,
        render_mode: str | None = None
    ):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode={render_mode!r} invalide. "
                f"Modes supportés: {self.metadata['render_modes']}"
            )

        self.xml_path = xml_path
        self.reward = reward
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.frame_skip = int(frame_skip)
        self.mujoco_step_periode = float(mujoco_step_periode)

        self.step_count = 0

        # Caméra RGB throttlée sur temps simulé
        self.camera_fps = float(camera_fps)
        self.camera_period = 1.0 / self.camera_fps
        self.next_frame_time = 0.0

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = mujoco_step_periode
        self.data = mujoco.MjData(self.model)

        # Vérifie que la caméra existe
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            raise ValueError(f'Camera "{camera_name}" introuvable dans le XML.')

        # Action space: commandes normalisées [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation space: Dict(image, state)
        state_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                ),
                "last_image": spaces.Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                ),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim*2,), dtype=np.float32
                ),
            }
        )

        # Renderer offscreen pour rgb_array / rgb_array_list / obs["image"]
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Viewer interactif pour human
        self.viewer = None

        # Paramètres dynamiques optionnels
        self._set_paddle_color(paddle_color)
        self._set_ball_mass(ball_mass)

        # Dernière image rendue
        self._prev_state = np.zeros(self.model.nq + self.model.nv, dtype=np.float32)
        self._prev_image = np.zeros((height, width, 3), dtype=np.uint8)
        self._last_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Buffer des frames pour render_mode="rgb_array_list"
        self._rendered_frames = []

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
    def _store_frame_if_needed(self, frame: np.ndarray):
        if self.render_mode == "rgb_array_list":
            self._rendered_frames.append(frame.copy())

    def _maybe_update_camera(self):
        """
        Met à jour l'image caméra UNIQUEMENT à camera_fps simulé.
        Entre deux frames, on conserve self._last_image.

        En mode rgb_array_list, chaque nouvelle frame effectivement calculée
        est aussi ajoutée dans self._rendered_frames.
        """
        if self.data.time < self.next_frame_time:
            return

        self.renderer.update_scene(self.data, camera=self.camera_name)
        self._last_image = self.renderer.render()

        self._store_frame_if_needed(self._last_image)

        # avancer le prochain créneau caméra
        self.next_frame_time += self.camera_period

        # sécurité si gros saut de temps simulé
        if self.data.time >= self.next_frame_time + 5 * self.camera_period:
            self.next_frame_time = self.data.time + self.camera_period

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _get_obs(self):
        self._maybe_update_camera()
        current_state = self._get_state()
        obs = {
            "image": self._last_image.copy(),
            "last_image": self._prev_image.copy(),
            "state": np.concatenate([self._prev_state, current_state], axis=0).astype(np.float32),
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

        mujoco.mj_resetData(self.model, self.data)

        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.ctrl[act_id] = self.data.qpos[qpos_index]

        mujoco.mj_forward(self.model, self.data)

        self.next_frame_time = self.data.time
        self._maybe_update_camera()

        current_state = self._get_state()

        # au reset, on duplique l'observation courante comme "précédente"
        self._prev_state = current_state.copy()
        self._prev_image = self._last_image.copy()

        obs = {
            "image": self._last_image.copy(),
            "last_image": self._prev_image.copy(),
            "state": np.concatenate([self._prev_state, current_state], axis=0).astype(np.float32),
        }

        if self.render_mode == "human":
            self._sync_human_viewer()

        return obs, {}

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

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._maybe_update_camera()

        current_image = self._last_image.copy()
        current_state = self._get_state()

        obs = {
            "image": current_image,
            "last_image": self._prev_image.copy(),
            "state": np.concatenate([self._prev_state, current_state], axis=0).astype(np.float32),
        }

        # mise à jour pour le prochain step
        self._prev_state = current_state.copy()
        self._prev_image = current_image.copy()

        if self.render_mode == "human":
            self._sync_human_viewer()

        if self.reward is not None:
            reward, reward_info = self.reward.compute(self.model, self.data)
        else:
            reward = 0.0
            reward_info = {}

        terminated = False
        truncated = False

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        if body_id != -1:
            ball_height = self.data.xpos[body_id, 2]
            if ball_height < 0.1:
                terminated = True

        if self.step_count >= self.max_episode_steps:
            truncated = True
            self.step_count = 0

        info = reward_info

        print(f"[Step] time={time.time() - t0:.3f}s | reward={reward:.3f}")
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            self._maybe_update_camera()
            return self._last_image

        if self.render_mode == "rgb_array_list":
            self._maybe_update_camera()
            temp_frames = self._rendered_frames.copy()
            self._rendered_frames = []
            return temp_frames

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