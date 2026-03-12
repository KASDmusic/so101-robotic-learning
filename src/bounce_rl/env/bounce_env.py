import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

from ..rewards.rewards import PingPongReward


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
        camera_name: str = "gripper_cam",
        width: int = 320,
        height: int = 240,
        mujoco_step_periode: float = 0.002,
        frame_skip: int = 20,
        camera_fps: int = 30,
        paddle_color=(0.9, 0.05, 0.05, 1),
        ball_mass: float = 0.003,
        render_mode: str | None = None,
    ):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode={render_mode!r} invalide. "
                f"Modes supportés: {self.metadata['render_modes']}"
            )

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.frame_skip = int(frame_skip)
        self.mujoco_step_periode = float(mujoco_step_periode)

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
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
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
        self._last_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Buffer des frames pour render_mode="rgb_array_list"
        self._rendered_frames = []

        self.reward_fn = PingPongReward(
            self.model,
            ball_body_name="ball",
            paddle_body_name="paddle_mount",
            target_ball_speed=2.0,
            speed_sigma=0.5,
            w_paddle_parallel=1.0,
            w_ball_vertical=0,
            w_ball_speed=0,
            w_ball_below_paddle=0,
            below_paddle_margin=0.05,
            paddle_normal_local=(0.0, -1.0, 0.0),  # à ajuster si besoin
        )

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
        return {"image": self._last_image, "state": self._get_state()}

    # -------------------------
    # Reset / Step
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.next_frame_time = 0.0
        self._rendered_frames = []

        mujoco.mj_resetData(self.model, self.data)

        # Optionnel: randomiser la balle
        #if self.model.nq >= 7:
        #    self.data.qpos[0:3] = np.array([0.5, np.random.uniform(-0.2, 0.2), 2.0])

        # IMPORTANT pour actuateurs position: aligner ctrl avec la pose
        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.ctrl[act_id] = self.data.qpos[qpos_index]

        mujoco.mj_forward(self.model, self.data)

        # Première image immédiate
        self.next_frame_time = self.data.time
        obs = self._get_obs()

        if self.render_mode == "human":
            self._sync_human_viewer()

        return obs, {}

    def step(self, action):

        # Rescale action de [-1, 1] à la plage de contrôle de MuJoCo
        ctrl_range = self.model.actuator_ctrlrange
        actions_rescaled = np.clip(action, -1, 1)  # sécurité
        actions_rescaled = 0.5 * (actions_rescaled + 1.0) * (ctrl_range[:, 1] - ctrl_range[:, 0]) + ctrl_range[:, 0]

        self.data.ctrl[:] = np.asarray(actions_rescaled, dtype=np.float32)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._maybe_update_camera()

        obs = self._get_obs()

        if self.render_mode == "human":
            self._sync_human_viewer()

        reward, reward_info = self.reward_fn.compute(self.model, self.data)
        terminated = False
        truncated = False
        info = reward_info

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