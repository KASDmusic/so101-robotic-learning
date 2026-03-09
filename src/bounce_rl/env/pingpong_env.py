import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import cv2


class PingPongEnv(gym.Env):
    """
    Gym env qui renvoie une observation Dict:
      - obs["image"]: image RGB uint8 (H, W, 3) depuis camera_name, CALCULÉE À 30 FPS SIMULÉS
      - obs["state"]: état float32 (qpos||qvel)

    Important:
      - AUCUN affichage (pas de cv2.imshow)
      - Le rendu de la caméra est "throttlé" via data.time : 30 images / seconde simulée
      - Entre deux instants caméra, on renvoie la DERNIÈRE image calculée (jamais None).
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        xml_path: str,
        camera_name: str = "gripper_cam",
        width: int = 320,
        height: int = 240,
        mujoco_step_periode: float = 0.002,
        frame_skip: int = 30,
        camera_fps: int = 30,
        paddle_color=(0.9, 0.05, 0.05, 1),
        ball_mass: float = 0.003,
    ):
        super().__init__()

        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.frame_skip = int(frame_skip)
        self.mujoco_step_periode = float(mujoco_step_periode)

        # Caméra: 30 FPS simulé via data.time
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

        # Renderer offscreen pour obtenir les images
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Paramètres dynamiques optionnels
        self._set_paddle_color(paddle_color)
        self._set_ball_mass(ball_mass)

        # Dernière image rendue (toujours une image valide retournée)
        self._last_image = np.zeros((height, width, 3), dtype=np.uint8)

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
    # Observations
    # -------------------------
    def _maybe_update_camera(self):
        """
        Met à jour l'image caméra UNIQUEMENT à 30Hz simulé.
        Entre deux frames, on conserve self._last_image.
        """
        if self.data.time < self.next_frame_time:
            return

        self.renderer.update_scene(self.data, camera=self.camera_name)
        self._last_image = self.renderer.render()  # RGB uint8 HxWx3

        # avancer le "slot" caméra (peut sauter si gros dt effectif)
        self.next_frame_time += self.camera_period

        # sécurité: si la simu avance beaucoup, on rattrape
        if self.data.time >= self.next_frame_time + 5 * self.camera_period:
            self.next_frame_time = self.data.time + self.camera_period

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _get_obs(self):
        # Toujours fournir une image valide, rendue à 30 Hz simulé
        self._maybe_update_camera()
        return {"image": self._last_image, "state": self._get_state()}

    # -------------------------
    # Reset / Step
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.next_frame_time = 0.0

        mujoco.mj_resetData(self.model, self.data)

        # Optionnel: randomiser la balle (si la balle est la 1ère freejoint)
        # freejoint = 7 dof : pos(3) + quat(4)
        if self.model.nq >= 7:
            self.data.qpos[0:3] = np.array([0.5, np.random.uniform(-0.2, 0.2), 0.5])

        # IMPORTANT pour actuateurs position: aligner ctrl avec la pose
        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.ctrl[act_id] = self.data.qpos[qpos_index]

        mujoco.mj_forward(self.model, self.data)

        # Force une première image immédiatement
        self.next_frame_time = self.data.time
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Applique ctrl
        self.data.ctrl[:] = np.asarray(action, dtype=np.float32)

        # Avance la simu
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # TODO: reward/termination (placeholders)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        # Pour compat Gym: renvoyer une image caméra (mise à jour à 30Hz simulé)
        self._maybe_update_camera()
        return self._last_image

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None