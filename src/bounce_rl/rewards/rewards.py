from __future__ import annotations

import numpy as np
import mujoco


class PingPongReward:
    """
    Reward shaping pour un environnement MuJoCo de ping-pong.

    Critères utilisés :
      1. La raquette doit être parallèle au sol
         -> sa normale doit être alignée avec l'axe vertical du monde.
      2. Le vecteur vitesse de la balle doit être perpendiculaire au sol
         -> la vitesse de la balle doit être majoritairement verticale.
      3. La vitesse de la balle doit être proche d'une vitesse cible.
      4. Malus si la balle est plus basse que la raquette.

    Notes importantes :
    - On lit l'orientation de la raquette via le body `paddle_mount`.
    - On lit la vitesse et la position de la balle via le body `ball`.
    - La normale locale de la raquette dépend de l'orientation réelle du mesh.
      Par défaut, elle est supposée être l'axe local Z = [0, 0, 1].
      Si la reward semble incohérente, change `paddle_normal_local`.

    Utilisation typique :
        reward_fn = PingPongReward(model)

        # dans env.step(...)
        reward, info = reward_fn.compute(model, data)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ball_body_name: str = "ball",
        paddle_body_name: str = "paddle_mount",
        target_ball_speed: float = 2.0,
        speed_sigma: float = 0.5,
        w_paddle_parallel: float = 0.4,
        w_ball_vertical: float = 0.3,
        w_ball_speed: float = 0.3,
        w_ball_below_paddle: float = 0.6,
        below_paddle_margin: float = 0.05,
        paddle_normal_local=(0, -1.0, 0.0),
        alive_bonus: float = 0.0,
    ):
        self.ball_body_name = ball_body_name
        self.paddle_body_name = paddle_body_name

        self.target_ball_speed = float(target_ball_speed)
        self.speed_sigma = float(speed_sigma)

        self.w_paddle_parallel = float(w_paddle_parallel)
        self.w_ball_vertical = float(w_ball_vertical)
        self.w_ball_speed = float(w_ball_speed)
        self.w_ball_below_paddle = float(w_ball_below_paddle)
        self.below_paddle_margin = float(below_paddle_margin)
        if self.below_paddle_margin <= 0:
            raise ValueError("below_paddle_margin doit être > 0.")

        self.paddle_normal_local = np.asarray(paddle_normal_local, dtype=np.float64)
        norm = np.linalg.norm(self.paddle_normal_local)
        if norm < 1e-8:
            raise ValueError("paddle_normal_local ne doit pas être nul.")
        self.paddle_normal_local /= norm

        self.alive_bonus = float(alive_bonus)

        self._world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self.ball_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.ball_body_name
        )
        if self.ball_body_id == -1:
            raise ValueError(
                f'Body "{self.ball_body_name}" introuvable dans le modèle MuJoCo.'
            )

        self.paddle_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.paddle_body_name
        )
        if self.paddle_body_id == -1:
            raise ValueError(
                f'Body "{self.paddle_body_name}" introuvable dans le modèle MuJoCo.'
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def _gaussian_reward(x: float, target: float, sigma: float) -> float:
        """
        Reward max = 1 quand x == target, puis décroissance gaussienne.
        """
        if sigma <= 0:
            raise ValueError("sigma doit être > 0.")
        return float(np.exp(-0.5 * ((x - target) / sigma) ** 2))

    def _get_body_linear_velocity_world(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_id: int,
    ) -> np.ndarray:
        """
        Retourne la vitesse linéaire du body en coordonnées monde.
        """
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel6,
            0,  # 0 => coordonnées monde
        )
        return vel6[3:].copy()

    def _get_paddle_normal_world(
        self,
        data: mujoco.MjData,
        body_id: int,
    ) -> np.ndarray:
        """
        Convertit une normale locale de la raquette en coordonnées monde.
        """
        xmat = data.xmat[body_id].reshape(3, 3)
        #print("xmat:", xmat)
        normal_world = xmat @ self.paddle_normal_local
        return self._safe_normalize(normal_world)

    def _get_body_position_world(
        self,
        data: mujoco.MjData,
        body_id: int,
    ) -> np.ndarray:
        """
        Retourne la position du body en coordonnées monde.
        """
        return data.xpos[body_id].copy()

    # ------------------------------------------------------------------
    # Reward principale
    # ------------------------------------------------------------------
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        """
        Calcule la reward courante.

        Retour :
            reward: float
            info: dict détaillant chaque terme
        """
        # ----- Orientation raquette -----
        paddle_normal_world = self._get_paddle_normal_world(data, self.paddle_body_id)
        #print("Paddle normal world:", paddle_normal_world)

        # Raquette parallèle au sol => normale alignée avec l'axe vertical
        # abs(...) : on accepte normale vers le haut ou vers le bas
        paddle_parallel_score = float(
            abs(np.dot(paddle_normal_world, self._world_up))
        )

        # ----- Vitesse balle -----
        ball_vel = self._get_body_linear_velocity_world(model, data, self.ball_body_id)
        ball_speed = float(np.linalg.norm(ball_vel))
        ball_vel_dir = self._safe_normalize(ball_vel)

        # Vitesse perpendiculaire au sol => direction alignée avec z
        # abs(...) : montée ou descente acceptées
        if ball_speed < 1e-8:
            ball_vertical_score = 0.0
        else:
            ball_vertical_score = float(abs(np.dot(ball_vel_dir, self._world_up)))

        # Vitesse proche d'une cible
        ball_speed_score = self._gaussian_reward(
            ball_speed, self.target_ball_speed, self.speed_sigma
        )

        # ----- Hauteur balle / raquette -----
        ball_pos = self._get_body_position_world(data, self.ball_body_id)
        paddle_pos = self._get_body_position_world(data, self.paddle_body_id)

        ball_height = float(ball_pos[2])
        paddle_height = float(paddle_pos[2])

        # profondeur > 0 si la balle est sous la raquette
        below_depth = max(0.0, paddle_height - ball_height)

        # malus normalisé entre 0 et 1
        ball_below_paddle_penalty = min(1.0, below_depth / self.below_paddle_margin)

        # ----- Combinaison -----
        reward = (
            self.w_paddle_parallel * paddle_parallel_score
            + self.w_ball_vertical * ball_vertical_score
            + self.w_ball_speed * ball_speed_score
            - self.w_ball_below_paddle * ball_below_paddle_penalty
            + self.alive_bonus
        )

        info = {
            "reward_total": float(reward),
            "reward_paddle_parallel": float(paddle_parallel_score),
            "reward_ball_vertical": float(ball_vertical_score),
            "reward_ball_speed": float(ball_speed_score),
            "penalty_ball_below_paddle": float(ball_below_paddle_penalty),
            "ball_speed": float(ball_speed),
            "ball_height": float(ball_height),
            "paddle_height": float(paddle_height),
            "height_difference_paddle_minus_ball": float(paddle_height - ball_height),
            "ball_velocity_world": ball_vel.copy(),
            "paddle_normal_world": paddle_normal_world.copy(),
        }

        return float(reward), info