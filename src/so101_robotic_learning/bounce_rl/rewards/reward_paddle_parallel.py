from __future__ import annotations

import numpy as np
import mujoco

from .reward_utils import safe_normalize, get_body_id_or_raise


class PaddleParallelReward:
    """
    Reward pour encourager la raquette à être parallèle au sol.
    Cela revient à aligner la normale de la raquette avec l'axe vertical du monde.
    """

    def __init__(
        self,
        paddle_body_name: str = "paddle_mount",
        paddle_normal_local=(0.0, -1.0, 0.0),
        paddle_site_name: str | None = None,
    ):
        self.paddle_body_name = paddle_body_name
        self.paddle_site_name = paddle_site_name

        self.paddle_normal_local = np.array(paddle_normal_local, dtype=np.float64, copy=True)
        norm = np.linalg.norm(self.paddle_normal_local)
        if norm < 1e-8:
            raise ValueError("paddle_normal_local ne doit pas être nul.")
        self.paddle_normal_local /= norm

        self.world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        if self.paddle_site_name is not None:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.paddle_site_name)
            if site_id < 0:
                raise ValueError(f"Paddle face site {self.paddle_site_name!r} not found.")
            paddle_normal_world = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
        else:
            paddle_body_id = get_body_id_or_raise(model, self.paddle_body_name)
            rotation = data.xmat[paddle_body_id].reshape(3, 3)
            paddle_normal_world = safe_normalize(rotation @ self.paddle_normal_local)
        # Only the chosen face pointing upward receives a positive score.
        score = float(np.clip(np.dot(paddle_normal_world, self.world_up), 0.0, 1.0))

        info = {
            "reward_paddle_parallel": score,
            "paddle_normal_world": paddle_normal_world.copy(),
        }
        return score, info
