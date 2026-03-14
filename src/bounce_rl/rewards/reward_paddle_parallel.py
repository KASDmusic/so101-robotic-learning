from __future__ import annotations

import numpy as np
import mujoco

from reward_utils import safe_normalize, get_body_id_or_raise


class PaddleParallelReward:
    """
    Reward pour encourager la raquette à être parallèle au sol.
    Cela revient à aligner la normale de la raquette avec l'axe vertical du monde.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        paddle_body_name: str = "paddle_mount",
        paddle_normal_local=(0.0, -1.0, 0.0),
    ):
        self.paddle_body_name = paddle_body_name
        self.paddle_body_id = get_body_id_or_raise(model, paddle_body_name)

        self.paddle_normal_local = np.asarray(paddle_normal_local, dtype=np.float64)
        norm = np.linalg.norm(self.paddle_normal_local)
        if norm < 1e-8:
            raise ValueError("paddle_normal_local ne doit pas être nul.")
        self.paddle_normal_local /= norm

        self.world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _get_paddle_normal_world(self, data: mujoco.MjData) -> np.ndarray:
        xmat = data.xmat[self.paddle_body_id].reshape(3, 3)
        normal_world = xmat @ self.paddle_normal_local
        return safe_normalize(normal_world)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        paddle_normal_world = self._get_paddle_normal_world(data)

        score = float(abs(np.dot(paddle_normal_world, self.world_up)))

        info = {
            "reward_paddle_parallel": score,
            "paddle_normal_world": paddle_normal_world.copy(),
        }
        return score, info