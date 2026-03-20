from __future__ import annotations

import numpy as np
import mujoco

from reward_utils import (
    safe_normalize,
    get_body_linear_velocity_world,
    get_body_id_or_raise,
)


class BallVerticalReward:
    """
    Reward pour encourager la balle à avoir une vitesse majoritairement verticale.
    """

    def __init__(
        self,
        ball_body_name: str = "ball",
    ):
        self.ball_body_name = ball_body_name
        self.world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        
        ball_body_id = get_body_id_or_raise(model, self.ball_body_name)

        ball_vel = get_body_linear_velocity_world(model, data, ball_body_id)
        ball_speed = float(np.linalg.norm(ball_vel))
        ball_vel_dir = safe_normalize(ball_vel)

        if ball_speed < 1e-8:
            score = 0.0
        else:
            score = float(abs(np.dot(ball_vel_dir, self.world_up)))

        info = {
            "reward_ball_vertical": score,
            "ball_velocity_world": ball_vel.copy(),
            "ball_speed": ball_speed,
        }
        return score, info