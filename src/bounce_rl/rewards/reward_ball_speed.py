from __future__ import annotations

import numpy as np
import mujoco

from reward_utils import (
    gaussian_reward,
    get_body_linear_velocity_world,
    get_body_id_or_raise,
)


class BallSpeedReward:
    """
    Reward pour encourager la vitesse de la balle à être proche d'une cible.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ball_body_name: str = "ball",
        target_ball_speed: float = 2.0,
        speed_sigma: float = 0.5,
    ):
        self.ball_body_name = ball_body_name
        self.ball_body_id = get_body_id_or_raise(model, ball_body_name)

        self.target_ball_speed = float(target_ball_speed)
        self.speed_sigma = float(speed_sigma)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        ball_vel = get_body_linear_velocity_world(model, data, self.ball_body_id)
        ball_speed = float(np.linalg.norm(ball_vel))

        score = gaussian_reward(
            ball_speed,
            self.target_ball_speed,
            self.speed_sigma,
        )

        info = {
            "reward_ball_speed": score,
            "ball_speed": ball_speed,
            "target_ball_speed": self.target_ball_speed,
        }
        return score, info