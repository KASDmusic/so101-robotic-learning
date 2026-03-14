from __future__ import annotations

import mujoco

from reward_utils import (
    get_body_position_world,
    get_body_id_or_raise,
)


class BallBelowPaddlePenalty:
    """
    Pénalité si la balle est plus basse que la raquette.
    La pénalité est normalisée entre 0 et 1.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ball_body_name: str = "ball",
        paddle_body_name: str = "paddle_mount",
        below_paddle_margin: float = 0.05,
    ):
        self.ball_body_name = ball_body_name
        self.paddle_body_name = paddle_body_name

        if below_paddle_margin <= 0:
            raise ValueError("below_paddle_margin doit être > 0.")
        self.below_paddle_margin = float(below_paddle_margin)

        self.ball_body_id = get_body_id_or_raise(model, ball_body_name)
        self.paddle_body_id = get_body_id_or_raise(model, paddle_body_name)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        ball_pos = get_body_position_world(data, self.ball_body_id)
        paddle_pos = get_body_position_world(data, self.paddle_body_id)

        ball_height = float(ball_pos[2])
        paddle_height = float(paddle_pos[2])

        below_depth = max(0.0, paddle_height - ball_height)
        penalty = min(1.0, below_depth / self.below_paddle_margin)

        info = {
            "penalty_ball_below_paddle": penalty,
            "ball_height": ball_height,
            "paddle_height": paddle_height,
            "height_difference_paddle_minus_ball": float(paddle_height - ball_height),
        }
        return penalty, info