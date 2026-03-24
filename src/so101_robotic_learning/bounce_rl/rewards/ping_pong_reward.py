from __future__ import annotations

import mujoco

from .reward_paddle_parallel import PaddleParallelReward
from .reward_ball_vertical import BallVerticalReward
from .reward_ball_speed import BallSpeedReward
from .penalty_ball_below_paddle import BallBelowPaddlePenalty


class PingPongReward:
    """
    Regroupe plusieurs rewards/penalties avec combinaison pondérée.
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
        paddle_normal_local=(0.0, -1.0, 0.0),
        alive_bonus: float = 0.0,
    ):
        self.w_paddle_parallel = float(w_paddle_parallel)
        self.w_ball_vertical = float(w_ball_vertical)
        self.w_ball_speed = float(w_ball_speed)
        self.w_ball_below_paddle = float(w_ball_below_paddle)
        self.alive_bonus = float(alive_bonus)

        self.paddle_parallel_reward = PaddleParallelReward(
            paddle_body_name=paddle_body_name,
            paddle_normal_local=paddle_normal_local,
        )

        self.ball_vertical_reward = BallVerticalReward(
            ball_body_name=ball_body_name,
        )

        self.ball_speed_reward = BallSpeedReward(
            ball_body_name=ball_body_name,
            target_ball_speed=target_ball_speed,
            speed_sigma=speed_sigma,
        )

        self.ball_below_paddle_penalty = BallBelowPaddlePenalty(
            ball_body_name=ball_body_name,
            paddle_body_name=paddle_body_name,
            below_paddle_margin=below_paddle_margin,
        )

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        paddle_parallel_score, info_paddle = self.paddle_parallel_reward.compute(model, data)
        ball_vertical_score, info_vertical = self.ball_vertical_reward.compute(model, data)
        ball_speed_score, info_speed = self.ball_speed_reward.compute(model, data)
        ball_below_penalty, info_penalty = self.ball_below_paddle_penalty.compute(model, data)

        reward = (
            self.w_paddle_parallel * paddle_parallel_score
            + self.w_ball_vertical * ball_vertical_score
            + self.w_ball_speed * ball_speed_score
            + self.w_ball_below_paddle * ball_below_penalty
            + self.alive_bonus
        )

        info = {
            "reward_total": float(reward),
            **info_paddle,
            **info_vertical,
            **info_speed,
            **info_penalty,
            "weight_paddle_parallel": self.w_paddle_parallel,
            "weight_ball_vertical": self.w_ball_vertical,
            "weight_ball_speed": self.w_ball_speed,
            "weight_ball_below_paddle": self.w_ball_below_paddle,
            "alive_bonus": self.alive_bonus,
        }

        return float(reward), info