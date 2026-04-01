from __future__ import annotations

import mujoco

from .reward_ball_in_interval import BallInInterval
from .reward_paddle_in_interval import PaddleInInterval


class RewardInterval:
    """
    Reward combinée :
    moyenne pondérée entre :
    - BallInInterval
    - PaddleInInterval
    """

    def __init__(
        self,
        ball_reward: BallInInterval,
        paddle_reward: PaddleInInterval,
        weight_ball: float = 0.5,
        weight_paddle: float = 0.5,
    ):
        assert weight_ball >= 0.0
        assert weight_paddle >= 0.0
        assert (weight_ball + weight_paddle) > 0.0

        self.ball_reward = ball_reward
        self.paddle_reward = paddle_reward

        self.weight_ball = float(weight_ball)
        self.weight_paddle = float(weight_paddle)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:

        ball_score, ball_info = self.ball_reward.compute(model, data)
        paddle_score, paddle_info = self.paddle_reward.compute(model, data)

        total_weight = self.weight_ball + self.weight_paddle

        score = (
            self.weight_ball * ball_score
            + self.weight_paddle * paddle_score
        ) / total_weight

        info = {
            "reward_ball": ball_score,
            "reward_paddle": paddle_score,
            "weight_ball": self.weight_ball,
            "weight_paddle": self.weight_paddle,
            "score": score,
            "ball_info": ball_info,
            "paddle_info": paddle_info,
        }

        return score, info