from __future__ import annotations

import numpy as np
from tensorboard import data
from bounce_rl.rewards.reward_paddle_parallel import PaddleParallelReward
import mujoco

from .reward_utils import get_body_id_or_raise


class BallAlignedOnZAndAbovePaddleReward:
    """
    Reward pour encourager la balle à être alignée avec la raquette sur l'axe z,
    tout en restant au-dessus de la raquette.

    - Reward maximale quand la balle est très proche de la raquette en z
    - Reward nulle si la balle est sous la raquette
    """

    def __init__(
        self,
        ball_body_name: str = "ball",
        paddle_body_name: str = "paddle_mount",
        z_tolerance: float = 0.1,
    ):
        self.ball_body_name = ball_body_name
        self.paddle_body_name = paddle_body_name
        self.z_tolerance = float(z_tolerance)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:
        
        ball_body_id = get_body_id_or_raise(model, self.ball_body_name)
        paddle_body_id = get_body_id_or_raise(model, self.paddle_body_name)
        
        ball_pos = data.xpos[ball_body_id]
        paddle_pos = data.xpos[paddle_body_id]

        paddle_pos[0] += 0.15  # Décalage pour viser le centre de la raquette (en m)

        ball_z = float(ball_pos[2])
        paddle_z = float(paddle_pos[2])

        # Distance cosinus entre le vecteur vertical (0, 0, 1) et le vecteur de la raquette à la balle
        vector_paddle_to_ball = ball_pos - paddle_pos
        vector_paddle_to_ball_norm = np.linalg.norm(vector_paddle_to_ball)
        if vector_paddle_to_ball_norm > 1e-6:
            cos_angle = vector_paddle_to_ball[2] / vector_paddle_to_ball_norm
        else:
            cos_angle = 1.0  # Si la balle est exactement au-dessus de la raquette

        # Reward basée sur l'alignement en z (cosine de l'angle)
        alignment_reward = max(0.0, cos_angle)  # Reward nulle si la balle est en dessous

        reward_paddle_parallel = PaddleParallelReward().compute(model, data)[0]
        score = alignment_reward + reward_paddle_parallel
        

        info = {
            "reward_ball_aligned_z_and_above_paddle": alignment_reward,
            "reward_paddle_parallel": reward_paddle_parallel,
            "score": score,
            "z_tolerance": self.z_tolerance,
            "ball_above_paddle": ball_z >= paddle_z,
        }

        return score, info