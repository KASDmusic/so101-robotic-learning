from __future__ import annotations

import math

import mujoco

from .reward_utils import get_body_id_or_raise


class PaddleInInterval:
    """
    Reward continue :
    - 1 si la raquette est au centre du pavé
    - 0 si la raquette est hors du pavé
    - entre 0 et 1 sinon, selon la distance au centre
    """

    def __init__(
        self,
        paddle_body_name: str = "paddle_mount",
        x_min: float = -5,
        x_max: float = 5,
        y_min: float = -5,
        y_max: float = 5,
        z_min: float = 0.2,
        z_max: float = 3,
        x_offset: float = 0.15
    ):
        self.paddle_body_name = paddle_body_name

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

        self.x_offset = float(x_offset)

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[float, dict]:

        paddle_body_id = get_body_id_or_raise(model, self.paddle_body_name)
        paddle_pos = data.xpos[paddle_body_id].copy()

        # Offset pour approximer le centre de gravité de la raquette
        paddle_pos[0] += self.x_offset

        x, y, z = float(paddle_pos[0]), float(paddle_pos[1]), float(paddle_pos[2])

        in_x = self.x_min <= x <= self.x_max
        in_y = self.y_min <= y <= self.y_max
        in_z = self.z_min <= z <= self.z_max

        inside = in_x and in_y and in_z

        x_center = 0.5 * (self.x_min + self.x_max)
        y_center = 0.5 * (self.y_min + self.y_max)
        z_center = 0.5 * (self.z_min + self.z_max)

        x_half = 0.5 * (self.x_max - self.x_min)
        y_half = 0.5 * (self.y_max - self.y_min)
        z_half = 0.5 * (self.z_max - self.z_min)

        if inside:
            # Coordonnées normalisées par rapport au centre du pavé
            dx = 0.0 if x_half == 0.0 else (x - x_center) / x_half
            dy = 0.0 if y_half == 0.0 else (y - y_center) / y_half
            dz = 0.0 if z_half == 0.0 else (z - z_center) / z_half

            # Distance euclidienne normalisée :
            # - 0 au centre
            # - sqrt(3) dans un coin
            normalized_distance = math.sqrt(dx * dx + dy * dy + dz * dz) / math.sqrt(3.0)

            reward = max(0.0, 1.0 - normalized_distance)
        else:
            dx = dy = dz = None
            normalized_distance = None
            reward = 0.0

        info = {
            "paddle_position": (x, y, z),
            "in_x_interval": in_x,
            "in_y_interval": in_y,
            "in_z_interval": in_z,
            "inside_interval": inside,
            "x_interval": (self.x_min, self.x_max),
            "y_interval": (self.y_min, self.y_max),
            "z_interval": (self.z_min, self.z_max),
            "center": (x_center, y_center, z_center),
            "normalized_offset": None if not inside else (dx, dy, dz),
            "normalized_distance_to_center": normalized_distance,
            "score": reward,
        }

        return reward, info