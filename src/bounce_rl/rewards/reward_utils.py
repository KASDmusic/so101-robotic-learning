from __future__ import annotations

import numpy as np
import mujoco


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def gaussian_reward(x: float, target: float, sigma: float) -> float:
    """
    Reward max = 1 quand x == target, puis décroissance gaussienne.
    """
    if sigma <= 0:
        raise ValueError("sigma doit être > 0.")
    return float(np.exp(-0.5 * ((x - target) / sigma) ** 2))


def get_body_linear_velocity_world(
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
        0,  # coordonnées monde
    )
    return vel6[3:].copy()


def get_body_position_world(
    data: mujoco.MjData,
    body_id: int,
) -> np.ndarray:
    """
    Retourne la position du body en coordonnées monde.
    """
    return data.xpos[body_id].copy()


def get_body_id_or_raise(
    model: mujoco.MjModel,
    body_name: str,
) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f'Body "{body_name}" introuvable dans le modèle MuJoCo.')
    return body_id