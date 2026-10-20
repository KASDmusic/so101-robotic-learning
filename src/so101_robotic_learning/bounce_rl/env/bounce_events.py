"""Physical bounce events and episode metrics, independent of reward shaping."""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class BounceConfig:
    ball_geom: str = "ball"
    paddle_geom: str = "paddle_col"
    paddle_site: str = "paddle_face"
    min_incoming_speed: float = 0.05  # m/s relative to the paddle
    min_outgoing_speed: float = 0.05
    min_flight_time: float = 0.2  # seconds without contact, filters chatter
    min_face_alignment: float = 0.7  # contact normal vs chosen face normal
    min_up_alignment: float = 0.5  # face normal vs world up
    fall_height: float = 0.1
    success_bounces: int = 5
    max_ball_paddle_distance: float = 2.0
    ground_geom: str | None = None  # None detects static world planes.

    def __post_init__(self):
        for name in ("min_incoming_speed", "min_outgoing_speed", "min_flight_time",
                     "max_ball_paddle_distance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("min_face_alignment", "min_up_alignment"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        if not np.isfinite(self.fall_height):
            raise ValueError("fall_height must be finite.")
        if not isinstance(self.success_bounces, int) or self.success_bounces < 1:
            raise ValueError("success_bounces must be a positive integer.")


class BounceTracker:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or BounceConfig()
        self.ball = self._id(mujoco.mjtObj.mjOBJ_GEOM, self.config.ball_geom)
        self.paddle = self._id(mujoco.mjtObj.mjOBJ_GEOM, self.config.paddle_geom)
        self.site = self._id(mujoco.mjtObj.mjOBJ_SITE, self.config.paddle_site)
        self.ground_geoms = (
            {self._id(mujoco.mjtObj.mjOBJ_GEOM, self.config.ground_geom)}
            if self.config.ground_geom is not None else
            set(np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
                               & (model.geom_bodyid == 0)).tolist())
        )
        self.ball_body = int(model.geom_bodyid[self.ball])
        self.paddle_body = int(model.geom_bodyid[self.paddle])
        if model.site_bodyid[self.site] != self.paddle_body:
            raise ValueError("The paddle face site must belong to the paddle body.")
        self._jac_ball = np.zeros((3, model.nv))
        self._jac_paddle = np.zeros((3, model.nv))

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Bounce tracking requires {name!r} in the MuJoCo scene.")
        return result

    def reset(self, data):
        self.start_time = float(data.time)
        self.valid_bounces = 0
        self.consecutive = 0
        self.max_consecutive = 0
        self.points = []
        self.time_to_fall = None
        self.termination_reason = None
        self.time_to_failure = None
        self.paddle_ground_contact = False
        self.ball_too_far = False
        self.candidate = None
        self.separated_since = None

    def sample(self, data):
        """Read a consistent mj_forward state before/after one physics step."""
        rotation = data.site_xmat[self.site].reshape(3, 3)
        normal = rotation[:, 2].copy()
        point = data.geom_xpos[self.ball]
        # Evaluate paddle velocity at the ball position: includes rotation.
        mujoco.mj_jac(self.model, data, self._jac_ball, None, point, self.ball_body)
        mujoco.mj_jac(self.model, data, self._jac_paddle, None, point, self.paddle_body)
        ball_velocity = self._jac_ball @ data.qvel
        relative_velocity = (self._jac_ball - self._jac_paddle) @ data.qvel
        contacts = []
        other_contact = False
        paddle_ground_contact = False
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if contact.dist <= 0 and self.paddle in pair and pair.intersection(self.ground_geoms):
                paddle_ground_contact = True
            if self.ball not in pair or contact.dist > 0:
                continue
            if self.paddle not in pair:
                other_contact = True
                continue
            contact_normal = np.array(contact.frame[:3], copy=True)
            if contact.geom1 == self.ball:
                contact_normal *= -1
            contacts.append({
                "face": bool(
                    np.dot(contact_normal, normal) >= self.config.min_face_alignment
                    and np.dot(point - data.site_xpos[self.site], normal) > 0
                    and normal[2] >= self.config.min_up_alignment
                ),
                "point": (rotation.T @ (contact.pos - data.site_xpos[self.site]))[:2].copy(),
            })
        return {
            "time": float(data.time),
            "height": float(point[2]),
            "normal": normal,
            "speed": float(relative_velocity @ normal),
            "ball_vz": float(ball_velocity[2]),
            "contacts": contacts,
            "other_contact": other_contact,
            "paddle_ground_contact": paddle_ground_contact,
            "ball_paddle_distance": float(np.linalg.norm(point - data.site_xpos[self.site])),
        }

    def update(self, before, after):
        """Return at most one event after confirmed flight following a contact."""
        cfg = self.config
        if after["height"] < cfg.fall_height and self.time_to_fall is None:
            self.time_to_fall = after["time"] - self.start_time
            self.consecutive = 0
        self.paddle_ground_contact |= bool(
            before.get("paddle_ground_contact", False) or after.get("paddle_ground_contact", False)
        )
        self.ball_too_far |= after.get("ball_paddle_distance", 0.) > cfg.max_ball_paddle_distance
        if self.termination_reason is None:
            if self.time_to_fall is not None:
                self.termination_reason = "ball_fell"
            elif self.paddle_ground_contact:
                self.termination_reason = "paddle_ground_contact"
            elif self.ball_too_far:
                self.termination_reason = "ball_too_far"
            if self.termination_reason is not None:
                self.time_to_failure = after["time"] - self.start_time
        if self.termination_reason is not None:
            self.consecutive = 0
            self.candidate = None
            self.separated_since = None
            return None
        if before["other_contact"] or after["other_contact"]:
            self.consecutive = 0
            self.candidate = None
            self.separated_since = None
            return None
        contacts = before["contacts"] or after["contacts"]
        if contacts:
            self.separated_since = None
            if self.candidate is None:
                chosen = next((c for c in contacts if c["face"]), contacts[0])
                self.candidate = {
                    "face": chosen["face"], "point": chosen["point"].copy(),
                    "incoming_speed": before["speed"],
                }
                if not chosen["face"] or before["speed"] > -cfg.min_incoming_speed:
                    self.consecutive = 0
            if not any(contact["face"] for contact in contacts):
                self.candidate["face"] = False
                self.consecutive = 0
            return None

        if self.candidate is None:
            return None
        if self.separated_since is None:
            self.separated_since = after["time"]
        if after["time"] - self.separated_since + 1e-12 < cfg.min_flight_time:
            return None

        candidate, self.candidate = self.candidate, None
        self.separated_since = None
        if not (candidate["face"]
                and candidate["incoming_speed"] <= -cfg.min_incoming_speed
                and after["speed"] >= cfg.min_outgoing_speed
                and after["ball_vz"] > 0):
            self.consecutive = 0
            return None

        self.valid_bounces += 1
        self.consecutive += 1
        self.max_consecutive = max(self.max_consecutive, self.consecutive)
        self.points.append(candidate["point"])
        return {
            "time_s": after["time"] - self.start_time,
            "incoming_speed_m_s": candidate["incoming_speed"],
            "outgoing_speed_m_s": after["speed"],
            "contact_position_local_m": candidate["point"].tolist(),
        }

    def metrics(self, data):
        dispersion = None
        if len(self.points) >= 2:
            points = np.asarray(self.points)
            dispersion = float(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))))
        return {
            "valid_bounces": self.valid_bounces,
            "consecutive_bounces": self.consecutive,
            "max_consecutive_bounces": self.max_consecutive,
            "elapsed_sim_time_s": float(data.time) - self.start_time,
            "time_to_fall_s": self.time_to_fall,
            "fell": self.time_to_fall is not None,
            "paddle_ground_contact": bool(self.paddle_ground_contact),
            "ball_too_far": bool(self.ball_too_far),
            "termination_reason": self.termination_reason,
            "time_to_failure_s": self.time_to_failure,
            "success": self.max_consecutive >= self.config.success_bounces,
            "success_target_bounces": self.config.success_bounces,
            "contact_dispersion_m": dispersion,
        }
