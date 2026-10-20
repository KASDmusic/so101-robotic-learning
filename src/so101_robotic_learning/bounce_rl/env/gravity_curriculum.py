"""Shared gravity stages, advanced by training successes and applied at reset."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class GravityCurriculumConfig:
    start_gravity: float = 3.0
    gravity_step: float = 1.0
    target_gravity: float = 9.81
    success_bounces: int = 5

    def __post_init__(self):
        for name in ("start_gravity", "gravity_step", "target_gravity"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive (m/s²).")
        if self.start_gravity > self.target_gravity:
            raise ValueError("start_gravity must not exceed target_gravity.")
        if type(self.success_bounces) is not int or self.success_bounces < 1:
            raise ValueError("success_bounces must be a positive integer.")


class GravityCurriculum:
    """One shared state for DummyVecEnv workers, with read-only evaluation."""

    def __init__(self, config=None, level=0):
        self.config = config or GravityCurriculumConfig()
        if not isinstance(self.config, GravityCurriculumConfig):
            raise TypeError("Expected GravityCurriculumConfig.")
        self.final_level = max(0, math.ceil(
            (self.config.target_gravity - self.config.start_gravity) / self.config.gravity_step - 1e-12
        ))
        if type(level) is not int or not 0 <= level <= self.final_level:
            raise ValueError("Invalid gravity curriculum level.")
        self.level = level

    @property
    def gravity(self):
        return min(self.config.start_gravity + self.level * self.config.gravity_step,
                   self.config.target_gravity)

    def advance(self, episode_level):
        # Successes from workers still running an easier stage cannot skip stages.
        if episode_level != self.level or self.level >= self.final_level:
            return False
        self.level += 1
        return True

    def state(self):
        return {"config": asdict(self.config), "level": self.level,
                "gravity_m_s2": self.gravity, "complete": self.level == self.final_level}

    def save(self, model_path):
        path = curriculum_path(model_path)
        path.write_text(json.dumps(self.state(), indent=2, allow_nan=False), encoding="utf-8")


def curriculum_path(model_path):
    path = Path(model_path)
    name = path.stem if path.suffix == ".zip" else path.name
    return path.with_name(name + "_curriculum.json")


def save_gravity_curriculum(curriculum, model_path):
    """Keep metadata consistent when an existing checkpoint is overwritten."""
    if curriculum is None:
        curriculum_path(model_path).unlink(missing_ok=True)
    else:
        curriculum.save(model_path)


def make_gravity_curriculum(config=None, resume_path=None):
    """Restore a checkpoint stage, or create an opt-in curriculum for new training."""
    if config is not None and not isinstance(config, GravityCurriculumConfig):
        raise TypeError("gravity_curriculum must be GravityCurriculumConfig or None.")
    path = curriculum_path(resume_path) if resume_path is not None else None
    if path is not None and path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        saved_config = GravityCurriculumConfig(**state["config"])
        if config is not None and config != saved_config:
            raise ValueError("Curriculum differs from the checkpoint. Use its saved configuration.")
        return GravityCurriculum(saved_config, level=state["level"])
    return GravityCurriculum(config) if config is not None else None
