"""Shared gravity stages, advanced by training successes and applied at reset."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from multiprocessing.reduction import ForkingPickler
from pathlib import Path


@dataclass(frozen=True)
class GravityCurriculumConfig:
    start_gravity: float = 3.0
    gravity_step: float = 1.0
    target_gravity: float = 9.81
    success_bounces: int = 5
    success_on_timeout: bool = True

    def __post_init__(self):
        for name in ("start_gravity", "gravity_step", "target_gravity"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive (m/s²).")
        if self.start_gravity > self.target_gravity:
            raise ValueError("start_gravity must not exceed target_gravity.")
        if type(self.success_bounces) is not int or self.success_bounces < 1:
            raise ValueError("success_bounces must be a positive integer.")
        if type(self.success_on_timeout) is not bool:
            raise ValueError("success_on_timeout must be a boolean.")


class GravityCurriculum:
    """One shared stage for training workers, with read-only evaluation."""

    def __init__(self, config=None, level=0):
        self.config = config or GravityCurriculumConfig()
        if not isinstance(self.config, GravityCurriculumConfig):
            raise TypeError("Expected GravityCurriculumConfig.")
        self.final_level = max(0, math.ceil(
            (self.config.target_gravity - self.config.start_gravity) / self.config.gravity_step - 1e-12
        ))
        if type(level) is not int or not 0 <= level <= self.final_level:
            raise ValueError("Invalid gravity curriculum level.")
        self._level = level
        self._shared_level = None

    def share(self, context):
        """Share the stage with workers started by this multiprocessing context."""
        if self._shared_level is None:
            self._shared_level = context.Value("q", self._level)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._shared_level is not None:
            # SB3 cloudpickles env factories. Use the multiprocessing reducer
            # for the counter so its backing memory is shared, not copied.
            state["_shared_level"] = bytes(ForkingPickler.dumps(self._shared_level))
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._shared_level is not None:
            self._shared_level = ForkingPickler.loads(self._shared_level)

    @property
    def level(self):
        return self._level if self._shared_level is None else self._shared_level.value

    def gravity_at(self, level):
        return min(self.config.start_gravity + level * self.config.gravity_step,
                   self.config.target_gravity)

    @property
    def gravity(self):
        return self.gravity_at(self.level)

    def advance(self, episode_level):
        # Successes from workers still running an easier stage cannot skip stages.
        shared = self._shared_level
        with shared.get_lock() if shared is not None else nullcontext():
            level = self.level
            if episode_level != level or level >= self.final_level:
                return False
            if shared is None:
                self._level = level + 1
            else:
                shared.value = level + 1
            return True

    def state(self):
        level = self.level
        return {"config": asdict(self.config), "level": level,
                "gravity_m_s2": self.gravity_at(level), "complete": level == self.final_level}

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
