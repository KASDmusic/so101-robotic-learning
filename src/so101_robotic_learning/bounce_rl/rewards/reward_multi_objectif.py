"""Composable weighted objectives, compatible with existing compute(model, data) rewards."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from math import fsum, isfinite


@dataclass(frozen=True)
class RewardContext:
    """Per-decision events and integrals already measured by the environment."""

    bounce_count: int = 0
    failed: bool = False
    energy_j: float = 0.0
    segment_distance_integral_m_s: float = 0.0


class RewardComponent(ABC):
    """Common interface for leaves and composites. compute must not mutate physics."""

    @abstractmethod
    def compute(self, model, data, context: RewardContext | None = None) -> tuple[float, dict]:
        raise NotImplementedError

    def reset(self, model, data):
        """Optional hook for stateful components, called at each environment reset."""


class StateRewardAdapter(RewardComponent):
    """Adapt existing two-argument rewards without changing their implementation."""

    def __init__(self, reward):
        if not callable(getattr(reward, "compute", None)):
            raise TypeError("A reward must implement compute(model, data).")
        self.reward = reward

    def compute(self, model, data, context=None):
        return self.reward.compute(model, data)


class ConstantReward(RewardComponent):
    def __init__(self, value):
        self.value = float(value)
        if not isfinite(self.value):
            raise ValueError("Constant reward must be finite.")

    def compute(self, model, data, context=None):
        return self.value, {}


class BounceReward(RewardComponent):
    def compute(self, model, data, context=None):
        return float((context or RewardContext()).bounce_count), {}


class FailurePenalty(RewardComponent):
    def compute(self, model, data, context=None):
        return float((context or RewardContext()).failed), {}


class EnergyPenalty(RewardComponent):
    def compute(self, model, data, context=None):
        return float((context or RewardContext()).energy_j), {}


class BallSegmentPenalty(RewardComponent):
    def compute(self, model, data, context=None):
        return float((context or RewardContext()).segment_distance_integral_m_s), {}


@dataclass(frozen=True)
class RewardTerm:
    name: str
    component: RewardComponent
    weight: float
    penalty: bool = False


class RewardMultiObjectif(RewardComponent):
    """Composite of named rewards, penalties or nested composites.

    Rewards preserve their sign. Penalties contribute -weight * abs(value),
    accepting either positive costs or legacy negative penalty scores.
    Weights are nonnegative and are never implicitly normalized.
    """

    def __init__(self):
        self._terms: dict[str, RewardTerm] = {}

    @property
    def components(self):
        return tuple(self._terms.values())

    @staticmethod
    def _weight(value):
        value = float(value)
        if not isfinite(value) or value < 0:
            raise ValueError("Reward weights must be finite and nonnegative.")
        return value

    def _contains(self, component):
        return self is component or any(
            isinstance(term.component, RewardMultiObjectif) and term.component._contains(component)
            for term in self.components
        )

    def _add(self, name, component, weight, penalty):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each objective requires a nonempty name.")
        if name in self._terms:
            raise ValueError(f"Duplicate objective name: {name!r}.")
        if isinstance(component, RewardMultiObjectif) and component._contains(self):
            raise ValueError("Reward composites cannot contain cycles.")
        adapted = component if isinstance(component, RewardComponent) else StateRewardAdapter(component)
        self._terms[name] = RewardTerm(name, adapted, self._weight(weight), penalty)
        return self

    def add_reward(self, name, component, weight=1.0):
        return self._add(name, component, weight, penalty=False)

    def add_penalty(self, name, component, weight=1.0):
        return self._add(name, component, weight, penalty=True)

    def set_weight(self, name, weight):
        self._terms[name] = replace(self._terms[name], weight=self._weight(weight))
        return self

    def remove(self, name):
        del self._terms[name]
        return self

    def reset(self, model, data):
        for term in self.components:
            term.component.reset(model, data)

    def compute(self, model, data, context=None):
        context = context or RewardContext()
        details = {}
        contributions = []
        for term in self.components:
            value, info = term.component.compute(model, data, context)
            value = float(value)
            if not isfinite(value):
                raise ValueError(f"Objective {term.name!r} returned a nonfinite score.")
            if not isinstance(info, dict):
                raise TypeError(f"Objective {term.name!r} must return a dict of details.")
            contribution = term.weight * (-abs(value) if term.penalty else value)
            if not isfinite(contribution):
                raise ValueError(f"Objective {term.name!r} has a nonfinite contribution.")
            contributions.append(contribution)
            details[term.name] = {
                "value": value, "weight": term.weight,
                "kind": "penalty" if term.penalty else "reward",
                "contribution": contribution, "details": info,
            }
        total = fsum(contributions)
        return total, {"reward_total": total, "components": details}


def make_bounce_reward(shaping, *, bounce_weight, shaping_weight, failure_weight,
                       energy_weight, segment_weight):
    """Default objective tree, preserving the existing BounceEnv reward formula."""
    return (RewardMultiObjectif()
            .add_reward("bounces", BounceReward(), bounce_weight)
            .add_reward("shaping", shaping if shaping is not None else ConstantReward(0), shaping_weight)
            .add_penalty("failure", FailurePenalty(), failure_weight)
            .add_penalty("energy", EnergyPenalty(), energy_weight)
            .add_penalty("ball_segment", BallSegmentPenalty(), segment_weight))
