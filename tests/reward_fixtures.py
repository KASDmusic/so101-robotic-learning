"""Explicit objectives for isolated tests of physics, events and individual costs."""
from so101_robotic_learning.bounce_rl.rewards import (
    BallSegmentPenalty, BounceReward, ConstantReward, EnergyPenalty, FailurePenalty,
    RewardMultiObjectif,
)


def task_reward(*, shaping=None, bounces=5., shaping_weight=.05,
                failure=5., energy=0., segment=0.):
    return (RewardMultiObjectif()
            .add_reward("bounces", BounceReward(), bounces)
            .add_reward("shaping", shaping if shaping is not None else ConstantReward(0), shaping_weight)
            .add_penalty("failure", FailurePenalty(), failure)
            .add_penalty("energy", EnergyPenalty(), energy)
            .add_penalty("ball_segment", BallSegmentPenalty(), segment))
