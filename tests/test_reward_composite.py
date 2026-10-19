"""Composite behavior, legacy adapters and full PPO/SAC objective integration."""
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.rewards import (
    BallSegmentPenalty, BounceReward, ConstantReward, EnergyPenalty, FailurePenalty,
    RewardComponent, RewardContext, RewardMultiObjectif, make_default_reward,
)
from so101_robotic_learning.bounce_rl.rl import train_ppo, train_sac
from so101_robotic_learning.bounce_rl.rl.rl_utils import summarize_task_metrics

from reward_fixtures import task_reward

FIXTURE = Path(__file__).parent / "assets/termination_fixture.xml"


def legacy(value, **details):
    return SimpleNamespace(compute=lambda model, data: (value, details))


class StatefulReward(RewardComponent):
    def __init__(self):
        self.resets = 0

    def reset(self, model, data):
        self.resets += 1

    def compute(self, model, data, context=None):
        return float(self.resets), {}


class RewardCompositeTests(unittest.TestCase):
    def test_weighted_rewards_preserve_sign_and_penalties_accept_both_conventions(self):
        objective = (RewardMultiObjectif()
                     .add_reward("gain", legacy(4), 2.)
                     .add_reward("signed_reward", legacy(-1), .5)
                     .add_penalty("positive_cost", legacy(3), .5)
                     .add_penalty("legacy_negative_penalty", legacy(-2), .5))
        score, info = objective.compute(None, None)
        self.assertEqual(score, 5.)
        self.assertEqual(info["reward_total"], 5.)
        self.assertEqual(info["components"]["legacy_negative_penalty"]["contribution"], -1.)
        self.assertEqual(info["components"]["positive_cost"]["kind"], "penalty")

    def test_nested_objectives_keep_colliding_details_separate(self):
        child = (RewardMultiObjectif().add_reward("a", legacy(1, score="first"))
                 .add_reward("b", legacy(3, score="second"), 2.))
        objective = (RewardMultiObjectif().add_reward("group", child, .5)
                     .add_penalty("cost", ConstantReward(2), .25))
        score, info = objective.compute(None, None)
        self.assertEqual(score, 3.)
        nested = info["components"]["group"]["details"]["components"]
        self.assertEqual(nested["a"]["details"]["score"], "first")
        self.assertEqual(nested["b"]["details"]["score"], "second")

    def test_empty_composite_weight_updates_and_removal(self):
        objective = RewardMultiObjectif()
        self.assertEqual(objective.compute(None, None), (0., {"reward_total": 0., "components": {}}))
        objective.add_reward("value", ConstantReward(2)).set_weight("value", 3.)
        self.assertEqual(objective.compute(None, None)[0], 6.)
        objective.set_weight("value", 0.)
        self.assertEqual(objective.compute(None, None)[1]["components"]["value"]["value"], 2.)
        objective.remove("value")
        self.assertEqual(objective.components, ())

    def test_cycles_and_duplicate_names_are_rejected(self):
        parent, child = RewardMultiObjectif(), RewardMultiObjectif()
        parent.add_reward("child", child)
        with self.assertRaises(ValueError):
            child.add_reward("parent", parent)
        with self.assertRaises(ValueError):
            parent.add_reward("self", parent)
        with self.assertRaises(ValueError):
            parent.add_reward("child", ConstantReward(1))

    def test_invalid_weights_components_and_values_are_rejected(self):
        for weight in (-1., np.nan, np.inf):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                RewardMultiObjectif().add_reward("value", ConstantReward(1), weight)
        with self.assertRaises(ValueError):
            RewardMultiObjectif().add_reward(" ", ConstantReward(1))
        with self.assertRaises(TypeError):
            RewardMultiObjectif().add_reward("missing_compute", object())
        for value in (np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RewardMultiObjectif().add_reward("invalid", legacy(value)).compute(None, None)
        with self.assertRaises(TypeError):
            RewardMultiObjectif().add_reward("bad_info", SimpleNamespace(
                compute=lambda model, data: (1., None))).compute(None, None)

    def test_step_context_reaches_nested_leaves_and_is_immutable(self):
        penalties = (RewardMultiObjectif().add_penalty("failure", FailurePenalty(), 5.)
                     .add_penalty("energy", EnergyPenalty(), .5)
                     .add_penalty("distance", BallSegmentPenalty(), 2.))
        objective = (RewardMultiObjectif().add_reward("bounces", BounceReward(), 5.)
                     .add_reward("costs", penalties))
        context = RewardContext(bounce_count=2, failed=True, energy_j=2.,
                                segment_distance_integral_m_s=.5)
        self.assertEqual(objective.compute(None, None, context)[0], 3.)
        self.assertEqual(objective.compute(None, None)[0], 0.)
        with self.assertRaises(FrozenInstanceError):
            context.bounce_count = 10

    def test_reset_propagates_to_nested_stateful_leaves(self):
        leaf = StatefulReward()
        objective = RewardMultiObjectif().add_reward("group", RewardMultiObjectif().add_reward("stateful", leaf))
        self.assertEqual(objective.compute(None, None)[0], 0.)
        objective.reset(None, None)
        self.assertEqual(objective.compute(None, None)[0], 1.)


class CompositeEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = BounceEnv(str(FIXTURE), render_profile="train_fast", **kwargs)
        self.addCleanup(env.close)
        env.reset()
        return env

    def place_ball(self, env, offset):
        joint = env.model.body_jntadr[env.bounce_tracker.ball_body]
        start = env.model.jnt_qposadr[joint]
        env.data.qpos[start:start + 3] = env.data.site_xpos[env.bounce_tracker.site] + offset
        mujoco.mj_forward(env.model, env.data)

    def test_none_builds_full_default_objective_and_fresh_trees(self):
        env = self.make_env()
        self.assertEqual({term.name: term.weight for term in env.reward.components}, {
            "bounces": 5., "shaping": .05, "failure": 5., "energy": .01, "ball_segment": 1.,
        })
        self.assertAlmostEqual(env.step(np.zeros(1))[1], .1)
        first, second = make_default_reward(), make_default_reward()
        first.set_weight("energy", 100.)
        self.assertEqual({term.name: term.weight for term in second.components}["energy"], .01)

    def test_explicit_composite_matches_chosen_formula_and_episode_contributions(self):
        objective = task_reward(shaping=legacy(2., marker="legacy details"), segment=3.)
        env = self.make_env(reward=objective)
        self.place_ball(env, np.array([.3, .4, .7]))
        _, reward, _, _, info = env.step(np.zeros(1))
        self.assertAlmostEqual(reward, .1 - 3. * .5 * .04)
        self.assertEqual(info["marker"], "legacy details")
        self.assertEqual(set(info["reward_components"]), {"bounces", "shaping", "failure", "energy", "ball_segment"})
        self.assertAlmostEqual(sum(part["contribution"] for part in info["reward_components"].values()), reward)
        self.assertAlmostEqual(sum(info["task_metrics"]["reward_components"].values()), reward)
        metrics = summarize_task_metrics([info["task_metrics"]])
        self.assertAlmostEqual(metrics["reward_components/ball_segment"], -.06)
        env.reset()
        self.assertEqual(env._task_metrics()["reward_components"], {})

    def test_nested_composite_is_the_complete_objective_without_outer_scaling(self):
        objective = (RewardMultiObjectif().add_reward("positive", ConstantReward(10))
                     .add_penalty("cost", ConstantReward(2)))
        env = self.make_env(reward=objective)
        _, reward, _, _, info = env.step(np.zeros(1))
        self.assertEqual(reward, 8.)
        self.assertEqual(set(info["reward_components"]), {"positive", "cost"})

    def test_custom_objective_keeps_failure_detection_without_adding_default_costs(self):
        objective = (RewardMultiObjectif().add_reward("constant", ConstantReward(7))
                     .add_penalty("failure", FailurePenalty(), 3.))
        env = self.make_env(reward=objective)
        self.place_ball(env, np.array([2.01, 0., .7]))
        _, reward, terminated, _, info = env.step(np.zeros(1))
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "ball_too_far")
        self.assertEqual(reward, 4.)
        self.assertEqual(info["penalty_failure"], -3.)
        self.assertEqual(set(info["reward_components"]), {"constant", "failure"})

    def test_rewards_are_isolated_between_vector_workers_and_factories(self):
        leaf = StatefulReward()
        objective = RewardMultiObjectif().add_reward("stateful", leaf)
        for factory in (train_ppo.make_env, train_sac.make_env):
            first, second = factory(FIXTURE, reward=objective)(), factory(FIXTURE, reward=objective)()
            try:
                first.reset()
                second.reset()
                first.reset()
                self.assertEqual(first.step(np.zeros(1))[1], 2.)
                self.assertEqual(second.step(np.zeros(1))[1], 1.)
                self.assertEqual(leaf.resets, 0)
            finally:
                first.close()
                second.close()

    def test_two_argument_leaf_is_a_full_objective_not_auxiliary_shaping(self):
        env = self.make_env(reward=legacy(2., marker="legacy details"))
        self.place_ball(env, np.array([2.01, 0., .7]))
        _, reward, terminated, _, info = env.step(np.zeros(1))
        self.assertTrue(terminated)
        self.assertEqual(reward, 2.)
        self.assertEqual(set(info["reward_components"]), {"custom"})
        self.assertEqual(info["reward_components"]["custom"]["details"]["marker"], "legacy details")

    def test_single_context_leaf_is_supported_as_full_objective(self):
        env = self.make_env(reward=ConstantReward(7.))
        self.assertEqual(env.step(np.zeros(1))[1], 7.)

    def test_removed_configuration_arguments_and_invalid_reward_are_rejected(self):
        for name in ("reward_composite", "bounce_reward_weight", "shaping_reward_weight",
                     "fall_penalty", "energy_penalty_weight", "ball_segment_penalty_weight"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                BounceEnv(str(FIXTURE), **{name: 1.})
        with self.assertRaises(TypeError):
            BounceEnv(str(FIXTURE), reward=object())


if __name__ == "__main__":
    unittest.main()
