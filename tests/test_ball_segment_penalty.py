"""Finite segment geometry and time-scaled distance penalties."""
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.rewards.reward_utils import distance_to_vertical_segment

FIXTURE = Path(__file__).parent / "assets/termination_fixture.xml"


class BallSegmentPenaltyTests(unittest.TestCase):
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

    def test_projection_uses_both_endpoints_and_does_not_modify_inputs(self):
        centre = np.array([1., 2., 3.])
        for offset, expected, height in (
            ([0., 0., .2], 0., .2),
            ([0., 0., 1.2], 0., 1.2),
            ([0., 0., .7], 0., .7),
            ([0., 0., .1], .1, .2),
            ([0., 0., 1.4], .2, 1.2),
            ([.3, .4, .7], .5, .7),
            ([.3, .4, 1.5], np.sqrt(.34), 1.2),
        ):
            with self.subTest(offset=offset):
                point = centre + offset
                original = point.copy()
                distance, closest = distance_to_vertical_segment(point, centre)
                self.assertAlmostEqual(distance, expected, places=12)
                np.testing.assert_allclose(closest, centre + [0., 0., height], atol=1e-12)
                np.testing.assert_array_equal(point, original)
                np.testing.assert_array_equal(centre, [1., 2., 3.])

    def test_constant_distance_has_expected_time_integrated_penalty(self):
        env = self.make_env(frame_skip=20, ball_segment_penalty_weight=2.)
        self.place_ball(env, np.array([.3, .4, .7]))
        _, reward, terminated, truncated, info = env.step(np.zeros(1))
        self.assertFalse(terminated or truncated)
        self.assertAlmostEqual(info["ball_segment_distance_m"], .5)
        self.assertAlmostEqual(info["segment_distance_integral_m_s"], .5 * .04)
        self.assertAlmostEqual(reward, -2. * .5 * .04)
        self.assertEqual(info["penalty_ball_segment"], reward)
        self.assertAlmostEqual(info["task_metrics"]["mean_ball_segment_distance_m"], .5)
        self.assertAlmostEqual(info["task_metrics"]["ball_segment_penalty"], reward)

    def test_ball_on_segment_has_no_penalty(self):
        env = self.make_env(ball_segment_penalty_weight=2.)
        self.place_ball(env, np.array([0., 0., .7]))
        _, reward, _, _, info = env.step(np.zeros(1))
        self.assertEqual(reward, 0.)
        self.assertEqual(info["ball_segment_distance_m"], 0.)

    def test_partition_and_zero_weight_preserve_physics_and_reset_clears_metrics(self):
        baseline = self.make_env(frame_skip=20)
        penalized = self.make_env(frame_skip=5, ball_segment_penalty_weight=2.)
        for env in (baseline, penalized):
            self.place_ball(env, np.array([.3, .4, .7]))
        baseline_obs, baseline_reward, _, _, baseline_info = baseline.step(np.zeros(1))
        total_reward = 0.
        for _ in range(4):
            obs, reward, _, _, info = penalized.step(np.zeros(1))
            total_reward += reward
        np.testing.assert_array_equal(baseline_obs["state"], obs["state"])
        self.assertEqual(baseline_reward, 0.)
        self.assertAlmostEqual(total_reward, -2. * baseline_info["segment_distance_integral_m_s"])
        self.assertAlmostEqual(info["task_metrics"]["segment_distance_integral_m_s"], .02)
        _, reset_info = penalized.reset()
        self.assertEqual(reset_info["task_metrics"]["segment_distance_integral_m_s"], 0.)
        self.assertEqual(reset_info["task_metrics"]["ball_segment_penalty"], 0.)
        self.assertIsNone(reset_info["task_metrics"]["mean_ball_segment_distance_m"])

    def test_segment_tracks_paddle_centre_and_custom_geometry(self):
        env = self.make_env(ball_segment_start_height=.3, ball_segment_length=.5)
        self.place_ball(env, np.array([.3, 0., .1]))
        env.data.qvel[0] = .5
        mujoco.mj_forward(env.model, env.data)
        _, _, _, _, info = env.step(np.zeros(1))
        centre = env.data.site_xpos[env.bounce_tracker.site]
        np.testing.assert_allclose(info["segment_closest_point"], centre + [0., 0., .3])
        self.assertAlmostEqual(info["ball_segment_distance_m"], np.sqrt(.3 ** 2 + .22 ** 2))

    def test_early_failure_integrates_only_one_substep_and_autoreset_preserves_report(self):
        env = self.make_env(frame_skip=20, ball_segment_penalty_weight=2.)
        vec = DummyVecEnv([lambda: env])
        self.addCleanup(vec.close)
        vec.reset()
        self.place_ball(env, np.array([2.01, 0., .7]))
        _, rewards, dones, infos = vec.step(np.zeros((1, 1)))
        self.assertTrue(dones[0])
        expected_integral = 2.01 * .002
        self.assertAlmostEqual(infos[0]["segment_distance_integral_m_s"], expected_integral)
        self.assertAlmostEqual(float(rewards[0]), -5. - 2. * expected_integral, places=6)
        self.assertAlmostEqual(infos[0]["task_metrics"]["mean_ball_segment_distance_m"], 2.01)
        self.assertEqual(env._episode_segment_distance_integral_m_s, 0.)

    def test_penalty_adds_to_energy_shaping_and_failure_costs(self):
        shaping = SimpleNamespace(compute=lambda model, data: (2., {}))
        env = self.make_env(reward=shaping, ball_segment_penalty_weight=3., energy_penalty_weight=2.)
        self.place_ball(env, np.array([2.01, 0., .7]))
        env.data.qvel[0] = .2
        mujoco.mj_forward(env.model, env.data)
        _, reward, terminated, _, info = env.step(np.array([.5]))
        self.assertTrue(terminated)
        self.assertLess(info["penalty_energy"], 0.)
        self.assertLess(info["penalty_ball_segment"], 0.)
        expected = (info["reward_bounces"] + .05 * info["reward_shaping"]
                    + info["penalty_failure"] + info["penalty_energy"] + info["penalty_ball_segment"])
        self.assertAlmostEqual(reward, expected)

    def test_invalid_geometry_and_weights_are_rejected(self):
        for kwargs in (
            {"ball_segment_penalty_weight": -1.}, {"ball_segment_penalty_weight": np.nan},
            {"ball_segment_penalty_weight": np.inf}, {"ball_segment_start_height": -.1},
            {"ball_segment_start_height": np.nan}, {"ball_segment_start_height": np.inf},
            {"ball_segment_length": 0.}, {"ball_segment_length": -1.},
            {"ball_segment_length": np.nan}, {"ball_segment_length": np.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BounceEnv(str(FIXTURE), **kwargs)


if __name__ == "__main__":
    unittest.main()
