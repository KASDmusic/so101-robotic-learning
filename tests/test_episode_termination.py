"""Failure boundaries must terminate at physics frequency and survive auto-reset."""
import unittest
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.env.bounce_events import BounceConfig
from so101_robotic_learning.bounce_rl.rl.rl_utils import summarize_task_metrics

from reward_fixtures import task_reward

FIXTURE = Path(__file__).parent / "assets/termination_fixture.xml"


class EpisodeTerminationTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = BounceEnv(str(FIXTURE), reward=task_reward(), render_profile="train_fast", **kwargs)
        self.addCleanup(env.close)
        env.reset()
        return env

    def place_ball(self, env, offset):
        joint = env.model.body_jntadr[env.bounce_tracker.ball_body]
        start = env.model.jnt_qposadr[joint]
        env.data.qpos[start:start + 3] = env.data.site_xpos[env.bounce_tracker.site] + offset
        mujoco.mj_forward(env.model, env.data)

    def test_distance_is_strictly_greater_than_two_metres_in_three_dimensions(self):
        env = self.make_env(frame_skip=20)
        for offset in ([2., 0., 0.], [2.001, 0., 0.], [0., 0., 2.001], [1.5, 1.5, 0.]):
            with self.subTest(offset=offset):
                env.reset()
                self.place_ball(env, np.asarray(offset))
                _, reward, terminated, truncated, info = env.step(np.zeros(1))
                outside = np.linalg.norm(offset) > 2.
                self.assertEqual(terminated, outside)
                self.assertFalse(truncated)
                self.assertEqual(reward, -5. if outside else 0.)
                self.assertEqual(info["task_metrics"]["ball_too_far"], outside)
                self.assertFalse(info["task_metrics"]["fell"])
                if outside:
                    self.assertEqual(info["termination_reason"], "ball_too_far")
                    self.assertAlmostEqual(env.data.time, .002)
                    self.assertEqual(info["penalty_fall"], 0.)
                    self.assertEqual(info["penalty_failure"], -5.)

    def test_real_paddle_floor_contact_terminates_at_first_substep(self):
        env = self.make_env(frame_skip=20)
        env.data.qpos[0] = -.497  # bottom of paddle penetrates floor by 2 mm
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env.bounce_tracker.sample(env.data)["paddle_ground_contact"])
        _, reward, terminated, truncated, info = env.step(np.zeros(1))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -5.)
        self.assertAlmostEqual(env.data.time, .002)
        self.assertEqual(info["termination_reason"], "paddle_ground_contact")
        self.assertTrue(info["task_metrics"]["paddle_ground_contact"])
        self.assertIsNone(info["task_metrics"]["time_to_fall_s"])
        env.reset()
        self.assertIsNone(env.bounce_tracker.termination_reason)
        self.assertFalse(env.bounce_tracker.paddle_ground_contact)
        self.assertFalse(env.step(np.zeros(1))[2])

    def test_distance_crossing_during_frame_skip_stops_without_waiting_for_decision_end(self):
        env = self.make_env(frame_skip=20)
        self.place_ball(env, np.array([1.95, 0., 0.]))
        joint = env.model.body_jntadr[env.bounce_tracker.ball_body]
        env.data.qvel[env.model.jnt_dofadr[joint]] = 10.
        mujoco.mj_forward(env.model, env.data)
        _, _, terminated, _, info = env.step(np.zeros(1))
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "ball_too_far")
        self.assertAlmostEqual(env.data.time, .006)
        self.assertEqual(info["task_metrics"]["time_to_failure_s"], env.data.time)

    def test_floor_contact_from_another_body_does_not_count_as_paddle_contact(self):
        env = self.make_env()
        self.place_ball(env, np.array([.5, 0., -.49]))
        sample = env.bounce_tracker.sample(env.data)
        self.assertTrue(sample["other_contact"])
        self.assertFalse(sample["paddle_ground_contact"])
        info = env.step(np.zeros(1))[4]
        self.assertEqual(info["termination_reason"], "ball_fell")

    def test_configurable_distance_and_named_ground_validation(self):
        env = self.make_env(bounce_config=BounceConfig(max_ball_paddle_distance=1., ground_geom="floor"))
        self.place_ball(env, np.array([1.01, 0., 0.]))
        self.assertTrue(env.step(np.zeros(1))[2])
        for distance in (0., -1., np.inf, np.nan):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                BounceConfig(max_ball_paddle_distance=distance)
        with self.assertRaisesRegex(ValueError, "missing_ground"):
            BounceEnv(str(FIXTURE), bounce_config=BounceConfig(ground_geom="missing_ground"))

    def test_vecenv_preserves_reason_and_terminal_observation_before_reset(self):
        env = self.make_env(record_video=True)
        vec = DummyVecEnv([lambda: env])
        self.addCleanup(vec.close)
        vec.reset()
        self.place_ball(env, np.array([2.01, 0., 0.]))
        obs, rewards, dones, infos = vec.step(np.zeros((1, 1)))
        self.assertTrue(dones[0])
        self.assertEqual(rewards[0], -5.)
        self.assertEqual(infos[0]["termination_reason"], "ball_too_far")
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertIn("terminal_render_frames", infos[0])
        self.assertIn("terminal_observation_frames", infos[0])
        self.assertFalse(np.array_equal(obs["state"][0], infos[0]["terminal_observation"]["state"]))
        self.assertIsNone(env.bounce_tracker.termination_reason)
        self.assertFalse(env.bounce_tracker.ball_too_far)
        metrics = summarize_task_metrics([infos[0]["task_metrics"]])
        self.assertEqual(metrics["ball_too_far_rate"], 1.)
        self.assertEqual(metrics["paddle_ground_contact_rate"], 0.)
        self.assertEqual(metrics["fall_rate"], 0.)


if __name__ == "__main__":
    unittest.main()
