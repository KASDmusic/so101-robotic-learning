"""Visualization camera must never change the policy's input or physics."""
import unittest
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.rl.rl_utils import evaluate_policy_on_env

XML = Path(__file__).resolve().parents[1] / "assets/mjcf/so101_new_calib copy.xml"


class VisualizationCameraTests(unittest.TestCase):
    def test_recording_does_not_change_observations_including_history(self):
        baseline = BounceEnv(str(XML), max_episode_steps=3, latent_steps=[1, 2])
        external = BounceEnv(str(XML), max_episode_steps=3, latent_steps=[1, 2],
                             video_camera_name="overview_cam", video_width=96,
                             video_height=64, record_video=True, render_mode="rgb_array")
        try:
            self.assertEqual(baseline.observation_space, external.observation_space)
            self.assertEqual(baseline.action_space, external.action_space)
            expected, _ = baseline.reset(seed=7)
            actual, _ = external.reset(seed=7)
            for index in range(3):
                np.testing.assert_array_equal(actual["image"], expected["image"])
                np.testing.assert_array_equal(actual["state"], expected["state"])
                self.assertEqual(actual["image"].shape, (84, 84, 9))
                self.assertEqual(external.render().shape, (64, 96, 3))
                action = np.zeros(baseline.model.nu, dtype=np.float32)
                expected, reward, done, truncated, _ = baseline.step(action)
                actual, actual_reward, actual_done, actual_truncated, info = external.step(action)
                self.assertEqual((reward, done, truncated), (actual_reward, actual_done, actual_truncated))
            self.assertTrue(all(frame.shape == (64, 96, 3) for frame in info["terminal_render_frames"]))
        finally:
            baseline.close()
            external.close()
        self.assertIsNone(external.video_renderer)

    def test_external_pixels_are_distinct_from_policy_pixels(self):
        env = BounceEnv(str(XML), video_camera_name="overview_cam", render_mode="rgb_array")
        try:
            obs, _ = env.reset()
            self.assertEqual(env.render().shape, obs["image"].shape)
            self.assertFalse(np.array_equal(env.render(), obs["image"]))
        finally:
            env.close()

    def test_training_does_not_render_external_camera(self):
        env = BounceEnv(str(XML), video_camera_name="overview_cam", max_episode_steps=2)
        try:
            env.reset()
            env.step(np.zeros(env.model.nu, dtype=np.float32))
            self.assertIsNone(env.video_renderer)
            self.assertEqual(env.pop_render_frames(), [])
        finally:
            env.close()

    def test_unknown_camera_is_rejected_before_renderer_creation(self):
        with self.assertRaisesRegex(ValueError, "Video camera"):
            BounceEnv(str(XML), video_camera_name="missing_camera")

    def test_both_streams_survive_terminal_step_and_history_is_not_recorded(self):
        env = DummyVecEnv([lambda: BounceEnv(
            str(XML), max_episode_steps=2, latent_steps=[1, 2], record_video=True,
            video_camera_name="overview_cam", video_width=96, video_height=64,
        )])
        class Policy:
            def predict(self, obs, deterministic=True):
                return np.zeros((1, 6), dtype=np.float32), None
        try:
            result = evaluate_policy_on_env(Policy(), env, n_eval_episodes=2,
                                            record_video_first_episode=True)
            self.assertEqual(len(result["video_frames"]), 3)
            self.assertEqual(len(result["gripper_video_frames"]), 3)
            self.assertTrue(all(frame.shape == (64, 96, 3) for frame in result["video_frames"]))
            self.assertTrue(all(frame.shape == (84, 84, 3) for frame in result["gripper_video_frames"]))
            # Auto-reset images and the second evaluation episode must not leak in.
            self.assertEqual(result["first_episode_length"], 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
