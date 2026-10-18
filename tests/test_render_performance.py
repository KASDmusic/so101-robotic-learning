"""Rendering savings must preserve physics, policy inputs and recorded episodes."""
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.rl.rl_utils import evaluate_policy_on_env
from so101_robotic_learning.bounce_rl.rl.train_ppo import make_env as make_ppo_env
from so101_robotic_learning.bounce_rl.rl.train_sac import make_env as make_sac_env

XML = Path(__file__).resolve().parents[1] / "assets/mjcf/so101_new_calib copy.xml"


class ZeroPolicy:
    def predict(self, obs, deterministic=True):
        return np.zeros((len(obs["state"]), 6), dtype=np.float32), None


class RenderPerformanceTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = BounceEnv(str(XML), render_profile="train_fast", **kwargs)
        self.addCleanup(env.close)
        return env

    def test_fast_profile_renders_once_per_decision_and_keeps_all_substeps(self):
        env = self.make_env()
        env.reset()
        self.assertEqual(env.camera_fps, 25)
        self.assertFalse(env.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW])
        self.assertFalse(env.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION])
        self.assertEqual(env.renderer._mjr_context.offSamples, 0)
        with patch.object(env.renderer, "render", wraps=env.renderer.render) as render, \
             patch.object(env.bounce_tracker, "update", wraps=env.bounce_tracker.update) as update:
            for _ in range(3):
                env.step(np.zeros(env.model.nu))
            self.assertEqual(render.call_count, 3)
            self.assertEqual(update.call_count, 3 * env.frame_skip)
        self.assertIsNone(env.video_renderer)

    def test_fast_profile_preserves_physics_and_rewards(self):
        standard = BounceEnv(str(XML))
        self.addCleanup(standard.close)
        fast = self.make_env()
        standard.reset(seed=7)
        fast.reset(seed=7)
        for _ in range(3):
            action = np.linspace(-0.1, 0.1, fast.model.nu)
            expected = standard.step(action)
            actual = fast.step(action)
            np.testing.assert_array_equal(expected[0]["state"], actual[0]["state"])
            self.assertEqual(expected[1:4], actual[1:4])
            self.assertEqual(expected[4]["task_metrics"], actual[4]["task_metrics"])

    def test_standard_video_does_not_change_fast_policy_pixels(self):
        baseline = self.make_env(latent_steps=[1, 2])
        recorded = self.make_env(latent_steps=[1, 2], record_video=True,
                                 video_camera_name="overview_cam", video_width=96, video_height=64)
        expected, _ = baseline.reset()
        actual, _ = recorded.reset()
        for _ in range(3):
            np.testing.assert_array_equal(actual["image"], expected["image"])
            recorded._maybe_update_camera()  # repeated display calls reuse pixels
            np.testing.assert_array_equal(recorded._last_image, actual["image"][:, :, :3])
            expected = baseline.step(np.zeros(baseline.model.nu))[0]
            actual = recorded.step(np.zeros(recorded.model.nu))[0]
        self.assertTrue(recorded.video_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW])
        self.assertEqual(recorded.video_renderer._mjr_context.offSamples, recorded._standard_offsamples)

    def test_only_first_worker_first_episode_renders_external_view(self):
        first = self.make_env(max_episode_steps=2, record_video=True, render_mode="rgb_array_list",
                              video_camera_name="overview_cam", video_width=96, video_height=64)
        second = self.make_env(max_episode_steps=4, record_video=True, render_mode="rgb_array_list",
                               video_camera_name="overview_cam", video_width=96, video_height=64)
        # Precreate renderers to count actual draws, including automatic resets.
        for worker in (first, second):
            worker.video_renderer = worker._create_renderer(96, 64, "standard")
        vec = DummyVecEnv([lambda: first, lambda: second])
        with patch.object(first.video_renderer, "render", wraps=first.video_renderer.render) as draw_first, \
             patch.object(second.video_renderer, "render", wraps=second.video_renderer.render) as draw_second:
            result = evaluate_policy_on_env(ZeroPolicy(), vec, n_eval_episodes=3,
                                            record_video_first_episode=True)
            self.assertEqual(len(result["rewards"]), 3)
            self.assertEqual(draw_first.call_count, 3)  # reset and two decisions
            self.assertEqual(draw_second.call_count, 0)
            self.assertEqual(len(result["video_frames"]), 3)
            self.assertEqual(len(result["gripper_video_frames"]), 3)
        self.assertIsNone(first._capture_episodes_remaining)
        first.reset()
        self.assertTrue(first._capture_this_episode)

    def test_disabled_recording_and_failure_restore_capture_budget(self):
        env = self.make_env(max_episode_steps=2, record_video=True, video_camera_name="overview_cam")
        vec = DummyVecEnv([lambda: env])
        evaluate_policy_on_env(ZeroPolicy(), vec, n_eval_episodes=1)
        self.assertIsNone(env.video_renderer)
        with patch.object(ZeroPolicy, "predict", side_effect=RuntimeError("policy failed")):
            with self.assertRaisesRegex(RuntimeError, "policy failed"):
                evaluate_policy_on_env(ZeroPolicy(), vec, record_video_first_episode=True)
        self.assertIsNone(env._capture_episodes_remaining)
        env.reset()
        self.assertTrue(env._capture_this_episode)

    def test_training_factories_default_to_fast_and_allow_standard(self):
        for factory in (make_ppo_env, make_sac_env):
            for kwargs, expected in ((None, "train_fast"), ({"render_profile": "standard"}, "standard")):
                env = factory(XML, env_kwargs=kwargs)()
                try:
                    self.assertEqual(env.unwrapped.render_profile, expected)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
