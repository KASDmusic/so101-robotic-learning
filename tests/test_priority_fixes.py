"""Run with python -m unittest discover -s tests -v (MuJoCo required)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from io import BytesIO
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import mujoco
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.rewards.reward_ball_aligned_on_z_and_above_paddle import BallAlignedOnZAndAbovePaddleReward
from so101_robotic_learning.bounce_rl.rewards import RewardMultiObjectif
from so101_robotic_learning.bounce_rl.rl import train_ppo, train_sac
from so101_robotic_learning.bounce_rl.rl.rl_utils import (
    EvalVideoSaveBestCallback, evaluate_policy_on_env, get_camera_fps, rollout_policy,
)

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/mjcf/so101_new_calib copy.xml"


class ZeroPolicy:
    def predict(self, obs, deterministic=True):
        if isinstance(obs, dict):
            return np.zeros(6, dtype=np.float32), None
        return np.zeros((len(obs), 1), dtype=np.float32), None


class CameraEpisode(gym.Env):
    """Two frames per step and an identifiable reset frame; auto-reset via SB3."""
    metadata = {"render_modes": ["rgb_array_list"]}
    render_mode = "rgb_array_list"
    camera_fps = 30

    def __init__(self, length, reward):
        self.length = length
        self.reward = reward
        self.action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-100., 100., (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.frames = [0]
        return np.zeros(1, dtype=np.float32), {}

    def pop_render_frames(self):
        frames, self.frames = self.frames, []
        return frames

    def step(self, action):
        self.steps += 1
        self.frames.extend([2 * self.steps - 1, 2 * self.steps])
        done = self.steps >= self.length
        info = {"terminal_render_frames": self.frames.copy()} if done else {}
        return np.array([self.steps], dtype=np.float32), self.reward, done, False, info


class EvaluationTests(unittest.TestCase):
    def test_all_workers_and_terminal_frames(self):
        env = VecMonitor(DummyVecEnv([lambda: CameraEpisode(2, 1.), lambda: CameraEpisode(4, 10.)]))
        try:
            self.assertEqual(get_camera_fps(env), 30)
            result = evaluate_policy_on_env(ZeroPolicy(), env, n_eval_episodes=3,
                                            record_video_first_episode=True)
            self.assertEqual(result["rewards"], [2., 40., 2.])
            self.assertEqual(result["video_frames"], [0, 1, 2, 3, 4])
            self.assertEqual(result["first_episode_length"], 2)
        finally:
            env.close()

    def test_step_cap_and_rollout_contract(self):
        env = DummyVecEnv([lambda: CameraEpisode(100, 2.), lambda: CameraEpisode(2, 3.)])
        try:
            result = evaluate_policy_on_env(ZeroPolicy(), env, n_eval_episodes=2, max_steps=3)
            self.assertEqual(result["rewards"], [6., 6.])
            self.assertEqual(result["episode_dones"], [False, True])
            rollout = rollout_policy(ZeroPolicy(), env, max_steps=2, capture_video=True)
            self.assertEqual(rollout["total_reward"], 4.)
            self.assertFalse(rollout["done"])
            self.assertEqual(rollout["frames"], [0, 1, 2, 3, 4])
        finally:
            env.close()

    def test_callback_crosses_episode_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = EvalVideoSaveBestCallback(None, Path(tmp) / "best", verbose=0)
            callback.episode_count = 9
            callback.locals = {"dones": np.array([True, True])}
            with patch.object(callback, "_evaluate_record_and_save") as evaluate:
                callback._on_step()
                callback._on_step()
                evaluate.assert_called_once()
            self.assertEqual(callback.next_eval_episode, 20)


class PhysicsTests(unittest.TestCase):
    def test_reward_is_repeatable_and_does_not_mutate_physics(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        before = data.xpos.copy()
        reward = BallAlignedOnZAndAbovePaddleReward()
        first = reward.compute(model, data)[0]
        second = reward.compute(model, data)[0]
        np.testing.assert_array_equal(data.xpos, before)
        self.assertEqual(first, second)

    def test_sac_honors_reward_and_matches_ppo_default(self):
        reward = BallAlignedOnZAndAbovePaddleReward()
        env = train_sac.make_env(XML, reward=reward)()
        try:
            obs, _ = env.reset()
            expected = reward.compute(env.unwrapped.model, env.unwrapped.data)[0]
            self.assertEqual(env.unwrapped.reward.compute(env.unwrapped.model, env.unwrapped.data)[0], expected)
            self.assertEqual([term.name for term in env.unwrapped.reward.components], ["custom"])
        finally:
            env.close()
        env = train_sac.make_env(XML)()
        try:
            self.assertIsInstance(env.unwrapped.reward, RewardMultiObjectif)
            self.assertEqual([term.name for term in env.unwrapped.reward.components],
                             ["bounces", "shaping", "failure", "energy", "ball_segment"])
        finally:
            env.close()

    def test_actual_vecenv_preserves_terminal_video(self):
        env = DummyVecEnv([lambda: BounceEnv(str(XML), max_episode_steps=2,
                                            record_video=True)])
        class Policy:
            def predict(self, obs, deterministic=True):
                return np.zeros((1, 6), dtype=np.float32), None
        try:
            result = evaluate_policy_on_env(Policy(), env, n_eval_episodes=1,
                                            record_video_first_episode=True)
            self.assertEqual(result["first_episode_length"], 2)
            # t=0, 1/30, 2/30; includes a frame generated during the terminal step.
            self.assertEqual(len(result["video_frames"]), 3)
            self.assertTrue(all(f.shape == (84, 84, 3) for f in result["video_frames"]))
        finally:
            env.close()

    def test_human_tests_capture_and_close_for_both_algorithms(self):
        for module, algorithm in [(train_ppo, "PPO"), (train_sac, "SAC")]:
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp:
                env = BounceEnv(str(XML), max_episode_steps=2, render_mode="human", record_video=True)
                with patch.object(env, "_sync_human_viewer"), patch.object(module, "make_env", return_value=lambda: env), patch.object(getattr(module, algorithm), "load", return_value=ZeroPolicy()):
                    module.test(XML, tmp, n_episodes=1, max_steps=2, verbose=0)
                    self.assertFalse(list(Path(tmp).rglob("*.mp4")))
                    event_file = next(Path(tmp).rglob("events.out.tfevents.*"))
                    events = EventAccumulator(str(event_file.parent))
                    events.Reload()
                    for tag in ("test/video", "test/gripper_cam", "test/cameras"):
                        videos = events.Images(tag)
                        self.assertEqual([video.step for video in videos], [1])
                        with Image.open(BytesIO(videos[0].encoded_image_string)) as image:
                            self.assertEqual(image.format, "GIF")
                    self.assertIsNone(env.renderer)


if __name__ == "__main__":
    unittest.main()
