"""Physical event, reward and evaluation regressions for the juggling objective."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.env.bounce_events import BounceConfig, BounceTracker
from so101_robotic_learning.bounce_rl.rewards.reward_paddle_parallel import PaddleParallelReward
from so101_robotic_learning.bounce_rl.rl.rl_utils import (
    EvalVideoSaveBestCallback, evaluate_policy_on_env, evaluation_score,
    summarize_task_metrics,
)
from test_priority_fixes import CameraEpisode, ZeroPolicy, XML

from reward_fixtures import task_reward

FIXTURE = Path(__file__).parent / "assets/bounce_fixture.xml"


def sample(time, contact=False, speed=-1., vz=1., face=True, other=False,
           height=0.6, point=(0., 0.)):
    return {
        "time": time, "height": height, "normal": np.array([0., 0., 1.]),
        "speed": speed, "ball_vz": vz, "other_contact": other,
        "contacts": [{"face": face, "point": np.array(point)}] if contact else [],
    }


class EventTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(FIXTURE))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.tracker = BounceTracker(self.model, BounceConfig(success_bounces=2))
        self.tracker.reset(self.data)

    def bounce(self, start=0., face=True, incoming=-1., outgoing=1., vz=1., point=(0., 0.)):
        self.tracker.update(sample(start, speed=incoming), sample(start + .002, True, face=face, point=point))
        self.tracker.update(sample(start + .002, True), sample(start + .004, speed=outgoing, vz=vz))
        self.tracker.update(sample(start + .004, speed=outgoing), sample(start + .006, speed=outgoing, vz=vz))
        return self.tracker.update(sample(start + .026, speed=outgoing), sample(start + .028, speed=outgoing, vz=vz))

    def test_sustained_contact_and_chatter_only_count_once(self):
        self.tracker.update(sample(0.), sample(.002, True))
        for i in range(1, 50):
            self.assertIsNone(self.tracker.update(sample(i * .002, True, speed=1.), sample((i + 1) * .002, True, speed=1.)))
        self.tracker.update(sample(.100, speed=1.), sample(.102, speed=1.))
        self.tracker.update(sample(.102, speed=1.), sample(.104, True, speed=1.))
        self.tracker.update(sample(.104, True, speed=1.), sample(.106, speed=1.))
        self.tracker.update(sample(.106, speed=1.), sample(.108, speed=1.))
        event = self.tracker.update(sample(.128, speed=1.), sample(.130, speed=1.))
        self.assertIsNotNone(event)
        self.assertEqual(self.tracker.valid_bounces, 1)
        self.assertIsNone(self.tracker.update(sample(.132), sample(.134)))
        self.assertEqual(self.tracker.valid_bounces, 1)

    def test_wrong_face_resting_ball_and_downward_departure_are_rejected(self):
        for kwargs in ({"face": False}, {"incoming": 0.}, {"outgoing": -1.}, {"vz": -1.}):
            with self.subTest(kwargs=kwargs):
                self.tracker.reset(self.data)
                self.assertIsNone(self.bounce(**kwargs))
                self.assertEqual(self.tracker.valid_bounces, 0)

    def test_streak_dispersion_fall_and_reset(self):
        self.bounce()
        self.bounce(start=.2, point=(.02, 0.))
        metrics = self.tracker.metrics(self.data)
        self.assertTrue(metrics["success"])
        self.assertAlmostEqual(metrics["contact_dispersion_m"], .01)
        self.tracker.update(sample(.4), sample(.402, other=True))
        self.assertEqual(self.tracker.consecutive, 0)
        self.assertEqual(self.tracker.max_consecutive, 2)
        self.tracker.update(sample(.5), sample(.502, height=.05))
        self.tracker.update(sample(.6), sample(.602, height=.05))
        self.assertAlmostEqual(self.tracker.time_to_fall, .502)
        self.tracker.reset(self.data)
        self.assertEqual(self.tracker.valid_bounces, 0)
        self.assertIsNone(self.tracker.metrics(self.data)["time_to_fall_s"])

    def test_config_rejects_invalid_thresholds(self):
        for kwargs in ({"min_flight_time": 0}, {"success_bounces": 0}, {"min_face_alignment": 2}):
            with self.assertRaises(ValueError):
                BounceConfig(**kwargs)

    def test_relative_velocity_includes_paddle_translation_and_rotation(self):
        xml = FIXTURE.read_text()
        for joint, velocity, expected in (
            ('<joint type="slide" axis="0 0 1"/>', -2., 2.),
            ('<joint type="hinge" axis="1 0 0"/>', 1., -.1),
        ):
            with self.subTest(joint=joint):
                model = mujoco.MjModel.from_xml_string(xml.replace(
                    '<body name="paddle_mount" pos="0 0 0.5">',
                    '<body name="paddle_mount" pos="0 0 0.5">' + joint,
                ))
                data = mujoco.MjData(model)
                data.qvel[0] = velocity
                # Paddle has one DOF before the ball free joint.
                data.qpos[2] = .1
                mujoco.mj_forward(model, data)
                tracker = BounceTracker(model)
                measured = tracker.sample(data)
                self.assertAlmostEqual(measured["speed"], expected)
                self.assertAlmostEqual(measured["ball_vz"], 0.)

    def test_real_physics_produces_three_bounces_and_then_settles(self):
        events = []
        for _ in range(1500):
            before = self.tracker.sample(self.data)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            event = self.tracker.update(before, self.tracker.sample(self.data))
            if event is not None:
                events.append(event)
        self.assertEqual(len(events), 3)
        self.assertEqual(self.tracker.max_consecutive, 3)
        self.assertTrue(all(e["incoming_speed_m_s"] < 0 < e["outgoing_speed_m_s"] for e in events))

    def test_underside_is_not_the_striking_face_in_actual_contacts(self):
        # Remove gravity, launch upward at the lower face of the paddle.
        self.model.opt.gravity[:] = 0
        self.data.qpos[:3] = [0., 0., .46]
        self.data.qvel[2] = 1.
        mujoco.mj_forward(self.model, self.data)
        contacts = []
        for _ in range(100):
            before = self.tracker.sample(self.data)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            after = self.tracker.sample(self.data)
            contacts.extend(after["contacts"])
            self.tracker.update(before, after)
        self.assertTrue(contacts)
        self.assertTrue(all(not c["face"] for c in contacts))
        self.assertEqual(self.tracker.valid_bounces, 0)


class ObjectiveIntegrationTests(unittest.TestCase):
    def test_substeps_accumulate_events_and_reward_uses_bounces(self):
        env = BounceEnv(str(FIXTURE), reward=task_reward(), frame_skip=500, bounce_config=BounceConfig(success_bounces=2))
        try:
            env.reset()
            _, reward, _, _, info = env.step(np.zeros(0, dtype=np.float32))
            count = len(info["bounce_events"])
            self.assertGreaterEqual(count, 2)
            self.assertEqual(info["task_metrics"]["valid_bounces"], count)
            self.assertEqual(reward, 5. * count)
            self.assertTrue(info["task_metrics"]["success"])
            env.reset()
            self.assertEqual(env.bounce_tracker.valid_bounces, 0)
        finally:
            env.close()

    def test_fall_stops_at_substep_and_reports_exact_time(self):
        env = BounceEnv(str(FIXTURE), reward=task_reward(), frame_skip=200)
        try:
            env.reset()
            env.data.qpos[:3] = [.5, 0., .11]  # miss the paddle
            env.data.qvel[2] = -1.
            mujoco.mj_forward(env.model, env.data)
            _, reward, terminated, _, info = env.step(np.zeros(0, dtype=np.float32))
            self.assertTrue(terminated)
            self.assertEqual(reward, -5.)
            self.assertLess(env.data.time, .02)
            self.assertEqual(info["task_metrics"]["time_to_fall_s"], env.data.time)
        finally:
            env.close()

    def test_chosen_face_normal_and_orientation_reward(self):
        model = mujoco.MjModel.from_xml_path(str(XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle_mount")
        np.testing.assert_allclose(data.site_xmat[site].reshape(3, 3)[:, 2], -data.xmat[body].reshape(3, 3)[:, 1], atol=1e-12)

        model = mujoco.MjModel.from_xml_path(str(FIXTURE))
        data = mujoco.MjData(model)
        reward = PaddleParallelReward(paddle_site_name="paddle_face")
        mujoco.mj_forward(model, data)
        self.assertEqual(reward.compute(model, data)[0], 1.)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "paddle_face")
        # Static site transforms are compiled by MuJoCo; compile the flipped scene.
        flipped = FIXTURE.read_text().replace(
            'name="paddle_face"', 'name="paddle_face" quat="0 1 0 0"'
        )
        model = mujoco.MjModel.from_xml_string(flipped)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(reward.compute(model, data)[0], 0.)


class TaskEpisode(CameraEpisode):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info["task_metrics"] = {
            "valid_bounces": self.steps, "max_consecutive_bounces": self.steps,
            "success": self.steps >= 3, "elapsed_sim_time_s": self.steps * .04,
            "fell": done, "time_to_fall_s": self.steps * .04 if done else None,
            "contact_dispersion_m": None,
        }
        return obs, reward, done, truncated, info


class EvaluationMetricTests(unittest.TestCase):
    def test_metrics_survive_autoreset_and_evaluation_cap(self):
        env = DummyVecEnv([lambda: TaskEpisode(2, 1.), lambda: TaskEpisode(10, 10.)])
        try:
            result = evaluate_policy_on_env(ZeroPolicy(), env, n_eval_episodes=2, max_steps=3)
            self.assertEqual(result["task_metrics"]["mean_valid_bounces"], 2.5)
            self.assertEqual(result["task_metrics"]["success_rate"], .5)
            self.assertEqual(result["task_metrics"]["fall_rate"], .5)
            self.assertAlmostEqual(result["task_metrics"]["mean_time_to_fall_s"], .08)
            self.assertTrue(result["episode_metrics"][1]["evaluation_censored"])
            self.assertIsNone(result["episode_metrics"][1]["time_to_fall_s"])
        finally:
            env.close()

    def test_best_model_prefers_task_success_and_saves_report(self):
        metrics = {"success": True, "valid_bounces": 5, "max_consecutive_bounces": 5,
                   "elapsed_sim_time_s": 2., "fell": False, "time_to_fall_s": None,
                   "contact_dispersion_m": .01}
        result = {"mean_reward": 1., "std_reward": 0., "rewards": [1.],
                  "task_metrics": summarize_task_metrics([metrics]), "episode_metrics": [metrics],
                  "video_frames": None, "first_episode_length": 50, "first_episode_time": .1}
        worse = dict(result, mean_reward=100., task_metrics=dict(result["task_metrics"], success_rate=0.))
        self.assertGreater(evaluation_score(result), evaluation_score(worse))
        with tempfile.TemporaryDirectory() as tmp:
            callback = EvalVideoSaveBestCallback(None, Path(tmp) / "best", save_video=False, verbose=0)
            callback.model = SimpleNamespace(logger=SimpleNamespace(record=Mock(), dump=Mock()), save=Mock())
            with patch("so101_robotic_learning.bounce_rl.rl.rl_utils.evaluate_policy_on_env", side_effect=[result, worse]):
                callback._evaluate_record_and_save()
                callback.episode_count = 10
                callback._evaluate_record_and_save()
            callback.model.save.assert_called_once()
            report = json.loads((Path(tmp) / "best_eval_000010.json").read_text())
            self.assertEqual(report["task_metrics"]["success_rate"], 0.)
            callback.model.logger.record.assert_any_call("eval/success_rate", 1.)


if __name__ == "__main__":
    unittest.main()
