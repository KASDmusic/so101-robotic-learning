"""Mechanical-work costs must integrate all substeps without changing physics."""
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv

FIXTURE = Path(__file__).parent / "assets/termination_fixture.xml"


class EnergyPenaltyTests(unittest.TestCase):
    def make_env(self, xml=FIXTURE, **kwargs):
        env = BounceEnv(str(xml), render_profile="train_fast", **kwargs)
        self.addCleanup(env.close)
        env.reset()
        env.data.qvel[0] = .2
        mujoco.mj_forward(env.model, env.data)
        return env

    def test_constant_force_has_expected_energy_and_reward(self):
        env = self.make_env(frame_skip=4, energy_penalty_weight=2.)
        _, reward, terminated, truncated, info = env.step(np.array([.5]))
        # F=0.5 N, m=0.1 kg, v0=0.2 m/s. End-of-substep quadrature.
        velocities = .2 + np.arange(1, 5) * .002 * .5 / .1
        expected_j = float(np.sum(.5 * velocities) * .002)
        self.assertAlmostEqual(info["energy_step_j"], expected_j, places=12)
        self.assertAlmostEqual(reward, -2. * expected_j, places=12)
        self.assertEqual(info["penalty_energy"], reward)
        self.assertEqual(info["reward_total"], reward)
        self.assertFalse(terminated or truncated)
        self.assertAlmostEqual(info["task_metrics"]["mechanical_energy_j"], expected_j)
        self.assertEqual(info["task_metrics"]["energy_penalty"], reward)

    def test_braking_is_a_cost_and_never_a_reward(self):
        env = self.make_env(frame_skip=4, energy_penalty_weight=2.)
        _, reward, _, _, info = env.step(np.array([-.5]))
        velocities = .2 - np.arange(1, 5) * .002 * .5 / .1
        expected_j = float(np.sum(.5 * velocities) * .002)
        self.assertLess(env.data.actuator_force[0] * env.data.actuator_velocity[0], 0.)
        self.assertAlmostEqual(info["energy_step_j"], expected_j, places=12)
        self.assertAlmostEqual(reward, -2. * expected_j, places=12)

    def test_opposite_motor_powers_do_not_cancel(self):
        xml = FIXTURE.read_text().replace(
            '</actuator>', '<motor joint="paddle_z" ctrlrange="-1 1" ctrllimited="true"/></actuator>'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two_motors.xml"
            path.write_text(xml)
            env = self.make_env(xml=path, frame_skip=4, energy_penalty_weight=1.)
            _, reward, _, _, info = env.step(np.array([1., -1.]))
            self.assertAlmostEqual(float(np.sum(env.data.actuator_force * env.data.actuator_velocity)), 0.)
            self.assertAlmostEqual(info["energy_step_j"], 2. * .2 * .002 * 4, places=12)
            self.assertAlmostEqual(reward, -info["energy_step_j"])

    def test_step_partition_preserves_physics_energy_and_reset_clears_totals(self):
        baseline = self.make_env(frame_skip=4)  # zero weight preserves previous reward
        penalized = self.make_env(frame_skip=1, energy_penalty_weight=2.)
        baseline_obs, baseline_reward, _, _, baseline_info = baseline.step(np.array([.5]))
        total_reward = 0.
        for _ in range(4):
            obs, reward, _, _, info = penalized.step(np.array([.5]))
            total_reward += reward
        np.testing.assert_array_equal(baseline_obs["state"], obs["state"])
        self.assertEqual(baseline_reward, 0.)
        self.assertGreater(baseline_info["energy_step_j"], 0.)
        self.assertAlmostEqual(info["task_metrics"]["mechanical_energy_j"], baseline_info["energy_step_j"], places=12)
        self.assertAlmostEqual(total_reward, -2. * baseline_info["energy_step_j"], places=12)
        _, reset_info = penalized.reset()
        self.assertEqual(reset_info["task_metrics"]["mechanical_energy_j"], 0.)
        self.assertEqual(reset_info["task_metrics"]["energy_penalty"], 0.)

    def test_early_failure_counts_only_executed_substeps_and_survives_autoreset(self):
        env = self.make_env(frame_skip=20, energy_penalty_weight=2.)
        vec = DummyVecEnv([lambda: env])
        self.addCleanup(vec.close)
        vec.reset()
        env.data.qvel[0] = .2
        joint = env.model.body_jntadr[env.bounce_tracker.ball_body]
        start = env.model.jnt_qposadr[joint]
        env.data.qpos[start:start + 3] = env.data.site_xpos[env.bounce_tracker.site] + [2.01, 0., 0.]
        mujoco.mj_forward(env.model, env.data)
        _, rewards, dones, infos = vec.step(np.array([[.5]]))
        expected_j = .002 * .5 * .21
        self.assertTrue(dones[0])
        self.assertAlmostEqual(infos[0]["energy_step_j"], expected_j, places=12)
        self.assertAlmostEqual(infos[0]["task_metrics"]["mechanical_energy_j"], expected_j, places=12)
        self.assertAlmostEqual(float(rewards[0]), -5. - 2. * expected_j, places=6)
        self.assertEqual(env._episode_mechanical_energy_j, 0.)

    def test_no_actuators_has_zero_mechanical_work(self):
        env = self.make_env(xml=FIXTURE.with_name("bounce_fixture.xml"), frame_skip=4,
                            energy_penalty_weight=2.)
        _, reward, _, _, info = env.step(np.zeros(0))
        self.assertEqual(info["energy_step_j"], 0.)
        self.assertEqual(info["penalty_energy"], 0.)
        self.assertEqual(reward, 0.)

    def test_negative_or_nonfinite_weight_is_rejected(self):
        for weight in (-1., np.inf, np.nan):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                BounceEnv(str(FIXTURE), energy_penalty_weight=weight)


if __name__ == "__main__":
    unittest.main()
