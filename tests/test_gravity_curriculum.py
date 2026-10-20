"""Gravity-only stages, shared workers, read-only evaluation and resumed training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv
from so101_robotic_learning.bounce_rl.env.bounce_events import BounceTracker
from so101_robotic_learning.bounce_rl.env.gravity_curriculum import (
    GravityCurriculum, GravityCurriculumConfig, curriculum_path, make_gravity_curriculum,
    save_gravity_curriculum,
)
from so101_robotic_learning.bounce_rl.rl import train_ppo, train_sac


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/mjcf/so101_new_calib copy.xml"
FIXTURE = Path(__file__).parent / "assets/bounce_fixture.xml"


def five_bounces(tracker, before, after):
    """Inject a confirmed streak to exercise curriculum wiring, not contact detection."""
    tracker.consecutive = tracker.max_consecutive = tracker.valid_bounces = 5
    return {"time_s": after["time"]}


class GravityCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def env(self, curriculum=None, training=True, **kwargs):
        return BounceEnv(str(FIXTURE), gravity_curriculum=curriculum,
                         curriculum_training=training, render_profile="train_fast",
                         frame_skip=1, **kwargs)

    def test_config_validation_and_exact_cap(self):
        for settings in ({"start_gravity": 0}, {"gravity_step": -1},
                         {"target_gravity": float("nan")}, {"start_gravity": 10},
                         {"success_bounces": 0}, {"success_bounces": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                GravityCurriculumConfig(**settings)
        curriculum = GravityCurriculum()
        gravities = [curriculum.gravity]
        while curriculum.advance(curriculum.level):
            gravities.append(curriculum.gravity)
        self.assertEqual(gravities, [3., 4., 5., 6., 7., 8., 9., 9.81])
        self.assertTrue(curriculum.state()["complete"])
        self.assertFalse(curriculum.advance(curriculum.level))
        ready = GravityCurriculum(GravityCurriculumConfig(start_gravity=9.81))
        self.assertTrue(ready.state()["complete"])

    def test_five_consecutive_bounces_schedule_one_change_at_next_reset_only(self):
        curriculum = GravityCurriculum()
        env = self.env(curriculum)
        try:
            env.reset()
            mass, dt = env.model.body_mass.copy(), env.model.opt.timestep
            bounce_config = env.bounce_tracker.config
            def four(tracker, before, after):
                tracker.consecutive = tracker.max_consecutive = 4
                tracker.valid_bounces = 20  # Total contacts do not qualify.
                return {"time_s": after["time"]}
            with patch.object(BounceTracker, "update", four):
                env.step(np.zeros(0))
            self.assertEqual(curriculum.level, 0)
            with patch.object(BounceTracker, "update", five_bounces):
                _, _, terminated, _, info = env.step(np.zeros(0))
                env.step(np.zeros(0))
            self.assertFalse(terminated)
            self.assertEqual(curriculum.level, 1)
            self.assertEqual(info["task_metrics"]["gravity_m_s2"], 3.)
            np.testing.assert_array_equal(env.model.opt.gravity, [0, 0, -3.])
            env.reset()
            np.testing.assert_array_equal(env.model.opt.gravity, [0, 0, -4.])
            np.testing.assert_array_equal(env.model.body_mass, mass)
            self.assertEqual(env.model.opt.timestep, dt)
            self.assertEqual(env.bounce_tracker.config, bounce_config)
            with patch.object(BounceTracker, "update", five_bounces):
                env.step(np.zeros(0))
            self.assertEqual(curriculum.level, 2)
        finally:
            env.close()

    def test_vector_autoreset_does_not_skip_levels_and_evaluation_never_advances(self):
        curriculum = GravityCurriculum(GravityCurriculumConfig(target_gravity=5.))
        env = DummyVecEnv([lambda: self.env(curriculum, max_episode_steps=1) for _ in range(2)])
        evaluation = self.env(curriculum, training=False)
        try:
            env.reset()
            with patch.object(BounceTracker, "update", five_bounces):
                _, _, dones, infos = env.step(np.zeros((2, 0)))
                self.assertTrue(all(dones))
                self.assertEqual(curriculum.level, 1)
                self.assertEqual([info["task_metrics"]["gravity_m_s2"] for info in infos], [3., 3.])
                self.assertTrue(all(worker.unwrapped.model.opt.gravity[2] == -4. for worker in env.envs))
                evaluation.reset()
                evaluation.step(np.zeros(0))
                self.assertEqual(curriculum.level, 1)
                env.step(np.zeros((2, 0)))
                self.assertEqual(curriculum.level, 2)
                self.assertEqual(curriculum.gravity, 5.)
                env.step(np.zeros((2, 0)))
                self.assertEqual(curriculum.level, 2)
        finally:
            env.close()
            evaluation.close()

    def test_disabled_curriculum_keeps_scene_gravity_and_observation_shape(self):
        ordinary, staged = self.env(), self.env(GravityCurriculumConfig())
        try:
            obs, info = ordinary.reset()
            staged_obs, _ = staged.reset()
            self.assertEqual(ordinary.model.opt.gravity[2], -9.81)
            self.assertNotIn("gravity_m_s2", info["task_metrics"])
            for name in obs:
                self.assertEqual(obs[name].shape, staged_obs[name].shape)
        finally:
            ordinary.close()
            staged.close()

    def test_sidecar_restores_stage_and_rejects_mismatched_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.zip"
            curriculum = GravityCurriculum(level=3)
            curriculum.save(checkpoint)
            restored = make_gravity_curriculum(resume_path=checkpoint)
            self.assertEqual(restored.state(), curriculum.state())
            self.assertEqual(curriculum_path(checkpoint), Path(tmp) / "model_curriculum.json")
            with self.assertRaisesRegex(ValueError, "differs"):
                make_gravity_curriculum(GravityCurriculumConfig(start_gravity=2.), checkpoint)
            self.assertIsNone(make_gravity_curriculum(resume_path=Path(tmp) / "old_model.zip"))
            save_gravity_curriculum(None, checkpoint)
            self.assertFalse(curriculum_path(checkpoint).exists())

    def test_ppo_and_sac_persist_log_and_resume_curriculum(self):
        for algorithm, module in (("ppo", train_ppo), ("sac", train_sac)):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp:
                config = dict(
                    xml_path=XML, root=tmp, model_dir_name="models/first",
                    total_timesteps=4, env_kwargs={"max_episode_steps": 2, "frame_skip": 1},
                    gravity_curriculum=GravityCurriculumConfig(start_gravity=8.),
                    n_eval_episodes=1, eval_max_steps=2, eval_every_timesteps=2,
                    evaluate_at_end=True, save_eval_video=False, eval_render_mode=None,
                    print_spaces=False, verbose=0, callback_verbose=0, device="cpu", pi_layers=(16,),
                )
                if algorithm == "ppo":
                    config.update(n_steps=2, batch_size=2, n_epochs=1, vf_layers=(16,))
                else:
                    config.update(buffer_size=16, learning_starts=100, batch_size=2, qf_layers=(16,))
                with patch.object(BounceTracker, "update", five_bounces):
                    result = module.train(**config)
                self.assertEqual(result["task_metrics"]["mean_gravity_m_s2"], 9.81)
                self.assertEqual(result["task_metrics"]["curriculum_complete_rate"], 1.)
                model_dir = Path(tmp) / "models/first"
                last = model_dir / f"{algorithm}_bounce_last.zip"
                for name in ("last", "best"):
                    state = json.loads(curriculum_path(model_dir / f"{algorithm}_bounce_{name}.zip").read_text())
                    self.assertEqual(state["level"], 2)
                    self.assertEqual(state["gravity_m_s2"], 9.81)
                reports = [json.loads(p.read_text()) for p in model_dir.glob("*_eval_step_*.json")]
                self.assertEqual([report["selection_score"][0] for report in sorted(reports, key=lambda r: r["training_timesteps"])], [1, 2])
                event_file = next((Path(tmp) / "runs").rglob("events.out.tfevents.*"))
                events = EventAccumulator(str(event_file.parent))
                events.Reload()
                self.assertEqual(events.Scalars("curriculum/next_gravity_m_s2")[-1].value,
                                 np.float32(9.81))
                self.assertIn("eval/mean_gravity_m_s2", events.Tags()["scalars"])
                self.assertTrue({1, 3}.issubset({event.step for event in events.Scalars("curriculum/level")}))
                config.update(resume_from=last, gravity_curriculum=None,
                              model_dir_name="models/resumed", total_timesteps=2)
                result = module.train(**config)
                self.assertEqual(result["task_metrics"]["mean_gravity_m_s2"], 9.81)
                state = json.loads(curriculum_path(Path(tmp) / f"models/resumed/{algorithm}_bounce_last.zip").read_text())
                self.assertEqual(state["level"], 2)


if __name__ == "__main__":
    unittest.main()
