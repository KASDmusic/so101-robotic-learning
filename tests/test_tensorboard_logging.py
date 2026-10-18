"""Verify persisted TensorBoard events, not only calls to logger.record()."""
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import numpy as np
from PIL import Image
from stable_baselines3.common.logger import configure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from so101_robotic_learning.bounce_rl.rl import train_ppo, train_sac
from so101_robotic_learning.bounce_rl.rl.rl_utils import (
    EvalVideoSaveBestCallback, combine_camera_frames, log_tensorboard_video, summarize_task_metrics,
)

XML = Path(__file__).resolve().parents[1] / "assets/mjcf/so101_new_calib copy.xml"


def read_scalars(directory):
    events = list(Path(directory).rglob("events.out.tfevents.*"))
    if not events:
        raise AssertionError(f"No TensorBoard event file in {directory}")
    accumulator = EventAccumulator(str(events[0].parent), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}


class TensorBoardLoggingTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def check_training(self, module, algorithm_kwargs, expected_steps, record_video=False):
        with tempfile.TemporaryDirectory() as tmp:
            module.train(
                xml_path=XML, root=tmp, tensorboard_log=True,
                env_kwargs={"max_episode_steps": 2, "frame_skip": 1,
                            "video_camera_name": "overview_cam",
                            "video_width": 96, "video_height": 64},
                eval_every_episodes=1, n_eval_episodes=1, eval_max_steps=2,
                save_eval_video=record_video, eval_render_mode=None,
                print_spaces=False, verbose=0, callback_verbose=0,
                device="cpu", **algorithm_kwargs,
            )
            self.assertFalse(list(Path(tmp).rglob("*.mp4")))
            self.assertFalse((Path(tmp) / "videos_sb3").exists())
            self.assertFalse((Path(tmp) / "videos_sac").exists())
            scalars = read_scalars(Path(tmp) / "runs")
            reports = [json.loads(path.read_text()) for path in sorted(
                (Path(tmp) / "models").glob("*_eval_*.json")
            )]
            self.assertEqual(len(reports), len(expected_steps))
            for metric in ("success_rate", "mean_valid_bounces", "mean_max_consecutive_bounces",
                           "mean_duration_s", "fall_rate", "mean_reward", "std_reward"):
                tag = f"eval/{metric}"
                self.assertIn(tag, scalars)
                self.assertEqual([event.step for event in scalars[tag]], expected_steps)
                for event, report in zip(scalars[tag], reports):
                    expected = report.get(metric, report["task_metrics"].get(metric))
                    self.assertAlmostEqual(event.value, expected, places=6)
            # No fall and no impacts in this tiny run: undefined is not zero.
            self.assertNotIn("eval/mean_time_to_fall_s", scalars)
            self.assertNotIn("eval/mean_contact_dispersion_m", scalars)
            if record_video:
                events = next((Path(tmp) / "runs").rglob("events.out.tfevents.*"))
                images = EventAccumulator(str(events.parent), size_guidance={"images": 0})
                images.Reload()
                videos = images.Images("eval/video")
                self.assertEqual([video.step for video in videos], expected_steps)
                for video in videos:
                    with Image.open(BytesIO(video.encoded_image_string)) as image:
                        self.assertEqual(image.format, "GIF")
                        self.assertEqual(image.size, (96, 64))
                gripper_videos = images.Images("eval/gripper_cam")
                self.assertEqual([video.step for video in gripper_videos], expected_steps)
                for video in gripper_videos:
                    with Image.open(BytesIO(video.encoded_image_string)) as image:
                        self.assertEqual(image.format, "GIF")
                        self.assertEqual(image.size, (84, 84))
                combined_videos = images.Images("eval/cameras")
                self.assertEqual([video.step for video in combined_videos], expected_steps)
                for video in combined_videos:
                    with Image.open(BytesIO(video.encoded_image_string)) as image:
                        self.assertEqual(image.size, (160, 84))
            return scalars

    def assert_finite_losses(self, scalars, tags, after_step):
        for tag in tags:
            self.assertIn(tag, scalars)
            self.assertTrue(scalars[tag])
            self.assertTrue(all(np.isfinite(event.value) for event in scalars[tag]))
            self.assertTrue(all(event.step > after_step for event in scalars[tag]))

    def test_ppo_persists_every_evaluation_within_one_rollout(self):
        self.check_training(train_ppo, {
            "total_timesteps": 8, "n_steps": 8, "batch_size": 8, "n_epochs": 1,
            "pi_layers": (16,), "vf_layers": (16,),
        }, [2, 4, 6, 8], record_video=True)

    def test_sac_persists_before_its_normal_log_interval(self):
        self.check_training(train_sac, {
            "total_timesteps": 4, "buffer_size": 16, "learning_starts": 100,
            "batch_size": 2, "pi_layers": (16,), "qf_layers": (16,),
        }, [2, 4], record_video=True)

    def test_tensorboard_video_is_animated_and_preserves_frame_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = configure(tmp, ["tensorboard"])
            frames = [np.full((12, 16, 3), value, dtype=np.uint8) for value in (0, 120, 255)]
            try:
                self.assertTrue(log_tensorboard_video(logger, frames, fps=30, step=42))
                accumulator = EventAccumulator(tmp)
                accumulator.Reload()
                event, = accumulator.Images("eval/video")
                self.assertEqual(event.step, 42)
                with Image.open(BytesIO(event.encoded_image_string)) as image:
                    self.assertEqual(image.n_frames, 3)
                    duration = 0
                    for index in range(image.n_frames):
                        image.seek(index)
                        duration += image.info["duration"]
                    self.assertEqual(duration, 100)  # 3 frames / 30 FPS
            finally:
                logger.close()

    def test_combined_video_keeps_camera_frames_paired(self):
        overview = [np.full((64, 96, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        gripper = [np.full((84, 84, 3), value, dtype=np.uint8) for value in (40, 50, 60)]
        combined = combine_camera_frames(overview, gripper)
        self.assertEqual(len(combined), 3)
        for index, frame in enumerate(combined):
            self.assertEqual(frame.shape, (84, 160, 3))
            np.testing.assert_array_equal(frame[20:, :96], overview[index])
            self.assertTrue(np.all(frame[20:, 96:] == 40 + index * 10))
        with tempfile.TemporaryDirectory() as tmp:
            logger = configure(tmp, ["tensorboard"])
            try:
                log_tensorboard_video(logger, combined, fps=30, step=7, tag="eval/cameras")
                events = EventAccumulator(tmp)
                events.Reload()
                event, = events.Images("eval/cameras")
                self.assertEqual(event.step, 7)
                with Image.open(BytesIO(event.encoded_image_string)) as image:
                    self.assertEqual(image.n_frames, 3)
            finally:
                logger.close()

    def test_combined_video_rejects_unsynchronized_stream_lengths(self):
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            combine_camera_frames([frame], [frame, frame])
        self.assertEqual(combine_camera_frames([frame], None), [])

    def test_sac_training_losses_coexist_with_evaluation_metrics(self):
        scalars = self.check_training(train_sac, {
            "total_timesteps": 8, "buffer_size": 16, "learning_starts": 2,
            "batch_size": 2, "gradient_steps": 1,
            "pi_layers": (16,), "qf_layers": (16,),
        }, [2, 4, 6, 8])
        self.assert_finite_losses(scalars, (
            "train/actor_loss", "train/critic_loss", "train/ent_coef_loss",
        ), after_step=2)

    def test_ppo_training_losses_coexist_with_evaluation_metrics(self):
        scalars = self.check_training(train_ppo, {
            "total_timesteps": 16, "n_steps": 8, "batch_size": 8, "n_epochs": 1,
            "pi_layers": (16,), "vf_layers": (16,),
        }, list(range(2, 17, 2)))
        self.assert_finite_losses(scalars, (
            "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss",
        ), after_step=8)

    def test_optional_metrics_are_flushed_and_not_reused_when_undefined(self):
        metrics = {"success": True, "valid_bounces": 5, "max_consecutive_bounces": 5,
                   "elapsed_sim_time_s": 2., "fell": True, "time_to_fall_s": 1.5,
                   "contact_dispersion_m": .0125}

        def result(episode):
            return {"mean_reward": 10., "std_reward": 0., "rewards": [10.],
                    "task_metrics": summarize_task_metrics([episode]),
                    "episode_metrics": [episode], "video_frames": None,
                    "first_episode_length": 50, "first_episode_time": .1}

        undefined = dict(metrics, fell=False, time_to_fall_s=None, contact_dispersion_m=None)
        with tempfile.TemporaryDirectory() as tmp:
            logger = configure(str(Path(tmp) / "events"), ["tensorboard"])
            callback = EvalVideoSaveBestCallback(None, Path(tmp) / "best", save_video=False, verbose=0)
            callback.model = SimpleNamespace(logger=logger, save=Mock())
            try:
                with patch("so101_robotic_learning.bounce_rl.rl.rl_utils.evaluate_policy_on_env",
                           side_effect=[result(metrics), result(undefined)]):
                    callback.num_timesteps = 10
                    callback._evaluate_record_and_save()
                    callback.num_timesteps = 20
                    callback._evaluate_record_and_save()
                # Read while the writer is open: the callback must already have flushed.
                scalars = read_scalars(Path(tmp) / "events")
                self.assertEqual(len(scalars), 9)
                for tag, value in (("eval/mean_time_to_fall_s", 1.5),
                                   ("eval/mean_contact_dispersion_m", .0125)):
                    self.assertEqual([e.step for e in scalars[tag]], [10])
                    self.assertAlmostEqual(scalars[tag][0].value, value)
                self.assertEqual([e.step for e in scalars["eval/success_rate"]], [10, 20])
            finally:
                logger.close()


if __name__ == "__main__":
    unittest.main()
