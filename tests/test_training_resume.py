"""Exercise real PPO/SAC checkpoints, optimizers, buffers and persisted logs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from so101_robotic_learning.bounce_rl.rl import train_ppo, train_sac
from so101_robotic_learning.bounce_rl.rl.checkpoints import replay_buffer_path


XML = Path(__file__).resolve().parents[1] / "assets/mjcf/so101_new_calib copy.xml"


class TrainingResumeTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self, root, algo):
        config = dict(
            xml_path=XML, root=root, model_dir_name="models/first", total_timesteps=4,
            env_kwargs={"max_episode_steps": 2, "frame_skip": 1},
            eval_every_episodes=1, n_eval_episodes=1, eval_max_steps=2,
            eval_render_mode=None, save_eval_video=False, print_spaces=False,
            verbose=0, callback_verbose=0, device="cpu", pi_layers=(16,),
        )
        if algo == "ppo":
            config.update(n_steps=4, batch_size=4, n_epochs=1, vf_layers=(16,))
        else:
            config.update(buffer_size=16, learning_starts=0, batch_size=2, qf_layers=(16,))
        return config

    def test_resume_both_algorithms_preserves_weights_optimizer_and_continues_updates(self):
        for algo, module, algorithm in (("ppo", train_ppo, PPO), ("sac", train_sac, SAC)):
            with self.subTest(algo=algo), tempfile.TemporaryDirectory() as tmp:
                config = self.config(tmp, algo)
                module.train(**config)
                first = Path(tmp) / f"models/first/{algo}_bounce_last.zip"
                original_bytes = first.read_bytes()
                old = algorithm.load(first, device="cpu")
                old_weights = copy.deepcopy(old.policy.state_dict())
                optimizer = old.policy.optimizer if algo == "ppo" else old.actor.optimizer
                old_steps = [int(state["step"]) for state in optimizer.state.values()]
                self.assertTrue(old_steps)

                load = module.load_training_model
                def checked_load(*args, **kwargs):
                    model = load(*args, **kwargs)
                    self.assertEqual(model.num_timesteps, 4)
                    self.assertEqual(model._n_updates, old._n_updates)
                    for name, value in old_weights.items():
                        torch.testing.assert_close(model.policy.state_dict()[name], value)
                    restored = model.policy.optimizer if algo == "ppo" else model.actor.optimizer
                    self.assertEqual([int(state["step"]) for state in restored.state.values()], old_steps)
                    return model

                config.update(resume_from=first.relative_to(tmp).with_suffix(""),
                              model_dir_name="models/resumed", learning_rate=.9, pi_layers=(32,))
                with patch.object(module, "load_training_model", side_effect=checked_load):
                    module.train(**config)
                resumed_path = Path(tmp) / f"models/resumed/{algo}_bounce_last.zip"
                resumed = algorithm.load(resumed_path, device="cpu")
                self.assertEqual(resumed.num_timesteps, 8)
                self.assertGreater(resumed._n_updates, old._n_updates)
                self.assertEqual(resumed.learning_rate, old.learning_rate)
                self.assertEqual(resumed.policy_kwargs, old.policy_kwargs)
                self.assertTrue(any(not torch.equal(value, resumed.policy.state_dict()[name])
                                    for name, value in old_weights.items()))
                self.assertEqual(first.read_bytes(), original_bytes)

                run_dir = Path(tmp) / f"runs/{algo}_bounce"
                self.assertEqual(sorted(path.name for path in run_dir.iterdir()),
                                 [f"{algo.upper()}_1", f"{algo.upper()}_2"])
                events = EventAccumulator(str(run_dir / f"{algo.upper()}_2"))
                events.Reload()
                self.assertEqual([item.step for item in events.Scalars("eval/mean_reward")], [6, 8])
                report = next((Path(tmp) / "models/resumed").glob("*_eval_*.json"))
                self.assertGreater(json.loads(report.read_text())["training_timesteps"], 4)

                if algo == "sac":
                    old.load_replay_buffer(replay_buffer_path(first))
                    resumed.load_replay_buffer(replay_buffer_path(resumed_path))
                    self.assertEqual(resumed.replay_buffer.size(), 8)
                    np.testing.assert_array_equal(resumed.replay_buffer.actions[:4], old.replay_buffer.actions[:4])
                    best = resumed_path.with_name("sac_bounce_best.zip")
                    self.assertTrue(best.is_file())
                    self.assertTrue(replay_buffer_path(best).is_file())

    def test_sac_old_checkpoint_without_buffer_collects_before_updating(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp, "sac")
            config.update(save_replay_buffer=False)
            train_sac.train(**config)
            first = Path(tmp) / "models/first/sac_bounce_last.zip"
            old = SAC.load(first, device="cpu")
            self.assertFalse(list(Path(tmp).rglob("*.pkl")))
            config.update(resume_from=first, model_dir_name="models/resumed", learning_starts=10)
            with self.assertWarnsRegex(UserWarning, "empty buffer"):
                train_sac.train(**config)
            resumed = SAC.load(Path(tmp) / "models/resumed/sac_bounce_last.zip", device="cpu")
            self.assertEqual(resumed.num_timesteps, 8)
            self.assertEqual(resumed._n_updates, old._n_updates)
            for name, value in old.policy.state_dict().items():
                torch.testing.assert_close(resumed.policy.state_dict()[name], value)
            self.assertFalse(list(Path(tmp).rglob("*.pkl")))
            config.update(model_dir_name="models/after_warmup", learning_starts=2,
                          tensorboard_log=False)
            with self.assertWarnsRegex(UserWarning, "empty buffer"):
                train_sac.train(**config)
            warmed = SAC.load(Path(tmp) / "models/after_warmup/sac_bounce_last.zip", device="cpu")
            self.assertEqual(warmed._n_updates, old._n_updates + 2)
            self.assertEqual(len(list((Path(tmp) / "runs/sac_bounce").iterdir())), 2)

    def test_sac_explicit_buffer_and_worker_count_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp, "sac")
            train_sac.train(**config)
            first = Path(tmp) / "models/first/sac_bounce_last.zip"
            alternate = Path(tmp) / "separate.pkl"
            replay_buffer_path(first).rename(alternate)
            config.update(resume_from=first, replay_buffer_from="separate.pkl",
                          model_dir_name="models/resumed")
            train_sac.train(**config)
            resumed = SAC.load(Path(tmp) / "models/resumed/sac_bounce_last.zip", device="cpu")
            resumed.load_replay_buffer(Path(tmp) / "models/resumed/sac_bounce_last_replay_buffer.pkl")
            self.assertEqual(resumed.replay_buffer.size(), 8)
            config.update(nb_vec_env_train=2)
            with self.assertRaisesRegex(ValueError, "counts must match"):
                train_sac.train(**config)

    def test_incompatible_observations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp, "ppo")
            train_ppo.train(**config)
            config.update(resume_from="models/first/ppo_bounce_last.zip",
                          model_dir_name="models/resumed",
                          env_kwargs={"latent_steps": [1], "frame_skip": 1})
            with self.assertRaisesRegex(ValueError, "Observation spaces do not match"):
                train_ppo.train(**config)

    def test_missing_paths_fail_before_creating_environments(self):
        for module in (train_ppo, train_sac):
            with self.subTest(module=module.__name__), patch.object(module, "make_env") as make:
                with self.assertRaises(FileNotFoundError):
                    module.train(XML, root=".", resume_from="missing_checkpoint.zip")
                make.assert_not_called()
        with self.assertRaisesRegex(ValueError, "requires resume_from"):
            train_sac.train(XML, root=".", replay_buffer_from="buffer.pkl")


if __name__ == "__main__":
    unittest.main()
