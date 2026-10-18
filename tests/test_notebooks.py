"""Execute notebook cells in real kernels with small budgets and isolated outputs."""
import json
import queue
import sys
import tempfile
import time
import unittest
from pathlib import Path

from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks/bounce_rl"


class NotebookTests(unittest.TestCase):
    def execute(self, client, code, timeout=120):
        message_id = client.execute(code, store_history=False)
        deadline = time.monotonic() + timeout
        errors = []
        while time.monotonic() < deadline:
            try:
                message = client.get_iopub_msg(timeout=1)
            except queue.Empty:
                continue
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            if message["msg_type"] == "error":
                errors.append("\n".join(message["content"]["traceback"]))
            if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
                self.assertFalse(errors, "\n".join(errors))
                return
        self.fail("Notebook cell exceeded its smoke-test time budget.")

    def check_notebook(self, filename, cwd, algorithm=None):
        notebook = json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))
        manager = KernelManager(kernel_name="python3")
        # Use the interpreter running the tests, without changing user kernel specs.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        with tempfile.TemporaryDirectory() as tmp:
            manager.start_kernel(cwd=str(cwd))
            client = manager.blocking_client()
            client.start_channels()
            try:
                client.wait_for_ready(timeout=60)
                self.execute(client, "import torch; torch.set_num_threads(1)\nimport matplotlib; matplotlib.use('Agg')")
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    code = "".join(cell["source"])
                    with self.subTest(notebook=filename, cell=index):
                        self.execute(client, code)
                        if "configuration" in cell["metadata"].get("tags", []):
                            if algorithm:
                                overrides = {
                                    "total_timesteps": 4, "eval_every_episodes": 1,
                                    "n_eval_episodes": 1, "eval_max_steps": 2,
                                    "device": "cpu", "verbose": 0,
                                    "callback_verbose": 0, "print_spaces": False,
                                }
                                if algorithm == "ppo":
                                    overrides.update(n_steps=4, batch_size=4, n_epochs=1,
                                                     pi_layers=(16,), vf_layers=(16,))
                                else:
                                    overrides.update(buffer_size=16, learning_starts=100,
                                                     batch_size=2, pi_layers=(16,), qf_layers=(16,))
                                self.execute(client, (
                                    f"run_root = Path({tmp!r})\n"
                                    "train_config['root'] = run_root\n"
                                    "env_kwargs.update(max_episode_steps=2, frame_skip=1, video_width=96, video_height=64)\n"
                                    f"train_config.update({overrides!r})\n"
                                    "RUN_POLICY_TEST = True\n"
                                ))
                            else:
                                self.execute(client, "env_kwargs.update(max_episode_steps=2, frame_skip=1, width=84, height=84, video_width=96, video_height=64)\nSIM_DURATION = 0.01")
                if algorithm:
                    self.execute(client, (
                        "assert root == Path(" + repr(str(ROOT)) + ")\n"
                        "model_dir = run_root / train_config['model_dir_name']\n"
                        "reports = [json.loads(path.read_text()) for path in sorted(model_dir.glob('*_eval_*.json'))]\n"
                        "latest = max(reports, key=lambda report: report['training_timesteps'])\n"
                        "assert len(reports) == 2\n"
                        "assert 'success_rate' in latest['task_metrics']\n"
                        "assert (model_dir / '" + algorithm + "_bounce_best.zip').is_file()\n"
                        "assert list((run_root / 'runs').rglob('events.out.tfevents.*'))\n"
                        "assert sorted(path.name for path in (run_root / 'runs').iterdir()) == ['" + algorithm + "_bounce']\n"
                    ))
                    if any("optional-test" in c.get("metadata", {}).get("tags", []) for c in notebook["cells"]):
                        self.execute(client, (
                            "assert len(list(run_root.rglob('test_episode_*.json'))) == 3\n"
                            "assert not list(run_root.rglob('*.mp4'))\n"
                            "assert list((run_root / 'runs').rglob('events.out.tfevents.*'))\n"
                        ))
                else:
                    self.execute(client, (
                        "assert frames_rgb and frames_list\n"
                        "assert metrics_rgb['valid_bounces'] == metrics_list['valid_bounces']\n"
                        "assert metrics_rgb['evaluation_censored']\n"
                    ))
            finally:
                client.stop_channels()
                manager.shutdown_kernel(now=True)

    def test_ppo_notebook_from_its_directory(self):
        self.check_notebook("notebook_train_ppo.ipynb", NOTEBOOKS, "ppo")

    def test_sac_notebook_from_repository_root(self):
        self.check_notebook("notebook_train_sac.ipynb", ROOT, "sac")

    def test_simulation_notebook_inline_outputs(self):
        self.check_notebook("test_ping_pong_env.ipynb", NOTEBOOKS)


if __name__ == "__main__":
    unittest.main()
