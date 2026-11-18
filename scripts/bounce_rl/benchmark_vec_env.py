"""Compare DummyVecEnv and SubprocVecEnv with an identical transition budget."""

import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import wraps
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from so101_robotic_learning.bounce_rl.rl.train_ppo import make_env
from so101_robotic_learning.bounce_rl.rl import train_ppo, train_sac
from so101_robotic_learning.bounce_rl.rl.rl_utils import EvalVideoSaveBestCallback
from so101_robotic_learning.bounce_rl.rl.vector_env import make_training_vec_env


ROOT = Path(__file__).resolve().parents[2]
TRAINING_PHASES = (
    "rollouts", "network updates", "evaluations",
    "train env startup", "eval env startup", "model init", "learning setup",
    "training logs", "final checkpoint", "env shutdown", "logger shutdown", "unattributed",
)
PHASE_WIDTH = max(map(len, TRAINING_PHASES))


class QuietStep(gym.Wrapper):
    """Keep per-step console output out of the measured workload in both modes."""

    def step(self, action):
        with redirect_stdout(None):
            return self.env.step(action)


class TrainingTimings:
    """Exclusive wall times, including when an evaluation interrupts a rollout."""

    def __init__(self):
        self.seconds = dict.fromkeys(TRAINING_PHASES, 0.)
        self.calls = dict.fromkeys(TRAINING_PHASES[:-1], 0)
        self.active = None
        self.started = 0.

    @contextmanager
    def measure(self, name, get_device):
        def synchronize():
            device = get_device() if get_device is not None else None
            if device is None:
                return
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()

        synchronize()
        boundary = perf_counter()
        parent = self.active
        if parent is not None:
            self.seconds[parent] += boundary - self.started
        self.active, self.started = name, boundary
        self.calls[name] += 1
        try:
            yield
        finally:
            synchronize()
            boundary = perf_counter()
            self.seconds[name] += boundary - self.started
            self.active, self.started = parent, boundary

    def wrap(self, method, phase, device_of=None, *, skip_during=None):
        @wraps(method)
        def timed(*args, **kwargs):
            if skip_during is not None and self.active == skip_during:
                return method(*args, **kwargs)
            get_device = (lambda: device_of(args[0])) if device_of is not None else None
            with self.measure(phase, get_device):
                return method(*args, **kwargs)

        return timed


def benchmark(backend, args, repeat):
    start = perf_counter()
    factory = make_env(
        xml_path=args.xml,
        env_kwargs={"frame_skip": args.frame_skip, "max_episode_steps": args.max_episode_steps},
    )
    with ExitStack() as resources:
        env = make_training_vec_env(
            [lambda: QuietStep(factory()) for _ in range(args.n_envs)],
            backend=backend, render_mode=None, curriculum=None, resources=resources,
        )
        env.seed(args.seed)
        env.reset()
        startup = perf_counter() - start

        # Recreate the same warmup and measured actions for both backends.
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, repeat]))
        shape = (args.n_envs, *env.action_space.shape)
        actions = rng.uniform(-1, 1, (args.warmup + args.steps // args.n_envs, *shape))
        actions = actions.astype(env.action_space.dtype, copy=False)
        for action in actions[:args.warmup]:
            env.step(action)

        start = perf_counter()
        for action in actions[args.warmup:]:
            env.step(action)
        elapsed = perf_counter() - start
    return startup, elapsed


def install_training_probes(probes, module, algorithm, timings):
    def probe(owner, name, phase, device_of=None, **options):
        wrapped = timings.wrap(getattr(owner, name), phase, device_of, **options)
        probes.enter_context(patch.object(owner, name, wrapped))

    def model_device(model):
        # The constructor does not have a device yet when the probe starts.
        return getattr(model, "device", None)

    probe(algorithm, "collect_rollouts", "rollouts", model_device)
    probe(algorithm, "train", "network updates", model_device)
    probe(EvalVideoSaveBestCallback, "_evaluate_record_and_save", "evaluations",
          lambda callback: callback.model.device)
    probe(module, "make_training_vec_env", "train env startup")
    probe(module, "DummyVecEnv", "eval env startup")
    probe(algorithm, "__init__", "model init", model_device)
    probe(algorithm, "_setup_learn", "learning setup", model_device)
    # Evaluation logs and best checkpoints already belong to evaluations.
    probe(Logger, "dump", "training logs", skip_during="evaluations")
    probe(algorithm, "save", "final checkpoint", skip_during="evaluations")
    probe(module, "save_gravity_curriculum", "final checkpoint")
    probe(DummyVecEnv, "close", "env shutdown")
    probe(SubprocVecEnv, "close", "env shutdown")
    probe(Logger, "close", "logger shutdown")


def benchmark_training(backend, args, repeat, root):
    module = train_ppo if args.mode == "ppo" else train_sac
    algorithm = train_ppo.PPO if args.mode == "ppo" else train_sac.SAC
    settings = (dict(n_steps=args.rollout_steps, n_epochs=args.ppo_epochs)
                if args.mode == "ppo" else
                dict(learning_starts=args.learning_starts, buffer_size=args.steps, save_replay_buffer=False))
    timings = TrainingTimings()
    # Instrument the real training methods in this process, only for this run.
    # Keeping the wrappers on the classes also keeps them out of model checkpoints.
    with ExitStack() as probes:
        install_training_probes(probes, module, algorithm, timings)
        start = perf_counter()
        result = module.train(
            xml_path=args.xml, root=root, vec_env_backend=backend,
            nb_vec_env_train=args.n_envs, nb_vec_env_test=1,
            total_timesteps=args.steps, batch_size=args.batch_size,
            env_kwargs=dict(frame_skip=args.frame_skip, max_episode_steps=args.max_episode_steps,
                            video_camera_name="overview_cam", video_width=320, video_height=240,
                            log_step_timing=False),
            seed=args.seed + repeat, device=args.device, eval_every_timesteps=args.eval_every or args.steps + 1,
            evaluate_at_end=True, eval_max_steps=args.eval_max_steps, n_eval_episodes=args.eval_episodes,
            eval_render_mode=None, save_eval_video=True, tensorboard_log=True,
            print_spaces=False, verbose=0, callback_verbose=0, **settings,
        )
        elapsed = perf_counter() - start
    if result["training_timesteps"] != args.steps:
        raise RuntimeError(f"{backend}: expected {args.steps} transitions, got {result['training_timesteps']}.")
    if not result["video_frames"] or not result["gripper_video_frames"]:
        raise RuntimeError(f"{backend}: evaluation did not capture both cameras.")
    timings.seconds["unattributed"] = elapsed - sum(timings.seconds.values())
    return elapsed, timings


def print_phase_times(elapsed, timings):
    for name, seconds in timings.seconds.items():
        calls = f" | {timings.calls[name]} calls" if name in timings.calls else ""
        print(f"    {name:<{PHASE_WIDTH}} {seconds:8.3f} s | {100 * seconds / elapsed:5.1f}%{calls}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=ROOT / "assets/mjcf/so101_new_calib copy.xml")
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000, help="Total measured transitions across all environments.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10, help="Unmeasured vector steps per backend and repeat.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=1024)
    parser.add_argument("--mode", choices=("collection", "ppo", "sac"), default="collection")
    parser.add_argument("--device", default="cpu", help="Training device (ppo/sac mode).")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=20, help="PPO steps per environment per rollout.")
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--learning-starts", type=int, default=50, help="SAC transitions before network updates.")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Additional evaluation interval in transitions. 0 evaluates only after training.")
    parser.add_argument("--eval-max-steps", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=1)
    args = parser.parse_args()
    for name in ("n_envs", "steps", "repeats", "frame_skip", "max_episode_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.steps % args.n_envs:
        parser.error("--steps must be divisible by --n-envs for an exact transition budget.")
    if min(args.warmup, args.seed) < 0:
        parser.error("--warmup and --seed must be non-negative.")
    args.xml = args.xml.resolve()
    if not args.xml.is_file():
        parser.error(f"Missing MuJoCo model: {args.xml}")

    training = args.mode != "collection"
    if training:
        for name in ("torch_threads", "batch_size", "rollout_steps", "ppo_epochs", "eval_max_steps", "eval_episodes"):
            if getattr(args, name) < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive.")
        if args.eval_every < 0 or args.eval_every % args.n_envs:
            parser.error("--eval-every must be zero or a positive multiple of --n-envs.")
        if args.mode == "ppo":
            rollout = args.n_envs * args.rollout_steps
            if args.steps % rollout or not 2 <= args.batch_size <= rollout or rollout % args.batch_size:
                parser.error("PPO needs --steps divisible by the rollout size (envs × rollout-steps), "
                             "and a --batch-size of at least 2 dividing that rollout size.")
        elif not 0 <= args.learning_starts < args.steps:
            parser.error("SAC requires 0 <= --learning-starts < --steps to update the network.")
        torch.set_num_threads(args.torch_threads)

    print(f"{args.n_envs} envs, {args.steps} transitions per run "
          f"({args.steps // args.n_envs} vector steps), frame_skip={args.frame_skip}, "
          f"{args.repeats} repeats. Mode: {args.mode}. Camera observations enabled.", flush=True)
    results = {"dummy": [], "subproc": []}
    phase_results = {"dummy": [], "subproc": []}
    label = "end-to-end" if training else "collection"
    for repeat in range(args.repeats):
        # Alternate the order to reduce systematic effects of the first run.
        order = ("dummy", "subproc") if repeat % 2 == 0 else ("subproc", "dummy")
        for backend in order:
            if training:
                with TemporaryDirectory(prefix="so101_vec_benchmark_") as root:
                    elapsed, timings = benchmark_training(backend, args, repeat, root)
                phase_results[backend].append(timings.seconds)
                startup_label = ""
            else:
                startup, elapsed = benchmark(backend, args, repeat)
                startup_label = f"startup: {startup:7.3f} s | "
            results[backend].append(elapsed)
            print(f"{repeat + 1}/{args.repeats} {backend:7} | {startup_label}"
                  f"{label}: {elapsed:7.3f} s | {args.steps / elapsed:7.1f} transitions/s", flush=True)
            if training:
                print_phase_times(elapsed, timings)

    dummy, subproc = (median(results[backend]) for backend in ("dummy", "subproc"))
    print(f"Median {label}: dummy={dummy:.3f} s | subproc={subproc:.3f} s")
    print(f"Time saved with subproc: {dummy - subproc:+.3f} s ({100 * (1 - subproc / dummy):+.1f}%)")
    print(f"Throughput ratio (subproc / dummy): {dummy / subproc:.2f}x")
    if training:
        print("\nMedian by phase (seconds):")
        print(f"{'Phase':<{PHASE_WIDTH}} {'dummy':>10} {'subproc':>10} {'time saved':>12}")
        for phase in TRAINING_PHASES:
            dummy_phase, subproc_phase = (
                median(run[phase] for run in phase_results[backend]) for backend in ("dummy", "subproc")
            )
            print(f"{phase:<{PHASE_WIDTH}} {dummy_phase:10.3f} {subproc_phase:10.3f} "
                  f"{dummy_phase - subproc_phase:+12.3f}")


if __name__ == "__main__":
    main()
