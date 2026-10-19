# so101-robotic-learning

<div align="center" style="max-width:900px; margin:auto;">

<img src="img/imageData.gif" width="100%"><br>

</div>

## Patch notes
- **bounce_rl:**
  - **Juggling objective**: bounce detection at every substep, bounce bonus, auxiliary reward shaping, and drop penalty.
  - **Episode failures**: terminate at the first physics substep with paddle-ground contact or a ball-to-paddle distance greater than 2 m. Apply the existing failure penalty once (-5 by default). Record each cause separately in reports and TensorBoard.
  - **Energy penalty**: integrate absolute actuator mechanical work at every physics substep. Configurable weight, enabled at `0.01` in PPO/SAC notebooks, with episode totals and penalties logged in TensorBoard.
  - **Ball alignment penalty**: penalize distance to the nearest point on a vertical segment from 0.2 m to 1.2 m above `paddle_face`. Integrate over simulated time and expose configuration and metrics in PPO/SAC notebooks and TensorBoard.
  - **Composite rewards**: `RewardMultiObjectif` combines named weighted rewards, penalties and nested composites. Existing rewards remain compatible. Per-component contributions are available in episode reports and TensorBoard.
  - **Corrected rewards**: computation without modifying the physics, local `paddle_face` frame, active face only, and a shared default reward for PPO/SAC.
  - **Metrics and checkpoints**: tracking of streaks, success, drops, and impact dispersion. Best-model selection based on juggling performance, with JSON reports.
  - **Reliable evaluation**: handling of vectorized environments, step limits, callback thresholds, and terminal frames before reset.
  - **TensorBoard**: metrics logged at every evaluation and PPO/SAC losses verified. Logs stored in `runs/ppo_bounce/` and `runs/sac_bounce/`, without `experiments`.
  - **Cameras**: external view reserved for visualization, gripper view retained for the agent, and synchronized GIFs of both views with a single slider. MP4 exports removed, including during tests.
  - **Updated notebooks**: current configurations, robust paths, dated runs, and old outputs cleared.
  - **Faster rendering**: `train_fast` profile, one frame per decision, and external camera limited to the recorded episode. Physics unchanged. Local speedup of approximately **6×**, excluding neural network training.
  - **Validation**: **75 tests pass**, covering composite objectives, segment-distance and energy penalties, episode termination, notebooks and TensorBoard files.
<!---

## Arborescence

```

├── README.md
├── .gitignore
│
├── assets/                     # Tout ce qui est physique
├── img/
├── notebooks/
|
└── src/
   └── bounce_rl/
       ├── envs/
       │   ├── bounce_env.py        # Gymnasium env principal
       │   ├── reset.py             # stratégies de reset
       │   ├── observations.py
       │   ├── rewards.py
       │   ├── events.py            # détection de rebond
       │   └── termination.py
       │
       ├── sim/
       │   ├── mujoco_loader.py
       │   ├── controllers.py       # PD / position control
       │   ├── domain_randomization.py
       │   └── physics_tuning.py    # restitution, friction…
       │
       ├── rl/
       │   ├── train.py
       │   ├── eval.py
       │   ├── callbacks.py
       │   └── wrappers.py
       │
       ├── real_robot/              
       │   ├── so101_interface.py
       │   ├── safety_limits.py
       │   └── calibration.py
       │
       ├── perception/
       │   ├── ball_tracking.py
       │   ├── kalman.py
       │   └── latency_compensation.py
       │
       ├── configs/
       │   ├── env.yaml
       │   ├── sac.yaml
       │   ├── randomization.yaml
       │   └── real_robot.yaml
       │
       ├── utils/
       │   ├── seeds.py
       │   ├── logger.py
       │   └── paths.py
       │
       └── __init__.py
```

-->

## Setup environnement

```
conda env create -f environment.yml
conda activate so101-robotic-learning
```

## Environnements

### bounce_rl

**bounce_rl** consists of a reinforcement learning gym environment designed around a bouncing-ball and paddle setup made with mujocco. The goal is to train an agent to control the paddle so as to maintain a stable and continuous bouncing behavior of the ball.

#### *Scripts*

```
cd scripts/bounce_rl
python [script_name].py
```

- **debug.py :** exemples to launch the **bounce_rl** environment.
- **train_ppo.py :** exemples to train a ppo policy on the **bounce_rl** environment.
- **train_sac.py :** exemples to train a sac policy on the **bounce_rl** environment.
- **test_ppo.py :** exemples to test and vizualise a ppo policy on the **bounce_rl** environment.
- **test_sac.py :** exemples to test and vizualise a sac policy on the **bounce_rl** environment.

#### *Rewards*

```mermaid
  graph TD;
      PaddleParallelReward e1@==> PingPongReward;
      BallVerticalReward e2@==> PingPongReward;
      BallSpeedReward e3@==> PingPongReward;
      BallBelowPaddle e4@==> PingPongReward;

      e1@{ animate: true }
      e2@{ animate: true }
      e3@{ animate: true }
      e4@{ animate: true }
```

##### *Solo*


- **PaddleParallelReward** : Evaluate if the paddle is parallel with the floor.
- **BallVerticalReward** : Evaluate if the ball’s velocity vector is perpendicular to the ground.
- **BallSpeedReward** : Evaluate if the ball’s velocity vector is close to a constent.
- **BallBelowPaddlePenalty** : Evaluate if the ball is bellow the paddle.

##### *Combination*

- **PingPongReward** : A multi-objective reward that combines several criteria.

#### TODO

- [ ] **Fix and implement rewards and penalities**
  - [x] Create a composite reward for `RewardMultiObjectif`.
  - [x] Implement additional rewards :
    - [x] Penalize ball distance to a vertical 1 m segment starting 20 cm above the paddle centre.
    - [x] End the episode if the paddle touches the ground or the ball moves more than 2 m from the paddle.
    - [x] Penalize absolute actuator mechanical work, including braking.
- [ ] **Add variations to environment initialization.**
  - [ ] Implement a seeding system.
  - [ ] Vary the ball’s position and velocity vector (its trajectory must head toward the paddle).
  - [ ] Vary the ball’s mass and other properties.
  - [ ] Vary the robot/paddle position.
  - [ ] Vary the lighting.
- [ ] **Run a hyperparameter grid search.**
- [ ] **When it is working, remove information unrelated to the arm (e.g., ball information).**
- [ ] **Add support for `SubprocVecEnv` ?**
- [ ] **Optimize environment execution?** (Test whether removing the paddle’s collision mesh and/or the arm’s mesh has a significant impact, reducing frame rendering and explore other optimizations, such as JAX.)
- [ ] **Sim2Real.**


<!---

#### *Benchmark*

|   | PPO  |  SAC |
|---|---|---|
| **PaddleParallelReward** | 120  | 150  |
| **BallVerticalReward**  | 32  | 40  |
| **BallSpeedReward** | 32  | 40  | 120
| **BallBelowPaddle** | 32  | 40  | 120
| **PingPongReward** | 32  | 40  | 120

-->

### Episode termination

PPO and SAC end an episode when the ball falls below 0.1 m, the collision paddle
contacts the ground, or the ball centre moves strictly more than 2 m from the
`paddle_face` site (3D Euclidean distance). All checks run at every physics substep.
The existing `fall_penalty` applies to each failure, once per terminal step.
Reaching the episode step limit remains a truncation without a failure penalty.

Both notebooks expose `BounceConfig(max_ball_paddle_distance=2.0)`. Ground planes
attached to the world are detected automatically, including unnamed planes.
For another ground geometry, set `ground_geom="geom_name"` in `BounceConfig`.

`info["termination_reason"]` distinguishes `ball_fell`, `paddle_ground_contact`
and `ball_too_far`. Episode reports retain these causes. TensorBoard logs
`eval/paddle_ground_contact_rate` and `eval/ball_too_far_rate` separately from
`eval/fall_rate`. Vector environments reset automatically after termination.
Direct users of `BounceEnv` must call `reset()` after `terminated=True`.

### Mechanical energy penalty

Each decision subtracts `energy_penalty_weight * energy_step_j` from its reward.
The energy proxy integrates `sum(abs(actuator_force * actuator_velocity)) * dt`
at every executed physics substep, using end-of-substep samples. Motor and
braking work both count positively. Opposite motor powers cannot cancel out.
An early episode termination counts only the substeps actually simulated.

Both notebooks set `energy_penalty_weight=0.01` in `env_kwargs`. This initial
weight subtracts 0.01 reward per joule of absolute mechanical work and should
be tuned using the task and energy metrics. Direct `BounceEnv` calls default
to `0.0` for compatibility. Setting this weight to zero disables the penalty
while retaining the energy measurements.

Step info contains `energy_step_j` and the signed contribution `penalty_energy`.
Episode `task_metrics` and JSON reports include cumulative `mechanical_energy_j`
and `energy_penalty`. TensorBoard records `eval/mean_mechanical_energy_j` and
`eval/mean_energy_penalty`. Totals reset at the start of every episode.

This measures mechanical work, not electrical consumption. Motor losses,
electronics and holding torque at zero velocity are not included. It does not
replace a separate action-smoothness or torque-squared penalty.

### Ball distance to the target segment

The target is a finite vertical segment, aligned with world Z (perpendicular
to the horizontal ground). It follows the current `paddle_face` position but
not the paddle orientation. Its lower endpoint is 0.2 m above that centre and
its upper endpoint is 1.2 m above it, giving a total length of 1 m.

For paddle centre `(px, py, pz)` and ball centre `(bx, by, bz)`, the nearest
point is `(px, py, clip(bz, pz + 0.2, pz + 1.2))`. The Euclidean distance to
this point is zero anywhere on the segment and positive outside it, including
above and below the endpoints.

Each physics substep adds `distance_m * dt` to a distance integral. The decision
reward subtracts `ball_segment_penalty_weight * integral_m_s`, independently
of the energy and failure penalties and the existing reward shaping. The time
integration avoids multiplying the penalty when reducing `frame_skip` and
includes only executed substeps when an episode ends early.

Both notebooks use these entries in `env_kwargs`:

```python
"ball_segment_penalty_weight": 1.0,
"ball_segment_start_height": 0.2,
"ball_segment_length": 1.0,
```

The weight is an initial tuning value in reward per metre-second. At weight
1.0, remaining 0.5 m away for one simulated second subtracts 0.5 reward.
Direct `BounceEnv` calls default to weight 0.0 for compatibility. Weight zero
disables the penalty while retaining the measurements.

Step info exposes `ball_segment_distance_m`, `segment_closest_point`, the
step's `segment_distance_integral_m_s` and its signed `penalty_ball_segment`.
Episode reports contain the cumulative integral, the time-weighted
`mean_ball_segment_distance_m` and cumulative signed `ball_segment_penalty`.
TensorBoard records `eval/mean_ball_segment_distance_m` (the mean of episode
averages) and `eval/mean_ball_segment_penalty`. Cumulative values reset with
every episode and the average distance is undefined before any simulated step.

### Composite multi-objective rewards

`RewardMultiObjectif` implements the Composite pattern through the common
`RewardComponent` interface. A child can be a leaf, an existing two-argument
reward (automatically adapted), or another composite. The score is a weighted
sum, with no implicit normalization:

```text
total = sum(weight * reward_value) - sum(weight * abs(penalty_value))
```

`add_penalty` accepts positive costs and existing negative penalty scores
without reversing the penalty into a bonus. Weights must be finite and
nonnegative. Names must be nonempty and unique within a composite. Cycles and
nonfinite scores are rejected. Weight zero disables the contribution but still
computes its diagnostics.

The default environment objective now uses the same composite with five
canonical names: `bounces`, `shaping`, `failure`, `energy` and `ball_segment`.
Existing scalar coefficients and notebook configurations retain their meanings.
The current formula and the physical event detection are preserved.

For a custom full objective, construct a tree and pass it in `env_kwargs`.
This example also adds a ball-speed preference to the auxiliary shaping:

```python
from so101_robotic_learning.bounce_rl.rewards import (
    RewardMultiObjectif, BounceReward, FailurePenalty,
    EnergyPenalty, BallSegmentPenalty,
)
from so101_robotic_learning.bounce_rl.rewards.reward_ball_aligned_on_z_and_above_paddle import (
    BallAlignedOnZAndAbovePaddleReward,
)
from so101_robotic_learning.bounce_rl.rewards.reward_ball_speed import BallSpeedReward

geometry = (
    RewardMultiObjectif()
    .add_reward("alignment", BallAlignedOnZAndAbovePaddleReward())
    .add_reward("speed", BallSpeedReward(target_ball_speed=2.0), weight=0.2)
)
objective = (
    RewardMultiObjectif()
    .add_reward("bounces", BounceReward(), weight=5.0)
    .add_reward("shaping", geometry, weight=0.05)
    .add_penalty("failure", FailurePenalty(), weight=5.0)
    .add_penalty("energy", EnergyPenalty(), weight=0.01)
    .add_penalty("ball_segment", BallSegmentPenalty(), weight=1.0)
)
env_kwargs["reward_composite"] = objective
train_config["reward"] = None
```

This works in both PPO and SAC. `reward_composite` replaces the entire default
objective, so the old scalar reward weights are not applied a second time.
Geometry settings such as `ball_segment_start_height` and `ball_segment_length`
still configure the measurements. Episode termination is independent of the
reward tree, so include `FailurePenalty` if failures should have a reward cost.

To customize only the auxiliary shaping instead, pass `geometry` as
`train_config["reward"]` and leave `reward_composite` unset. The existing
`shaping_reward_weight` then scales that whole group, while the environment
retains the other objectives. Providing both options raises an error rather
than silently counting components twice.

Use `set_weight(name, weight)` or `remove(name)` to edit a tree. When changing
an environment already created, edit `env.reward_composite`. Custom composites
are deep-copied for each environment, including when used as shaping. New
stateful leaves can inherit `RewardComponent` and implement `reset(model, data)`.
Legacy two-argument rewards remain supported as stateless leaves.

The environment supplies an immutable `RewardContext` with the current
`bounce_count`, `failed`, `energy_j` and `segment_distance_integral_m_s`. Thus
energy, distance and event leaves reuse measurements from all executed physics
substeps instead of estimating them from only the final state.

`info["reward_components"]` exposes each child's raw `value`, `weight`, `kind`,
signed `contribution` and `details`. Nested composites retain their own tree in
`details`, avoiding collisions between child diagnostic keys. Existing flat
channels such as `penalty_energy` follow the canonical top-level names. For
arbitrary names or nested groups, use the component tree for the full breakdown.

Episode reports store cumulative top-level contributions in
`task_metrics["reward_components"]`. TensorBoard automatically records their
mean episode sums under `eval/reward_components/<name>`, including custom names.
These contributions sum to the mean episode reward.
