# so101-robotic-learning

<div align="center" style="max-width:900px; margin:auto;">

<img src="img/imageData.gif" width="100%"><br>

</div>

## Patch notes
- **bounce_rl:**
  - **Juggling objective**: bounce detection at every substep, bounce bonus, auxiliary reward shaping, and drop penalty.
  - **Episode failures**: terminate at the first physics substep with paddle-ground contact or a ball-to-paddle distance greater than 2 m. Apply the existing failure penalty once (-5 by default). Record each cause separately in reports and TensorBoard.
  - **Energy penalty**: integrate absolute actuator mechanical work at every physics substep. Configurable weight, enabled at `0.01` in PPO/SAC notebooks, with episode totals and penalties logged in TensorBoard.
  - **Corrected rewards**: computation without modifying the physics, local `paddle_face` frame, active face only, and a shared default reward for PPO/SAC.
  - **Metrics and checkpoints**: tracking of streaks, success, drops, and impact dispersion. Best-model selection based on juggling performance, with JSON reports.
  - **Reliable evaluation**: handling of vectorized environments, step limits, callback thresholds, and terminal frames before reset.
  - **TensorBoard**: metrics logged at every evaluation and PPO/SAC losses verified. Logs stored in `runs/ppo_bounce/` and `runs/sac_bounce/`, without `experiments`.
  - **Cameras**: external view reserved for visualization, gripper view retained for the agent, and synchronized GIFs of both views with a single slider. MP4 exports removed, including during tests.
  - **Updated notebooks**: current configurations, robust paths, dated runs, and old outputs cleared.
  - **Faster rendering**: `train_fast` profile, one frame per decision, and external camera limited to the recorded episode. Physics unchanged. Local speedup of approximately **6×**, excluding neural network training.
  - **Validation**: **54 tests pass**, covering energy penalties, episode termination, notebooks and TensorBoard files.
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
  - [ ] Create a composite reward for `RewardMultiObjectif`.
  - [ ] Implement additional rewards :
    - [ ] Penality depending of the distance with the closest point of a line beetween the paddle;
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
