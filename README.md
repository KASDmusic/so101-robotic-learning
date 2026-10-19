# so101-robotic-learning

<div align="center" style="max-width:900px; margin:auto;">

<img src="img/imageData.gif" width="100%"><br>

</div>

## Patch notes
- **bounce_rl:**
  - **Juggling objective**: bounce detection at every substep, bounce bonus, auxiliary reward shaping, and drop penalty.
  - **Corrected rewards**: computation without modifying the physics, local `paddle_face` frame, active face only, and a shared default reward for PPO/SAC.
  - **Metrics and checkpoints**: tracking of streaks, success, drops, and impact dispersion. Best-model selection based on juggling performance, with JSON reports.
  - **Reliable evaluation**: handling of vectorized environments, step limits, callback thresholds, and terminal frames before reset.
  - **TensorBoard**: metrics logged at every evaluation and PPO/SAC losses verified. Logs stored in `runs/ppo_bounce/` and `runs/sac_bounce/`, without `experiments`.
  - **Cameras**: external view reserved for visualization, gripper view retained for the agent, and synchronized GIFs of both views with a single slider. MP4 exports removed, including during tests.
  - **Updated notebooks**: current configurations, robust paths, dated runs, and old outputs cleared.
  - **Faster rendering**: `train_fast` profile, one frame per decision, and external camera limited to the recorded episode. Physics unchanged. Local speedup of approximately **6×**, excluding neural network training.
  - **Validation**: **41 tests pass**, covering notebooks and TensorBoard files.
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
    - [x] Reset the environnement also if the paddle touch the ground or if the ball is too far of the robot (because the model actually learn to launch the ball far away one time);
    - [ ] Penalty depending of the 'energy" consumed.
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
