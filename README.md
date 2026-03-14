# so101-robotic-learning

<div align="center" style="max-width:900px; margin:auto;">

<img src="img/global_mujoco.GIF" width="100%">
<p>Env mujoco</p>

<table align="center" width="100%">
  <tr>
  <td align="center" width="50%">
    <img src="img/ppo.gif" width="100%"><br>
    PPO raquette droite
  </td>

  <td align="center" width="50%">
    <img src="img/ppo.gif" width="100%"><br>
    PPO raquette droite
    </td>
</tr>
</table>

</div>

<!---

## TODO

- [x] package setup (requirements.txt);
- [x] Refactoring code (rename env en bounce_env + cleaner train.py et rewards.py);
- [x] push
- [ ] maj readme.txt
- [ ] push
- [ ] tensorboard;
- [ ] Log temps exec chaque étape PPO;
- [ ] Corriger reward (séparer en une reward par evaluation).

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

**bounce_rl** consists of a reinforcement learning environment designed around a bouncing-ball and paddle setup. The goal is to train an agent to control the paddle so as to maintain a stable and continuous bouncing behavior of the ball.

#### *Rewards*


- **PaddleParallelReward** : Evaluate if the paddle is parallel with the floor.
- **BallVerticalReward** : Evaluate if the ball’s velocity vector is perpendicular to the ground.
- **BallSpeedReward** : Evaluate if the ball’s velocity vector is close to a constent.
- **BallBelowPaddle** : Evaluate if the ball is bellow the paddle.
- **PingPongReward** : A multi-objective reward that combines several criteria.

#### *Benchmark*

|   |  REINFORCE | Q-learning  | PPO  |  SAC |
|---|---|---|---|---|
| **PaddleParallelReward** | 32  | 40  | 120  | 150  |
| **BallVerticalReward**  | 32  | 40  | 120  | 150  |
| **BallSpeedReward** | 32  | 40  | 120  | 150  |
| **BallBelowPaddle** | 32  | 40  | 120  | 150  |
| **PingPongReward** | 32  | 40  | 120  | 150  |
