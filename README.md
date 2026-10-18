# so101-robotic-learning

<div align="center" style="max-width:900px; margin:auto;">

<img src="img/imageData.gif" width="100%"><br>

</div>

## Patch notes
- **bounce_rl  :** 
  - **Objectif de jonglage** : détection des rebonds à chaque sous-pas, bonus de rebond, shaping auxiliaire et pénalité de chute.
  - **Récompenses corrigées** : calcul sans modifier la physique, repère local `paddle_face`, face utile uniquement et récompense par défaut commune à PPO/SAC.
  - **Métriques et checkpoints** : suivi des séries, réussite, chutes et dispersion des impacts. sélection du meilleur modèle selon les performances de jonglage et rapports JSON.
  - **Évaluation fiabilisée** : gestion des environnements vectorisés, limites de pas, seuils du callback et images terminales avant reset.
  - **TensorBoard** : métriques enregistrées à chaque évaluation et pertes PPO/SAC vérifiées. logs dans `runs/ppo_bounce/` et `runs/sac_bounce/`, sans `experiments`.
  - **Caméras** : vue externe réservée à la visualisation, vue de pince conservée pour l'agent et GIF des deux vues synchronisées avec un seul slider. Exports MP4 supprimés, y compris pendant les tests.
  - **Notebooks adaptés** : configurations à jour, chemins robustes, essais datés et anciennes sorties nettoyées.
  - **Rendu accéléré** : profil `train_fast`, une image par décision et caméra externe limitée à l'épisode enregistré. Physique inchangée. gain local d'environ **×6**, hors apprentissage du réseau.
  - **Validation** : **41 tests passent**, incluant les notebooks et les fichiers TensorBoard.

<!---

## TODO

- [x] Enregistrer checkpoints model (best_reward)
- [x] Faire en sorte que les entrainement puisse être lancés depuis un notebook (passer en paramètre fichier de config)
- [x] Ajouter etat précédent dans observations env;
- [x] Régler problèmes import en fonction de l'endroit execution fichier;
- [x] Régler problème enregistrement vidéos;
- [ ] Rendre nombres d'états latents dynamiques (prends en paramètre une liste de taille (nb_states_latents), où liste[0] = step à prendre)
- [ ] Ajouter date à noms models et vidéos;
- [ ] Cleaner fichiers train (evolutif, multi env, etc)
  - [x] Mettre tout les paramètres dans trains et tests;
  - [x] Implémenter pour DummyVecEnv;
  - [ ] Implémenter pour SubprocVecEnv;
- [ ] Corriger et faire rewards
  - [ ] Créer une composite à RewardMultiObjectif;
  - [ ] Faire autres rewards ...
    - [x] Reward position balle comprise dans un espace donné;
    - [x] Reward position raquette comprise dans un espace donné;
  - [x] Corriger raquette droite pour que ça ne concerne qu'une face de la raquette;
- [x] Détecter les rebonds valides à chaque sous-pas, sans compter plusieurs fois les contacts prolongés;
- [x] Ajouter bonus de rebond, shaping auxiliaire et pénalité de chute;
- [x] Mesurer séries de rebonds, temps avant chute et dispersion des impacts;
- [x] Sélectionner les checkpoints selon les métriques de jonglage et enregistrer les rapports;
- [ ] Tester entrainement avec muli env; 
- [ ] Ajouter variations initialisation env;
  - [ ] implémenter système seed;
  - [ ] Position balle + vecteur vitesse (la traj de la balle doit aller vers la raquette);
  - [ ] masse balle, proprité de la balle
  - [ ] Position robot / raquette;
  - [ ] lumières
- [ ] Gridsearch Hyperparamètres;
- [ ] Supprimer les informations non relatives au bras (balle par ex)
- [ ] Optimiser execution env ? (tester enlever mesh collision raquette et/ou mesh bras pour voir si grand impact + voir si autres optimisations ex : jax)

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
mais des pixels différents ; conserver le même profil à l'entraînement et au test.
Redémarrer le kernel avant un nouveau lancement. Les runs existants ne sont
ni déplacés ni modifiés, et les entraînements en cours ne changent pas de profil.
