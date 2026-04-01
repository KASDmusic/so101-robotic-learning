from pathlib import Path

from so101_robotic_learning.bounce_rl.rl import train_ppo

# Paths
root = Path("../../")
xml_path = root / "assets/mjcf/so101_new_calib copy.xml"
model_path = root / "models/ppo_bounce_best.zip"

# Train the PPO agent with explicit parameters
"""
train_ppo.train(
    xml_path=xml_path,
    root=root,

    # Entraînement
    total_timesteps=100_000,

    # PPO hyperparameters
    learning_rate=1e-4,
    n_steps=256,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,

    # Architecture réseau
    pi_layers=(256, 128),
    vf_layers=(256, 128),

    # Évaluation
    eval_every_episodes=5,
    n_eval_episodes=5,
    eval_max_steps=1500,
    deterministic_eval=True,
    save_eval_video=True,

    # Divers
    device="auto",
    verbose=1,
)
"""

train_ppo.train(
    xml_path=xml_path,
    root=root,
    total_timesteps=50_000,
    n_steps=128,
    batch_size=64,
    nb_vec_env_train=5
)