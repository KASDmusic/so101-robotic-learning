from pathlib import Path
from so101_robotic_learning.bounce_rl.rl import train_sac

# Paths sécurisés
root = Path("../../")
xml_path = root / "assets/mjcf/so101_new_calib copy.xml"

model_path = root / "models/sac_bounce_best.zip"

# Train SAC avec paramètres explicites
"""
train_sac.train(
    xml_path=xml_path,
    root=root,

    # Entraînement
    total_timesteps=100_000,

    # Hyperparamètres SAC
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",

    # Architecture réseau
    pi_layers=(256, 256, 128),
    qf_layers=(256, 256, 128),

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

train_sac.train(
    xml_path=xml_path,
    root=root,
    total_timesteps=50_000,
    batch_size=128,
)