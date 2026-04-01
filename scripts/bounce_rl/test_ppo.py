from pathlib import Path

from so101_robotic_learning.bounce_rl.rl import train_ppo

# Paths
root = Path("../../")
xml_path = root / "assets/mjcf/so101_new_calib copy.xml"
model_path = root / "models/ppo_bounce_best.zip"

train_ppo.test(
    xml_path,
    root,
    model_path=model_path,
    reward=None,
    render_mode="human",
    deterministic=True,
    n_episodes=5,
    max_steps=1024,
    video_dir_name="videos_sb3_test",
    video_prefix="test_episode",
    save_test_video=True,
    fps=None,
    verbose=1,
)