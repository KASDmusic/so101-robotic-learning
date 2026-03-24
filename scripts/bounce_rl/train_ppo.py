from so101_robotic_learning.bounce_rl.rl import train_ppo

# train the PPO agent
root = "../../"
xml_path = root + "assets/mjcf/so101_new_calib copy.xml"
train_ppo.train(xml_path=xml_path, root=root)