from so101_robotic_learning.bounce_rl.rl import train_sac

# train the PPO agent
root = "../../"
xml_path = root + "assets/mjcf/so101_new_calib copy.xml"
train_sac.train(xml_path=xml_path, root=root)