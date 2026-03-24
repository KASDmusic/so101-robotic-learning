import time
import mujoco
import mujoco.viewer

import sys
from pathlib import Path

from so101_robotic_learning.bounce_rl.rewards.reward_ball_aligned_on_z_and_above_paddle import BallAlignedOnZAndAbovePaddleReward

root = "../../"
xml_path = root + "assets/mjcf/so101_new_calib copy.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(model.actuator_ctrlrange)

reward_fn = BallAlignedOnZAndAbovePaddleReward()

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        mujoco.mj_step(model, data)
        print(reward_fn.compute(model, data))

        # Important en mode passif pour refléter l'état courant
        viewer.sync()

        # Respect approximatif du timestep
        time.sleep(model.opt.timestep)