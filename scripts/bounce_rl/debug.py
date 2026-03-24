import time
import mujoco
import mujoco.viewer

import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent.parent.parent
sys.path.append(str(ROOT / "src"))

from bounce_rl.rewards.rewards import PingPongReward
from bounce_rl.rewards.penalty_ball_below_paddle import BallBelowPaddlePenalty
from bounce_rl.rewards.reward_ball_aligned_on_z_and_above_paddle import BallAlignedOnZAndAbovePaddleReward

def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "mjcf" / "so101_new_calib copy.xml"))
    data = mujoco.MjData(model)

    print(model.actuator_ctrlrange)

    reward_fn = BallAlignedOnZAndAbovePaddleReward(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        while viewer.is_running():
            mujoco.mj_step(model, data)
            print(reward_fn.compute(model, data))

            # Important en mode passif pour refléter l'état courant
            viewer.sync()

            # Respect approximatif du timestep
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()