import time
import mujoco
import mujoco.viewer

import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent.parent.parent
sys.path.append(str(ROOT / "src"))

from bounce_rl.rewards.rewards import PingPongReward

def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "mjcf" / "so101_new_calib copy.xml"))
    data = mujoco.MjData(model)

    print(model.actuator_ctrlrange)

    reward_fn = reward_fn = PingPongReward(
            model,
            ball_body_name="ball",
            paddle_body_name="paddle_mount",
            target_ball_speed=2.0,
            speed_sigma=0.5,
            w_paddle_parallel=1.0,
            w_ball_vertical=0,
            w_ball_speed=0,
            w_ball_below_paddle=0,
            below_paddle_margin=0.05,
            paddle_normal_local=(0.0, -1.0, 0.0),  # à ajuster si besoin
        )

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