import mujoco
import mujoco.viewer
import numpy as np
import cv2

XML_PATH = "so101_new_calib.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Applique la pose initiale "par défaut"
mujoco.mj_resetData(model, data)

# Mets wrist_roll à -1.5 (index 15 d'après ton print)
data.qpos[15] = -1.5

# ⭐ CRITIQUE : aligner ctrl avec la pose actuelle
for act_id in range(model.nu):

    joint_id = model.actuator_trnid[act_id, 0]
    qpos_index = model.jnt_qposadr[joint_id]

    data.ctrl[act_id] = data.qpos[qpos_index]

# sécurité supplémentaire
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

cam_name = "gripper_cam"

CAMERA_FPS = 30
camera_period = 1.0 / CAMERA_FPS
next_frame_time = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1

    while viewer.is_running():

        mujoco.mj_step(model, data)

        # Capture uniquement quand le temps simulé dépasse le prochain frame time
        if data.time >= next_frame_time:

            renderer.update_scene(data, camera=cam_name)
            img = renderer.render()

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("gripper_cam", img_bgr)
            cv2.waitKey(1)

            next_frame_time += camera_period

        viewer.sync()

cv2.destroyAllWindows()
