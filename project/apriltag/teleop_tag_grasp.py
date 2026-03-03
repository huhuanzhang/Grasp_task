#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teleop + D455 AprilTag + FK “仪表盘”抓取测试脚本（不做IK，全靠你遥操对准）
- 实时检测 tag(id可选)，把 tag 位姿从 cam 坐标变到 base 坐标：T_base_tag = T_base_cam @ T_cam_tag
- 用 Pinocchio 根据 robot.get_observation() 的 5DoF 关节角做 FK 得到夹爪当前点 p_cur(base)
- 定义目标点 p_goal(base)（默认 tag 上方 z_offset 米），打印误差 err = p_goal - p_cur
- 你根据 err 遥操把 dx/dy/dz 调到接近 0，再下探夹取

按键（在 OpenCV 窗口获得焦点时）：
- q: 退出
- g: 切换“预抓取/抓取”两种高度（默认 +10cm / +2cm）
- o / l: 增加 / 减少 Z 偏移（每次 5mm）
- c: 显示/隐藏更多调试信息

运行示例：
python teleop_tag_grasp_dashboard.py --remote_ip 172.16.0.14 --urdf_path so101_new_calib.urdf --tag_id 1
"""

import argparse
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
import pinocchio as pin

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def T_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def make_T_from_pose(pose_R: np.ndarray, pose_t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose_R
    T[:3, 3] = pose_t.reshape(3)
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_ip", type=str, default="172.16.0.14")
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi")
    parser.add_argument("--leader_profile", type=str, default="so-arm-5dof", choices=["so-arm-5dof", "am-arm-6dof"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no_leader", action="store_true")
    parser.add_argument("--no_robot", action="store_true")

    # Camera / AprilTag
    parser.add_argument("--cam_w", type=int, default=1280)
    parser.add_argument("--cam_h", type=int, default=720)
    parser.add_argument("--cam_fps", type=int, default=30)
    parser.add_argument("--tag_size", type=float, default=0.04)
    parser.add_argument("--tag_id", type=int, default=1)
    parser.add_argument("--min_decision_margin", type=float, default=25.0)
    parser.add_argument("--max_hamming", type=int, default=0)

    # FK
    parser.add_argument("--urdf_path", type=str, required=True, help="path to so101_new_calib.urdf")
    parser.add_argument("--ee_frame", type=str, default="gripper_frame_link", help="Pinocchio frame name for end-effector")

    # Grasp offsets
    parser.add_argument("--pregrasp_z", type=float, default=0.10, help="meters above tag for pregrasp")
    parser.add_argument("--grasp_z", type=float, default=0.02, help="meters above tag for grasp/down")

    args = parser.parse_args()

    NO_ROBOT = args.no_robot
    NO_LEADER = args.no_leader
    FPS = args.fps

    # ====== 你的手眼标定结果：T_base_cam ======
    # 直接用你最新一次标定的 Z（T_base_cam）
    T_base_cam = np.array(
        [
            [-0.26296128, -0.32847891, 0.90716755, -0.04672538],
            [-0.95438312, 0.22639856, -0.19467038, 0.15929978],
            [-0.14143632, -0.91697617, -0.37302878, 0.25363927],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # ====== Robot + teleop setup ======
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
    bi_cfg = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(port="/dev/ttyACM0", arm_profile=args.leader_profile),
        right_arm_config=SOLeaderConfig(port="/dev/ttyACM1", arm_profile=args.leader_profile),
        id=args.leader_id,
    )

    leader = BiSOLeader(bi_cfg)
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="my_laptop_keyboard"))
    robot = LeKiwiClient(robot_config)

    if not NO_ROBOT:
        robot.connect()
    if not NO_LEADER:
        leader.connect()
    keyboard.connect()

    init_rerun(session_name="lekiwi_grasp_dashboard")

    # ====== Camera + AprilTag ======
    at_detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, args.cam_w, args.cam_h, rs.format.bgr8, args.cam_fps)
    rs_profile = rs_pipeline.start(rs_config)

    intr = rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    print(f"[Camera] fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.ppx:.2f} cy={intr.ppy:.2f}")

    # ====== FK (Pinocchio) ======
    model = pin.buildModelFromUrdf(args.urdf_path)
    data = model.createData()
    frame_id = model.getFrameId(args.ee_frame)
    if frame_id == len(model.frames):
        raise ValueError(f"Frame {args.ee_frame} not found in URDF. Try ee_frame=gripper_link or wrist_link")

    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    OBS_KEYS = [
        "arm_left_shoulder_pan.pos",
        "arm_left_shoulder_lift.pos",
        "arm_left_elbow_flex.pos",
        "arm_left_wrist_flex.pos",
        "arm_left_wrist_roll.pos",
    ]

    def fk_gripper_pos_from_obs(observation: dict) -> np.ndarray:
        q = pin.neutral(model)
        for jn, ok in zip(JOINT_NAMES, OBS_KEYS):
            ang_deg = float(observation[ok])
            jid = model.getJointId(jn)
            q[model.joints[jid].idx_q] = np.deg2rad(ang_deg)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T_bg = data.oMf[frame_id].homogeneous
        return T_bg[:3, 3].copy()

    # ====== Runtime state ======
    show_debug = True
    use_grasp = False
    z_pre = float(args.pregrasp_z)
    z_grasp = float(args.grasp_z)

    print("\nControls (focus OpenCV window):")
    print("  q: quit")
    print("  g: toggle pregrasp/grasp height")
    print("  o/l: z_offset += 5mm / -= 5mm (current mode)")
    print("  c: toggle debug text\n")

    # fixed lift (你之前一直用的)
    lift_action = {"lift_axis.height_mm": 500.0}

    last_tag_info = None
    try:
        while True:
            t0 = time.perf_counter()

            # ---- Robot state/action ----
            observation = robot.get_observation() if not NO_ROBOT else {}
            arm_actions = leader.get_action() if not NO_LEADER else {}
            arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
            keyboard_keys = keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)

            action = {**arm_actions, **base_action, **lift_action}
            log_rerun_data(observation, action)

            if not NO_ROBOT:
                print("TELEOP KEYS:", sorted(action.keys()))
                robot.send_action(action)

            # ---- Camera frame ----
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
                continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # AprilTag detect
            dets = at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=args.tag_size,
            )

            best = None
            for d in dets:
                if d.tag_id != args.tag_id:
                    continue
                if d.decision_margin < args.min_decision_margin:
                    continue
                if d.hamming > args.max_hamming:
                    continue
                if (best is None) or (d.decision_margin > best.decision_margin):
                    best = d

            if best is not None:
                T_cam_tag = make_T_from_pose(best.pose_R, best.pose_t)
                T_base_tag = T_base_cam @ T_cam_tag
                p_tag = T_base_tag[:3, 3]
                last_tag_info = {
                    "T_cam_tag": T_cam_tag,
                    "p_tag": p_tag,
                    "dm": float(best.decision_margin),
                    "ham": int(best.hamming),
                    "corners": best.corners.astype(np.int32),
                }
            else:
                T_cam_tag = None
                T_base_tag = None
                p_tag = None

            # ---- FK current gripper pos ----
            p_cur = None
            if (not NO_ROBOT) and observation:
                try:
                    p_cur = fk_gripper_pos_from_obs(observation)
                except Exception:
                    p_cur = None

            # ---- Compute goal + error ----
            err = None
            p_goal = None
            mode_z = (z_grasp if use_grasp else z_pre)
            mode_name = ("GRASP" if use_grasp else "PRE")

            if last_tag_info is not None and p_cur is not None:
                p_goal = last_tag_info["p_tag"] + np.array([0.0, 0.0, mode_z], dtype=np.float64)
                err = p_goal - p_cur

            # ---- Draw / UI ----
            if last_tag_info is not None:
                cv2.polylines(frame, [last_tag_info["corners"]], True, (0, 255, 0), 2)
                x, y, z = last_tag_info["T_cam_tag"][:3, 3]
                cv2.putText(
                    frame,
                    f"tag {args.tag_id} cam xyz=({x:.3f},{y:.3f},{z:.3f}) dm={last_tag_info['dm']:.1f} ham={last_tag_info['ham']}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"tag {args.tag_id} not found / low conf",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            if show_debug:
                cv2.putText(
                    frame,
                    f"MODE={mode_name}  z_offset={mode_z:.3f}m  (g toggle, o/l +/-5mm)",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                if p_cur is not None:
                    cv2.putText(
                        frame,
                        f"p_cur(base)=({p_cur[0]:+.3f},{p_cur[1]:+.3f},{p_cur[2]:+.3f})",
                        (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                if last_tag_info is not None:
                    pt = last_tag_info["p_tag"]
                    cv2.putText(
                        frame,
                        f"p_tag(base)=({pt[0]:+.3f},{pt[1]:+.3f},{pt[2]:+.3f})",
                        (10, 109),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                if err is not None:
                    cv2.putText(
                        frame,
                        f"err=goal-cur (m): dx={err[0]:+.3f} dy={err[1]:+.3f} dz={err[2]:+.3f}",
                        (10, 136),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "err: (need tag + robot obs)",
                        (10, 136),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            disp = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.imshow("handeye_capture", disp)

            # ---- Key handling ----
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("g"):
                use_grasp = not use_grasp
            elif k == ord("c"):
                show_debug = not show_debug
            elif k == ord("o"):
                # increase current mode offset by 5mm
                if use_grasp:
                    z_grasp += 0.005
                else:
                    z_pre += 0.005
            elif k == ord("l"):
                if use_grasp:
                    z_grasp -= 0.005
                else:
                    z_pre -= 0.005

            # Also print a compact one-line dashboard to terminal
            if err is not None:
                print(
                    f"\r[{mode_name}] err(m): dx={err[0]:+.3f} dy={err[1]:+.3f} dz={err[2]:+.3f} | tag_z_cam={last_tag_info['T_cam_tag'][2,3]:.3f}",
                    end="",
                )

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    finally:
        try:
            rs_pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("\nExit.")


if __name__ == "__main__":
    main()