#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import json
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============ Parameter Section ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--no_robot", action="store_true")
parser.add_argument("--no_leader", action="store_true")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--remote_ip", type=str, default="172.16.0.14")
parser.add_argument("--leader_id", type=str, default="so101_leader_bi")
parser.add_argument(
    "--leader_profile",
    type=str,
    default="so-arm-5dof",
    choices=["so-arm-5dof", "am-arm-6dof"],
)

# ===== NEW: calibration sampling params =====
parser.add_argument("--tag_size", type=float, default=0.04, help="AprilTag size in meters (e.g., 0.04 for 4cm)")
parser.add_argument("--tag_id", type=int, default=1, help="Target AprilTag ID to record")
parser.add_argument("--out_jsonl", type=str, default="samples.jsonl", help="Output jsonl file for hand-eye samples")
parser.add_argument("--min_decision_margin", type=float, default=30.0, help="Reject detections with low confidence")
parser.add_argument("--max_hamming", type=int, default=0, help="Reject detections with hamming > this value")
parser.add_argument("--cam_w", type=int, default=1280)
parser.add_argument("--cam_h", type=int, default=720)
parser.add_argument("--cam_fps", type=int, default=30)

args = parser.parse_args()

NO_ROBOT = args.no_robot
NO_LEADER = args.no_leader
FPS = args.fps

# ===== NEW =====
TAG_SIZE = float(args.tag_size)
TAG_ID_TARGET = int(args.tag_id)
OUT_JSONL = Path(args.out_jsonl)
MIN_DECISION_MARGIN = float(args.min_decision_margin)
MAX_HAMMING = int(args.max_hamming)

# ========================================== #

# Create configs
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
bi_cfg = BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(
        port="/dev/ttyACM0",
        arm_profile=args.leader_profile,
    ),
    right_arm_config=SOLeaderConfig(
        port="/dev/ttyACM1",
        arm_profile=args.leader_profile,
    ),
    id=args.leader_id,
)
leader = BiSOLeader(bi_cfg)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
robot = LeKiwiClient(robot_config)

# Connect
if not NO_ROBOT:
    robot.connect()
if not NO_LEADER:
    leader.connect()
keyboard.connect()

init_rerun(session_name="lekiwi_teleop")

# ===== NEW: init D455 + AprilTag detector =====
print("Starting RealSense + AprilTag ...")
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
print(f"Camera intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")

# Output file
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
f_out = open(OUT_JSONL, "a", encoding="utf-8")
print(f"Saving samples to: {OUT_JSONL.resolve()}")

# simple key debounce
_last_save_t = 0.0
SAVE_DEBOUNCE_SEC = 0.35

def make_T_from_pose(pose_R: np.ndarray, pose_t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose_R
    T[:3, 3] = pose_t.reshape(3)
    return T

def try_get_target_tag_T_cam_tag(frame_bgr: np.ndarray) -> dict | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dets = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE,
    )
    best = None
    for d in dets:
        if d.tag_id != TAG_ID_TARGET:
            continue
        if d.decision_margin < MIN_DECISION_MARGIN:
            continue
        if d.hamming > MAX_HAMMING:
            continue
        # 选 decision_margin 最大的那个
        if (best is None) or (d.decision_margin > best.decision_margin):
            best = d

    if best is None:
        return None

    T_cam_tag = make_T_from_pose(best.pose_R, best.pose_t)
    corners = best.corners.astype(np.int32)
    return {
        "tag_id": int(best.tag_id),
        "decision_margin": float(best.decision_margin),
        "hamming": int(best.hamming),
        "pose_err": float(getattr(best, "pose_err", 0.0)),
        "T_cam_tag": T_cam_tag,
        "center": [float(best.center[0]), float(best.center[1])],
        "corners":corners,
    }

def get_left_arm_q_deg_from_observation(obs: dict) -> list[float]:
    # so-arm-5dof keys (from your LeKiwi config)
    keys = [
        "arm_left_shoulder_pan.pos",
        "arm_left_shoulder_lift.pos",
        "arm_left_elbow_flex.pos",
        "arm_left_wrist_flex.pos",
        "arm_left_wrist_roll.pos",
    ]
    return [float(obs[k]) for k in keys]

# ===== end NEW =====

target_height_mm = 500.0
lift_action = {"lift_axis.height_mm": target_height_mm}

print("\nControls:")
print("  - Press 's' to save ONE calibration sample ( robot joints + T_cam_tag for id=%d)" % TAG_ID_TARGET)
print("  - Press 'q' in the OpenCV window to quit (or Ctrl+C)\n")

try:
    while True:
        t0 = time.perf_counter()

        # ---- Robot/teleop ----
        observation = robot.get_observation() if not NO_ROBOT else {}
        arm_actions = leader.get_action() if not NO_LEADER else {}
        arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        action = {**arm_actions, **base_action, **lift_action}
        log_rerun_data(observation, action)

        if not NO_ROBOT:
            robot.send_action(action)

        # ---- Camera frame ----
        frames = rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            continue

        frame = np.asanyarray(color_frame.get_data())

        # draw detection (optional)
        tag_info = try_get_target_tag_T_cam_tag(frame)
        if tag_info is not None:
            # visualize
            corners = tag_info["corners"]
            cv2.polylines(frame,[corners],isClosed=True,color=(0,255,0),thickness=2)  # pupil-apriltags returns corners in detection obj; we didn't keep it for best
            T_cam_tag = tag_info["T_cam_tag"]
            x, y, z = T_cam_tag[:3, 3]
            cv2.putText(
                frame,
                f"tag {tag_info['tag_id']}  xyz(m)=({x:.3f},{y:.3f},{z:.3f})  dm={tag_info['decision_margin']:.1f} ham={tag_info['hamming']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"tag {TAG_ID_TARGET} not found / low confidence (dm<{MIN_DECISION_MARGIN} or ham>{MAX_HAMMING})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        disp = cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA)
        cv2.imshow("handeye_capture", disp)
        k=cv2.waitKey(1)& 0xFF
        save_pressed = (k == ord("p"))
        quit_pressed = (k == ord("q"))
        # ---- Save sample on 's' ----
        # KeyboardTeleop 的键名不一定叫 key_s；稳妥起见：你先打印 keyboard_keys 看字段。
        # 这里我们兼容几种常见写法：
        #s_pressed = bool(keyboard_keys.get("key_p", False) or keyboard_keys.get("p", False) or keyboard_keys.get("SAVE", False))

        now = time.time()
        if save_pressed and (now - _last_save_t) > SAVE_DEBOUNCE_SEC:
            _last_save_t = now

            if NO_ROBOT:
                print("[WARN] NO_ROBOT enabled: cannot read observation for joints. Skip save.")
            elif tag_info is None:
                print("[WARN] tag not stable/visible, skip save. (Try better lighting, closer distance, slower motion.)")
            else:
                try:
                    q_deg = get_left_arm_q_deg_from_observation(observation)
                except KeyError as e:
                    print(f"[ERROR] observation missing key: {e}. Print observation.keys() to confirm.")
                    q_deg = None

                if q_deg is not None:
                    sample = {
                        "t": now,
                        "arm_profile": args.leader_profile,
                        "q_deg_left_5dof": q_deg,
                        "tag_id": tag_info["tag_id"],
                        "decision_margin": tag_info["decision_margin"],
                        "hamming": tag_info["hamming"],
                        "pose_err": tag_info["pose_err"],
                        "T_cam_tag": tag_info["T_cam_tag"].tolist(),
                        "camera_params": list(camera_params),
                        "tag_size_m": TAG_SIZE,
                    }
                    f_out.write(json.dumps(sample) + "\n")
                    f_out.flush()
                    print(f"[SAVED] q_deg={['%.2f'%v for v in q_deg]}  z={tag_info['T_cam_tag'][2,3]:.3f} m  dm={tag_info['decision_margin']:.1f}")

        # quit with 'q' in OpenCV window
        if quit_pressed:
            break

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

finally:
    try:
        f_out.close()
    except Exception:
        pass
    try:
        rs_pipeline.stop()
    except Exception:
        pass