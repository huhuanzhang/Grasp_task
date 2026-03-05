#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep


def capture_d455_image(path: Path, w=1280, h=720, fps=30):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipeline.start(cfg)
    try:
        frames = None
        for _ in range(20):
            frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
    finally:
        pipeline.stop()


def idw_interpolate_q(xy_cm: np.ndarray, calib_xy_cm: np.ndarray, calib_q: np.ndarray, power=2.0, eps=1e-6):
    d = np.linalg.norm(calib_xy_cm - xy_cm.reshape(1, 2), axis=1)
    j = int(np.argmin(d))
    if d[j] < 1e-4:
        return calib_q[j]
    w = 1.0 / np.power(d + eps, power)
    w = w / np.sum(w)
    q = (w.reshape(-1, 1) * calib_q).sum(axis=0)
    return q


def send_action(robot: LeKiwiClient, obs: dict, q_deg5: np.ndarray, lift_mm: float):
    action = {
        "arm_left_shoulder_pan.pos": float(q_deg5[0]),
        "arm_left_shoulder_lift.pos": float(q_deg5[1]),
        "arm_left_elbow_flex.pos": float(q_deg5[2]),
        "arm_left_wrist_flex.pos": float(q_deg5[3]),
        "arm_left_wrist_roll.pos": float(q_deg5[4]),
        "arm_left_gripper.pos": float(obs.get("arm_left_gripper.pos", 50.0)),

        "arm_right_shoulder_pan.pos": float(obs.get("arm_right_shoulder_pan.pos", 0.0)),
        "arm_right_shoulder_lift.pos": float(obs.get("arm_right_shoulder_lift.pos", 0.0)),
        "arm_right_elbow_flex.pos": float(obs.get("arm_right_elbow_flex.pos", 0.0)),
        "arm_right_wrist_flex.pos": float(obs.get("arm_right_wrist_flex.pos", 0.0)),
        "arm_right_wrist_roll.pos": float(obs.get("arm_right_wrist_roll.pos", 0.0)),
        "arm_right_gripper.pos": float(obs.get("arm_right_gripper.pos", 50.0)),

        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
        "lift_axis.height_mm": float(lift_mm),
    }
    robot.send_action(action)


def load_q_from_samples(samples: Path, n: int):
    lines = [json.loads(x) for x in samples.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < n:
        raise ValueError(f"samples lines({len(lines)}) < required points({n})")
    q = np.array([lines[i]["q_deg_left_5dof"] for i in range(n)], dtype=np.float64)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote_ip", default="172.16.0.14")
    ap.add_argument("--project", default="/home/hhz/.openclaw/workspace/project/apriltag/table_guidance_project_v8")
    ap.add_argument("--prompt", default="green stick")
    ap.add_argument("--weights", default="FastSAM-x.pt")

    ap.add_argument("--samples", default="/home/hhz/.openclaw/workspace/project/apriltag/samples.jsonl")
    ap.add_argument("--calib_points_cm", default="10,10;0,20;20,20;10,20;0,10;20,10")
    ap.add_argument("--power", type=float, default=2.0)

    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lift_mm", type=float, default=500.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    pts = []
    for it in args.calib_points_cm.split(";"):
        x, y = it.split(",")
        pts.append([float(x), float(y)])
    calib_xy_cm = np.array(pts, dtype=np.float64)
    calib_q = load_q_from_samples(Path(args.samples), len(calib_xy_cm))

    project = Path(args.project)
    image = project / "assets" / "d455_live.jpg"
    out_dir = project / "outputs"
    out_json = out_dir / "result.json"

    print("[1/4] Capturing D455 image...")
    capture_d455_image(image)
    print(f"Saved image: {image}")

    print("[2/4] Running v8 run_demo...")
    cmd = [
        "/home/hhz/miniconda3/envs/lerobot_alohamini/bin/python",
        str(project / "scripts" / "run_demo.py"),
        "--image", str(image),
        "--config", str(project / "config" / "demo.yaml"),
        "--fastsam-weights", str(project / args.weights),
        "--text-prompt", args.prompt,
        "--device", "cpu",
        "--out", str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    result = json.loads(out_json.read_text(encoding="utf-8"))
    tag_xy_m = np.array(result["target_xy_m_in_tag_plane"], dtype=np.float64)
    xy_cm = tag_xy_m * 100.0
    print(f"Detected target_xy_cm = {xy_cm.tolist()}")

    q_cmd = idw_interpolate_q(xy_cm, calib_xy_cm, calib_q, power=float(args.power))
    print(f"[3/4] Interpolated q_deg5 = {q_cmd.tolist()}")

    if args.dry_run:
        return

    print("[4/4] Streaming commands...")
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
    robot.connect()

    n = max(1, int(float(args.duration) * max(args.fps, 1)))
    for i in range(n):
        obs = robot.get_observation()
        send_action(robot, obs, q_cmd, lift_mm=args.lift_mm)
        if i % max(1, args.fps // 2) == 0:
            print(f"[SEND] {i+1}/{n}")
        precise_sleep(1.0 / max(args.fps, 1))

    print("[DONE] streamed interpolated target")


if __name__ == "__main__":
    main()
