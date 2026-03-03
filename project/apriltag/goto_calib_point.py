#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Move robot to a previously calibrated point by replaying recorded joint angles.
This bypasses SE2/IK and uses one-to-one correspondence directly.

Compared to one-shot send, this version streams commands for a duration
(similar to teleop-style control), which is more reliable on this controller.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep


def parse_plane_cm(s: str):
    pts = []
    for item in s.strip().split(";"):
        if not item.strip():
            continue
        x, y = item.split(",")
        pts.append((float(x), float(y)))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote_ip", default="172.16.0.14")
    ap.add_argument("--samples", default="samples.jsonl")
    ap.add_argument("--plane-cm", required=True, help='e.g. "10,10;0,20;20,20;10,20;0,10;20,10"')
    ap.add_argument("--x_cm", type=float, required=True)
    ap.add_argument("--y_cm", type=float, required=True)
    ap.add_argument("--lift_mm", type=float, default=500.0)
    ap.add_argument("--duration", type=float, default=2.5, help="seconds to stream command")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ramp", type=float, default=0.6, help="0~1 blend from current to target; 1 means direct")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    plane_pts = parse_plane_cm(args.plane_cm)
    lines = [json.loads(x) for x in Path(args.samples).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < len(plane_pts):
        raise ValueError(f"samples lines({len(lines)}) < plane points({len(plane_pts)})")

    target = (float(args.x_cm), float(args.y_cm))
    try:
        idx = plane_pts.index(target)
    except ValueError:
        raise ValueError(f"target {target} not found in plane list {plane_pts}")

    q_tar = [float(v) for v in lines[idx]["q_deg_left_5dof"]]
    print(f"[INFO] target={target} -> sample_index={idx} -> q_target={q_tar}")

    if args.dry_run:
        return

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
    robot.connect()

    obs0 = robot.get_observation()
    q_cur = [
        float(obs0.get("arm_left_shoulder_pan.pos", q_tar[0])),
        float(obs0.get("arm_left_shoulder_lift.pos", q_tar[1])),
        float(obs0.get("arm_left_elbow_flex.pos", q_tar[2])),
        float(obs0.get("arm_left_wrist_flex.pos", q_tar[3])),
        float(obs0.get("arm_left_wrist_roll.pos", q_tar[4])),
    ]

    a = max(0.0, min(1.0, float(args.ramp)))
    q_cmd = [(1.0 - a) * c + a * t for c, t in zip(q_cur, q_tar)]

    print(f"[INFO] q_current={q_cur}")
    print(f"[INFO] q_command={q_cmd}  (ramp={a})")

    n = max(1, int(args.duration * max(args.fps, 1)))
    for i in range(n):
        obs = robot.get_observation()
        action = {
            "arm_left_shoulder_pan.pos": float(q_cmd[0]),
            "arm_left_shoulder_lift.pos": float(q_cmd[1]),
            "arm_left_elbow_flex.pos": float(q_cmd[2]),
            "arm_left_wrist_flex.pos": float(q_cmd[3]),
            "arm_left_wrist_roll.pos": float(q_cmd[4]),
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
            "lift_axis.height_mm": float(args.lift_mm),
        }
        robot.send_action(action)

        if i % max(1, args.fps // 2) == 0:
            print(f"[SEND] {i+1}/{n}")

        precise_sleep(1.0 / max(args.fps, 1))

    print("[DONE] streamed calibrated joint target")


if __name__ == "__main__":
    main()
