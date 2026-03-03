#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Move robot to a previously calibrated point by replaying recorded joint angles.
This bypasses SE2/IK and uses one-to-one correspondence directly.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig


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

    q = lines[idx]["q_deg_left_5dof"]
    print(f"[INFO] target={target} -> sample_index={idx} -> q={q}")

    if args.dry_run:
        return

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
    robot.connect()
    obs = robot.get_observation()

    action = {
        "arm_left_shoulder_pan.pos": float(q[0]),
        "arm_left_shoulder_lift.pos": float(q[1]),
        "arm_left_elbow_flex.pos": float(q[2]),
        "arm_left_wrist_flex.pos": float(q[3]),
        "arm_left_wrist_roll.pos": float(q[4]),
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
    print("[DONE] sent calibrated joint target")


if __name__ == "__main__":
    main()
