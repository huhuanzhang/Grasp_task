#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move robot EE to a target (x,y) defined in tag-plane coordinates.

Mapping used:
  p_base_xy = R @ p_tag_xy + t
where (R,t) is loaded from a SE2 json (e.g. outputs/T_base_tag_se2_from_q_first4.json).

Example:
  /home/hhz/miniconda3/envs/lerobot_alohamini/bin/python move_to_tag_xy.py \
    --remote_ip 172.16.0.14 \
    --urdf so101_new_calib.urdf \
    --se2 outputs/T_base_tag_se2_from_q_first4.json \
    --x_cm 10 --y_cm 15 --keep_z
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep


def dls(J, e, lam=0.03):
    JJt = J @ J.T
    A = JJt + (lam * lam) * np.eye(3)
    return J.T @ np.linalg.solve(A, e)


class IKPos5Dof:
    def __init__(self, urdf_path: str, ee_frame: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(ee_frame)
        if self.frame_id == len(self.model.frames):
            raise ValueError(f"Frame {ee_frame} not found in URDF")

        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.joint_ids = [self.model.getJointId(n) for n in self.joint_names]
        self.q_idx = [self.model.joints[j].idx_q for j in self.joint_ids]
        self.v_idx = [self.model.joints[j].idx_v for j in self.joint_ids]

    def q_from_obs_deg(self, obs: dict) -> np.ndarray:
        q = pin.neutral(self.model)
        keys = [
            "arm_left_shoulder_pan.pos",
            "arm_left_shoulder_lift.pos",
            "arm_left_elbow_flex.pos",
            "arm_left_wrist_flex.pos",
            "arm_left_wrist_roll.pos",
        ]
        for idx_q, k in zip(self.q_idx, keys):
            q[idx_q] = np.deg2rad(float(obs[k]))
        return q

    def q_to_deg5(self, q: np.ndarray) -> list[float]:
        return [float(np.rad2deg(q[idx])) for idx in self.q_idx]

    def fk_pos(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.frame_id].homogeneous
        return T[:3, 3].copy()

    def jac_pos(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J6[:3, :][:, self.v_idx]

    def step_towards(self, q: np.ndarray, p_goal: np.ndarray, step_scale=0.6, lam=0.03):
        p = self.fk_pos(q)
        e = p_goal - p
        err = float(np.linalg.norm(e))
        J = self.jac_pos(q)
        dq = dls(J, e, lam=lam)
        dq = np.clip(dq, -0.12, 0.12)
        q2 = q.copy()
        for idx, dqi in zip(self.q_idx, dq):
            q2[idx] += step_scale * dqi
        return q2, err, p


def load_se2(path: str):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    R = np.array(d["R"], dtype=np.float64)
    t = np.array(d["t"], dtype=np.float64)
    return R, t


def send_left_arm(robot: LeKiwiClient, obs: dict, q_deg5: list[float], lift_mm: float):
    action = {
        "arm_left_shoulder_pan.pos": q_deg5[0],
        "arm_left_shoulder_lift.pos": q_deg5[1],
        "arm_left_elbow_flex.pos": q_deg5[2],
        "arm_left_wrist_flex.pos": q_deg5[3],
        "arm_left_wrist_roll.pos": q_deg5[4],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote_ip", default="172.16.0.14")
    ap.add_argument("--urdf", default="so101_new_calib.urdf")
    ap.add_argument("--ee-frame", default="gripper_frame_link")
    ap.add_argument("--se2", default="outputs/T_base_tag_se2_from_q_first4.json")

    ap.add_argument("--x_cm", type=float, required=True, help="target x in local input frame, cm")
    ap.add_argument("--y_cm", type=float, required=True, help="target y in local input frame, cm")
    ap.add_argument("--origin_x_cm", type=float, default=0.0, help="origin x (cm) in calibrated tag-plane frame")
    ap.add_argument("--origin_y_cm", type=float, default=0.0, help="origin y (cm) in calibrated tag-plane frame")
    ap.add_argument("--input_theta_deg", type=float, default=0.0, help="rotate input frame into calibrated tag-plane frame (deg)")

    ap.add_argument("--keep_z", action="store_true", help="keep current EE z (recommended default usage)")
    ap.add_argument("--z_base", type=float, default=None, help="target base z in meters if not keep_z")

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=220)
    ap.add_argument("--tol_m", type=float, default=0.008)
    ap.add_argument("--lift_mm", type=float, default=500.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    R, t = load_se2(args.se2)

    # Input point is local to user-defined frame; convert to calibrated tag-plane frame first
    p_local = np.array([args.x_cm / 100.0, args.y_cm / 100.0], dtype=np.float64)
    th = np.deg2rad(float(args.input_theta_deg))
    c, s = np.cos(th), np.sin(th)
    R_in = np.array([[c, -s], [s, c]], dtype=np.float64)
    origin = np.array([args.origin_x_cm / 100.0, args.origin_y_cm / 100.0], dtype=np.float64)
    p_tag = origin + (R_in @ p_local)

    p_base_xy = R @ p_tag + t

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
    robot.connect()

    ik = IKPos5Dof(args.urdf, args.ee_frame)

    obs0 = robot.get_observation()
    q_cmd = ik.q_from_obs_deg(obs0)
    p0 = ik.fk_pos(q_cmd)

    if args.keep_z:
        z_goal = float(p0[2])
    else:
        if args.z_base is None:
            raise ValueError("Use --keep_z or provide --z_base")
        z_goal = float(args.z_base)

    p_goal = np.array([p_base_xy[0], p_base_xy[1], z_goal], dtype=np.float64)

    print("[INFO] input local xy (m):", p_local.tolist())
    print("[INFO] input->tag origin (m):", origin.tolist(), " theta_deg=", float(args.input_theta_deg))
    print("[INFO] target tag xy (m):", p_tag.tolist())
    print("[INFO] target base xyz (m):", p_goal.tolist())
    print("[INFO] current base xyz (m):", p0.tolist())

    if args.dry_run:
        return

    for i in range(args.iters):
        q_cmd, err, p_cur = ik.step_towards(q_cmd, p_goal, step_scale=0.6, lam=0.03)

        obs = robot.get_observation()
        q_deg5 = ik.q_to_deg5(q_cmd)
        send_left_arm(robot, obs, q_deg5, lift_mm=args.lift_mm)

        if i % 10 == 0:
            print(f"iter={i:03d} err={err:.4f} cur={p_cur.tolist()}")

        if err < args.tol_m:
            print(f"[DONE] reached tolerance: err={err:.4f} m at iter={i}")
            break

        precise_sleep(1.0 / max(args.fps, 1))

    print("[END] movement finished")


if __name__ == "__main__":
    main()
