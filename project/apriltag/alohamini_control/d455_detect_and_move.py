#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
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
            raise ValueError(f"Frame {ee_frame} not found")

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
        return self.data.oMf[self.frame_id].homogeneous[:3, 3].copy()

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


def load_se2(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    R = np.array(d["R"], dtype=np.float64)
    t = np.array(d["t"], dtype=np.float64)
    return R, t


def send_action(robot: LeKiwiClient, obs: dict, q_deg5: list[float], lift_mm: float):
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
    ap.add_argument("--project", default="/home/hhz/.openclaw/workspace/project/apriltag/table_guidance_project_v8")
    ap.add_argument("--prompt", default="green stick")
    ap.add_argument("--weights", default="FastSAM-x.pt")

    ap.add_argument("--se2", default="/home/hhz/.openclaw/workspace/project/apriltag/outputs/T_base_tag_se2_from_q_6pts_fixed_order.json")
    ap.add_argument("--urdf", default="/home/hhz/.openclaw/workspace/project/apriltag/so101_new_calib.urdf")
    ap.add_argument("--ee-frame", default="gripper_frame_link")

    ap.add_argument("--z_base", type=float, default=-0.054)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=220)
    ap.add_argument("--tol_m", type=float, default=0.008)
    ap.add_argument("--lift_mm", type=float, default=500.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

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
    tag_xy = np.array(result["target_xy_m_in_tag_plane"], dtype=np.float64)
    print(f"Detected target_xy_m_in_tag_plane = {tag_xy.tolist()}")

    print("[3/4] Converting tag XY -> base XY...")
    R, t = load_se2(Path(args.se2))
    base_xy = R @ tag_xy + t
    p_goal = np.array([base_xy[0], base_xy[1], float(args.z_base)], dtype=np.float64)
    print(f"Goal base xyz = {p_goal.tolist()}")

    if args.dry_run:
        return

    print("[4/4] IK move...")
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
    robot.connect()
    ik = IKPos5Dof(args.urdf, args.ee_frame)

    obs = robot.get_observation()
    q_cmd = ik.q_from_obs_deg(obs)

    for i in range(args.iters):
        q_cmd, err, p_cur = ik.step_towards(q_cmd, p_goal, step_scale=0.6, lam=0.03)
        obs = robot.get_observation()
        send_action(robot, obs, ik.q_to_deg5(q_cmd), lift_mm=args.lift_mm)
        if i % 10 == 0:
            print(f"iter={i:03d} err={err:.4f} cur={p_cur.tolist()}")
        if err < args.tol_m:
            print(f"[DONE] reached tolerance: err={err:.4f} m")
            break
        precise_sleep(1.0 / max(args.fps, 1))


if __name__ == "__main__":
    main()
