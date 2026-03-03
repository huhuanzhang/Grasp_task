#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given joint-angle samples (jsonl) and corresponding plane points (cm),
compute T_base_tag SE2 mapping:
    p_base_xy = R @ p_tag_xy + t

Usage example:
python3 fit_base_tag_from_q_and_plane.py \
  --samples samples_no_tag_ok.jsonl \
  --urdf so101_new_calib.urdf \
  --ee-frame gripper_frame_link \
  --plane-cm "10,10;0,20;20,20;10,20;0,10;20,10" \
  --out outputs/T_base_tag_se2_from_q.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pinocchio as pin

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def parse_plane_cm(s: str) -> np.ndarray:
    pts = []
    for item in s.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        x, y = item.split(",")
        pts.append([float(x) / 100.0, float(y) / 100.0])
    arr = np.array(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Invalid --plane-cm format")
    return arr


def load_q_list(jsonl: Path) -> list[list[float]]:
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        q = s.get("q_deg_left_5dof")
        if q is None:
            continue
        if len(q) != 5:
            raise ValueError("Each q_deg_left_5dof must have length 5")
        out.append([float(v) for v in q])
    return out


def fk_xy(model, data, frame_id: int, q_deg_5: list[float]) -> np.ndarray:
    q = pin.neutral(model)
    for jn, ang_deg in zip(JOINT_NAMES, q_deg_5):
        jid = model.getJointId(jn)
        idx = model.joints[jid].idx_q
        q[idx] = np.deg2rad(float(ang_deg))
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T = data.oMf[frame_id].homogeneous
    p = T[:3, 3]
    return np.array([float(p[0]), float(p[1])], dtype=np.float64)


def estimate_se2(src_xy: np.ndarray, dst_xy: np.ndarray):
    """Estimate dst = R @ src + t (2D rigid transform)."""
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    c1 = src.mean(axis=0)
    c2 = dst.mean(axis=0)
    X = src - c1
    Y = dst - c2
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c2 - R @ c1
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--ee-frame", default="gripper_frame_link")
    ap.add_argument("--plane-cm", required=True, help='e.g. "10,10;0,20;20,20;10,20;0,10;20,10"')
    ap.add_argument("--out", default="outputs/T_base_tag_se2_from_q.json")
    args = ap.parse_args()

    samples_path = Path(args.samples)
    q_list = load_q_list(samples_path)
    plane_xy = parse_plane_cm(args.plane_cm)

    if len(q_list) != len(plane_xy):
        raise ValueError(f"Count mismatch: {len(q_list)} joint samples vs {len(plane_xy)} plane points")
    if len(q_list) < 3:
        raise ValueError("Need at least 3 points")

    model = pin.buildModelFromUrdf(str(args.urdf))
    data = model.createData()
    frame_id = model.getFrameId(args.ee_frame)
    if frame_id == len(model.frames):
        raise ValueError(f"EE frame not found: {args.ee_frame}")

    base_xy = np.array([fk_xy(model, data, frame_id, q) for q in q_list], dtype=np.float64)

    R, t = estimate_se2(plane_xy, base_xy)
    pred = (R @ plane_xy.T).T + t.reshape(1, 2)
    err = np.linalg.norm(pred - base_xy, axis=1)

    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    out = {
        "R": R.tolist(),
        "t": t.tolist(),
        "yaw_rad": yaw,
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "max_err_m": float(np.max(err)),
        "n": int(len(err)),
        "src_tag_xy_m": plane_xy.tolist(),
        "dst_base_xy_m": base_xy.tolist(),
        "per_point_err_m": err.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
