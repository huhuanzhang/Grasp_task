#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge utility: convert supervisor vision output (tag-plane xy) to robot base xy,
then generate pregrasp/grasp targets.

Input:
  - result.json from table_guidance_project_v3/scripts/run_demo.py
  - T_base_tag_se2.json from calibrate_base_from_points.py

Output:
  - target JSON with base-frame targets
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transform_tag_xy_to_base_xy(tag_xy: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # p_base = R * p_tag + t
    return R @ tag_xy + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="result.json from table_guidance")
    ap.add_argument("--se2", required=True, help="T_base_tag_se2.json from point-pair calibration")
    ap.add_argument("--table-z", type=float, required=True, help="table height in base frame (meters)")
    ap.add_argument("--pre-z", type=float, default=0.10, help="pregrasp offset above table (m)")
    ap.add_argument("--grasp-z", type=float, default=0.02, help="grasp offset above table (m)")
    ap.add_argument("--out", default="target_base.json", help="output path")
    args = ap.parse_args()

    result = load_json(args.result)
    se2 = load_json(args.se2)

    tag_xy = np.array(result["target_xy_m_in_tag_plane"], dtype=np.float64)
    R = np.array(se2["R"], dtype=np.float64)
    t = np.array(se2["t"], dtype=np.float64)

    base_xy = transform_tag_xy_to_base_xy(tag_xy, R, t)

    yaw_tag = result.get("target_yaw_rad_in_plane", None)
    yaw_base = None
    if yaw_tag is not None:
        # For pure planar rotation: yaw_base = yaw_tag + yaw(R)
        yaw_R = float(np.arctan2(R[1, 0], R[0, 0]))
        yaw_base = float(yaw_tag + yaw_R)

    table_z = float(args.table_z)
    pre_xyz = [float(base_xy[0]), float(base_xy[1]), table_z + float(args.pre_z)]
    grasp_xyz = [float(base_xy[0]), float(base_xy[1]), table_z + float(args.grasp_z)]

    out = {
        "source": {
            "result_json": str(Path(args.result).resolve()),
            "se2_json": str(Path(args.se2).resolve()),
        },
        "target_xy_base_m": [float(base_xy[0]), float(base_xy[1])],
        "target_yaw_base_rad": yaw_base,
        "table_z_base_m": table_z,
        "pregrasp_xyz_base_m": pre_xyz,
        "grasp_xyz_base_m": grasp_xyz,
    }

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[SAVED] {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
