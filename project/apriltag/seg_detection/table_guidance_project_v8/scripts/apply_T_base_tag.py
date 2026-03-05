#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_T_base_tag.py
Read result.json + T_base_tag_se2.json -> base XY(mm) + pregrasp/grasp (relative Z).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Dict
import numpy as np

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--T", required=True)
    ap.add_argument("--tag-xy-units", choices=["m","mm"], default="m")
    ap.add_argument("--T-units", choices=["mm","m"], default="mm")
    ap.add_argument("--paper-z-mm", type=float, default=None)
    ap.add_argument("--pregrasp-dz-mm", type=float, default=80.0)
    ap.add_argument("--grasp-dz-mm", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = _load_json(args.result)
    T = _load_json(args.T)
    R = np.array(T["R"], dtype=np.float64).reshape(2,2)
    t = np.array(T["t"], dtype=np.float64).reshape(2,)

    x,y = map(float, result["target_xy_m_in_tag_plane"])
    xy = np.array([x,y], dtype=np.float64)
    if args.tag_xy_units=="m" and args.T_units=="mm":
        xy_T = xy*1000.0
    elif args.tag_xy_units=="mm" and args.T_units=="m":
        xy_T = xy/1000.0
    else:
        xy_T = xy

    base_xy = (R@xy_T)+t
    base_xy_mm = base_xy*1000.0 if args.T_units=="m" else base_xy

    yaw_base = None
    if result.get("target_yaw_rad_in_plane", None) is not None:
        yaw_tag = float(result["target_yaw_rad_in_plane"])
        yaw_R = float(math.atan2(R[1,0], R[0,0]))
        yaw_base = _wrap_pi(yaw_R + yaw_tag)

    out: Dict[str, Any] = {
        "target_base_xy_mm": [float(base_xy_mm[0]), float(base_xy_mm[1])],
        "pregrasp_base_xy_mm_and_dz_mm": [float(base_xy_mm[0]), float(base_xy_mm[1]), float(args.pregrasp_dz_mm)],
        "grasp_base_xy_mm_and_dz_mm": [float(base_xy_mm[0]), float(base_xy_mm[1]), float(args.grasp_dz_mm)],
    }
    if yaw_base is not None:
        out["target_yaw_rad_in_base_plane"] = float(yaw_base)

    if args.paper_z_mm is not None:
        z0 = float(args.paper_z_mm)
        out["pregrasp_base_xyz_mm"] = [float(base_xy_mm[0]), float(base_xy_mm[1]), z0+float(args.pregrasp_dz_mm)]
        out["grasp_base_xyz_mm"] = [float(base_xy_mm[0]), float(base_xy_mm[1]), z0+float(args.grasp_dz_mm)]

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
        print(f"Saved: {p}")

if __name__=="__main__":
    main()
