#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pairs_tool.py

用途：
1) 从 samples_no_tag_ok.jsonl 生成 pairs_template.json
2) 校验并生成 pairs.json（严格一一对应）
3) 可选：直接调用 calibrate_base_from_points.py 求 T_base_tag_se2.json
"""

from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_template(jsonl_path: Path, out_path: Path, require_visible: bool = True):
    if not jsonl_path.exists():
        raise SystemExit(f"[ERROR] jsonl not found: {jsonl_path}")

    src: List[List[float]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        visible = bool(d.get("tag_visible"))
        if require_visible and not visible:
            continue
        T = d.get("T_cam_tag")
        if not T:
            continue
        # 模板点：取 tag 原点在相机坐标系的 x,y 供人工对齐（后续可手工改）
        src.append([float(T[0][3]), float(T[1][3])])

    if len(src) < 2:
        raise SystemExit(f"[ERROR] usable samples too few: {len(src)}")

    tpl = {
        "src_tag_xy": src,
        "dst_base_xy": [[None, None] for _ in src],
        "note": "手动填写 dst_base_xy，与 src_tag_xy 行数严格一致、一一对应"
    }
    save_json(out_path, tpl)
    print(f"[OK] template saved: {out_path} (N={len(src)})")


def validate_and_export(template_path: Path, out_pairs: Path):
    if not template_path.exists():
        raise SystemExit(f"[ERROR] template not found: {template_path}")

    d = load_json(template_path)
    src = d.get("src_tag_xy", [])
    dst = d.get("dst_base_xy", [])

    if not isinstance(src, list) or not isinstance(dst, list):
        raise SystemExit("[ERROR] src_tag_xy / dst_base_xy must be lists")

    if len(src) != len(dst):
        raise SystemExit(f"[ERROR] count mismatch: src={len(src)} dst={len(dst)}")

    if len(src) < 2:
        raise SystemExit("[ERROR] need at least 2 pairs")

    for i, (a, b) in enumerate(zip(src, dst)):
        if (not isinstance(a, list)) or len(a) != 2:
            raise SystemExit(f"[ERROR] row {i}: src_tag_xy must be [x, y]")
        if (not isinstance(b, list)) or len(b) != 2:
            raise SystemExit(f"[ERROR] row {i}: dst_base_xy must be [x, y]")
        if any(v is None for v in b):
            raise SystemExit(f"[ERROR] row {i}: dst_base_xy has None")
        try:
            src[i] = [float(a[0]), float(a[1])]
            dst[i] = [float(b[0]), float(b[1])]
        except Exception:
            raise SystemExit(f"[ERROR] row {i}: values must be numeric")

    out = {"src_tag_xy": src, "dst_base_xy": dst}
    save_json(out_pairs, out)
    print(f"[OK] pairs saved: {out_pairs} (N={len(src)})")


def solve_se2(pairs_path: Path, out_se2: Path, project_v8: Path, python_bin: str):
    script = project_v8 / "scripts" / "calibrate_base_from_points.py"
    if not script.exists():
        raise SystemExit(f"[ERROR] script not found: {script}")

    cmd = [
        python_bin,
        str(script),
        "--pairs", str(pairs_path),
        "--out", str(out_se2),
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(project_v8 / "src")
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description="jsonl/pairs/se2 helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("template", help="Generate pairs template from jsonl")
    a.add_argument("--jsonl", default="samples_no_tag_ok.jsonl")
    a.add_argument("--out", default="pairs_template.json")
    a.add_argument("--allow-invisible", action="store_true", help="Allow rows with tag_visible=false")

    b = sub.add_parser("finalize", help="Validate template and export pairs.json")
    b.add_argument("--template", default="pairs_template.json")
    b.add_argument("--out", default="pairs.json")

    c = sub.add_parser("solve", help="Solve T_base_tag_se2 from pairs.json")
    c.add_argument("--pairs", default="pairs.json")
    c.add_argument("--out", default="outputs/T_base_tag_se2.json")
    c.add_argument("--project-v8", default="seg_detection/table_guidance_project_v8")
    c.add_argument("--python", default="/home/hhz/miniconda3/envs/lerobot_alohamini/bin/python")

    args = ap.parse_args()

    if args.cmd == "template":
        build_template(Path(args.jsonl), Path(args.out), require_visible=not args.allow_invisible)
    elif args.cmd == "finalize":
        validate_and_export(Path(args.template), Path(args.out))
    elif args.cmd == "solve":
        solve_se2(Path(args.pairs), Path(args.out), Path(args.project_v8), args.python)


if __name__ == "__main__":
    main()
