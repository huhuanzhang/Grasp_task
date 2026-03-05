from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml

from table_guidance.apriltag_backend import apriltag_detect
from table_guidance.transforms import cam_to_ref_tag
from table_guidance.plane_fusion import build_plane_model_from_tags, pixel_to_plane_via_ray, PlaneModel, project_point_to_plane
from table_guidance.viz import draw_tags

def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def _fallback_plane_from_ref(ref) -> PlaneModel:
    n = (np.asarray(ref.pose_R) @ np.array([0.0,0.0,1.0])).reshape(3,)
    n = n/(np.linalg.norm(n)+1e-12)
    d = -float(n @ np.asarray(ref.pose_t).reshape(3,))
    origin = project_point_to_plane(np.asarray(ref.pose_t).reshape(3,), n, d)
    ref_x = (np.asarray(ref.pose_R) @ np.array([1.0,0.0,0.0])).reshape(3,)
    x_proj = ref_x - float(ref_x @ n) * n
    x_hat = x_proj/(np.linalg.norm(x_proj)+1e-12)
    y_hat = np.cross(n, x_hat); y_hat=y_hat/(np.linalg.norm(y_hat)+1e-12)
    return PlaneModel(n=n, d=d, origin_cam=origin, x_hat=x_hat, y_hat=y_hat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ref_id = int(cfg["reference_tag_id"])
    tag_size_m = float(cfg["tag_size_m"])
    tag_family = str(cfg.get("tag_family", "tag36h11"))
    cam = cfg["camera"]
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])

    pf = cfg.get("plane_fit", {}) or {}
    pf_enabled = bool(pf.get("enabled", True))
    use_tag_corners = bool(pf.get("use_tag_corners", True))
    min_tags = int(pf.get("min_tags", 2))
    weight_by_dm = bool(pf.get("weight_by_decision_margin", True))

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    tags = apriltag_detect(img, tag_family=tag_family, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=tag_size_m)
    tags_pose = [t for t in tags if t.pose_R is not None and t.pose_t is not None]
    if not tags_pose:
        raise RuntimeError("No tags with pose found.")
    ref = next((t for t in tags_pose if t.tag_id == ref_id), None)
    if ref is None:
        raise RuntimeError("Reference tag pose missing")

    if pf_enabled and len(tags_pose) >= min_tags:
        plane_model = build_plane_model_from_tags(tags_pose, ref_tag_id=ref_id, tag_size_m=tag_size_m,
                                                  fx=fx, fy=fy, cx=cx, cy=cy,
                                                  use_tag_corners=use_tag_corners,
                                                  weight_by_decision_margin=weight_by_dm)
        mode = "multi_tag_ls"
    else:
        plane_model = _fallback_plane_from_ref(ref)
        mode = "ref_only"

    print(f"Detected tag IDs: {[t.tag_id for t in tags_pose]}")
    print(f"Reference tag ID: {ref_id}")
    print(f"Plane mode: {mode}\n")

    rows = []
    for t in tags_pose:
        if t.tag_id == ref_id:
            continue
        # pose-based other tag origin in ref tag frame
        p_ref = cam_to_ref_tag(np.asarray(t.pose_t).reshape(3,), np.asarray(ref.pose_R), np.asarray(ref.pose_t))
        pose_xy = p_ref[:2]
        pose_z = float(p_ref[2])

        # ray-plane-based from pixel center
        _, (x_ray, y_ray) = pixel_to_plane_via_ray(float(t.center[0]), float(t.center[1]), fx, fy, cx, cy, plane_model)
        diff_mm = (np.array([x_ray, y_ray]) - pose_xy) * 1000.0
        rows.append((t.tag_id, pose_xy[0], pose_xy[1], pose_z, x_ray, y_ray, diff_mm[0], diff_mm[1]))

    print("Compare OTHER TAG CENTERS (pose-based vs ray-plane-based):")
    print("  tag_id | pose_xy(m) pose_z(m) | rayplane_xy(m) | diff_xy(mm)")
    for r in rows:
        print(f"{r[0]:7d} | ({r[1]:+0.4f},{r[2]:+0.4f}) z={r[3]:+0.4f} | ({r[4]:+0.4f},{r[5]:+0.4f}) | ({r[6]:+0.1f},{r[7]:+0.1f})")

    overlay = draw_tags(img, tags_pose)
    for t in tags_pose:
        u,v = int(t.center[0]), int(t.center[1])
        cv2.circle(overlay, (u,v), 5, (0,255,0), -1)

    for r in rows:
        tag_id, *_rest, dx, dy = r
        t = next(tt for tt in tags_pose if tt.tag_id == tag_id)
        u,v = int(t.center[0]), int(t.center[1])
        mag = float(np.hypot(dx, dy))
        rad = int(min(40, max(8, mag/5.0)))
        cv2.circle(overlay, (u,v), rad, (0,0,255), 2)

    out_path = out_dir/"validate_overlay.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"\nSaved: {out_path}")
    print("Legend: green dot=detected tag centers; red ring size ~ diff(mm).")

if __name__ == "__main__":
    main()
