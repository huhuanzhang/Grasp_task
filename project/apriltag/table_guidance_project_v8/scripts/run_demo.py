from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import yaml

from table_guidance.apriltag_backend import apriltag_detect
from table_guidance.segmentation import segment_target_mask
from table_guidance.pose import largest_connected_component, mask_centroid_and_axis
from table_guidance.viz import draw_tags, draw_target
from table_guidance.plane_fusion import build_plane_model_from_tags, pixel_to_plane_via_ray, PlaneModel, project_point_to_plane

def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def _fallback_plane_from_ref(ref, tag_size_m: float) -> PlaneModel:
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
    ap.add_argument("--mask", default=None)
    ap.add_argument("--fastsam-weights", default=None)
    ap.add_argument("--text-prompt", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ref_id = int(cfg["reference_tag_id"])
    tag_size_m = float(cfg["tag_size_m"])
    tag_family = str(cfg.get("tag_family", "tag36h11"))
    cam = cfg["camera"]
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    cam_w, cam_h = int(cam.get("width", 0)), int(cam.get("height", 0))
    pregrasp_offset_m = float(cfg.get("pregrasp_offset_m", 0.08))

    pf = cfg.get("plane_fit", {}) or {}
    pf_enabled = bool(pf.get("enabled", True))
    use_tag_corners = bool(pf.get("use_tag_corners", True))
    min_tags = int(pf.get("min_tags", 2))
    weight_by_dm = bool(pf.get("weight_by_decision_margin", True))

    fastsam_cfg = cfg.get("fastsam", {}) or {}
    imgsz = int(fastsam_cfg.get("imgsz", 640))
    conf = float(fastsam_cfg.get("conf", 0.25))
    iou = float(fastsam_cfg.get("iou", 0.9))

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    H_img, W_img = img.shape[:2]
    if cam_w and cam_h and (W_img != cam_w or H_img != cam_h):
        print(f"[WARN] Image is {W_img}x{H_img} but intrinsics are for {cam_w}x{cam_h}. Pose may be inaccurate.")

    tags = apriltag_detect(img, tag_family=tag_family, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=tag_size_m)
    tags_pose = [t for t in tags if t.pose_R is not None and t.pose_t is not None]
    if not tags_pose:
        raise RuntimeError("No tags with pose found.")
    ref = next((t for t in tags_pose if t.tag_id == ref_id), None)
    if ref is None:
        raise RuntimeError(f"Reference tag {ref_id} not found. Detected: {[t.tag_id for t in tags_pose]}")

    plane_info = {"enabled": pf_enabled, "mode": None}
    if pf_enabled and len(tags_pose) >= min_tags:
        plane_model = build_plane_model_from_tags(tags_pose, ref_tag_id=ref_id, tag_size_m=tag_size_m,
                                                  fx=fx, fy=fy, cx=cx, cy=cy,
                                                  use_tag_corners=use_tag_corners,
                                                  weight_by_decision_margin=weight_by_dm)
        plane_info.update({
            "mode": "multi_tag_ls",
            "num_tags": len(tags_pose),
            "use_tag_corners": use_tag_corners,
            "weight_by_decision_margin": weight_by_dm,
            "n": plane_model.n.tolist(),
            "d": float(plane_model.d),
        })
    else:
        plane_model = _fallback_plane_from_ref(ref, tag_size_m)
        plane_info.update({"mode": "ref_only", "num_tags": len(tags_pose), "n": plane_model.n.tolist(), "d": float(plane_model.d)})

    # Segment target
    mask = segment_target_mask(img, mask_path=args.mask, fastsam_weights=args.fastsam_weights, text_prompt=args.text_prompt,
                               device=args.device, imgsz=imgsz, conf=conf, iou=iou)
    mask = largest_connected_component(mask)
    (cu, cv), axis_uv = mask_centroid_and_axis(mask)

    p_cam, (x_m, y_m) = pixel_to_plane_via_ray(cu, cv, fx, fy, cx, cy, plane_model)

    yaw = None
    if axis_uv is not None:
        du, dv = float(axis_uv[0]), float(axis_uv[1])
        _, (x2, y2) = pixel_to_plane_via_ray(cu + du*30.0, cv + dv*30.0, fx, fy, cx, cy, plane_model)
        v_plane = np.array([x2-x_m, y2-y_m], dtype=np.float64)
        if float(np.linalg.norm(v_plane)) > 1e-9:
            yaw = float(np.arctan2(v_plane[1], v_plane[0]))

    n_cam = plane_model.n
    p_pre = p_cam + n_cam * pregrasp_offset_m

    overlay = draw_tags(img, tags)
    overlay = draw_target(overlay, (cu, cv), axis_uv, (x_m, y_m), yaw, extra=f"plane:{plane_info['mode']}")
    cv2.imwrite(str(out_dir/"overlay.jpg"), overlay)
    cv2.imwrite(str(out_dir/"mask.png"), mask)

    result = {
        "reference_tag_id": ref_id,
        "tag_size_m": tag_size_m,
        "pixel_xy": [cu, cv],
        "target_xy_m_in_tag_plane": [float(x_m), float(y_m)],
        "target_yaw_rad_in_plane": yaw,
        "target_xyz_m_in_camera": [float(p_cam[0]), float(p_cam[1]), float(p_cam[2])],
        "pregrasp_xyz_m_in_camera": [float(p_pre[0]), float(p_pre[1]), float(p_pre[2])],
        "table_normal_in_camera": [float(n_cam[0]), float(n_cam[1]), float(n_cam[2])],
        "detected_tag_ids": [t.tag_id for t in tags],
        "plane_fit": plane_info,
        "fastsam": {"device": args.device, "imgsz": imgsz, "conf": conf, "iou": iou},
        "camera_params": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": cam_w, "height": cam_h},
        "xy_method": "multi_tag_ls_plane_ray_intersection"
    }
    (out_dir/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir/'overlay.jpg'}")
    print(f"Saved: {out_dir/'result.json'}")

if __name__ == "__main__":
    main()
