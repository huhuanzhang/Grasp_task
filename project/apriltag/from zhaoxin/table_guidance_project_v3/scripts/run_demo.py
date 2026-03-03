from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from table_guidance.apriltag_backend import apriltag_detect
from table_guidance.geom import homography_from_tag, pixel_to_plane, plane_vec_from_pixel_vec
from table_guidance.segmentation import segment_target_mask
from table_guidance.pose import largest_connected_component, mask_centroid_and_axis
from table_guidance.viz import draw_tags, draw_target

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--config", required=True, help="yaml config path")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--seg-backend", default="fastsam", choices=["color", "fastsam"], help="segmentation backend")
    ap.add_argument("--mask", default=None, help="optional precomputed mask image (white=object)")
    ap.add_argument("--fastsam-weights", default=None, help="FastSAM weights path, e.g. FastSAM-x.pt")
    ap.add_argument("--text-prompt", default=None, help="FastSAM text prompt, e.g. 'pink building block' or 'cucumber'")
    ap.add_argument("--device", default=None, help="ultralytics device: cpu/cuda/mps")
    ap.add_argument("--imgsz", type=int, default=None, help="FastSAM imgsz")
    ap.add_argument("--conf", type=float, default=None, help="FastSAM conf threshold")
    ap.add_argument("--iou", type=float, default=None, help="FastSAM iou threshold")
    ap.add_argument("--tag-id", type=int, default=None, help="override reference_tag_id in config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ref_id = int(args.tag_id) if args.tag_id is not None else int(cfg.get("reference_tag_id", 0))
    tag_size_m = float(cfg.get("tag_size_m", 0.06))
    tag_family = str(cfg.get("tag_family", "tag36h11"))

    fastsam_cfg = cfg.get("fastsam", {}) or {}
    imgsz = int(args.imgsz) if args.imgsz is not None else int(fastsam_cfg.get("imgsz", 640))
    conf = float(args.conf) if args.conf is not None else float(fastsam_cfg.get("conf", 0.25))
    iou = float(args.iou) if args.iou is not None else float(fastsam_cfg.get("iou", 0.9))
    device = str(args.device) if args.device is not None else "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    # 1) AprilTag detection
    tags = apriltag_detect(img, tag_family=tag_family)
    if len(tags) == 0:
        raise RuntimeError("No AprilTags detected. Check lighting / tag_family.")
    ref = next((t for t in tags if t.tag_id == ref_id), None)
    if ref is None:
        raise RuntimeError(f"Reference tag id {ref_id} not found. Detected: {[t.tag_id for t in tags]}")

    # 2) Image->plane homography (single tag)
    H_img2plane = homography_from_tag(ref.corners, tag_size_m=tag_size_m)

    # 3) Segmentation -> mask
    mask = segment_target_mask(
        img,
        backend=args.seg_backend,
        mask_path=args.mask,
        fastsam_weights=args.fastsam_weights,
        text_prompt=args.text_prompt,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    mask = largest_connected_component(mask)
    (cu, cv), axis_uv = mask_centroid_and_axis(mask)

    # 4) Pixel -> plane coords
    x_m, y_m = pixel_to_plane(H_img2plane, (cu, cv))

    # 5) Orientation (optional): map pixel axis to plane yaw
    yaw = None
    if axis_uv is not None:
        du, dv = float(axis_uv[0]), float(axis_uv[1])
        vec_plane = plane_vec_from_pixel_vec(H_img2plane, (cu, cv), (du*50.0, dv*50.0))
        if np.linalg.norm(vec_plane) > 1e-9:
            yaw = float(np.arctan2(vec_plane[1], vec_plane[0]))

    # 6) Save outputs
    overlay = draw_tags(img, tags)
    overlay = draw_target(overlay, (cu, cv), axis_uv, (x_m, y_m), yaw)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)
    cv2.imwrite(str(out_dir / "mask.png"), mask)

    result = {
        "reference_tag_id": ref_id,
        "tag_size_m": tag_size_m,
        "pixel_xy": [cu, cv],
        "target_xy_m_in_tag_plane": [x_m, y_m],
        "target_yaw_rad_in_plane": yaw,
        "detected_tag_ids": [t.tag_id for t in tags],
        "fastsam": {"device": device, "imgsz": imgsz, "conf": conf, "iou": iou},
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir/'overlay.jpg'}")
    print(f"Saved: {out_dir/'result.json'}")

if __name__ == "__main__":
    main()
