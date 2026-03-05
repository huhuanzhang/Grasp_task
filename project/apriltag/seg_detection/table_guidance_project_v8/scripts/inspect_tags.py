from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from table_guidance.apriltag_backend import apriltag_detect
from table_guidance.viz import draw_tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    tags = apriltag_detect(img, tag_family=args.tag_family, estimate_tag_pose=False)
    cv2.imwrite(str(out_dir/"inspect_tags.jpg"), draw_tags(img, tags))

    info = [{"id": t.tag_id, "center": [float(t.center[0]), float(t.center[1])]} for t in tags]
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir/'inspect_tags.jpg'}")

if __name__ == "__main__":
    main()
