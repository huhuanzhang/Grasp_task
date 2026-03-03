from __future__ import annotations
import argparse
from pathlib import Path
import json
import cv2

from table_guidance.apriltag_backend import apriltag_detect
from table_guidance.viz import draw_tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    tags = apriltag_detect(img, tag_family=args.tag_family)
    overlay = draw_tags(img, tags)
    cv2.imwrite(str(out_dir / "inspect_tags.jpg"), overlay)

    info = [{
        "id": t.tag_id,
        "center": [float(t.center[0]), float(t.center[1])],
        "corners": [[float(x), float(y)] for x, y in t.corners],
        "decision_margin": t.decision_margin,
    } for t in tags]

    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir/'inspect_tags.jpg'}")

if __name__ == "__main__":
    main()
