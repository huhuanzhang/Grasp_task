from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from table_guidance.calibration import estimate_se2_from_points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="outputs/T_base_tag_se2.json")
    args = ap.parse_args()

    data = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    src = np.array(data["src_tag_xy"], dtype=np.float64)
    dst = np.array(data["dst_base_xy"], dtype=np.float64)

    R, t = estimate_se2_from_points(src, dst)
    out = {"R": R.tolist(), "t": t.tolist()}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {p}")

if __name__ == "__main__":
    main()
