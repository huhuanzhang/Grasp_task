# Table Guidance (AprilTag + Ultralytics FastSAM)

This project connects:

1) **AprilTag detection** (for metric scale on the table plane)  
2) **FastSAM segmentation** (Ultralytics `FastSAMPredictor` + text prompt, matching your working code)  
3) **Planar coordinate guidance**: pixel -> (x,y) meters in the **reference tag plane**  
4) Visualization + JSON output

Your setup is **eye-to-hand** (camera fixed in the environment), which is perfect for tabletop guiding.

---

## Quick start

### Install
```bash
cd table_guidance_project_v2
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

> On macOS (M1/M2/M3), install PyTorch with MPS support following PyTorch official instructions.
> Then run with `--device mps`.

---

## Step 1: inspect tag IDs
```bash
python scripts/inspect_tags.py --image assets/table_scene.jpg --out outputs
```
This prints detected tag IDs and saves `outputs/inspect_tags.jpg`.

---

## Step 2: set your reference tag and tag size
Edit `config/demo.yaml`:

- `reference_tag_id`: choose one of the detected IDs (prefer a clear, unoccluded tag).
- `tag_size_m`: black square edge length in meters (e.g. 6cm -> 0.06)

---

## Step 3: full pipeline (FastSAM prompt)
This uses **your exact Ultralytics workflow**:

- `FastSAM('FastSAM-x.pt')`
- `FastSAMPredictor(overrides=...)`
- `everything_results = predictor(img)`
- `results = predictor.prompt(everything_results, texts=...)`

Run:
```bash
python scripts/run_demo.py \
  --image assets/table_scene.jpg \
  --config config/demo.yaml \
  --seg-backend fastsam \
  --fastsam-weights FastSAM-x.pt \
  --text-prompt "pink building block" \
  --device mps \
  --out outputs
```

Outputs:
- `outputs/overlay.jpg` : tag boxes + target point + text with (x,y) meters
- `outputs/mask.png` : selected target mask
- `outputs/result.json` : machine-readable coordinates

---

## Fallback segmentation (for quick debug)
If you want to test without FastSAM, use a simple HSV color threshold:
```bash
python scripts/run_demo.py --image assets/table_scene.jpg --config config/demo.yaml --seg-backend color --out outputs
```

---

## Coordinate convention (reference tag plane)
Output `(x, y)` is in **meters** in the **reference tag plane**:

- Origin: **center of the reference tag**
- +x: tag's **right** direction in its printed orientation
- +y: tag's **down** direction
- z is assumed 0 on the table plane

To command the robot, you typically need a fixed transform `T_base_tag` (eye-to-hand):
- Either measure it (touch tag corners with TCP once),
- Or solve it from 3+ point correspondences.

A helper script is included: `scripts/calibrate_base_from_points.py`.

---

## Accuracy notes
- Use a larger tag size or bring camera closer to reduce noise.
- If you have multiple tags and you know their layout on the table, you can extend to multi-tag homography (not required for the current demo).
- In real robot grasping: move to `(x,y,z_pre)` first, re-run vision, then descend.


## Troubleshooting: mask looks right but point is wrong

If `mask.png` looks correct but the blue point is far away, it usually means the mask was in 640x640 (letterboxed) coordinates while the image is larger. v0.2.1 fixes this by building masks from `results[0].masks.xy` (original pixel polygons).
