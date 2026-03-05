# table_guidance_project_v8 (Multi-tag least-squares plane fitting)

This version improves stability by using **all detected AprilTags** to fit the tabletop plane via **least squares**,
instead of relying on only one reference tag.

## What changes from v7
- v7: Use reference tag pose only to define the table plane (ray–plane intersection).
- v8: Use **multiple tags** to estimate a single best-fit plane in camera frame:
  - Collect many 3D points on the table plane from **all tags** (tag corners in camera frame).
  - Fit plane `n·p + d = 0` by SVD (least squares).
  - Keep the **table X/Y axes** aligned with the *reference tag* x-axis (projected onto the fitted plane),
    so your output coordinate system stays consistent for robot mapping.

This reduces noise in plane normal/offset and usually improves validation diffs.

---

## Requirements
- Python >= 3.9
- D455 RGB frames (we do NOT use depth)
- AprilTag family: `tag36h11`
- Tag size: **4cm** black square edge -> `tag_size_m = 0.04`

---

## 0) Install

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

---

## 1) Configure config/demo.yaml

`config/demo.yaml` is filled from your D455 RGB intrinsics:
- resolution: 1280x720
- fx=645.6649169921875, fy=644.830322265625, cx=642.4880981445312, cy=362.65289306640625
- tag_size_m=0.04

Plane fitting options:
- `plane_fit.enabled`: use multi-tag least squares
- `use_tag_corners`: use 4 corners per tag (more robust than just centers)
- `min_tags`: minimum number of tags required to enable plane fitting
- `weight_by_decision_margin`: weight each tag's corners by AprilTag detection confidence (decision_margin)

⚠️ Important:
- The image you process MUST have the same resolution as `camera.width/height`.
- If you change resolution, update intrinsics accordingly.

---

## 2) Detect AprilTags (sanity check)

```bash
python scripts/inspect_tags.py --image assets/cucumber.jpg --out outputs
```

Outputs:
- `outputs/inspect_tags.jpg`
- printed tag IDs

Choose a stable `reference_tag_id` and set it in `config/demo.yaml`.

---

## 3) Run the full pipeline (FastSAM + multi-tag plane fit)

```bash
python scripts/run_demo.py \
  --image assets/cucumber.jpg \
  --config config/demo.yaml \
  --fastsam-weights FastSAM-x.pt \
  --text-prompt "cucumber" \
  --device mps \
  --out outputs
```

Outputs:
- `outputs/overlay.jpg`
- `outputs/mask.png`
- `outputs/result.json` containing:
  - `target_xy_m_in_tag_plane` (meters, pose-consistent, using multi-tag plane fit)
  - `target_xyz_m_in_camera` (meters)
  - `pregrasp_xyz_m_in_camera` (meters)
  - `plane_fit` (plane normal/offset, number of tags/points used)

---

## 4) Validate transforms (pose vs multi-tag plane)

```bash
python scripts/validate_transforms.py --image assets/cucumber.jpg --config config/demo.yaml --out outputs
```

What it checks:
- For each non-reference tag:
  - pose chaining gives tag-center in reference tag frame (`pose_xy`)
  - ray–plane intersection + fitted plane gives (`rayplane_xy`)
- Prints diff in mm and writes `outputs/validate_overlay.jpg`.

---

## 5) Calibrate tag-plane -> robot base plane (SE(2), mm)

Edit `config/point_pairs_mm.json`:
- `src_tag_xy`: tag-frame points in **mm**
  - for 4cm tag: half = 20mm
  - corners: (-20,-20), (20,-20), (20,20), (-20,20), center (0,0)
- `dst_base_xy`: robot controller base XY readouts in **mm** when TCP aligns to those points

Then run:
```bash
python scripts/calibrate_base_from_points.py \
  --pairs config/point_pairs_mm.json \
  --out outputs/T_base_tag_se2.json
```

---

## 6) Produce final robot targets (base XY mm + relative-to-paper Z)

```bash
python scripts/apply_T_base_tag.py \
  --result outputs/result.json \
  --T outputs/T_base_tag_se2.json \
  --tag-xy-units m \
  --T-units mm \
  --pregrasp-dz-mm 80 \
  --grasp-dz-mm 10 \
  --out outputs/robot_targets.json
```

---

## Notes
- Distortion: this pipeline assumes pinhole `(fx,fy,cx,cy)` without explicit undistortion. If you see edge errors,
  consider undistorting RGB frames before processing.
- If fewer than `min_tags` tags are detected, the pipeline falls back to using the reference tag plane only.

---

## Scripts
- `scripts/inspect_tags.py`
- `scripts/run_demo.py`
- `scripts/validate_transforms.py`
- `scripts/calibrate_base_from_points.py`
- `scripts/apply_T_base_tag.py`
