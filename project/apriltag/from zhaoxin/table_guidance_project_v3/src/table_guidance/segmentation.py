from __future__ import annotations
from typing import Optional, Tuple, Literal, Dict, Any
import numpy as np
import cv2
import os
import tempfile

SegBackend = Literal["color", "fastsam"]

def _segment_color_pink(image_bgr: np.ndarray) -> np.ndarray:
    """HSV threshold for the pink demo block. Returns binary mask uint8 {0,255}."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 60, 50])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([150, 60, 50])
    upper2 = np.array([179, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def _load_mask(mask_path: str, image_shape: Tuple[int,int]) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if m.shape[:2] != image_shape:
        m = cv2.resize(m, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def _segment_fastsam_ultralytics(
    image_bgr: np.ndarray,
    weights: str,
    text_prompt: str,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.9,
) -> np.ndarray:
    """Ultralytics FastSAM implementation that matches your working code,
    and returns a mask in the ORIGINAL image coordinate system.

    IMPORTANT:
    - Ultralytics internally letterboxes the image to a square (e.g. 640x640).
      The raw tensor masks are in that resized space.
    - To guide the robot, we must map the mask back to the original image size.
      We do this using `results[0].masks.xy` polygons (original pixel coordinates).
    """
    try:
        from ultralytics import FastSAM  # type: ignore
        from ultralytics.models.fastsam import FastSAMPredictor  # type: ignore
    except Exception as e:
        raise ImportError("ultralytics not installed. pip install ultralytics") from e

    orig_h, orig_w = image_bgr.shape[:2]

    # Ensure weights file exists or let Ultralytics handle download if supported
    _ = FastSAM(weights)

    overrides: Dict[str, Any] = dict(
        conf=float(conf),
        iou=float(iou),
        task="segment",
        mode="predict",
        model=str(weights),
        save=False,
        imgsz=int(imgsz),
        device=str(device),
    )
    predictor = FastSAMPredictor(overrides=overrides)

    # predictor expects a path for widest compatibility
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, image_bgr)
        everything_results = predictor(tmp)
        prompted = predictor.prompt(everything_results, texts=str(text_prompt))
        if not prompted or prompted[0] is None:
            raise RuntimeError("FastSAM prompt returned empty results.")
        r0 = prompted[0]
        if r0.masks is None:
            raise RuntimeError("FastSAM returned no masks.")

        n = len(r0.masks)
        if n == 0:
            raise RuntimeError("FastSAM returned zero masks.")

        # Select the target instance index
        idx = 0
        if getattr(r0, "boxes", None) is not None and len(r0.boxes) == n:
            confs = r0.boxes.conf.cpu().numpy()
            idx = int(np.argmax(confs))
        else:
            # fallback: choose largest area in tensor mask space
            masks_np = r0.masks.data.cpu().numpy()
            areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))

        # --- KEY FIX: build mask in ORIGINAL image coordinates ---
        # Preferred: polygon coordinates already in original pixels.
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        polys = getattr(r0.masks, "xy", None)
        if polys is not None and len(polys) > idx and polys[idx] is not None and len(polys[idx]) >= 3:
            poly = np.asarray(polys[idx], dtype=np.float32)
            poly[:, 0] = np.clip(poly[:, 0], 0, orig_w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, orig_h - 1)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            return mask

        # Fallback: resize raw mask to original size (less accurate if letterbox padding exists)
        masks_np = r0.masks.data.cpu().numpy()
        raw = (masks_np[idx] > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def segment_target_mask(
    image_bgr: np.ndarray,
    backend: SegBackend = "fastsam",
    mask_path: Optional[str] = None,
    fastsam_weights: Optional[str] = None,
    text_prompt: Optional[str] = None,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.9,
) -> np.ndarray:
    """Return binary mask uint8 {0,255} of target object."""
    h, w = image_bgr.shape[:2]
    if mask_path:
        return _load_mask(mask_path, (h, w))
    backend = backend.lower()
    if backend == "color":
        return _segment_color_pink(image_bgr)
    if backend == "fastsam":
        if not fastsam_weights:
            raise ValueError("fastsam_weights is required when backend='fastsam'.")
        if not text_prompt:
            raise ValueError("text_prompt is required for FastSAM. e.g. 'pink building block' or 'cucumber'.")
        return _segment_fastsam_ultralytics(
            image_bgr,
            weights=fastsam_weights,
            text_prompt=text_prompt,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
    raise ValueError(f"Unknown seg backend: {backend}")
