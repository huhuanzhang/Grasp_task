from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
import os
import tempfile

def _load_mask(mask_path: str, image_shape: Tuple[int,int]) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.shape[:2] != image_shape:
        m = cv2.resize(m, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def _segment_fastsam_ultralytics(image_bgr: np.ndarray, weights: str, text_prompt: str,
                                device: str="cpu", imgsz: int=640, conf: float=0.25, iou: float=0.9) -> np.ndarray:
    """Return mask in ORIGINAL image pixels using masks.xy polygons."""
    try:
        from ultralytics import FastSAM  # type: ignore
        from ultralytics.models.fastsam import FastSAMPredictor  # type: ignore
    except Exception as e:
        raise ImportError("pip install ultralytics") from e

    orig_h, orig_w = image_bgr.shape[:2]
    _ = FastSAM(weights)

    overrides: Dict[str, Any] = dict(conf=float(conf), iou=float(iou), task="segment", mode="predict",
                                    model=str(weights), save=False, imgsz=int(imgsz), device=str(device))
    predictor = FastSAMPredictor(overrides=overrides)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, image_bgr)
        everything = predictor(tmp)
        prompted = predictor.prompt(everything, texts=str(text_prompt))
        r0 = prompted[0]
        if r0.masks is None:
            raise RuntimeError("no masks")
        n = len(r0.masks)
        if n == 0:
            raise RuntimeError("zero masks")

        idx = 0
        if getattr(r0, "boxes", None) is not None and len(r0.boxes) == n:
            idx = int(np.argmax(r0.boxes.conf.cpu().numpy()))
        else:
            mnp = r0.masks.data.cpu().numpy()
            idx = int(np.argmax(mnp.reshape(mnp.shape[0], -1).sum(axis=1)))

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        polys = getattr(r0.masks, "xy", None)
        if polys is not None and len(polys) > idx and polys[idx] is not None and len(polys[idx]) >= 3:
            poly = np.asarray(polys[idx], dtype=np.float32)
            poly[:, 0] = np.clip(poly[:, 0], 0, orig_w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, orig_h - 1)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            return mask

        raw = (r0.masks.data.cpu().numpy()[idx] > 0.5).astype(np.uint8) * 255
        return cv2.resize(raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

def segment_target_mask(image_bgr: np.ndarray, mask_path: Optional[str]=None,
                        fastsam_weights: Optional[str]=None, text_prompt: Optional[str]=None,
                        device: str="cpu", imgsz: int=640, conf: float=0.25, iou: float=0.9) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mask_path:
        return _load_mask(mask_path, (h, w))
    if not fastsam_weights or not text_prompt:
        raise ValueError("fastsam_weights and text_prompt required")
    return _segment_fastsam_ultralytics(image_bgr, fastsam_weights, text_prompt, device, imgsz, conf, iou)
