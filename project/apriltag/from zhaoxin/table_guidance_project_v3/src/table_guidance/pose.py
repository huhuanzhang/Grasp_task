from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2

def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component in a binary mask."""
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (labels == idx).astype(np.uint8) * 255
    return out

def mask_centroid_and_axis(mask: np.ndarray) -> Tuple[Tuple[float,float], Optional[np.ndarray]]:
    """Return centroid (u,v) and main axis direction (du,dv) in image pixels."""
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) < 50:
        if len(xs) == 0:
            return (0.0, 0.0), None
        return (float(xs.mean()), float(ys.mean())), None

    cu = float(xs.mean())
    cv = float(ys.mean())

    pts = np.stack([xs - cu, ys - cv], axis=1).astype(np.float64)
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    v = v / (np.linalg.norm(v) + 1e-9)
    return (cu, cv), v
