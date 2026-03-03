from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple

def order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)

def homography_from_tag(corners_img: np.ndarray, tag_size_m: float) -> np.ndarray:
    """Compute H_img2tagplane using a single tag (metric because tag_size_m is known).

    Tag plane coordinates (meters):
      origin at tag center.
      x right, y down.

    Plane corner coordinates used:
      TL(-s/2,-s/2), TR(s/2,-s/2), BR(s/2,s/2), BL(-s/2,s/2)
    """
    s = float(tag_size_m)
    half = s / 2.0
    corners_plane = np.array(
        [[-half, -half],
         [ half, -half],
         [ half,  half],
         [-half,  half]],
        dtype=np.float32,
    )
    c_img = order_corners_tl_tr_br_bl(corners_img)
    H, _ = cv2.findHomography(c_img, corners_plane, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed for the reference tag.")
    return H

def pixel_to_plane(H_img2plane: np.ndarray, uv: Tuple[float, float]) -> Tuple[float, float]:
    u, v = float(uv[0]), float(uv[1])
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H_img2plane @ p
    q = q / q[2]
    return float(q[0]), float(q[1])

def plane_vec_from_pixel_vec(H_img2plane: np.ndarray, uv: Tuple[float, float], d_uv: Tuple[float, float]) -> np.ndarray:
    """Map a small pixel direction vector to plane direction by mapping two points and subtracting."""
    u, v = uv
    du, dv = d_uv
    p0 = np.array(pixel_to_plane(H_img2plane, (u, v)), dtype=np.float64)
    p1 = np.array(pixel_to_plane(H_img2plane, (u + du, v + dv)), dtype=np.float64)
    return p1 - p0
