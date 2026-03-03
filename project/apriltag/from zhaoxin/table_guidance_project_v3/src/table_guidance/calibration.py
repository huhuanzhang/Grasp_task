from __future__ import annotations
from typing import Tuple
import numpy as np

def estimate_se2_from_points(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate 2D rigid transform (SE(2)) that maps src -> dst.

    src_xy: (N,2) points in reference tag plane (meters)
    dst_xy: (N,2) points in robot base plane (meters, or mm if you keep consistent)

    Returns:
      R: (2,2) rotation
      t: (2,) translation

    Requirement: N >= 2 (N>=3 recommended).
    """
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if src.shape[0] < 2:
        raise ValueError("Need at least 2 point correspondences (3+ recommended).")
    if src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError("src_xy and dst_xy must be (N,2) with same shape.")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # enforce det=+1
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    return R, t
