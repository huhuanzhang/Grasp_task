from __future__ import annotations
from typing import Tuple
import numpy as np

def estimate_se2_from_points(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if src.shape[0] < 2 or src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError("src_xy and dst_xy must be (N,2) with N>=2")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    X = src - src_mean
    Y = dst - dst_mean
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    return R, t
