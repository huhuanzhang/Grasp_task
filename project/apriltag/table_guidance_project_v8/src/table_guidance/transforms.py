from __future__ import annotations
import numpy as np

def cam_to_ref_tag(p_cam: np.ndarray, R_ref: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    R = np.asarray(R_ref, dtype=np.float64).reshape(3,3)
    t = np.asarray(t_ref, dtype=np.float64).reshape(3,)
    return R.T @ (np.asarray(p_cam, dtype=np.float64).reshape(3,) - t)
