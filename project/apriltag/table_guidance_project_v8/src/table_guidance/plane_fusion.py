from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class PlaneModel:
    # plane in camera frame: n·p + d = 0, with ||n||=1
    n: np.ndarray  # (3,)
    d: float
    origin_cam: np.ndarray  # (3,) plane-frame origin in camera
    x_hat: np.ndarray  # (3,)
    y_hat: np.ndarray  # (3,)

def fit_plane_svd(points: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < 3:
        raise ValueError("Need >=3 points to fit plane.")
    if weights is None:
        centroid = P.mean(axis=0)
        Q = P - centroid
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        n = vh[-1, :]
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        w = np.maximum(w, 1e-12)
        centroid = (P * w).sum(axis=0) / float(w.sum())
        Q = (P - centroid) * np.sqrt(w)
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        n = vh[-1, :]

    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ centroid)
    return n, d

def project_point_to_plane(p: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(3,)
    n = np.asarray(n, dtype=np.float64).reshape(3,)
    dist = float(n @ p + d)
    return p - dist * n

def build_plane_model_from_tags(
    tags: List,
    ref_tag_id: int,
    tag_size_m: float,
    fx: float, fy: float, cx: float, cy: float,  # unused here but kept for future
    use_tag_corners: bool = True,
    weight_by_decision_margin: bool = True,
) -> PlaneModel:
    ref = next((t for t in tags if t.tag_id == ref_tag_id), None)
    if ref is None or ref.pose_R is None or ref.pose_t is None:
        raise RuntimeError("Reference tag pose missing for plane model.")

    half = float(tag_size_m) / 2.0
    corners_tag = np.array([[-half, -half, 0.0],
                            [ half, -half, 0.0],
                            [ half,  half, 0.0],
                            [-half,  half, 0.0]], dtype=np.float64)

    pts = []
    wts = []
    for t in tags:
        if t.pose_R is None or t.pose_t is None:
            continue
        R = np.asarray(t.pose_R, dtype=np.float64).reshape(3,3)
        tt = np.asarray(t.pose_t, dtype=np.float64).reshape(3,)
        dm = float(t.decision_margin) if (weight_by_decision_margin and t.decision_margin is not None) else 1.0
        if use_tag_corners:
            pc = (R @ corners_tag.T).T + tt.reshape(1,3)
            pts.append(pc)
            wts.append(np.full((pc.shape[0],), dm, dtype=np.float64))
        else:
            pts.append(tt.reshape(1,3))
            wts.append(np.array([dm], dtype=np.float64))

    P = np.concatenate(pts, axis=0)
    W = np.concatenate(wts, axis=0) if weight_by_decision_margin else None

    n, d = fit_plane_svd(P, W)

    # orient n to match ref tag +Z (stability)
    ref_z = (np.asarray(ref.pose_R, dtype=np.float64).reshape(3,3) @ np.array([0.0,0.0,1.0])).reshape(3,)
    if float(n @ ref_z) < 0:
        n = -n
        d = -d

    origin = project_point_to_plane(np.asarray(ref.pose_t).reshape(3,), n, d)

    ref_x = (np.asarray(ref.pose_R, dtype=np.float64).reshape(3,3) @ np.array([1.0,0.0,0.0])).reshape(3,)
    x_proj = ref_x - float(ref_x @ n) * n
    x_hat = x_proj / (np.linalg.norm(x_proj) + 1e-12)
    y_hat = np.cross(n, x_hat)
    y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-12)

    return PlaneModel(n=n, d=float(d), origin_cam=origin, x_hat=x_hat, y_hat=y_hat)

def cam_point_to_plane_xy(p_cam: np.ndarray, plane: PlaneModel) -> Tuple[float, float]:
    p = np.asarray(p_cam, dtype=np.float64).reshape(3,) - plane.origin_cam.reshape(3,)
    x = float(p @ plane.x_hat)
    y = float(p @ plane.y_hat)
    return x, y

def pixel_to_plane_via_ray(u: float, v: float, fx: float, fy: float, cx: float, cy: float, plane: PlaneModel) -> Tuple[np.ndarray, Tuple[float,float]]:
    d_ray = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=np.float64)
    d_ray = d_ray / (np.linalg.norm(d_ray) + 1e-12)
    denom = float(plane.n @ d_ray)
    if abs(denom) < 1e-9:
        raise RuntimeError("Ray parallel to plane.")
    s = -plane.d / denom
    p_cam = s * d_ray
    xy = cam_point_to_plane_xy(p_cam, plane)
    return p_cam, xy
