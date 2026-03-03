from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import cv2
from .types import TagDetection

def apriltag_detect(
    image_bgr: np.ndarray,
    tag_family: str = "tag36h11",
    nthreads: int = 2,
    quad_decimate: float = 1.0,
    quad_sigma: float = 0.0,
    refine_edges: int = 1,
    decode_sharpening: float = 0.25,
    estimate_tag_pose: bool = False,
    camera_params: Optional[Tuple[float, float, float, float]] = None,  # fx,fy,cx,cy
    tag_size: Optional[float] = None,
) -> List[TagDetection]:
    """Detect AprilTags; optional pose (tag->camera): p_cam = R @ p_tag + t."""
    try:
        from pupil_apriltags import Detector  # type: ignore
    except Exception as e:
        raise ImportError("Install pupil-apriltags: pip install pupil-apriltags") from e

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    det = Detector(
        families=tag_family,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
    )

    if estimate_tag_pose:
        if camera_params is None or tag_size is None:
            raise ValueError("camera_params and tag_size required for pose estimation")
        results = det.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    else:
        results = det.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

    out: List[TagDetection] = []
    for r in results:
        corners = np.array(r.corners, dtype=np.float32).reshape(4, 2)
        center = np.array(r.center, dtype=np.float32).reshape(2,)
        dm = getattr(r, "decision_margin", None)

        R = None
        t = None
        if estimate_tag_pose:
            if hasattr(r, "pose_R") and hasattr(r, "pose_t"):
                R = np.array(r.pose_R, dtype=np.float64).reshape(3, 3)
                t = np.array(r.pose_t, dtype=np.float64).reshape(3,)
            elif hasattr(r, "R") and hasattr(r, "t"):
                R = np.array(r.R, dtype=np.float64).reshape(3, 3)
                t = np.array(r.t, dtype=np.float64).reshape(3,)

        out.append(TagDetection(int(r.tag_id), corners, center, dm, R, t))
    return out
