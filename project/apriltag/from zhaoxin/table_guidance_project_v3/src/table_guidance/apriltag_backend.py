from __future__ import annotations
from typing import List
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
) -> List[TagDetection]:
    """Detect AprilTags and return their pixel corners/centers.

    Backend: pupil-apriltags.
    Install: pip install pupil-apriltags
    """
    try:
        from pupil_apriltags import Detector  # type: ignore
    except Exception as e:
        raise ImportError(
            "AprilTag backend not found. Install with: pip install pupil-apriltags"
        ) from e

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    det = Detector(
        families=tag_family,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
    )
    results = det.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

    out: List[TagDetection] = []
    for r in results:
        corners = np.array(r.corners, dtype=np.float32).reshape(4, 2)
        center = np.array(r.center, dtype=np.float32).reshape(2,)
        dm = getattr(r, "decision_margin", None)
        out.append(TagDetection(tag_id=int(r.tag_id), corners=corners, center=center, decision_margin=dm))
    return out
