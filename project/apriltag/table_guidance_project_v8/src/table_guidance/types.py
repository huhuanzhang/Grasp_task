from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    corners: np.ndarray  # (4,2) pixels
    center: np.ndarray   # (2,) pixels
    decision_margin: Optional[float] = None
    pose_R: Optional[np.ndarray] = None  # (3,3) tag->camera
    pose_t: Optional[np.ndarray] = None  # (3,)  tag origin in camera
