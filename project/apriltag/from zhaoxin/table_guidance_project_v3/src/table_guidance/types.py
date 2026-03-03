from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    corners: np.ndarray  # shape (4,2), float32
    center: np.ndarray   # shape (2,)
    decision_margin: Optional[float] = None

@dataclass(frozen=True)
class GuidanceResult:
    ref_tag_id: int
    target_xy_m: Tuple[float, float]
    target_yaw_rad: Optional[float]
    pixel_xy: Tuple[float, float]
