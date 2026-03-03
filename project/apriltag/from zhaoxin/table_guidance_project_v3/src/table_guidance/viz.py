from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2
from .types import TagDetection

def draw_tags(image_bgr: np.ndarray, tags: List[TagDetection]) -> np.ndarray:
    out = image_bgr.copy()
    for t in tags:
        c = t.corners.astype(int).reshape(4,2)
        cv2.polylines(out, [c], isClosed=True, color=(0,255,0), thickness=2)
        center = tuple(t.center.astype(int))
        cv2.putText(out, f"id={t.tag_id}", (center[0]+5, center[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.circle(out, center, 3, (0,255,0), -1)
    return out

def draw_target(
    image_bgr: np.ndarray,
    pixel_xy: Tuple[float,float],
    axis_uv: Optional[np.ndarray],
    plane_xy: Tuple[float,float],
    yaw_rad: Optional[float],
) -> np.ndarray:
    out = image_bgr.copy()
    u, v = int(pixel_xy[0]), int(pixel_xy[1])
    cv2.circle(out, (u,v), 6, (255,0,0), -1)
    if axis_uv is not None:
        du, dv = float(axis_uv[0]), float(axis_uv[1])
        p2 = (int(u + du*80), int(v + dv*80))
        cv2.arrowedLine(out, (u,v), p2, (255,0,0), 3, tipLength=0.15)

    cv2.putText(out, f"plane(x,y)=({plane_xy[0]:.3f},{plane_xy[1]:.3f}) m",
                (max(10,u-260), max(30,v-20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
    if yaw_rad is not None:
        cv2.putText(out, f"yaw={yaw_rad:.3f} rad",
                    (max(10,u-260), max(60,v+10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
    return out
