import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# =============== Config ===============
SAVE_DIR = "./saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

REF_TAG_ID = 0
TAG_SIZE_M = 0.06  # 参考tag黑色方块边长（米），按你上司方法用于建立米制平面

# AprilTag detector
at_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)


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
    """H_img2plane: image pixel -> reference tag plane (meters).

    Tag plane convention (same as supervisor):
      origin at tag center, x right, y down.
    """
    s = float(tag_size_m)
    half = s / 2.0
    corners_plane = np.array(
        [[-half, -half],
         [half, -half],
         [half, half],
         [-half, half]],
        dtype=np.float32,
    )
    c_img = order_corners_tl_tr_br_bl(corners_img)
    H, _ = cv2.findHomography(c_img, corners_plane, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed for the reference tag.")
    return H


def pixel_to_plane(H_img2plane: np.ndarray, uv) -> tuple[float, float]:
    u, v = float(uv[0]), float(uv[1])
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H_img2plane @ p
    q = q / q[2]
    return float(q[0]), float(q[1])


def plane_to_pixel(H_plane2img: np.ndarray, xy) -> tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H_plane2img @ p
    q = q / q[2]
    return int(round(q[0])), int(round(q[1]))


def draw_tag_boxes(bgr: np.ndarray, tags) -> np.ndarray:
    out = bgr.copy()
    for tag in tags:
        corners = tag.corners.astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(out, pt1, pt2, (0, 255, 0), 2)

        center = tuple(tag.center.astype(int))
        cv2.circle(out, center, 4, (0, 255, 255), -1)
        cv2.putText(
            out,
            f"id={tag.tag_id}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
    return out


# RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("启动 RealSense...")
pipeline.start(config)

print("\nControls:")
print("  - Press 's' to save the annotated image")
print("  - Press 'r' to save the raw image")
print("  - Press 'q' to quit\n")
print(f"Reference tag id = {REF_TAG_ID}, tag_size_m = {TAG_SIZE_M}")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = at_detector.detect(gray)
        vis = draw_tag_boxes(frame, tags)

        ref = next((t for t in tags if int(t.tag_id) == REF_TAG_ID), None)
        if ref is not None:
            try:
                H_img2plane = homography_from_tag(ref.corners, TAG_SIZE_M)
                H_plane2img = np.linalg.inv(H_img2plane)

                # 画出参考坐标轴（和上司约定一致：+x右，+y下）
                origin_px = plane_to_pixel(H_plane2img, (0.0, 0.0))
                x_axis_px = plane_to_pixel(H_plane2img, (0.03, 0.0))
                y_axis_px = plane_to_pixel(H_plane2img, (0.0, 0.03))
                cv2.arrowedLine(vis, origin_px, x_axis_px, (0, 0, 255), 2, tipLength=0.2)  # x: red
                cv2.arrowedLine(vis, origin_px, y_axis_px, (255, 0, 0), 2, tipLength=0.2)  # y: blue
                cv2.putText(vis, "O", (origin_px[0] + 4, origin_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis, "x", (x_axis_px[0] + 4, x_axis_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(vis, "y", (y_axis_px[0] + 4, y_axis_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 显示各tag中心在参考平面中的坐标
                y0 = 60
                cv2.putText(
                    vis,
                    f"Ref tag: id={REF_TAG_ID} origin=(0,0)",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                y0 += 28

                for t in sorted(tags, key=lambda z: int(z.tag_id)):
                    x_m, y_m = pixel_to_plane(H_img2plane, t.center)
                    c = tuple(t.center.astype(int))
                    txt = f"id={int(t.tag_id)}: ({x_m:+.3f}, {y_m:+.3f}) m"

                    cv2.putText(
                        vis,
                        txt,
                        (c[0] + 8, c[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        vis,
                        txt,
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 255, 0),
                        2,
                    )
                    y0 += 24

            except Exception as e:
                cv2.putText(
                    vis,
                    f"Homography error: {e}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                vis,
                f"Reference tag id={REF_TAG_ID} not found",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            vis,
            f"Tags: {len(tags)}   (s=save, r=raw, q=quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("AprilTag Detection (ref-tag plane)", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(SAVE_DIR, f"tag_{ts}.png")
            cv2.imwrite(fname, vis)
            print(f"[SAVED] {fname}")

        elif key == ord("r"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(SAVE_DIR, f"raw_{ts}.png")
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
