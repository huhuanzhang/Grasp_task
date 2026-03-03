import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# =============== Config ===============
SAVE_DIR = "./saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

TAG_SIZE_M = 0.04   # 你的tag黑色方块边长（米）
MIN_DM = 20.0       # decision_margin 过滤阈值

# 固定桌面坐标系（按你的实际摆放改这里）
# 格式: tag_id: (x_m, y_m, yaw_rad)
# 下面默认是 20cm 方形布局示例：
# 0(原点) ---- 2(+x方向)
#   |           |
# 4(+y方向) -- 3
TAG_LAYOUT = {
    0: (0.00, 0.00, 0.0),
    2: (0.20, 0.00, 0.0),
    4: (0.00, 0.20, 0.0),
    3: (0.20, 0.20, 0.0),
}

# AprilTag detector
at_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)


def order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def tag_local_corners(tag_size_m: float) -> np.ndarray:
    h = float(tag_size_m) / 2.0
    # TL, TR, BR, BL
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)


def world_corners_from_layout(tag_id: int, tag_size_m: float) -> np.ndarray:
    x, y, yaw = TAG_LAYOUT[tag_id]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    local = tag_local_corners(tag_size_m)
    world = (R @ local.T).T + np.array([x, y], dtype=np.float64)
    return world


def pixel_to_world(H_img2world: np.ndarray, uv) -> tuple[float, float]:
    p = np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64)
    q = H_img2world @ p
    q = q / q[2]
    return float(q[0]), float(q[1])


def world_to_pixel(H_world2img: np.ndarray, xy) -> tuple[int, int]:
    p = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
    q = H_world2img @ p
    q = q / q[2]
    return int(round(q[0])), int(round(q[1]))


def draw_tag_boxes(bgr: np.ndarray, tags) -> np.ndarray:
    out = bgr.copy()
    for tag in tags:
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(out, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
        c = tuple(tag.center.astype(int))
        dm = float(getattr(tag, "decision_margin", 0.0))
        cv2.circle(out, c, 4, (0, 255, 255), -1)
        cv2.putText(out, f"id={tag.tag_id} dm={dm:.1f}", (c[0] + 6, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
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
print(f"TAG_SIZE_M={TAG_SIZE_M}, MIN_DM={MIN_DM}")
print(f"TAG_LAYOUT IDs={sorted(TAG_LAYOUT.keys())}")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags_all = at_detector.detect(gray)
        tags = [t for t in tags_all if float(getattr(t, "decision_margin", 0.0)) >= MIN_DM]
        vis = draw_tag_boxes(frame, tags)

        # 多tag硬约束融合：图像角点 -> 已知桌面角点
        src_img_list, dst_world_list, used_ids = [], [], []
        for t in tags:
            tid = int(t.tag_id)
            if tid not in TAG_LAYOUT:
                continue
            img_c = order_corners_tl_tr_br_bl(t.corners).astype(np.float64)
            world_c = world_corners_from_layout(tid, TAG_SIZE_M)
            src_img_list.append(img_c)
            dst_world_list.append(world_c)
            used_ids.append(tid)

        H_img2world, inliers = None, 0
        if len(src_img_list) >= 1:
            src = np.concatenate(src_img_list, axis=0).astype(np.float32)
            dst = np.concatenate(dst_world_list, axis=0).astype(np.float32)
            H_img2world, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.01)
            if mask is not None:
                inliers = int(mask.ravel().sum())

        y0 = 58
        cv2.putText(vis, f"Tags(raw/used): {len(tags_all)}/{len(tags)}  layout_used={sorted(set(used_ids))}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        y0 += 24

        if H_img2world is not None:
            H_world2img = np.linalg.inv(H_img2world)

            # 画桌面世界坐标轴（原点在tag0中心）
            origin_px = world_to_pixel(H_world2img, (0.0, 0.0))
            x_axis_px = world_to_pixel(H_world2img, (0.05, 0.0))
            y_axis_px = world_to_pixel(H_world2img, (0.0, 0.05))
            cv2.arrowedLine(vis, origin_px, x_axis_px, (0, 0, 255), 2, tipLength=0.2)   # +x 红
            cv2.arrowedLine(vis, origin_px, y_axis_px, (255, 0, 0), 2, tipLength=0.2)   # +y 蓝
            cv2.putText(vis, "O", (origin_px[0] + 4, origin_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, "x", (x_axis_px[0] + 4, x_axis_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, "y", (y_axis_px[0] + 4, y_axis_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(vis, f"FUSED H ok, inliers={inliers}/{len(used_ids)*4}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y0 += 24

            # 显示每个tag中心坐标 + 与期望布局误差
            for t in sorted(tags, key=lambda z: int(z.tag_id)):
                tid = int(t.tag_id)
                x_m, y_m = pixel_to_world(H_img2world, t.center)
                c = tuple(t.center.astype(int))

                if tid in TAG_LAYOUT:
                    gx, gy, _ = TAG_LAYOUT[tid]
                    err_cm = 100.0 * np.hypot(x_m - gx, y_m - gy)
                    txt = f"id={tid}: ({x_m:+.3f},{y_m:+.3f})m  err={err_cm:.1f}cm"
                else:
                    txt = f"id={tid}: ({x_m:+.3f},{y_m:+.3f})m"

                cv2.putText(vis, txt, (c[0] + 8, c[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(vis, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 0), 2)
                y0 += 22
        else:
            cv2.putText(vis, "Homography unavailable (need >=1 known tag)",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y0 += 24

        cv2.putText(vis, "(s=save, r=raw, q=quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("AprilTag Multi-Tag (fixed layout)", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(SAVE_DIR, f"tag_fused_{ts}.png")
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
