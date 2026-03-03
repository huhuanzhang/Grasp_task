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
TAG_SIZE_M = 0.06  # tag黑色方块边长（米）
MIN_DM = 20.0      # decision_margin 过滤阈值

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
    s = float(tag_size_m)
    h = s / 2.0
    # TL, TR, BR, BL in tag-local plane
    return np.array(
        [[-h, -h], [h, -h], [h, h], [-h, h]],
        dtype=np.float64,
    )


def homography_from_tag(corners_img: np.ndarray, tag_size_m: float) -> np.ndarray:
    c_img = order_corners_tl_tr_br_bl(corners_img).astype(np.float64)
    c_tag = tag_local_corners(tag_size_m)
    H, _ = cv2.findHomography(c_img, c_tag.astype(np.float32), method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed")
    return H


def pixel_to_plane(H_img2plane: np.ndarray, uv) -> tuple[float, float]:
    p = np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64)
    q = H_img2plane @ p
    q = q / q[2]
    return float(q[0]), float(q[1])


def plane_to_pixel(H_plane2img: np.ndarray, xy) -> tuple[int, int]:
    p = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
    q = H_plane2img @ p
    q = q / q[2]
    return int(round(q[0])), int(round(q[1]))


def estimate_se2(src_xy: np.ndarray, dst_xy: np.ndarray):
    """Estimate dst = R @ src + t (2D rigid transform)."""
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    c1 = src.mean(axis=0)
    c2 = dst.mean(axis=0)
    X = src - c1
    Y = dst - c2
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c2 - R @ c1
    return R, t


def apply_se2(points_xy: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points_xy.T).T + t.reshape(1, 2)


def draw_tag_boxes(bgr: np.ndarray, tags) -> np.ndarray:
    out = bgr.copy()
    for tag in tags:
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(out, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
        c = tuple(tag.center.astype(int))
        cv2.circle(out, c, 4, (0, 255, 255), -1)
        dm = float(getattr(tag, "decision_margin", 0.0))
        cv2.putText(out, f"id={tag.tag_id} dm={dm:.1f}", (c[0] + 6, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return out


# --- Runtime cache: T_ref_from_tag[id] = (R, t), map tag-local -> ref(id=0) plane ---
T_ref_from_tag: dict[int, tuple[np.ndarray, np.ndarray]] = {
    REF_TAG_ID: (np.eye(2), np.zeros(2, dtype=np.float64))
}

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
print(f"Reference tag id = {REF_TAG_ID}, tag_size_m = {TAG_SIZE_M}, min_dm = {MIN_DM}")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags_all = at_detector.detect(gray)

        # 质量过滤
        tags = [t for t in tags_all if float(getattr(t, "decision_margin", 0.0)) >= MIN_DM]
        vis = draw_tag_boxes(frame, tags)

        # Step 1) 如果本帧看到了 ref，则更新其它tag->ref 的 SE2
        ref = next((t for t in tags if int(t.tag_id) == REF_TAG_ID), None)
        if ref is not None:
            H_img2ref = homography_from_tag(ref.corners, TAG_SIZE_M)
            local = tag_local_corners(TAG_SIZE_M)
            for t in tags:
                tid = int(t.tag_id)
                if tid == REF_TAG_ID:
                    continue
                # t 的4角：图像 -> ref平面
                img_c = order_corners_tl_tr_br_bl(t.corners).astype(np.float64)
                dst_ref = np.array([pixel_to_plane(H_img2ref, p) for p in img_c], dtype=np.float64)
                # t 本地角点 local 对应到 ref
                R, tt = estimate_se2(local, dst_ref)
                T_ref_from_tag[tid] = (R, tt)

        # Step 2) 多tag融合：拼接所有“图像角点 -> ref平面角点”对应，RANSAC求H
        src_img = []
        dst_ref = []
        used_ids = []
        local = tag_local_corners(TAG_SIZE_M)
        for t in tags:
            tid = int(t.tag_id)
            if tid not in T_ref_from_tag:
                continue  # 尚未学到该tag相对ref的变换
            R, tt = T_ref_from_tag[tid]
            ref_corners = apply_se2(local, R, tt)
            img_corners = order_corners_tl_tr_br_bl(t.corners).astype(np.float64)
            src_img.append(img_corners)
            dst_ref.append(ref_corners)
            used_ids.append(tid)

        H_img2ref_fused = None
        inlier_count = 0
        if len(src_img) > 0:
            src = np.concatenate(src_img, axis=0).astype(np.float32)
            dst = np.concatenate(dst_ref, axis=0).astype(np.float32)
            H_img2ref_fused, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.005)
            if mask is not None:
                inlier_count = int(mask.ravel().sum())

        # Step 3) 显示坐标
        y0 = 58
        cv2.putText(vis, f"Tags(raw/used): {len(tags_all)}/{len(tags)}  used={sorted(set(used_ids))}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        y0 += 24
        cv2.putText(vis, f"Learned transforms: {sorted(T_ref_from_tag.keys())}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        y0 += 24

        if H_img2ref_fused is not None:
            H_ref2img = np.linalg.inv(H_img2ref_fused)
            origin_px = plane_to_pixel(H_ref2img, (0.0, 0.0))
            x_axis_px = plane_to_pixel(H_ref2img, (0.03, 0.0))
            y_axis_px = plane_to_pixel(H_ref2img, (0.0, 0.03))
            cv2.arrowedLine(vis, origin_px, x_axis_px, (0, 0, 255), 2, tipLength=0.2)
            cv2.arrowedLine(vis, origin_px, y_axis_px, (255, 0, 0), 2, tipLength=0.2)
            cv2.putText(vis, "O", (origin_px[0] + 4, origin_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, "x", (x_axis_px[0] + 4, x_axis_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, "y", (y_axis_px[0] + 4, y_axis_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(vis, f"FUSED H ok, inliers={inlier_count}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y0 += 24

            for t in sorted(tags, key=lambda z: int(z.tag_id)):
                x_m, y_m = pixel_to_plane(H_img2ref_fused, t.center)
                c = tuple(t.center.astype(int))
                txt = f"id={int(t.tag_id)}: ({x_m:+.3f}, {y_m:+.3f}) m"
                cv2.putText(vis, txt, (c[0] + 8, c[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(vis, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)
                y0 += 22
        else:
            cv2.putText(vis, "FUSED H unavailable (need learned tags / enough correspondences)",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(vis, "(s=save, r=raw, q=quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("AprilTag Multi-Tag Fusion (ref id=0)", vis)
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
