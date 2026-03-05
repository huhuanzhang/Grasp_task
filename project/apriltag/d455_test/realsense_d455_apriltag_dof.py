import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# ========== 配置 ==========
TAG_SIZE = 0.04  # ← 改成你的标签实际边长（米）！
TAG_ID_TARGET = 0  # ← 你要抓的那个标签ID

# 初始化检测器
at_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 同时开启深度流（后续需要）
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

# 对齐深度和彩色
align = rs.align(rs.stream.color)

# 获取相机内参
profile = pipeline.get_active_profile()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_params = [intr.fx, intr.fy, intr.ppx, intr.ppy]
print(f"相机内参: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")

# 深度缩放（米/单位）
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测 + 位姿估计
        tags = at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE
        )

        for tag in tags:
            corners = tag.corners.astype(int)
            center = (int(tag.center[0]), int(tag.center[1]))
            
            # 画边框和ID
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i+1) % 4])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # 获取深度距离（验证用）
            depth = depth_frame.get_distance(center[0], center[1])
            
            # 位姿估计结果
            dist = np.linalg.norm(tag.pose_t)
            x, y, z = tag.pose_t.flatten()
            
            # 高亮目标标签
            color = (0, 255, 255) if tag.tag_id == TAG_ID_TARGET else (255, 255, 0)
            
            text1 = f"ID:{tag.tag_id} {dist*100:.1f}cm"
            text2 = f"XYZ:[{x*100:.1f}, {y*100:.1f}, {z*100:.1f}]cm"
            cv2.putText(frame, text1, (center[0]+10, center[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, text2, (center[0]+10, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 打印目标标签信息
            if tag.tag_id == TAG_ID_TARGET:
                print(f"\r目标标签 {tag.tag_id}: 距离={dist:.3f}m, 相机坐标=({x:.3f}, {y:.3f}, {z:.3f})", end='')

        cv2.imshow('AprilTag Grasp Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()