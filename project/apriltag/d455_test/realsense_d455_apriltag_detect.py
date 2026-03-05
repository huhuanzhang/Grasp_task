import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# 初始化 AprilTag 检测器
at_detector = Detector(
    families="tag36h11",
    nthreads=4,               # PC 上可以多线程
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

print("启动 RealSense...")
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 转换为 OpenCV 格式
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测 AprilTag
        tags = at_detector.detect(gray)

        # 绘制结果
        for tag in tags:
            corners = tag.corners.astype(int)
            # 画边框
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i+1) % 4])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # 画中心和ID
            center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{tag.tag_id}", 
                       (center[0]+10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('AprilTag Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()