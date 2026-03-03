import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

TAG_SIZE = 0.04  # 你的标签边长（米）

# 初始化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# 获取实际内参
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_params = [intr.fx, intr.fy, intr.ppx, intr.ppy]
print(f"分辨率: {intr.width}x{intr.height}")
print(f"内参: fx={intr.fx:.2f}, fy={intr.fy:.2f}")

at_detector = Detector(families="tag36h11", nthreads=4)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        
        if not color or not depth:
            continue

        frame = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        tags = at_detector.detect(gray, estimate_tag_pose=True, 
                                  camera_params=camera_params, tag_size=TAG_SIZE)

        for tag in tags:
            # 画框
            corners = tag.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i+1)%4]), (0,255,0), 2)
            
            center = (int(tag.center[0]), int(tag.center[1]))
            
            # 两种距离对比
            dist_pose = np.linalg.norm(tag.pose_t)  # AprilTag 估计的距离
            dist_depth = depth.get_distance(center[0], center[1])  # RealSense 深度
            
            x, y, z = tag.pose_t.flatten()
            
            text = f"ID:{tag.tag_id} Pose:{dist_pose*100:.1f}cm Depth:{dist_depth*100:.1f}cm"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            print(f"\rPose:{dist_pose:.3f}m | Depth:{dist_depth:.3f}m | Diff:{abs(dist_pose-dist_depth)*100:.1f}cm", end='')

        cv2.imshow('Verify Distance', frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()