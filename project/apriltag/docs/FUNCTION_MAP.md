# FUNCTION_MAP（函数定位速查）

> 目标：让任何人 1 分钟内找到 FK / IK / 分割 / 坐标转换核心函数。

## FK（正运动学）

- `alohamini_control/teleop_tag_grasp.py`
  - 关键调用：`pin.forwardKinematics(model, data, q)`
- `alohamini_control/FK_transfer.py`
  - `T_inv(T)`
  - `make_T(rvec, t)`
  - `se3_log(T)`
  - `set_joint(q, joint_name, angle_rad)`

## IK（逆运动学）

- `alohamini_control/grasp.py`
  - `dls(J, e, lam=0.03)`
  - `class IKPos5Dof`
- `alohamini_control/d455_detect_and_move.py`
  - `dls(J, e, lam=0.03)`
  - `class IKPos5Dof`

## 分割与目标提取

- `seg_detection/table_guidance_project_v8/src/table_guidance/segmentation.py`
  - `segment_target_mask(...)`
  - `_segment_fastsam_ultralytics(...)`
  - `_load_mask(...)`
- `seg_detection/table_guidance_project_v8/src/table_guidance/pose.py`
  - `largest_connected_component(mask)`
  - `mask_centroid_and_axis(mask)`

## AprilTag 检测

- `seg_detection/table_guidance_project_v8/src/table_guidance/apriltag_backend.py`
  - `apriltag_detect(...)`

## 平面拟合 + 坐标转换

- `seg_detection/table_guidance_project_v8/src/table_guidance/plane_fusion.py`
  - `fit_plane_svd(points, weights=None)`
  - `build_plane_model_from_tags(...)`
  - `pixel_to_plane_via_ray(u, v, fx, fy, cx, cy, plane)`
  - `cam_point_to_plane_xy(p_cam, plane)`
- `seg_detection/table_guidance_project_v8/src/table_guidance/transforms.py`
  - `cam_to_ref_tag(p_cam, R_ref, t_ref)`
- `alohamini_control/bridge_tag_to_base.py`
  - `transform_tag_xy_to_base_xy(tag_xy, R, t)`

## 标定与应用

- `seg_detection/table_guidance_project_v8/src/table_guidance/calibration.py`
  - `estimate_se2_from_points(src_xy, dst_xy)`
- `seg_detection/table_guidance_project_v8/scripts/calibrate_base_from_points.py`
- `seg_detection/table_guidance_project_v8/scripts/apply_T_base_tag.py`

## 主入口脚本

- 视觉主入口：`seg_detection/table_guidance_project_v8/scripts/run_demo.py`
- 视觉验证：`seg_detection/table_guidance_project_v8/scripts/validate_transforms.py`
- 机械臂执行（IK）：`alohamini_control/d455_detect_and_move.py`
- 机械臂执行（插值）：`alohamini_control/d455_detect_and_move_qinterp.py`
