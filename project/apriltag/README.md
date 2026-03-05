# Grasp Task

基于 D455 + AprilTag + 分割检测 + 机械臂控制的抓取项目。

## 1. 项目流程

1. 相机采图（D455）
2. AprilTag 检测与位姿估计
3. 目标分割并提取目标像素位置
4. 像素坐标转换到桌面平面坐标 / 机器人基坐标
5. 机械臂执行（IK 或关节插值）

---

## 2. 目录结构

- `d455_test/`：相机与 Tag 基础测试
- `seg_detection/`：分割、检测、平面拟合、坐标计算
- `alohamini_control/`：机械臂控制（FK/IK/执行）
- `docs/`：补充文档
- `archive/`：历史数据与旧文件

---

## 3. 核心函数索引

### FK（正运动学）
- `alohamini_control/teleop_tag_grasp.py`
  - `pin.forwardKinematics(...)`
- `alohamini_control/FK_transfer.py`
  - `T_inv(...)`
  - `make_T(...)`
  - `se3_log(...)`
  - `set_joint(...)`

### IK（逆运动学）
- `alohamini_control/grasp.py`
  - `dls(...)`
  - `class IKPos5Dof`
- `alohamini_control/d455_detect_and_move.py`
  - `dls(...)`
  - `class IKPos5Dof`

### 分割
- `seg_detection/table_guidance_project_v8/src/table_guidance/segmentation.py`
  - `segment_target_mask(...)`
  - `_segment_fastsam_ultralytics(...)`

### 坐标转换
- `seg_detection/table_guidance_project_v8/src/table_guidance/plane_fusion.py`
  - `build_plane_model_from_tags(...)`
  - `pixel_to_plane_via_ray(...)`
  - `cam_point_to_plane_xy(...)`
- `seg_detection/table_guidance_project_v8/src/table_guidance/transforms.py`
  - `cam_to_ref_tag(...)`
- `alohamini_control/bridge_tag_to_base.py`
  - `transform_tag_xy_to_base_xy(...)`

### AprilTag 检测
- `seg_detection/table_guidance_project_v8/src/table_guidance/apriltag_backend.py`
  - `apriltag_detect(...)`

---

## 4. 快速使用

### 4.1 相机与Tag测试
```bash
cd d455_test
python d455_cam.py
python d455_intrinsics.py
python realsense_d455_apriltag_detect.py
```

### 4.2 视觉主流程（分割 + 坐标输出）
```bash
cd seg_detection/table_guidance_project_v8
python scripts/run_demo.py \
  --image assets/d455_live.jpg \
  --config config/demo.yaml \
  --fastsam-weights FastSAM-x.pt \
  --device cpu \
  --out outputs
```

输出文件：
- `outputs/overlay.jpg`
- `outputs/mask.png`
- `outputs/result.json`

### 4.3 变换验证
```bash
cd seg_detection/table_guidance_project_v8
python scripts/validate_transforms.py \
  --image assets/d455_live.jpg \
  --config config/demo.yaml \
  --out outputs
```

### 4.4 机械臂执行
```bash
cd alohamini_control
python d455_detect_and_move.py --help
python d455_detect_and_move_qinterp.py --help
python teleop_tag_grasp.py --help
```

---

## 5. 常见产出

- `overlay.jpg`：检测可视化
- `mask.png`：分割结果
- `result.json`：目标位置、姿态角、预抓取点、平面信息

---

## 6. 备注

- 详细函数清单见：`docs/FUNCTION_MAP.md`
- `archive/` 目录默认不参与主流程运行
