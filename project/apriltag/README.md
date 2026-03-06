# Grasp Task（AprilTag + D455 + 机械臂抓取）

基于 D455 + AprilTag + 分割检测 + 机械臂控制的抓取项目。

---

## 1. 主流程（推荐理解顺序）

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

## 3. 核心函数说明（含作用）

> 详细索引见 `docs/FUNCTION_MAP.md`，这里给“能直接上手”的解释。

### 3.1 FK（正运动学）

- `alohamini_control/teleop_tag_grasp.py`
  - `pin.forwardKinematics(model, data, q)`：根据关节角 `q` 计算各连杆位姿。
- `alohamini_control/FK_transfer.py`
  - `T_inv(T)`：4x4 齐次变换求逆。
  - `make_T(rvec, t)`：由旋转向量 + 平移构造 4x4 变换矩阵。
  - `se3_log(T)`：SE(3) 误差映射到李代数（用于误差度量/控制）。
  - `set_joint(q, joint_name, angle_rad)`：按名字设置关节角。

### 3.2 IK（逆运动学）

- `alohamini_control/grasp.py`
  - `dls(J, e, lam=0.03)`：阻尼最小二乘解 IK 增量（提升奇异位姿稳定性）。
  - `class IKPos5Dof`：5自由度位置/姿态求解器。
- `alohamini_control/d455_detect_and_move.py`
  - `dls(...)`、`IKPos5Dof`：同上，集成在检测到执行的主流程中。

### 3.3 分割与目标提取

- `seg_detection/table_guidance_project_v8/src/table_guidance/segmentation.py`
  - `segment_target_mask(...)`：输入图像与 prompt，输出目标掩码。
  - `_segment_fastsam_ultralytics(...)`：FastSAM 实际推理实现。
  - `_load_mask(...)`：从外部 mask 文件读取掩码。
- `seg_detection/table_guidance_project_v8/src/table_guidance/pose.py`
  - `largest_connected_component(mask)`：保留最大连通域，抑制噪声。
  - `mask_centroid_and_axis(mask)`：计算目标中心和主轴方向。

### 3.4 AprilTag 检测

- `seg_detection/table_guidance_project_v8/src/table_guidance/apriltag_backend.py`
  - `apriltag_detect(...)`：检测 Tag，按需估计 `tag->camera` 位姿。

### 3.5 平面拟合与坐标转换

- `seg_detection/table_guidance_project_v8/src/table_guidance/plane_fusion.py`
  - `fit_plane_svd(points, weights=None)`：SVD 拟合平面。
  - `build_plane_model_from_tags(...)`：用多 Tag 角点拟合桌面平面。
  - `pixel_to_plane_via_ray(u, v, fx, fy, cx, cy, plane)`：像素射线与平面求交。
  - `cam_point_to_plane_xy(p_cam, plane)`：相机点投影到平面 XY。
- `seg_detection/table_guidance_project_v8/src/table_guidance/transforms.py`
  - `cam_to_ref_tag(p_cam, R_ref, t_ref)`：相机系点转到参考 Tag 坐标系。
- `alohamini_control/bridge_tag_to_base.py`
  - `transform_tag_xy_to_base_xy(tag_xy, R, t)`：Tag 平面坐标映射到机器人 base 坐标。

### 3.6 标定相关

- `seg_detection/table_guidance_project_v8/src/table_guidance/calibration.py`
  - `estimate_se2_from_points(src_xy, dst_xy)`：由对应点估计平面 SE2（旋转+平移）。
- 相关脚本：
  - `seg_detection/table_guidance_project_v8/scripts/calibrate_base_from_points.py`
  - `seg_detection/table_guidance_project_v8/scripts/apply_T_base_tag.py`

---

## 4. 运行指令（按使用场景）

> 以下命令默认在仓库根目录 `project/apriltag` 下执行。

### 4.1 相机与 Tag 基础测试

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

### 4.3 视觉变换验证

```bash
cd seg_detection/table_guidance_project_v8
python scripts/validate_transforms.py \
  --image assets/d455_live.jpg \
  --config config/demo.yaml \
  --out outputs
```

### 4.4 机械臂执行（查看参数）

```bash
cd alohamini_control
python d455_detect_and_move.py --help
python d455_detect_and_move_qinterp.py --help
python teleop_tag_grasp.py --help
```

### 4.5 当前推荐一键主流程（qinterp）

```bash
cd /home/hhz/.openclaw/workspace/project/apriltag && /home/hhz/miniconda3/envs/lerobot_alohamini/bin/python d455_detect_and_move_qinterp.py --remote_ip 172.16.0.14 --prompt "green stick"
```

### 4.6 CAL 标定（你当前常用）

标准标定：
```bash
RERUN=OFF python3 seg_detection/realsense_d455_apriltag_cal.py \
  --remote_ip 172.16.0.14 \
  --tag_size 0.04 \
  --tag_id 0 \
  --out_jsonl samples.jsonl
```

允许“无 Tag 也保存关节角”的标定：
```bash
RERUN=OFF python3 seg_detection/realsense_d455_apriltag_cal_save_joints.py \
  --remote_ip 172.16.0.14 \
  --tag_size 0.04 \
  --tag_id 0 \
  --out_jsonl samples_no_tag_ok.jsonl
```

---

## 5. 常见产出

- `overlay.jpg`：检测可视化
- `mask.png`：分割结果
- `result.json`：目标位置、姿态角、预抓取点、平面信息
- `samples*.jsonl`：标定采样数据（关节角 + 观测）

---

## 6. 备注

- `archive/` 目录默认不参与主流程运行。
- 若要做函数级排查，先看：`docs/FUNCTION_MAP.md`。
