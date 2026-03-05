# Grasp Task（管理者快速阅读版）

这是一个“相机看到目标 → 算出目标坐标 → 机械臂去抓”的项目。

一句话流程：
1. D455 相机采图
2. AprilTag + 分割找到目标
3. 把像素点换算成桌面坐标 / 机械臂坐标
4. 用 FK/IK 或插值控制机械臂执行

---

## 一、目录怎么找（最重要）

- `d455_test/`：相机与 AprilTag 基础测试（先确认相机、内参、识别都正常）
- `seg_detection/`：分割检测与坐标计算主流程
- `alohamini_control/`：AlohaMini 机械臂控制（含 FK / IK）
- `docs/`：文档说明
- `archive/`：历史数据与旧文件（不参与主流程）

---

## 二、建议演示路径（最稳）

### Step 1：先测相机
看 `d455_test/`：
- `d455_cam.py`（相机画面是否正常）
- `d455_intrinsics.py`（内参读取）
- `realsense_d455_apriltag_detect.py`（Tag 是否能识别）

### Step 2：跑视觉主流程
看 `seg_detection/table_guidance_project_v8/`：
- `scripts/run_demo.py`：主入口
- 输出：`overlay.jpg`、`mask.png`、`result.json`

### Step 3：做机械臂动作
看 `alohamini_control/`：
- `d455_detect_and_move.py`（IK方式）
- `d455_detect_and_move_qinterp.py`（关节插值方式）
- `teleop_tag_grasp.py`（遥操作 + FK仪表盘）

---

## 三、你上司关心的函数在哪（FK / IK / 分割 / 坐标转换）

> 下面是“直接能定位代码”的函数索引。

### 1) FK（正运动学）
- `alohamini_control/teleop_tag_grasp.py`
  - 使用 `pin.forwardKinematics(...)` 做当前夹爪位姿计算
- `alohamini_control/FK_transfer.py`
  - `set_joint(...)`
  - `residual(...)`（配合位姿误差优化）
  - `T_inv(...)`, `make_T(...)`, `se3_log(...)`（SE3工具）

### 2) IK（逆运动学）
- `alohamini_control/grasp.py`
  - `class IKPos5Dof`
  - `dls(J, e, lam=0.03)`（阻尼最小二乘）
- `alohamini_control/d455_detect_and_move.py`
  - `class IKPos5Dof`
  - `dls(...)`

### 3) 分割（Segmentation）
- `seg_detection/table_guidance_project_v8/src/table_guidance/segmentation.py`
  - `segment_target_mask(...)`（分割总入口）
  - `_segment_fastsam_ultralytics(...)`（FastSAM）
  - `_load_mask(...)`（读取外部mask）

### 4) 坐标转换（关键）
- `seg_detection/table_guidance_project_v8/src/table_guidance/plane_fusion.py`
  - `pixel_to_plane_via_ray(...)`（像素 -> 桌面平面点）
  - `cam_point_to_plane_xy(...)`
  - `build_plane_model_from_tags(...)`（多Tag拟合平面）
- `seg_detection/table_guidance_project_v8/src/table_guidance/transforms.py`
  - `cam_to_ref_tag(...)`（相机系 -> 参考Tag系）
- `alohamini_control/bridge_tag_to_base.py`
  - `transform_tag_xy_to_base_xy(...)`（Tag平面 -> 机械臂base平面）

### 5) AprilTag检测
- `seg_detection/table_guidance_project_v8/src/table_guidance/apriltag_backend.py`
  - `apriltag_detect(...)`

---

## 四、产出文件（汇报时常用）

视觉主流程输出（在 `run_demo.py` 后）：
- `overlay.jpg`：检测可视化
- `mask.png`：分割mask
- `result.json`：目标点坐标、yaw、预抓取点、平面法向量等

---

## 五、当前状态（汇报模板）

可以这样对上汇报：

- 已完成：D455采集、AprilTag检测、目标分割、多Tag平面拟合、像素到桌面坐标转换。
- 已打通：视觉结果到机械臂控制代码链路（IK与插值两种执行方式）。
- 待收口：生产环境参数固化、机械臂末端策略与误差回归测试。

---

## 六、快速答疑

- **Q: FK/IK在哪？**
  - FK：`teleop_tag_grasp.py` / `FK_transfer.py`
  - IK：`grasp.py` / `d455_detect_and_move.py`
- **Q: 分割在哪？**
  - `segmentation.py::segment_target_mask`
- **Q: 坐标转换在哪？**
  - `plane_fusion.py::pixel_to_plane_via_ray`
  - `bridge_tag_to_base.py::transform_tag_xy_to_base_xy`

---

如果需要给老板“PPT一页图”，建议直接画成：
**相机帧 → Tag检测/分割 → 平面拟合 → 坐标转换 → 机械臂执行（FK监控/IK求解）**。
