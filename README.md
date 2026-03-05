# Grasp Task / AprilTag Workspace

这个目录已按功能重构为 3 大模块，方便后续上传 GitHub 与维护。

## 目录结构

- `d455_test/`：D455 相机与 AprilTag 基础测试代码
- `seg_detection/`：分割与检测主流程（含 `table_guidance_project_v8`）
- `alohamini_control/`：AlohaMini 机械臂控制相关代码
- `docs/`：说明文档与手记
- `archive/`：历史产物、样本、旧版本与非核心文件（暂存，待确认后可清理）

---

## 1) d455_test

用于相机侧验证：

- `d455_cam.py`：D455 采集/画面测试
- `d455_intrinsics.py` / `d455_intrinsics.json`：内参读取与保存
- `d455_depth_compare.py`：深度对比测试
- `realsense_d455_apriltag_detect.py`：AprilTag 检测基础脚本
- `realsense_d455_apriltag_dof.py`：AprilTag 位姿/自由度相关测试

## 2) seg_detection

视觉主链路（分割 + Tag + 平面拟合）：

- `table_guidance_project_v8/`：当前主版本
  - 多 AprilTag 最小二乘拟合桌面平面
  - 目标分割（FastSAM）
  - 输出目标点与预抓取点（相机系）
- `realsense_d455_apriltag_cal.py`：标定相关脚本
- `realsense_d455_apriltag_cal_save_joints.py`：标定并保存关节位

## 3) alohamini_control

机械臂控制与抓取执行：

- `d455_detect_and_move.py`
- `d455_detect_and_move_qinterp.py`
- `teleop_tag_grasp.py`
- `grasp.py`
- `bridge_tag_to_base.py`
- `FK_transfer.py`
- `so101_new_calib.urdf`

---

## 当前状态（简）

- 视觉侧核心流程已跑通（见 `seg_detection/table_guidance_project_v8/outputs` 历史结果）
- 机械臂侧代码已独立分组，便于后续做 base 坐标系闭环

---

## 建议的下一步（上传 GitHub 前）

1. 在 `archive/` 中二次筛选：删除大文件与无关内容
2. 为 `alohamini_control/` 补充最小可运行说明（依赖、IP、机械臂参数）
3. 统一入口脚本命名（例如 `run_*.py`）
4. 若仓库 public，确认不包含敏感信息（IP、账号、私有数据）

---

## 备注

本次重构以“先分层、再精修”为目标；`archive/` 中内容未彻底删除，保留以防误删。确认后可统一清理。
