# Grasp Task（AprilTag + D455 + 机械臂抓取）

基于 D455 + AprilTag + 分割检测 + 机械臂控制的抓取项目。

---

## 1. 主流程函数与逻辑（重点）

> 先标定，再跑在线抓取。

### Step A：标定采样（离线准备）
- 脚本：`seg_detection/realsense_d455_apriltag_cal_save_joints.py`
- 作用：采集标定样本（机械臂关节角 + 相机观测）。
- 特点：**看不见 Tag 也保存关节角**（当前采用的简化流程）。

### Step B：视觉感知（在线）
- 主入口：`seg_detection/table_guidance_project_v8/scripts/run_demo.py`
- 关键函数链：
  1. `apriltag_detect(...)`（`apriltag_backend.py`）
  2. `segment_target_mask(...)`（`segmentation.py`）
  3. `build_plane_model_from_tags(...)`（`plane_fusion.py`）
  4. `pixel_to_plane_via_ray(...)` / `cam_point_to_plane_xy(...)`（`plane_fusion.py`）
- 输出：`outputs/result.json`（目标点、姿态、平面信息）

### Step C：坐标系桥接（在线）
- 函数：`transform_tag_xy_to_base_xy(...)`（`alohamini_control/bridge_tag_to_base.py`）
- 作用：Tag/桌面坐标 → 机器人 base 坐标。

### Step D：机械臂执行（在线）
- 主入口（当前常用）：`alohamini_control/d455_detect_and_move_qinterp.py`
- 备选入口：`alohamini_control/d455_detect_and_move.py`
- IK 核心（IK版本）：`dls(...)`、`IKPos5Dof`

---

## 2. 目录结构

- `d455_test/`：相机与 Tag 基础测试
- `seg_detection/`：分割、检测、平面拟合、坐标计算、标定
- `alohamini_control/`：机械臂控制（FK/IK/执行）
- `docs/`：补充文档
- `archive/`：历史数据与旧文件

---

## 3. 关键函数速查（按模块）

### 3.1 AprilTag 检测
- `seg_detection/table_guidance_project_v8/src/table_guidance/apriltag_backend.py`
  - `apriltag_detect(...)`：检测 Tag，按需估计 `tag->camera` 位姿。

### 3.2 分割与目标提取
- `seg_detection/table_guidance_project_v8/src/table_guidance/segmentation.py`
  - `segment_target_mask(...)`：根据文本 prompt 输出目标掩码。
  - `_segment_fastsam_ultralytics(...)`：FastSAM 推理实现。
- `seg_detection/table_guidance_project_v8/src/table_guidance/pose.py`
  - `largest_connected_component(mask)`：去噪保留主体。
  - `mask_centroid_and_axis(mask)`：计算目标中心与方向。

### 3.3 平面拟合与坐标变换
- `seg_detection/table_guidance_project_v8/src/table_guidance/plane_fusion.py`
  - `fit_plane_svd(...)`：SVD 平面拟合。
  - `build_plane_model_from_tags(...)`：多 Tag 联合建桌面平面。
  - `pixel_to_plane_via_ray(...)`：像素射线与平面求交。
  - `cam_point_to_plane_xy(...)`：相机点转桌面平面 XY。
- `seg_detection/table_guidance_project_v8/src/table_guidance/transforms.py`
  - `cam_to_ref_tag(...)`：相机系 → 参考 Tag 系。
- `alohamini_control/bridge_tag_to_base.py`
  - `transform_tag_xy_to_base_xy(...)`：Tag 平面坐标 → base 坐标。

### 3.4 FK / IK / 执行
- `alohamini_control/FK_transfer.py`
  - `T_inv(...)`、`make_T(...)`、`se3_log(...)`、`set_joint(...)`
- `alohamini_control/grasp.py`
  - `dls(...)`、`IKPos5Dof`
- `alohamini_control/d455_detect_and_move.py`
  - `dls(...)`、`IKPos5Dof`（检测到执行的一体流程）

---

## 4. 运行指令（按实际使用顺序）

> 以下命令默认在仓库根目录 `project/apriltag` 下执行。

### 4.1 相机与 Tag 基础测试（先确认环境）

```bash
cd d455_test
python d455_cam.py
python d455_intrinsics.py
python realsense_d455_apriltag_detect.py
```

- `d455_cam.py`：D455 实时画面 + 多 Tag 可视化（用于快速确认相机/Tag是否正常）。
- `d455_intrinsics.py`：打印/检查相机内参（fx/fy/cx/cy）。
- `realsense_d455_apriltag_detect.py`：最小化 AprilTag 检测脚本（只看检测是否稳定）。

### 4.2 CAL 标定（当前简化版，优先）

> 当前流程只保留“无 Tag 也保存关节角”的版本。

```bash
RERUN=OFF python3 seg_detection/realsense_d455_apriltag_cal_save_joints.py \
  --remote_ip 172.16.0.14 \
  --tag_size 0.04 \
  --tag_id 0 \
  --out_jsonl samples_no_tag_ok.jsonl
```

### 4.3 jsonl 转 pairs.json（手动补坐标，严格一一对应）

> 用途：从 `samples_no_tag_ok.jsonl` 里提取可见 Tag 的采样点，生成 `pairs_template.json`，你再手动填入每个点对应的 `dst_base_xy`。脚本会强制检查：数量必须一致，不能少/不能多。

```bash
cd /home/hhz/.openclaw/workspace/project/apriltag
python3 - <<'PY'
import json
from pathlib import Path

jsonl_path = Path('samples_no_tag_ok.jsonl')
out_path = Path('pairs_template.json')

if not jsonl_path.exists():
    raise SystemExit(f"[ERROR] not found: {jsonl_path}")

src = []
for line in jsonl_path.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    if d.get('tag_visible') and d.get('T_cam_tag'):
        T = d['T_cam_tag']
        # 简化：取 tag 原点在相机坐标系的 x,y 作为可编辑模板点
        src.append([float(T[0][3]), float(T[1][3])])

if len(src) < 2:
    raise SystemExit(f"[ERROR] visible tag samples too few: {len(src)}")

pairs = {
    "src_tag_xy": src,
    "dst_base_xy": [[None, None] for _ in src],
    "note": "手动把 dst_base_xy 每一行填成对应的 base 坐标；必须与 src_tag_xy 一一对应"
}
out_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[OK] template saved: {out_path} (N={len(src)})")
PY
```

手动填写完成后，先做严格校验（不允许多/少/空值）：

```bash
cd /home/hhz/.openclaw/workspace/project/apriltag
python3 - <<'PY'
import json
from pathlib import Path

p = Path('pairs_template.json')
if not p.exists():
    raise SystemExit('[ERROR] pairs_template.json not found')

d = json.loads(p.read_text(encoding='utf-8'))
src = d.get('src_tag_xy', [])
dst = d.get('dst_base_xy', [])

if len(src) != len(dst):
    raise SystemExit(f"[ERROR] count mismatch: src={len(src)} dst={len(dst)}")

for i, (a, b) in enumerate(zip(src, dst)):
    if (not isinstance(a, list)) or (not isinstance(b, list)) or len(a) != 2 or len(b) != 2:
        raise SystemExit(f"[ERROR] row {i}: each point must be [x, y]")
    if any(v is None for v in b):
        raise SystemExit(f"[ERROR] row {i}: dst_base_xy has None, please fill it")

out = Path('pairs.json')
out.write_text(json.dumps({"src_tag_xy": src, "dst_base_xy": dst}, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[OK] pairs.json ready: {out} (N={len(src)})")
PY
```

然后执行标定矩阵求解：

```bash
cd seg_detection/table_guidance_project_v8
PYTHONPATH=src /home/hhz/miniconda3/envs/lerobot_alohamini/bin/python scripts/calibrate_base_from_points.py \
  --pairs /home/hhz/.openclaw/workspace/project/apriltag/pairs.json \
  --out /home/hhz/.openclaw/workspace/project/apriltag/outputs/T_base_tag_se2.json
```

### 4.4 视觉主流程（分割 + 坐标输出）

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

### 4.4 视觉变换验证

```bash
cd seg_detection/table_guidance_project_v8
python scripts/validate_transforms.py \
  --image assets/d455_live.jpg \
  --config config/demo.yaml \
  --out outputs
```

### 4.5 机械臂执行（查看参数）

```bash
cd alohamini_control
python d455_detect_and_move.py --help
python d455_detect_and_move_qinterp.py --help
python teleop_tag_grasp.py --help
```

- `d455_detect_and_move.py`：检测结果 → IK 求解 → 机械臂执行的主流程版本。
- `d455_detect_and_move_qinterp.py`：检测结果 → 关节空间插值执行（当前主用，更稳）。
- `teleop_tag_grasp.py`：遥操作 + D455 + Tag 可视化联调脚本，用于人工对位/调试。

### 4.6 当前推荐一键主流程（qinterp）

```bash
cd /home/hhz/.openclaw/workspace/project/apriltag && /home/hhz/miniconda3/envs/lerobot_alohamini/bin/python d455_detect_and_move_qinterp.py --remote_ip 172.16.0.14 --prompt "green stick"
```

---

## 5. 常见产出

- `overlay.jpg`：检测可视化
- `mask.png`：分割结果
- `result.json`：目标位置、姿态角、预抓取点、平面信息
- `samples_no_tag_ok.jsonl`：标定采样数据（含“无 Tag 也记录”的关节角样本）

---

## 6. 备注

- `archive/` 目录默认不参与主流程运行。
- 函数级排查建议参考：`docs/FUNCTION_MAP.md`。
