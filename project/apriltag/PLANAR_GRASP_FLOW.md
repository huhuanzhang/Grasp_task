# 平面抓取落地流程（从定位到机械臂指令）

## 目标
把视觉输出的 `tag平面坐标 (x,y,yaw)` 转成机器人 `base坐标 (X,Y,Z,yaw)`，再喂给你现有抓取控制（`grasp.py` / IK / send_action）。

---

## A. 一次性标定（平面版，建议先做）

1. 用上司脚本拿到若干点：`src_tag_xy`
2. 用机器人示教/触碰拿到对应点：`dst_base_xy`
3. 运行：

```bash
python from\ zhaoxin/table_guidance_project_v3/scripts/calibrate_base_from_points.py \
  --pairs from\ zhaoxin/table_guidance_project_v3/config/example_point_pairs.json \
  --out outputs/T_base_tag_se2.json
```

得到：
- `R` (2x2): tag平面到base平面的旋转
- `t` (2,): 平移

---

## B. 每次抓取在线流程

1. 视觉定位（上司脚本）：

```bash
python from\ zhaoxin/table_guidance_project_v3/scripts/run_demo.py \
  --image from\ zhaoxin/table_guidance_project_v3/assets/table_scene.jpg \
  --config from\ zhaoxin/table_guidance_project_v3/config/demo.yaml \
  --seg-backend fastsam \
  --fastsam-weights FastSAM-x.pt \
  --text-prompt "pink building block" \
  --out outputs
```

输出：`outputs/result.json`

2. 坐标桥接（本仓库新增脚本）：

```bash
python bridge_tag_to_base.py \
  --result outputs/result.json \
  --se2 outputs/T_base_tag_se2.json \
  --table-z 0.03 \
  --pre-z 0.10 \
  --grasp-z 0.02 \
  --out outputs/target_base.json
```

输出：
- `pregrasp_xyz_base_m`
- `grasp_xyz_base_m`
- `target_yaw_base_rad`（如果视觉有yaw）

3. 控制执行（接你现有控制）：
- 先到 `pregrasp_xyz_base_m`
- 再下探到 `grasp_xyz_base_m`
- 闭夹爪
- 抬升并视觉复检

---

## C. 如何确认 Z 抓取有效（实操要点）

1. 先做一次 `table_z` 标定（慢速下探到接触）
2. 设置：
- `Z_pre = table_z + 8~12cm`
- `Z_grasp = table_z + 1~2cm`（根据物体厚度调）
3. 成功判定至少用两项：
- 夹爪闭合后位置未闭到底
- 抬升后目标跟随（视觉复检）

---

## D. 单位/坐标坑位清单

- `lift_axis.height_mm` 是 **mm**，视觉/位姿大多是 **m**
- `yaw` 常是 **rad**，发关节时注意你控制链是否要 **deg**
- tag平面 +y 方向和base +y 方向可能相反，靠 SE2 标定吸收
