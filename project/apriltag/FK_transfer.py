#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

SAMPLES_JSONL = "samples.jsonl"
URDF_PATH = "so101_new_calib.urdf"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_FRAME = "gripper_frame_link"   # 也可试 "gripper_link" / "wrist_link"

def T_inv(T):
    Rm = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = Rm.T
    Ti[:3,3]  = -Rm.T @ t
    return Ti

def make_T(rvec, t):
    T = np.eye(4)
    T[:3,:3] = R.from_rotvec(rvec).as_matrix()
    T[:3,3] = t
    return T

def se3_log(T):
    """Return 6D log of SE3: [omega(3), v(3)] in a simple approximation.
       For calibration, this is good enough; if you want exact, can use pin.log6.
    """
    rot = R.from_matrix(T[:3,:3]).as_rotvec()
    trans = T[:3,3]
    return np.hstack([rot, trans])

# 1) 读 samples
samples = []
with open(SAMPLES_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        s = json.loads(line)
        if int(s["tag_id"]) != 1:
            continue
        samples.append(s)
assert len(samples) >= 15, "样本建议>=15，最好40+。"

# 2) FK 得到 T_base_gripper 和 T_cam_tag
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()
frame_id = model.getFrameId(EE_FRAME)
if frame_id == len(model.frames):
    raise ValueError(f"Frame {EE_FRAME} not found.")

def set_joint(q, joint_name, angle_rad):
    jid = model.getJointId(joint_name)
    idx = model.joints[jid].idx_q
    q[idx] = angle_rad

T_bg_list = []
T_ct_list = []

for s in samples:
    q = pin.neutral(model)
    for jn, ang_deg in zip(JOINT_NAMES, s["q_deg_left_5dof"]):
        set_joint(q, jn, np.deg2rad(float(ang_deg)))

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    T_bg = data.oMf[frame_id].homogeneous.copy()              # base->gripper
    T_ct = np.array(s["T_cam_tag"], dtype=np.float64)         # cam->tag

    T_bg_list.append(T_bg)
    T_ct_list.append(T_ct)

# 3) 最小二乘：未知 x=[rvecX,tX,rvecZ,tZ] 共12维
#    让 inv(T_bg*X) * (Z*T_ct) ≈ I
def residual(x):
    rX = x[0:3]; tX = x[3:6]
    rZ = x[6:9]; tZ = x[9:12]
    X = make_T(rX, tX)   # gripper->tag
    Z = make_T(rZ, tZ)   # base->cam

    res = []
    for T_bg, T_ct in zip(T_bg_list, T_ct_list):
        E = T_inv(T_bg @ X) @ (Z @ T_ct)   # should be I
        res.append(se3_log(E))
    return np.concatenate(res, axis=0)

# 初值：X,Z 都设为单位（也可以把你之前算的结果当初值）
x0 = np.zeros(12, dtype=np.float64)

sol = least_squares(residual, x0, verbose=2, max_nfev=200)

x = sol.x
X = make_T(x[0:3], x[3:6])
Z = make_T(x[6:9], x[9:12])

print("\n=== Optimized Results ===")
print("T_gripper_tag (X)=\n", X)
print("\nT_base_cam (Z)=\n", Z)

# 4) 正确一致性检查：gripper->tag 应该近似常数
T_gt_list = []
for T_bg, T_ct in zip(T_bg_list, T_ct_list):
    T_gt = T_inv(T_bg) @ Z @ T_ct
    T_gt_list.append(T_gt)

ref = T_gt_list[0]
t_err = [np.linalg.norm(T[:3,3] - ref[:3,3]) for T in T_gt_list]
R_err = []
for T in T_gt_list:
    dR = ref[:3,:3].T @ T[:3,:3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
    R_err.append(ang)

print("\n[CHECK] gripper->tag consistency:")
print("  translation error (m): mean=%.4f max=%.4f" % (np.mean(t_err), np.max(t_err)))
print("  rotation error (deg): mean=%.2f max=%.2f" % (np.mean(R_err), np.max(R_err)))