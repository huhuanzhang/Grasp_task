#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
import pinocchio as pin

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep


def make_T_from_pose(pose_R: np.ndarray, pose_t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose_R
    T[:3, 3] = pose_t.reshape(3)
    return T


def dls(J, e, lam=0.03):
    # dq = J^T (J J^T + lam^2 I)^{-1} e
    JJt = J @ J.T
    A = JJt + (lam * lam) * np.eye(3)
    return J.T @ np.linalg.solve(A, e)


class IKPos5Dof:
    def __init__(self, urdf_path: str, ee_frame: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(ee_frame)
        if self.frame_id == len(self.model.frames):
            raise ValueError(f"Frame {ee_frame} not found in URDF")

        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.joint_ids = [self.model.getJointId(n) for n in self.joint_names]
        for n, jid in zip(self.joint_names, self.joint_ids):
            if jid == 0:
                raise ValueError(f"Joint {n} not found in URDF")

        self.q_idx = [self.model.joints[j].idx_q for j in self.joint_ids]
        self.v_idx = [self.model.joints[j].idx_v for j in self.joint_ids]

    def q_from_obs_deg(self, obs: dict) -> np.ndarray:
        q = pin.neutral(self.model)
        keys = [
            "arm_left_shoulder_pan.pos",
            "arm_left_shoulder_lift.pos",
            "arm_left_elbow_flex.pos",
            "arm_left_wrist_flex.pos",
            "arm_left_wrist_roll.pos",
        ]
        for idx_q, k in zip(self.q_idx, keys):
            q[idx_q] = np.deg2rad(float(obs[k]))
        return q

    def q_to_deg5(self, q: np.ndarray) -> list[float]:
        return [float(np.rad2deg(q[idx])) for idx in self.q_idx]

    def fk_pos(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.frame_id].homogeneous
        return T[:3, 3].copy()

    def jac_pos(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(
            self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J6[:3, :][:, self.v_idx]  # (3,5)

    def step_towards(self, q: np.ndarray, p_goal: np.ndarray, step_scale=0.6, lam=0.03) -> tuple[np.ndarray, float]:
        """单步 IK（每帧跑1-2次即可）"""
        p = self.fk_pos(q)
        e = p_goal - p
        err = float(np.linalg.norm(e))
        J = self.jac_pos(q)
        dq = dls(J, e, lam=lam)
        dq = np.clip(dq, -0.12, 0.12)  # 每帧步长限制（rad）
        q2 = q.copy()
        for idx, dqi in zip(self.q_idx, dq):
            q2[idx] += step_scale * dqi
        return q2, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_ip", type=str, default="172.16.0.14")
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--ee_frame", type=str, default="gripper_frame_link")

    parser.add_argument("--tag_id", type=int, default=1)
    parser.add_argument("--tag_size", type=float, default=0.04)
    parser.add_argument("--min_decision_margin", type=float, default=30.0)
    parser.add_argument("--max_hamming", type=int, default=0)

    parser.add_argument("--cam_w", type=int, default=1280)
    parser.add_argument("--cam_h", type=int, default=720)
    parser.add_argument("--cam_fps", type=int, default=30)

    parser.add_argument("--pre_z", type=float, default=0.10)
    parser.add_argument("--grasp_z", type=float, default=0.02)
    parser.add_argument("--lift_mm", type=float, default=500.0)

    args = parser.parse_args()

    # 你的 hand-eye 结果：T_base_cam
    T_base_cam = np.array([
        [-0.26296128, -0.32847891,  0.90716755, -0.04672538],
        [-0.95438312,  0.22639856, -0.19467038,  0.15929978],
        [-0.14143632, -0.91697617, -0.37302878,  0.25363927],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
    robot.connect()

    ik = IKPos5Dof(args.urdf_path, args.ee_frame)

    # RealSense + AprilTag
    at_detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, args.cam_w, args.cam_h, rs.format.bgr8, args.cam_fps)
    rs_profile = rs_pipeline.start(rs_config)
    intr = rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)

    # runtime states
    pre_z = args.pre_z
    grasp_z = args.grasp_z
    use_grasp = False
    follow = False
    show_debug = True

    # IK internal q (kept between frames)
    q_cmd = None
    last_tag = None

    print("\nKeys (focus OpenCV window):")
    print("  q quit")
    print("  f toggle FOLLOW (real-time)")
    print("  g toggle PRE/GRASP")
    print("  o/l: z_offset +=5mm / -=5mm (current mode)")
    print("  c toggle debug overlay\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1) get obs
            obs = robot.get_observation()

            # init q_cmd from obs
            if q_cmd is None:
                q_cmd = ik.q_from_obs_deg(obs)

            # 2) camera + detect tag
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            dets = at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=args.tag_size,
            )
            best = None
            for d in dets:
                if d.tag_id != args.tag_id:
                    continue
                if d.decision_margin < args.min_decision_margin:
                    continue
                if d.hamming > args.max_hamming:
                    continue
                if best is None or d.decision_margin > best.decision_margin:
                    best = d

            p_goal = None
            if best is not None:
                T_cam_tag = make_T_from_pose(best.pose_R, best.pose_t)
                T_base_tag = T_base_cam @ T_cam_tag
                p_tag = T_base_tag[:3, 3]
                z_off = (grasp_z if use_grasp else pre_z)
                p_goal = p_tag + np.array([0.0, 0.0, z_off], dtype=np.float64)

                last_tag = {
                    "corners": best.corners.astype(np.int32),
                    "dm": float(best.decision_margin),
                    "p_tag": p_tag,
                    "z_cam": float(T_cam_tag[2, 3]),
                }

            # 3) handle keys
            disp = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.imshow("follow_grasp", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("f"):
                follow = not follow
                print("\nFOLLOW =", follow)
            elif k == ord("g"):
                use_grasp = not use_grasp
            elif k == ord("c"):
                show_debug = not show_debug
            elif k == ord("o"):
                if use_grasp: grasp_z += 0.005
                else: pre_z += 0.005
            elif k == ord("l"):
                if use_grasp: grasp_z -= 0.005
                else: pre_z -= 0.005

            # 4) follow: IK step(s) + send_action every frame
            err = None
            if follow and (p_goal is not None):
                # 每帧跑两小步更稳
                q_cmd, err = ik.step_towards(q_cmd, p_goal, step_scale=0.6, lam=0.03)
                q_cmd, err = ik.step_towards(q_cmd, p_goal, step_scale=0.6, lam=0.03)

                q_deg5 = ik.q_to_deg5(q_cmd)

                # action keys 必须带 .pos（与你 TELEOP KEYS 一致）
                action = {
                    "arm_left_shoulder_pan.pos":  q_deg5[0],
                    "arm_left_shoulder_lift.pos": q_deg5[1],
                    "arm_left_elbow_flex.pos":    q_deg5[2],
                    "arm_left_wrist_flex.pos":    q_deg5[3],
                    "arm_left_wrist_roll.pos":    q_deg5[4],

                    # gripper 保持当前（你想自动开合再改）
                    "arm_left_gripper.pos": float(obs.get("arm_left_gripper.pos", 50.0)),

                    # 右臂保持当前
                    "arm_right_shoulder_pan.pos":  float(obs.get("arm_right_shoulder_pan.pos", 0.0)),
                    "arm_right_shoulder_lift.pos": float(obs.get("arm_right_shoulder_lift.pos", 0.0)),
                    "arm_right_elbow_flex.pos":    float(obs.get("arm_right_elbow_flex.pos", 0.0)),
                    "arm_right_wrist_flex.pos":    float(obs.get("arm_right_wrist_flex.pos", 0.0)),
                    "arm_right_wrist_roll.pos":    float(obs.get("arm_right_wrist_roll.pos", 0.0)),
                    "arm_right_gripper.pos":       float(obs.get("arm_right_gripper.pos", 50.0)),

                    # base + lift
                    "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0,
                    "lift_axis.height_mm": float(args.lift_mm),
                }

                robot.send_action(action)

            # 5) draw overlay (after key handling, redraw on original frame for clarity)
            # (Simple overlay without re-resize complexity)
            if last_tag is not None:
                cv2.polylines(frame, [last_tag["corners"]], True, (0, 255, 0), 2)
                cv2.putText(frame, f"tag {args.tag_id} dm={last_tag['dm']:.1f} z_cam={last_tag['z_cam']:.3f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if show_debug:
                mode = "GRASP" if use_grasp else "PRE"
                z_off = grasp_z if use_grasp else pre_z
                cv2.putText(frame, f"FOLLOW={follow}  MODE={mode}  z_off={z_off:.3f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if err is not None:
                    cv2.putText(frame, f"IK err={err:.3f} m",
                                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            disp2 = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.imshow("follow_grasp", disp2)

            # keep fps
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

    finally:
        try:
            rs_pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()