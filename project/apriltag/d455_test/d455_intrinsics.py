import json
import pyrealsense2 as rs

W, H, FPS = 1280, 720, 30   # 你实际用的分辨率/帧率（要和你项目一致）

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

profile = pipeline.start(config)

def intrinsics_to_dict(intr: rs.intrinsics):
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,   # principal point x
        "cy": intr.ppy,   # principal point y
        "distortion_model": str(intr.model),
        "distortion_coeffs": list(intr.coeffs),  # [k1,k2,p1,p2,k3,...]
    }

try:
    # 获取当前启用流的内参
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()

    # 获取 depth->color 外参（可选，但很多人也会一起要）
    extr = depth_stream.get_extrinsics_to(color_stream)
    extr_dict = {
        "rotation_3x3": [extr.rotation[i] for i in range(9)],  # row-major
        "translation_3": [extr.translation[i] for i in range(3)],
    }

    out = {
        "device": "Intel RealSense D455F",
        "streams": {
            "color": intrinsics_to_dict(color_intr),
            "depth": intrinsics_to_dict(depth_intr),
        },
        "extrinsics_depth_to_color": extr_dict,
        "note": "Intrinsics depend on resolution/FPS. Ensure these match the settings used in your pipeline.",
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    with open("d455_intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n[SAVED] d455_intrinsics.json")

finally:
    pipeline.stop()