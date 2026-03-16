from pathlib import Path
f_path = Path(__file__).absolute()
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))

import os
os.environ["OPEN3D_CPU_RENDERING"] = "true"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ.setdefault("XDG_RUNTIME_DIR", f"/tmp/{os.getenv('USER', 'user')}-runtime")

import re
import argparse
import numpy as np
import open3d as o3d
import av


FRAME_RE = re.compile(r"frame_(\d+)$")


def list_frame_meshes(root_dir: str, fn:str):
    items = []
    for name in os.listdir(root_dir):
        m = FRAME_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        mesh_path = os.path.join(root_dir, name, fn)
        if os.path.isfile(mesh_path):
            items.append((idx, mesh_path))
    items.sort(key=lambda x: x[0])
    return items


def load_o3d_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        return mesh
    # 如果没有法向就算一下
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def compute_global_bbox(mesh_paths):
    bb_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bb_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    ok = 0
    for p in mesh_paths:
        m = load_o3d_mesh(p)
        if m.is_empty():
            continue
        aabb = m.get_axis_aligned_bounding_box()
        bb_min = np.minimum(bb_min, aabb.get_min_bound())
        bb_max = np.maximum(bb_max, aabb.get_max_bound())
        ok += 1
    if ok == 0:
        raise RuntimeError("All meshes are empty / failed to load.")
    return bb_min, bb_max


def pick_camera(bb_min, bb_max, view: str):
    center = (bb_min + bb_max) * 0.5
    extent = (bb_max - bb_min)
    radius = float(np.linalg.norm(extent) * 0.5)
    dist = max(1e-6, radius * 1.)

    if view == "front":
        direction = np.array([0.0, -1.0, 0.2], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif view == "side":
        direction = np.array([1.0, 0.0, 0.2], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif view == "top":
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:  # isometric
        direction = np.array([1.0, -1.0, 0.8], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    direction /= np.linalg.norm(direction)
    eye = center + direction * dist
    return eye, center, up


class PyAVWriter:
    def __init__(self, out_path: str, fps: int):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.container = av.open(out_path, mode="w")
        self.stream = self.container.add_stream("h264", rate=fps)
        self.stream.pix_fmt = "yuv420p"
        self._inited = False

    def write_rgb(self, img_rgb_uint8: np.ndarray):
        if not self._inited:
            h, w = img_rgb_uint8.shape[:2]
            self.stream.width = w
            self.stream.height = h
            self._inited = True

        frame = av.VideoFrame.from_ndarray(img_rgb_uint8, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="render_cond 目录（包含 frame_xxxx 子目录）")
    ap.add_argument("--out", required=True, help="输出 mp4 路径")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--view", type=str, default="isometric",
                    choices=["front", "side", "top", "isometric"])
    ap.add_argument("--use_global_bbox", action="store_true",
                    help="用所有帧的 bbox 来定相机（推荐：镜头完全不抖）")
    ap.add_argument("--unlit", action="store_true",
                    help="无光照材质（更稳定，但阴影层次少）")
    ap.add_argument("--bg", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help="背景色 RGB (0~1)，例如 --bg 1 1 1")
    ap.add_argument("--fn", type=str, default="mesh.ply")
    args = ap.parse_args()

    frames = list_frame_meshes(args.root_dir, fn=args.fn)
    if not frames:
        raise FileNotFoundError(f"No frame_xxxx/{args.fn} found under: {args.root_dir}")

    indices, mesh_paths = zip(*frames)
    print(f"Found {len(mesh_paths)} meshes: frame {indices[0]} .. {indices[-1]}")

    # 选 bbox（决定相机）
    if args.use_global_bbox:
        bb_min, bb_max = compute_global_bbox(mesh_paths)
    else:
        m0 = load_o3d_mesh(mesh_paths[0])
        aabb = m0.get_axis_aligned_bounding_box()
        bb_min, bb_max = aabb.get_min_bound(), aabb.get_max_bound()

    eye, center, up = pick_camera(np.asarray(bb_min), np.asarray(bb_max), args.view)

    # Open3D 离屏渲染器
    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    scene = renderer.scene
    scene.set_background([args.bg[0], args.bg[1], args.bg[2], 1.0])

    # 材质
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit" if args.unlit else "defaultLit"
    mat.base_color = [0.9, 0.9, 0.9, 1.0]

    # 尝试加点光（不同 Open3D 版本 API 可能有差异，所以 try）
    try:
        scene.scene.set_sun_light(direction=[-1.0, -1.0, -1.0],
                                  color=[1.0, 1.0, 1.0],
                                  intensity=75000.0)
        scene.scene.enable_sun_light(True)
    except Exception:
        pass

    scene.camera.look_at(center, eye, up)

    writer = PyAVWriter(args.out, fps=args.fps)

    # 逐帧渲染 + 写视频
    for idx, mesh_path in frames:
        mesh = load_o3d_mesh(mesh_path)
        if mesh.is_empty():
            print(f"[warn] empty mesh: {mesh_path}")
            continue

        if scene.has_geometry("mesh"):
            scene.remove_geometry("mesh")
        scene.add_geometry("mesh", mesh, mat)

        img = renderer.render_to_image()
        rgb = np.asarray(img)[:, :, :3].copy()  # HxWx3 uint8 RGB
        writer.write_rgb(rgb)

    writer.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()