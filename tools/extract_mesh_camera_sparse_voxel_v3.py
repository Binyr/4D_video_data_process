#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: process_geometry_with_bpy.py
# Description:
#   1) Load an animated .glb / .gltf in Blender
#   2) Remove root-body translation per frame by subtracting each frame's bbox center
#   3) Rescale the whole sequence so all shapes fit inside a unit bounding box
#   4) Render static multi-view cameras per frame
#   5) Save one shared-topology mesh npz: faces once + per-frame vertices
#   6) Save normalized mesh as one .ply per keyframe
#   7) Optionally export the normalized animated scene + static cameras as a new GLB
#   8) Optionally render the canonical scene box
#   9) Optionally randomize camera intrinsics while keeping fovx = fovy
#  10) Lighting:
#        - with probability sunlight_prob, use SunLight
#        - otherwise, randomly choose one HDR map from hdr_dir
#        - record the chosen lighting config into metadata
#
#  New camera distance logic:
#     For each sampled view (azim/elev),
#       - rotate all normalized mesh frames into camera-aligned space
#       - compute a sequence-level scene box in that camera space
#       - compute a view-specific normalization scale from that camera-space scene box
#       - estimate tight distance from the normalized camera-space box
#       - convert the distance back to the shared world-normalized scene
#
# Output layout:
#   <prefix>_rgb/
#       view_00/frame_0001.png
#       view_00/frame_0002.png
#       ...
#       view_01/frame_0001.png
#       ...
#
#   <prefix>_normal/
#       view_00/frame_0001.png
#       view_00/frame_0002.png
#       ...
#       view_01/frame_0001.png
#       ...
#
#   <prefix>_mesh_ply/
#       frame_0001.ply
#       frame_0008.ply
#       ...
#
# Example:
# blender --background --python process_geometry_with_bpy.py -- \
#   --object_path /path/to/model.glb \
#   --output_file /path/to/result.json \
#   --normalized_glb_path /path/to/result_normalized.glb \
#   --resolution 512 \
#   --transparent_bg \
#   --hdr_dir data/hdr \
#   --hdr_strength 1.0 \
#   --sunlight_prob 0.05 \
#   --sunlight_energy 3.0 \
#   --camera_frame_padding 0.01 \
#   --camera_fit_safety 1.01 \
#   --camera_distance_jitter_scale 1.01 \
#   --render_scene_box \
#   --randomize_camera_intrinsics \
#   --camera_fov_min_deg 30 \
#   --camera_fov_max_deg 70

import argparse
import sys
import os
import math
import json
from pathlib import Path
from typing import Dict, Callable, List

import numpy as np
import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view


# =================================================================================
#  0. DEBUG
# =================================================================================

def debug_project_world_points(cam_obj, points_world: np.ndarray, title: str = ""):
    scene = bpy.context.scene

    xs, ys, zs = [], [], []
    print(f"Projected points for camera: {cam_obj.name} | {title}")
    for i, p in enumerate(points_world):
        co_ndc = world_to_camera_view(scene, cam_obj, Vector(np.asarray(p, dtype=np.float32).tolist()))
        x, y, z = float(co_ndc.x), float(co_ndc.y), float(co_ndc.z)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        print(f"  point {i}: x={x:.6f}, y={y:.6f}, z={z:.6f}")

    print(f"x range: [{min(xs):.6f}, {max(xs):.6f}]")
    print(f"y range: [{min(ys):.6f}, {max(ys):.6f}]")
    print(f"z range: [{min(zs):.6f}, {max(zs):.6f}]")


def debug_project_bbox_corners(cam_obj, bbox_min, bbox_max):
    corners = get_bbox_corners(
        np.asarray(bbox_min, dtype=np.float32),
        np.asarray(bbox_max, dtype=np.float32),
    )
    debug_project_world_points(cam_obj, corners, title="world canonical bbox")


def debug_project_camera_space_bbox(cam_obj, bbox_min_cam, bbox_max_cam, azim: float, elev: float):
    corners_world = camera_aligned_bbox_corners_to_world(
        np.asarray(bbox_min_cam, dtype=np.float32),
        np.asarray(bbox_max_cam, dtype=np.float32),
        azim=azim,
        elev=elev,
    )
    debug_project_world_points(cam_obj, corners_world, title="camera-space sequence bbox mapped to world")


# =================================================================================
#  1. BASIC HELPERS
# =================================================================================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
}


def get_cli_argv():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return argv[1:]


def init_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh, do_unlink=True)


def load_object(object_path: str) -> None:
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    IMPORT_FUNCTIONS[file_extension](filepath=object_path)


def remove_all_light_objects():
    removed = []
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    if len(removed) > 0:
        print(f"Removed imported light objects: {removed}")
    return removed

def summarize_existing_light_objects():
    infos = []
    for obj in bpy.context.scene.objects:
        if obj.type != "LIGHT":
            continue
        light_type = None
        light_energy = None
        if getattr(obj, "data", None) is not None:
            light_type = getattr(obj.data, "type", None)
            light_energy = getattr(obj.data, "energy", None)
        infos.append({
            "name": obj.name,
            "light_type": str(light_type) if light_type is not None else None,
            "energy": float(light_energy) if light_energy is not None else None,
        })
    return infos


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def look_at(
    cam_pos: np.ndarray,
    target: np.ndarray,
    up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
):
    """
    Return cam2world whose columns are:
      x-axis = right
      y-axis = up
      z-axis = camera local +Z

    Blender camera looks toward local -Z, so we store -forward in the 3rd column.
    """
    forward = normalize(target - cam_pos)
    right = normalize(np.cross(forward, up))
    if np.linalg.norm(right) < 1e-6:
        up_alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, up_alt))
    true_up = normalize(np.cross(right, forward))

    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = true_up
    cam2world[:3, 2] = -forward
    cam2world[:3, 3] = cam_pos
    return cam2world


def orbit_offset(radius: float, azim: float, elev: float):
    x = radius * math.cos(elev) * math.cos(azim)
    y = radius * math.cos(elev) * math.sin(azim)
    z = radius * math.sin(elev)
    return np.array([x, y, z], dtype=np.float32)


def create_camera(name="TrackingCamera", lens=50.0, sensor_width=36.0, sensor_height=36.0):
    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "PERSP"
    cam_data.lens = lens
    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_height
    cam_data.sensor_fit = "HORIZONTAL"

    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def set_camera_from_cam2world(cam_obj, cam2world: np.ndarray):
    cam_obj.matrix_world = Matrix(cam2world.tolist())


def set_camera_intrinsics_from_fov(cam_obj, fov_deg: float, sensor_size: float = 36.0):
    """
    通过设置单一 FOV 来随机相机内参，并保持:
      - sensor_width = sensor_height
      - 因此在正方形输出分辨率下，fovx = fovy
    """
    fov_deg = float(np.clip(fov_deg, 1.0, 179.0))
    fov_rad = math.radians(fov_deg)

    cam_data = cam_obj.data
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = float(sensor_size)
    cam_data.sensor_height = float(sensor_size)

    lens_mm = 0.5 * float(sensor_size) / max(math.tan(0.5 * fov_rad), 1e-8)
    cam_data.lens = float(lens_mm)


def get_camera_intrinsics_dict(cam_obj, resolution: int):
    """
    从 Blender camera 读出等价内参信息。
    注意这里默认输出是 resolution x resolution 的正方形图像。
    """
    cam_data = cam_obj.data

    angle_x = float(cam_data.angle_x)
    angle_y = float(cam_data.angle_y)

    fx = 0.5 * float(resolution) / max(math.tan(0.5 * angle_x), 1e-8)
    fy = 0.5 * float(resolution) / max(math.tan(0.5 * angle_y), 1e-8)
    cx = 0.5 * float(resolution)
    cy = 0.5 * float(resolution)

    return {
        "lens_mm": float(cam_data.lens),
        "sensor_width_mm": float(cam_data.sensor_width),
        "sensor_height_mm": float(cam_data.sensor_height),
        "fov_x_deg": float(math.degrees(angle_x)),
        "fov_y_deg": float(math.degrees(angle_y)),
        "fx_px": float(fx),
        "fy_px": float(fy),
        "cx_px": float(cx),
        "cy_px": float(cy),
        "fovx_equals_fovy": bool(abs(angle_x - angle_y) < 1e-8),
    }


# =================================================================================
#  2. GEOMETRY EXTRACTION
# =================================================================================

def extract_merged_mesh_world(mesh_objs):
    """
    提取当前帧所有 mesh，并合并成一个 world-space mesh。

    Returns:
        merged_vertices: np.ndarray [N, 3], float32
        merged_faces: np.ndarray [F, 3], int32
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    all_vertices = []
    all_faces = []
    vert_offset = 0

    for obj in mesh_objs:
        if obj.type != "MESH":
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        temp_mesh = obj_eval.to_mesh()

        if temp_mesh is None:
            continue

        temp_mesh.calc_loop_triangles()

        if len(temp_mesh.vertices) == 0 or len(temp_mesh.loop_triangles) == 0:
            obj_eval.to_mesh_clear()
            continue

        world_mat = obj_eval.matrix_world.copy()

        verts = np.array([world_mat @ v.co for v in temp_mesh.vertices], dtype=np.float32)
        faces = np.array([lt.vertices[:] for lt in temp_mesh.loop_triangles], dtype=np.int32)

        all_vertices.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)

        obj_eval.to_mesh_clear()

    if len(all_vertices) == 0:
        raise RuntimeError("No valid mesh found in current frame.")

    merged_vertices = np.concatenate(all_vertices, axis=0)
    merged_faces = np.concatenate(all_faces, axis=0)
    return merged_vertices, merged_faces


def compute_bbox_center(vertices_world: np.ndarray):
    bbox_min = vertices_world.min(axis=0)
    bbox_max = vertices_world.max(axis=0)
    return 0.5 * (bbox_min + bbox_max)


def collect_keyframe_frames(frame_start: int, frame_end: int) -> List[int]:
    """
    从所有 action / fcurve 中提取真实关键帧编号。
    只保留 [frame_start, frame_end] 范围内的整帧。
    """
    keyframes = set()

    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            for kp in fcurve.keyframe_points:
                frame = int(round(float(kp.co.x)))
                if frame_start <= frame <= frame_end:
                    keyframes.add(frame)

    keyframes = sorted(keyframes)
    return keyframes


def save_mesh_as_ply(vertices: np.ndarray, faces: np.ndarray, ply_path: str):
    """
    保存三角 mesh 为 binary little endian PLY。
    vertices: [N, 3], float32/float16/float64
    faces:    [F, 3], int32/int64
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))

        vertices.astype("<f4", copy=False).tofile(f)

        face_dtype = np.dtype([
            ("count", "u1"),
            ("idx", "<i4", (3,)),
        ])
        face_data = np.empty(len(faces), dtype=face_dtype)
        face_data["count"] = 3
        face_data["idx"] = faces.astype("<i4", copy=False)
        face_data.tofile(f)


# =================================================================================
#  3. SEQUENCE NORMALIZATION
# =================================================================================

def create_sequence_normalizer():
    """
    创建一个空物体作为额外父节点。
    后续逐帧设置其 scale / translation，实现：
        x' = s * (x - center_t)
    """
    root_objs = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(root_objs) == 0:
        raise RuntimeError("No root objects found in the scene.")

    normalizer = bpy.data.objects.new("SequenceNormalizer", None)
    bpy.context.scene.collection.objects.link(normalizer)
    normalizer.location = (0.0, 0.0, 0.0)
    normalizer.rotation_euler = (0.0, 0.0, 0.0)
    normalizer.scale = (1.0, 1.0, 1.0)

    for obj in root_objs:
        world_mat = obj.matrix_world.copy()
        obj.parent = normalizer
        obj.matrix_parent_inverse = normalizer.matrix_world.inverted()
        obj.matrix_world = world_mat

    bpy.context.view_layer.update()
    return normalizer


def compute_sequence_normalization_params(mesh_objs, frame_indices: np.ndarray):
    """
    第一遍扫描序列：
      1) 检查拓扑是否一致
      2) 计算每帧 bbox center
      3) 统计去平移后的全序列 bbox，用于求全局 scale
    """
    shared_faces = None
    centers_seq = []

    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    for i in frame_indices:
        bpy.context.scene.frame_set(int(i))
        bpy.context.view_layer.update()

        raw_vertices, raw_faces = extract_merged_mesh_world(mesh_objs)

        if shared_faces is None:
            shared_faces = raw_faces.copy()
            print(
                f"Reference topology set from frame {int(i)}: "
                f"{shared_faces.shape[0]} faces, {raw_vertices.shape[0]} vertices"
            )
        else:
            if raw_faces.shape != shared_faces.shape or not np.array_equal(raw_faces, shared_faces):
                raise RuntimeError(
                    f"Topology changed at frame {int(i)}. "
                    f"Cannot store one shared face array for all frames. "
                    f"Reference faces shape={shared_faces.shape}, current faces shape={raw_faces.shape}"
                )

        center = compute_bbox_center(raw_vertices).astype(np.float32)
        centers_seq.append(center)

        centered = raw_vertices - center[None, :]
        global_min = np.minimum(global_min, centered.min(axis=0))
        global_max = np.maximum(global_max, centered.max(axis=0))

    centers_seq = np.stack(centers_seq, axis=0)
    shared_faces = np.asarray(shared_faces, dtype=np.int32)

    extent = global_max - global_min
    box_size = float(np.max(extent))
    sequence_scale = 1.0 / box_size if box_size > 1e-6 else 1.0

    canonical_bbox_min = (global_min * sequence_scale).astype(np.float32)
    canonical_bbox_max = (global_max * sequence_scale).astype(np.float32)

    return shared_faces, centers_seq, sequence_scale, canonical_bbox_min, canonical_bbox_max


def apply_sequence_normalization(normalizer_obj, frame_center: np.ndarray, global_scale: float):
    """
    对当前帧应用：
        x' = global_scale * (x - frame_center)
    """
    normalizer_obj.scale = (global_scale, global_scale, global_scale)
    normalizer_obj.location = tuple((-global_scale * frame_center).tolist())
    bpy.context.view_layer.update()


def bake_sequence_normalization_to_keyframes(
    normalizer_obj,
    frame_indices: np.ndarray,
    centers_seq: np.ndarray,
    global_scale: float,
):
    """
    把逐帧 normalization 烘焙到 normalizer 的 keyframes 上，方便导出成带动画的 GLB。
    """
    scene = bpy.context.scene
    current_frame = scene.frame_current

    scene.frame_start = int(frame_indices[0])
    scene.frame_end = int(frame_indices[-1])

    if normalizer_obj.animation_data is not None:
        normalizer_obj.animation_data_clear()

    for local_frame_idx, source_frame in enumerate(frame_indices):
        scene.frame_set(int(source_frame))

        normalizer_obj.scale = (global_scale, global_scale, global_scale)
        normalizer_obj.location = tuple(
            (-global_scale * centers_seq[local_frame_idx]).tolist()
        )

        normalizer_obj.keyframe_insert(data_path="location", frame=int(source_frame))
        normalizer_obj.keyframe_insert(data_path="scale", frame=int(source_frame))

    if normalizer_obj.animation_data is not None and normalizer_obj.animation_data.action is not None:
        for fcurve in normalizer_obj.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = "LINEAR"

    scene.frame_set(current_frame)
    bpy.context.view_layer.update()


def precompute_normalized_mesh_sequence(
    mesh_objs,
    frame_indices: np.ndarray,
    normalizer_obj,
    centers_seq: np.ndarray,
    global_scale: float,
    shared_faces: np.ndarray,
    export_keyframe_ply: bool = False,
    keyframe_frame_set=None,
    mesh_ply_dir: str = "",
):
    """
    第二遍扫描序列：
      - 应用全局 world normalization
      - 提取每帧 normalized vertices
      - 可选导出 keyframe ply

    返回:
      vertices_seq_list: List[np.ndarray], each [N,3], float16
      keyframe_ply_records: dict(frame_int -> ply_path)
    """
    from tqdm import tqdm

    if keyframe_frame_set is None:
        keyframe_frame_set = set()

    vertices_seq_list = []
    keyframe_ply_records = {}

    for local_frame_idx, source_frame in enumerate(tqdm(frame_indices, desc="Precomputing normalized mesh sequence")):
        frame_int = int(source_frame)
        bpy.context.scene.frame_set(frame_int)

        apply_sequence_normalization(
            normalizer_obj,
            frame_center=centers_seq[local_frame_idx],
            global_scale=global_scale,
        )
        bpy.context.view_layer.update()

        norm_vertices, norm_faces = extract_merged_mesh_world(mesh_objs)
        if norm_faces.shape != shared_faces.shape or not np.array_equal(norm_faces, shared_faces):
            raise RuntimeError(
                f"Topology unexpectedly changed after normalization at frame {frame_int}."
            )

        vertices_seq_list.append(norm_vertices.astype(np.float16))

        if export_keyframe_ply and (frame_int in keyframe_frame_set):
            mesh_ply_path = os.path.join(mesh_ply_dir, f"frame_{frame_int:04d}.ply")
            save_mesh_as_ply(norm_vertices, shared_faces, mesh_ply_path)
            keyframe_ply_records[frame_int] = mesh_ply_path

    return vertices_seq_list, keyframe_ply_records


# =================================================================================
#  4. RENDERING / LIGHTING
# =================================================================================

def enable_cycles_acceleration(backend: str = "CUDA"):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    backend = backend.upper()
    prefs.compute_device_type = backend

    try:
        prefs.refresh_devices()
    except AttributeError:
        prefs.get_devices()

    found_gpu = False
    print("=== Cycles device discovery ===")
    print("Requested backend:", backend)

    for d in prefs.devices:
        d.use = (d.type != "CPU")
        print(f"  name={d.name}, type={d.type}, use={d.use}")
        if d.type != "CPU" and d.use:
            found_gpu = True

    scene.cycles.device = "GPU"

    print("scene.render.engine =", scene.render.engine)
    print("scene.cycles.device =", scene.cycles.device)
    print("prefs.compute_device_type =", prefs.compute_device_type)

    if not found_gpu:
        raise RuntimeError(
            f"No usable GPU found for Cycles backend={backend}. "
            f"Try backend='CUDA', check nvidia-smi, and verify Blender can see the GPU."
        )


def setup_renderer(
    resolution=512,
    engine="BLENDER_EEVEE",
    transparent_bg=True
):
    scene = bpy.context.scene
    scene.render.engine = engine

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent_bg else "RGB"
    scene.render.film_transparent = transparent_bg
    scene.render.use_file_extension = True

    if engine == "CYCLES":
        scene.cycles.device = "GPU"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True

    elif engine == "BLENDER_EEVEE":
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 64


def create_sun_light(name="SunLight", energy=3.0):
    light_data = bpy.data.lights.new(name=name, type="SUN")
    light_data.energy = float(energy)
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    light_obj.location = (2.0, -2.0, 3.0)
    light_obj.rotation_euler = (math.radians(35.0), 0.0, math.radians(45.0))
    return light_obj


def resolve_lighting_seed(args):
    if args.hdr_seed is not None:
        return int(args.hdr_seed)
    return int(args.traj_seed + args.traj_id * 10007 + 424242)


def list_hdr_files(hdr_dir: str):
    hdr_dir = Path(hdr_dir)
    if not hdr_dir.exists():
        raise FileNotFoundError(f"HDR directory not found: {hdr_dir}")

    hdr_files = []
    for ext in ("*.exr", "*.hdr", "*.EXR", "*.HDR"):
        hdr_files.extend(hdr_dir.glob(ext))

    hdr_files = sorted(set(p.resolve() for p in hdr_files if p.is_file()))
    if len(hdr_files) == 0:
        raise RuntimeError(f"No HDR files (*.exr / *.hdr) found in directory: {hdr_dir}")

    return hdr_files


def setup_random_lighting(
    scene,
    hdr_dir: str,
    hdr_strength: float,
    seed: int,
    sunlight_prob: float = 0.05,
    sunlight_energy: float = 3.0,
):
    """
    以 sunlight_prob 的概率使用 sunlight，否则随机选择一个 HDR 环境光。

    兼容逻辑：
      - sunlight 分支：完全复刻旧版代码行为
            * 不删除 imported lights
            * 不修改 / 不接管 scene.world
            * 只额外创建一盏 SunLight
      - HDR 分支：保持新版代码行为
            * 删除 imported lights
            * 接管 world nodes
            * 随机选择一个 HDR 环境光
    """
    sunlight_prob = float(np.clip(sunlight_prob, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    use_sunlight = bool(rng.random() < sunlight_prob)

    if use_sunlight:
        # ---------------------------------------------------------------------
        # Legacy sunlight mode: exactly like the old script
        #   1) do NOT remove imported light objects
        #   2) do NOT modify scene.world
        #   3) only create one extra SunLight
        # ---------------------------------------------------------------------
        existing_lights_before = summarize_existing_light_objects()

        sun_obj = create_sun_light(energy=float(sunlight_energy))

        existing_lights_after = summarize_existing_light_objects()

        lighting_info = {
            "lighting_type": "sunlight",
            "lighting_seed": int(seed),
            "lighting_compatibility_mode": "legacy_sunlight_exact",

            "sunlight_prob": float(sunlight_prob),
            "sunlight_energy": float(sunlight_energy),
            "sunlight_name": sun_obj.name,

            "hdr_dir": str(Path(hdr_dir).resolve()),
            "selected_hdr_index": None,
            "selected_hdr_map": None,
            "selected_hdr_basename": None,
            "num_available_hdr_maps": None,
            "hdr_strength": None,

            "imported_lights_removed": False,
            "removed_imported_lights": [],
            "existing_lights_before_sunlight": existing_lights_before,
            "existing_lights_after_sunlight": existing_lights_after,

            "world_modified_for_lighting": False,
        }

        print("Lighting mode: sunlight (legacy exact)")
        print(f"  sunlight_energy = {lighting_info['sunlight_energy']}")
        print(f"  lighting_seed = {lighting_info['lighting_seed']}")
        print(f"  sunlight_prob = {lighting_info['sunlight_prob']}")
        print(f"  preserved_existing_lights = {[x['name'] for x in existing_lights_before]}")
        print(f"  created_sunlight = {sun_obj.name}")

        return lighting_info, []

    # -------------------------------------------------------------------------
    # New HDR mode: exactly like the new script
    #   1) remove imported light objects
    #   2) rebuild world nodes
    #   3) load one random HDR
    # -------------------------------------------------------------------------
    removed_lights = remove_all_light_objects()

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True

    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    hdr_files = list_hdr_files(hdr_dir)
    chosen_idx = int(rng.integers(low=0, high=len(hdr_files)))
    chosen_hdr = hdr_files[chosen_idx]

    node_texcoord = nodes.new(type="ShaderNodeTexCoord")
    node_mapping = nodes.new(type="ShaderNodeMapping")
    node_env = nodes.new(type="ShaderNodeTexEnvironment")
    node_bg = nodes.new(type="ShaderNodeBackground")
    node_out = nodes.new(type="ShaderNodeOutputWorld")

    node_texcoord.location = (-800, 0)
    node_mapping.location = (-600, 0)
    node_env.location = (-350, 0)
    node_bg.location = (-100, 0)
    node_out.location = (150, 0)

    node_env.image = bpy.data.images.load(str(chosen_hdr), check_existing=True)
    node_bg.inputs["Strength"].default_value = float(hdr_strength)

    links.new(node_texcoord.outputs["Generated"], node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_env.inputs["Vector"])
    links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
    links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])

    lighting_info = {
        "lighting_type": "random_hdr_environment_map",
        "lighting_seed": int(seed),
        "lighting_compatibility_mode": "new_hdr_exact",

        "sunlight_prob": float(sunlight_prob),
        "sunlight_energy": None,
        "sunlight_name": None,

        "hdr_dir": str(Path(hdr_dir).resolve()),
        "selected_hdr_index": int(chosen_idx),
        "selected_hdr_map": str(chosen_hdr),
        "selected_hdr_basename": chosen_hdr.name,
        "num_available_hdr_maps": int(len(hdr_files)),
        "hdr_strength": float(hdr_strength),

        "imported_lights_removed": True,
        "removed_imported_lights": list(removed_lights),

        "world_modified_for_lighting": True,
    }

    print("Lighting mode: HDR environment (new exact)")
    print(f"  hdr_dir = {lighting_info['hdr_dir']}")
    print(f"  selected_hdr_index = {lighting_info['selected_hdr_index']}")
    print(f"  selected_hdr_map = {lighting_info['selected_hdr_map']}")
    print(f"  hdr_strength = {lighting_info['hdr_strength']}")
    print(f"  lighting_seed = {lighting_info['lighting_seed']}")
    print(f"  sunlight_prob = {lighting_info['sunlight_prob']}")
    print(f"  removed_imported_lights = {removed_lights}")

    return lighting_info, removed_lights

def render_frame(output_path: str):
    scene = bpy.context.scene
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def setup_normal_output(normal_root_dir: str):
    """
    用 compositor 把 normal pass 输出到 PNG。
    返回 file_output node，后续每次 render 前动态改 base_path / prefix。
    """
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    view_layer.use_pass_normal = True

    scene.use_nodes = True
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True

    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    rlayers = nodes.new(type="CompositorNodeRLayers")
    rlayers.location = (-500, 0)

    composite = nodes.new(type="CompositorNodeComposite")
    composite.location = (350, 120)
    links.new(rlayers.outputs["Image"], composite.inputs["Image"])

    normal_mul = nodes.new(type="CompositorNodeMixRGB")
    normal_mul.blend_type = "MULTIPLY"
    normal_mul.inputs[0].default_value = 1.0
    normal_mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    normal_mul.location = (-150, -80)
    links.new(rlayers.outputs["Normal"], normal_mul.inputs[1])

    normal_add = nodes.new(type="CompositorNodeMixRGB")
    normal_add.blend_type = "ADD"
    normal_add.inputs[0].default_value = 1.0
    normal_add.inputs[2].default_value = (0.5, 0.5, 0.5, 0.0)
    normal_add.location = (80, -80)
    links.new(normal_mul.outputs["Image"], normal_add.inputs[1])

    set_alpha = nodes.new(type="CompositorNodeSetAlpha")
    set_alpha.location = (300, -80)
    links.new(normal_add.outputs["Image"], set_alpha.inputs["Image"])
    links.new(rlayers.outputs["Alpha"], set_alpha.inputs["Alpha"])

    file_output = nodes.new(type="CompositorNodeOutputFile")
    file_output.location = (550, -80)
    file_output.base_path = normal_root_dir

    slot = file_output.file_slots[0]
    slot.path = "frame_"
    slot.use_node_format = True
    slot.save_as_render = False

    file_output.format.file_format = "PNG"
    file_output.format.color_mode = "RGBA"
    file_output.format.color_depth = "16"

    links.new(set_alpha.outputs["Image"], file_output.inputs[0])

    print(f"Normal output is enabled. Files will be written under: {normal_root_dir}")
    return file_output


def update_normal_output_path(file_output_node, base_dir: str, prefix: str):
    os.makedirs(base_dir, exist_ok=True)
    file_output_node.base_path = base_dir
    file_output_node.file_slots[0].path = prefix


def export_normalized_scene_as_glb(output_glb_path: str):
    """
    导出当前场景为 GLB，包含：
      - 原始动画对象
      - baked 后的 SequenceNormalizer 动画
      - 静态 cameras
    """
    os.makedirs(os.path.dirname(output_glb_path) or ".", exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=output_glb_path,
        export_format="GLB",
        use_selection=False,
        export_cameras=True,
        export_animations=True,
        export_frame_range=True,
        export_frame_step=1,
        export_force_sampling=True,
        export_apply=False,
    )

    print(f"Normalized animated scene + cameras exported to: {output_glb_path}")


def create_emission_material(name: str, color=(1.0, 0.2, 0.2, 1.0), strength: float = 1.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new(type="ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = float(strength)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (220, 0)

    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def create_scene_box_object(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    name: str = "CanonicalSceneBox",
    thickness: float = 0.004,
    color=(1.0, 0.2, 0.2),
    emission_strength: float = 1.5,
):
    """
    创建一个渲染用的 canonical scene box（wireframe cube）。
    注意：它只是可视化，不参与 mesh 导出。
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)

    corners = get_bbox_corners(bbox_min, bbox_max)

    faces = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ]

    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(corners.tolist(), [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    wire_mod = obj.modifiers.new(name="Wireframe", type="WIREFRAME")
    wire_mod.thickness = float(thickness)
    wire_mod.use_replace = True
    if hasattr(wire_mod, "use_even_offset"):
        wire_mod.use_even_offset = True
    if hasattr(wire_mod, "use_relative_offset"):
        wire_mod.use_relative_offset = False

    rgba = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    mat = create_emission_material(
        name=name + "Mat",
        color=rgba,
        strength=float(emission_strength),
    )
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    obj.hide_render = True
    obj.hide_viewport = False
    return obj


def cleanup_temp_render_file(base_path_no_ext: str):
    candidates = [
        base_path_no_ext,
        base_path_no_ext + ".png",
        base_path_no_ext + ".PNG",
    ]
    for p in candidates:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def render_rgb_and_normal(
    rgb_path: str,
    normal_output_node,
    scene_box_obj=None,
    render_scene_box: bool = False,
    scene_box_affect_normal: bool = False,
):
    """
    默认行为：
      - 不开 scene box: 一次 render，同时得到 rgb + normal
      - 开 scene box 且不想污染 normal:
            第一次：scene box 可见，只写 rgb
            第二次：scene box 隐藏，只写 normal（会产生一个临时 rgb，再删掉）
      - 开 scene box 且允许进入 normal:
            一次 render，同时得到 rgb + normal
    """
    if (scene_box_obj is None) or (not render_scene_box):
        if scene_box_obj is not None:
            scene_box_obj.hide_render = True
        render_frame(rgb_path)
        return

    if scene_box_affect_normal:
        scene_box_obj.hide_render = False
        render_frame(rgb_path)
        scene_box_obj.hide_render = True
        return

    prev_mute = bool(normal_output_node.mute)

    scene_box_obj.hide_render = False
    normal_output_node.mute = True
    render_frame(rgb_path)

    scene_box_obj.hide_render = True
    normal_output_node.mute = prev_mute

    tmp_rgb_base = os.path.splitext(rgb_path)[0] + "__normal_only__"
    render_frame(tmp_rgb_base)
    cleanup_temp_render_file(tmp_rgb_base)


# =================================================================================
#  5. STATIC MULTI-VIEW CAMERAS (CAMERA-SPACE TIGHT FIT)
# =================================================================================

def get_bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray):
    x0, y0, z0 = bbox_min.tolist()
    x1, y1, z1 = bbox_max.tolist()
    corners = np.array([
        [x0, y0, z0],
        [x0, y0, z1],
        [x0, y1, z0],
        [x0, y1, z1],
        [x1, y0, z0],
        [x1, y0, z1],
        [x1, y1, z0],
        [x1, y1, z1],
    ], dtype=np.float32)
    return corners


def compute_camera_axes_from_angles(azim: float, elev: float):
    """
    给定 orbit azim/elev，返回相机坐标系在 world 中的:
      right, up, forward
    其中 forward 是“相机看向 target”的方向。
    """
    cam_pos_unit = orbit_offset(1.0, azim, elev)
    forward = normalize(-cam_pos_unit)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = normalize(np.cross(forward, world_up))
    if np.linalg.norm(right) < 1e-6:
        up_alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, up_alt))

    true_up = normalize(np.cross(right, forward))
    return right, true_up, forward


def compute_world_to_camera_aligned_rotation(azim: float, elev: float):
    """
    返回一个 3x3 矩阵 R，使得:
        p_cam_aligned = R @ p_world
    其中 p_cam_aligned 的三个坐标轴分别是:
        x = right
        y = up
        z = forward
    """
    right, up, forward = compute_camera_axes_from_angles(azim, elev)
    R = np.stack([right, up, forward], axis=0).astype(np.float32)
    return R, right, up, forward


def camera_aligned_bbox_corners_to_world(
    bbox_min_cam: np.ndarray,
    bbox_max_cam: np.ndarray,
    azim: float,
    elev: float,
):
    """
    把 camera-aligned bbox 的 8 个角点映射回 world，用于 debug 投影。
    """
    corners_cam = get_bbox_corners(
        np.asarray(bbox_min_cam, dtype=np.float32),
        np.asarray(bbox_max_cam, dtype=np.float32),
    )
    right, up, forward = compute_camera_axes_from_angles(azim, elev)
    basis_cols = np.stack([right, up, forward], axis=1).astype(np.float32)  # [3,3]
    corners_world = corners_cam @ basis_cols.T
    return corners_world.astype(np.float32)


def compute_camera_space_scene_box_from_vertices_seq(
    vertices_seq_list: List[np.ndarray],
    rot_world_to_cam_aligned: np.ndarray,
):
    """
    对整个 normalized 序列，旋转到当前 view 的 camera-aligned space，
    并聚合出序列级别的 scene box。
    """
    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    rot_t = rot_world_to_cam_aligned.T.astype(np.float32)

    for verts in vertices_seq_list:
        verts32 = np.asarray(verts, dtype=np.float32)
        verts_cam = verts32 @ rot_t
        global_min = np.minimum(global_min, verts_cam.min(axis=0))
        global_max = np.maximum(global_max, verts_cam.max(axis=0))

    return global_min.astype(np.float32), global_max.astype(np.float32)


def compute_bbox_unit_normalization_scale(bbox_min: np.ndarray, bbox_max: np.ndarray):
    """
    对 axis-aligned bbox 求一个 uniform scale，使其最大边长归一化到 1。
    """
    extent = np.asarray(bbox_max, dtype=np.float32) - np.asarray(bbox_min, dtype=np.float32)
    max_extent = float(np.max(extent))
    scale = 1.0 / max_extent if max_extent > 1e-6 else 1.0
    return float(scale), extent.astype(np.float32)


def compute_tight_camera_distance_for_aligned_bbox(
    cam_obj,
    bbox_min_aligned: np.ndarray,
    bbox_max_aligned: np.ndarray,
    frame_padding: float = 0.03,
    fit_safety: float = 1.02,
):
    """
    对“已经处于 camera-aligned space 的 bbox”求 tight camera distance。
    这里不再需要 azim/elev，因为 bbox 已经被旋到了该 camera 的坐标系下。

    假设 camera 位于 +distance * cam_pos_unit 方向，物体中心仍在 world 原点。
    对应到 aligned 坐标中，每个点 p=(x,y,z)，相机看到的深度是 (distance - z)。
    """
    bbox_min_aligned = np.asarray(bbox_min_aligned, dtype=np.float32)
    bbox_max_aligned = np.asarray(bbox_max_aligned, dtype=np.float32)
    corners = get_bbox_corners(bbox_min_aligned, bbox_max_aligned)

    half_fov_x = 0.5 * float(cam_obj.data.angle_x)
    half_fov_y = 0.5 * float(cam_obj.data.angle_y)

    tan_half_fov_x = max(math.tan(half_fov_x), 1e-6)
    tan_half_fov_y = max(math.tan(half_fov_y), 1e-6)

    fill_ratio = max(1e-3, 1.0 - float(frame_padding))
    d_required = 0.0

    for p in corners:
        px = abs(float(p[0]))
        py = abs(float(p[1]))
        pz = float(p[2])

        req_x = px / (fill_ratio * tan_half_fov_x) - pz
        req_y = py / (fill_ratio * tan_half_fov_y) - pz
        d_required = max(d_required, req_x, req_y)

    min_pz = float(np.min(corners[:, 2]))
    d_required = max(d_required, -min_pz + 1e-4)

    d_required = max(d_required * float(fit_safety), 1e-4)
    return float(d_required)


def create_static_multiview_cameras(
    num_cameras: int,
    seed: int,
    normalized_vertices_seq: List[np.ndarray],
    resolution: int,
    elev_min_deg: float = 0.0,
    elev_max_deg: float = 80.0,
    frame_padding: float = 0.03,
    fit_safety: float = 1.02,
    distance_jitter_scale: float = 1.04,
    randomize_camera_intrinsics: bool = False,
    camera_fov_min_deg: float = 35.0,
    camera_fov_max_deg: float = 70.0,
    camera_sensor_size: float = 36.0,
):
    """
    固定多视角相机：
      - 第 0 个 azimuth 随机
      - 后续 azimuth 均匀分布
      - 每个 camera 的 elevation ~ U([elev_min_deg, elev_max_deg])
      - 每个 camera 的 distance:
            1) 把整个 normalized sequence 旋转到 camera-aligned space
            2) 聚合出 scene_box_camera_space
            3) 基于该 box 再做一个 view-specific normalization
            4) 在该 normalized camera-space box 上求 tight distance
            5) 再把 distance 映射回共享 world-normalized scene
      - 全部看向原点 (0,0,0)

    额外支持：
      - 每个 camera 随机一个共同 FOV，保持 fovx = fovy
    """
    rng = np.random.default_rng(seed)
    distance_jitter_scale = max(1.0, float(distance_jitter_scale))

    if camera_fov_min_deg > camera_fov_max_deg:
        raise ValueError("camera_fov_min_deg must be <= camera_fov_max_deg")

    azim0 = float(rng.uniform(0.0, 2.0 * math.pi))
    azims = azim0 + np.arange(num_cameras, dtype=np.float32) * (2.0 * math.pi / num_cameras)
    elevs_deg = rng.uniform(elev_min_deg, elev_max_deg, size=num_cameras).astype(np.float32)

    camera_objs = []
    camera_infos = []
    target = np.zeros(3, dtype=np.float32)

    for k in range(num_cameras):
        cam_obj = create_camera(
            name=f"TrackingCamera_{k:02d}",
            lens=50.0,
            sensor_width=float(camera_sensor_size),
            sensor_height=float(camera_sensor_size),
        )

        if randomize_camera_intrinsics:
            sampled_fov_deg = float(rng.uniform(camera_fov_min_deg, camera_fov_max_deg))
            set_camera_intrinsics_from_fov(
                cam_obj,
                fov_deg=sampled_fov_deg,
                sensor_size=float(camera_sensor_size),
            )
        else:
            sampled_fov_deg = float(math.degrees(cam_obj.data.angle_x))
            cam_obj.data.sensor_fit = "HORIZONTAL"
            cam_obj.data.sensor_width = float(camera_sensor_size)
            cam_obj.data.sensor_height = float(camera_sensor_size)

        azim = float(azims[k] % (2.0 * math.pi))
        elev_deg = float(elevs_deg[k])
        elev = math.radians(elev_deg)

        rot_world_to_cam_aligned, right, up, forward = compute_world_to_camera_aligned_rotation(
            azim=azim,
            elev=elev,
        )

        camera_space_bbox_min, camera_space_bbox_max = compute_camera_space_scene_box_from_vertices_seq(
            normalized_vertices_seq,
            rot_world_to_cam_aligned=rot_world_to_cam_aligned,
        )

        view_specific_scale, camera_space_extent = compute_bbox_unit_normalization_scale(
            camera_space_bbox_min,
            camera_space_bbox_max,
        )

        normalized_camera_space_bbox_min = (camera_space_bbox_min * view_specific_scale).astype(np.float32)
        normalized_camera_space_bbox_max = (camera_space_bbox_max * view_specific_scale).astype(np.float32)

        tight_distance_in_view_normalized_space = compute_tight_camera_distance_for_aligned_bbox(
            cam_obj=cam_obj,
            bbox_min_aligned=normalized_camera_space_bbox_min,
            bbox_max_aligned=normalized_camera_space_bbox_max,
            frame_padding=frame_padding,
            fit_safety=fit_safety,
        )

        tight_distance = float(
            tight_distance_in_view_normalized_space / max(view_specific_scale, 1e-8)
        )

        distance = tight_distance * float(rng.uniform(1.0, distance_jitter_scale))

        cam_pos = orbit_offset(distance, azim, elev)
        cam2world = look_at(cam_pos, target)
        set_camera_from_cam2world(cam_obj, cam2world)

        intrinsics = get_camera_intrinsics_dict(cam_obj, resolution=resolution)

        camera_info = {
            "camera_name": cam_obj.name,
            "view_index": int(k),
            "azimuth_deg": float(np.degrees(azim)),
            "elevation_deg": float(elev_deg),

            "camera_space_scene_box_min": camera_space_bbox_min.astype(np.float32).tolist(),
            "camera_space_scene_box_max": camera_space_bbox_max.astype(np.float32).tolist(),
            "camera_space_scene_box_extent": camera_space_extent.astype(np.float32).tolist(),

            "camera_space_sequence_scale_for_unit_box": float(view_specific_scale),
            "normalized_camera_space_scene_box_min": normalized_camera_space_bbox_min.tolist(),
            "normalized_camera_space_scene_box_max": normalized_camera_space_bbox_max.tolist(),

            "tight_fit_distance_in_view_normalized_space": float(tight_distance_in_view_normalized_space),
            "tight_fit_distance": float(tight_distance),
            "distance": float(distance),
            "distance_scale_from_tight_fit": float(distance / max(tight_distance, 1e-8)),

            "world_to_camera_aligned_rotation": rot_world_to_cam_aligned.astype(np.float32).tolist(),
            "camera_right_world": right.astype(np.float32).tolist(),
            "camera_up_world": up.astype(np.float32).tolist(),
            "camera_forward_world": forward.astype(np.float32).tolist(),

            "camera_c2w": cam2world.astype(np.float32).tolist(),
            "camera_pos": cam_pos.astype(np.float32).tolist(),
            "camera_target": target.astype(np.float32).tolist(),

            "sampled_common_fov_deg": float(sampled_fov_deg),
            "intrinsics": intrinsics,

            "distance_estimation_method": (
                "rotate_normalized_sequence_to_camera_space -> "
                "aggregate_sequence_bbox_in_camera_space -> "
                "view_specific_bbox_normalization -> "
                "tight_distance_on_normalized_camera_space_bbox"
            ),
        }

        camera_objs.append(cam_obj)
        camera_infos.append(camera_info)

    return camera_objs, camera_infos


# =================================================================================
#  6. CORE WORKER
# =================================================================================

def process_geometry(args):
    print("--- Starting Geometry Processing ---")

    output_file = args.output_file
    normalized_glb_path = args.normalized_glb_path.strip()

    rgb_dir = os.path.splitext(output_file)[0] + "_rgb"
    normal_dir = os.path.splitext(output_file)[0] + "_normal"
    mesh_npz_path = os.path.splitext(output_file)[0] + "_mesh.npz"
    mesh_ply_dir = os.path.splitext(output_file)[0] + "_mesh_ply"

    json_exists = os.path.exists(output_file)
    glb_exists = (normalized_glb_path == "") or os.path.exists(normalized_glb_path)
    mesh_npz_exists = os.path.exists(mesh_npz_path)
    ply_dir_exists = (not args.export_keyframe_ply) or os.path.isdir(mesh_ply_dir)

    if (not args.render_scene_box) and json_exists and glb_exists and mesh_npz_exists and ply_dir_exists:
        print(
            f"✅ Skipped (requested outputs already exist): "
            f"json={output_file}, glb={normalized_glb_path or 'N/A'}, "
            f"mesh_npz={mesh_npz_path}, mesh_ply_dir={mesh_ply_dir if args.export_keyframe_ply else 'disabled'}"
        )
        return

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    if args.export_keyframe_ply:
        os.makedirs(mesh_ply_dir, exist_ok=True)

    print(f"Loading object from: {args.object_path}")
    init_scene()
    load_object(args.object_path)

    # removed_lights = remove_all_light_objects()

    sequence_normalizer = create_sequence_normalizer()
    print("Sequence normalizer created.")

    # if args.render_engine == "CYCLES":
    #     enable_cycles_acceleration(args.cycles_backend)

    setup_renderer(
        resolution=args.resolution,
        engine=args.render_engine,
        transparent_bg=args.transparent_bg,
    )

    lighting_seed = resolve_lighting_seed(args)
    lighting_info, removed_lights = setup_random_lighting(
        scene=bpy.context.scene,
        hdr_dir=args.hdr_dir,
        hdr_strength=args.hdr_strength,
        seed=lighting_seed,
        sunlight_prob=args.sunlight_prob,
        sunlight_energy=args.sunlight_energy,
    )

    normal_output_node = setup_normal_output(normal_dir)

    mesh_objs = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render:
            continue
        if not obj.visible_get(view_layer=bpy.context.view_layer):
            continue
        mesh_objs.append(obj)

    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after loading.")

    print("Found mesh objects:")
    for obj in mesh_objs:
        print(
            f"  name={obj.name}, "
            f"type={obj.type}, "
            f"hide_render={obj.hide_render}, "
            f"hide_viewport={obj.hide_viewport}, "
            f"visible_get={obj.visible_get(view_layer=bpy.context.view_layer)}"
        )

    actions = bpy.data.actions
    frame_start, frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        frame_start = int(min(r[0] for r in ranges))
        frame_end = int(max(r[1] for r in ranges))

    frame_indices = np.arange(frame_start, frame_end + 1, dtype=np.int32)

    keyframe_frames = collect_keyframe_frames(frame_start, frame_end)
    if len(keyframe_frames) == 0:
        print("Warning: no explicit keyframes found in actions/fcurves. No per-keyframe PLY will be exported.")
    else:
        print(f"Detected {len(keyframe_frames)} keyframes: {keyframe_frames}")
    keyframe_frame_set = set(keyframe_frames)

    shared_faces, centers_seq, sequence_scale, canonical_bbox_min, canonical_bbox_max = \
        compute_sequence_normalization_params(mesh_objs, frame_indices)

    print("Sequence normalization:")
    print(f"  sequence_scale = {sequence_scale}")
    print(f"  canonical_bbox_min = {canonical_bbox_min.tolist()}")
    print(f"  canonical_bbox_max = {canonical_bbox_max.tolist()}")

    print("Precomputing normalized mesh sequence for camera fitting / mesh export...")
    vertices_seq_list, keyframe_ply_records = precompute_normalized_mesh_sequence(
        mesh_objs=mesh_objs,
        frame_indices=frame_indices,
        normalizer_obj=sequence_normalizer,
        centers_seq=centers_seq,
        global_scale=sequence_scale,
        shared_faces=shared_faces,
        export_keyframe_ply=args.export_keyframe_ply,
        keyframe_frame_set=keyframe_frame_set,
        mesh_ply_dir=mesh_ply_dir,
    )

    camera_seed = int(args.traj_seed + args.traj_id * 9973)
    camera_objs, camera_infos = create_static_multiview_cameras(
        num_cameras=args.num_cameras,
        seed=camera_seed,
        normalized_vertices_seq=vertices_seq_list,
        resolution=args.resolution,
        elev_min_deg=args.camera_elev_min_deg,
        elev_max_deg=args.camera_elev_max_deg,
        frame_padding=args.camera_frame_padding,
        fit_safety=args.camera_fit_safety,
        distance_jitter_scale=args.camera_distance_jitter_scale,
        randomize_camera_intrinsics=args.randomize_camera_intrinsics,
        camera_fov_min_deg=args.camera_fov_min_deg,
        camera_fov_max_deg=args.camera_fov_max_deg,
        camera_sensor_size=args.camera_sensor_size,
    )

    bpy.context.scene.camera = camera_objs[0]

    print("Static multi-view camera setup:")
    for info in camera_infos:
        intr = info["intrinsics"]
        print(
            f"  view {info['view_index']:02d}: "
            f"azim={info['azimuth_deg']:.2f} deg, "
            f"elev={info['elevation_deg']:.2f} deg, "
            f"tight_dist={info['tight_fit_distance']:.4f}, "
            f"dist={info['distance']:.4f}, "
            f"view_scale={info['camera_space_sequence_scale_for_unit_box']:.4f}, "
            f"fovx={intr['fov_x_deg']:.2f}, "
            f"fovy={intr['fov_y_deg']:.2f}, "
            f"fx={intr['fx_px']:.2f}, "
            f"fy={intr['fy_px']:.2f}"
        )

    for view_idx in range(args.num_cameras):
        os.makedirs(os.path.join(rgb_dir, f"view_{view_idx:02d}"), exist_ok=True)
        os.makedirs(os.path.join(normal_dir, f"view_{view_idx:02d}"), exist_ok=True)

    if normalized_glb_path:
        print("Baking sequence normalization to keyframes for GLB export...")
        bake_sequence_normalization_to_keyframes(
            normalizer_obj=sequence_normalizer,
            frame_indices=frame_indices,
            centers_seq=centers_seq,
            global_scale=sequence_scale,
        )

        bpy.context.scene.frame_set(int(frame_indices[0]))
        bpy.context.view_layer.update()
        export_normalized_scene_as_glb(normalized_glb_path)

    scene_box_obj = None
    if args.render_scene_box:
        scene_box_obj = create_scene_box_object(
            bbox_min=canonical_bbox_min,
            bbox_max=canonical_bbox_max,
            name="CanonicalSceneBox",
            thickness=args.scene_box_thickness,
            color=tuple(args.scene_box_color),
            emission_strength=args.scene_box_emission_strength,
        )
        print("Scene box rendering is enabled.")
        print(f"  scene_box_thickness = {args.scene_box_thickness}")
        print(f"  scene_box_color = {args.scene_box_color}")
        print(f"  scene_box_emission_strength = {args.scene_box_emission_strength}")
        print(f"  scene_box_affect_normal = {args.scene_box_affect_normal}")
        print("  note: rendered scene box is still the shared world-canonical box.")

    final_watertight_data = {}
    from tqdm import tqdm

    for local_frame_idx, source_frame in enumerate(tqdm(frame_indices, desc="Rendering frames")):
        frame_int = int(source_frame)
        frame_key = f"frame_{frame_int:04d}"
        print(f"Processing {frame_key} ({local_frame_idx + 1}/{len(frame_indices)})")

        bpy.context.scene.frame_set(frame_int)

        apply_sequence_normalization(
            sequence_normalizer,
            frame_center=centers_seq[local_frame_idx],
            global_scale=sequence_scale,
        )
        bpy.context.view_layer.update()

        mesh_ply_path = None
        is_keyframe = frame_int in keyframe_frame_set
        if args.export_keyframe_ply and is_keyframe:
            mesh_ply_path = keyframe_ply_records.get(frame_int, None)

        views = []

        for view_idx, (camera_obj, camera_info) in enumerate(zip(camera_objs, camera_infos)):
            bpy.context.scene.camera = camera_obj

            if args.debug_camera_projection and local_frame_idx == 0:
                debug_project_camera_space_bbox(
                    camera_obj,
                    np.asarray(camera_info["camera_space_scene_box_min"], dtype=np.float32),
                    np.asarray(camera_info["camera_space_scene_box_max"], dtype=np.float32),
                    azim=math.radians(float(camera_info["azimuth_deg"])),
                    elev=math.radians(float(camera_info["elevation_deg"])),
                )

            view_rgb_dir = os.path.join(rgb_dir, f"view_{view_idx:02d}")
            view_normal_dir = os.path.join(normal_dir, f"view_{view_idx:02d}")

            rgb_path = os.path.join(view_rgb_dir, f"frame_{frame_int:04d}.png")

            normal_prefix = "frame_"
            update_normal_output_path(
                normal_output_node,
                base_dir=view_normal_dir,
                prefix=normal_prefix,
            )
            normal_path = os.path.join(view_normal_dir, f"frame_{frame_int:04d}.png")

            render_rgb_and_normal(
                rgb_path=rgb_path,
                normal_output_node=normal_output_node,
                scene_box_obj=scene_box_obj,
                render_scene_box=args.render_scene_box,
                scene_box_affect_normal=args.scene_box_affect_normal,
            )

            views.append({
                "view_index": int(view_idx),
                "rgb_path": rgb_path,
                "normal_path": normal_path,
                "camera_c2w": camera_info["camera_c2w"],
                "camera_pos": camera_info["camera_pos"],
                "camera_target": camera_info["camera_target"],
                "azimuth_deg": camera_info["azimuth_deg"],
                "elevation_deg": camera_info["elevation_deg"],

                "camera_space_scene_box_min": camera_info["camera_space_scene_box_min"],
                "camera_space_scene_box_max": camera_info["camera_space_scene_box_max"],
                "camera_space_sequence_scale_for_unit_box": camera_info["camera_space_sequence_scale_for_unit_box"],
                "normalized_camera_space_scene_box_min": camera_info["normalized_camera_space_scene_box_min"],
                "normalized_camera_space_scene_box_max": camera_info["normalized_camera_space_scene_box_max"],

                "tight_fit_distance_in_view_normalized_space": camera_info["tight_fit_distance_in_view_normalized_space"],
                "tight_fit_distance": camera_info["tight_fit_distance"],
                "distance": camera_info["distance"],
                "distance_scale_from_tight_fit": camera_info["distance_scale_from_tight_fit"],

                "sampled_common_fov_deg": camera_info["sampled_common_fov_deg"],
                "intrinsics": camera_info["intrinsics"],
                "distance_estimation_method": camera_info["distance_estimation_method"],
            })

        final_watertight_data[frame_key] = {
            "mesh_npz_path": mesh_npz_path,
            "mesh_frame_index": int(local_frame_idx),
            "frame_index": frame_int,
            "is_keyframe": bool(is_keyframe),
            "mesh_ply_path": mesh_ply_path,
            "frame_bbox_center_before_normalization": centers_seq[local_frame_idx].astype(np.float32).tolist(),
            "canonical_bbox_center": [0.0, 0.0, 0.0],
            "views": views,
        }

    vertices_seq = np.stack(vertices_seq_list, axis=0)

    print(f"Saving mesh sequence to {mesh_npz_path} ...")
    os.makedirs(os.path.dirname(mesh_npz_path) or ".", exist_ok=True)
    np.savez_compressed(
        mesh_npz_path,
        vertices=vertices_seq,
        faces=shared_faces,
        frame_indices=frame_indices,
    )
    print(
        f"Mesh npz saved. vertices.shape={vertices_seq.shape}, dtype={vertices_seq.dtype}; "
        f"faces.shape={shared_faces.shape}, dtype={shared_faces.dtype}"
    )

    final_watertight_data["_global"] = {
        "mesh_npz_path": mesh_npz_path,
        "mesh_ply_dir": mesh_ply_dir if args.export_keyframe_ply else None,
        "keyframe_frames": [int(x) for x in keyframe_frames],
        "num_keyframe_mesh_ply": int(len(keyframe_ply_records)),
        "normalized_glb_path": normalized_glb_path if normalized_glb_path else None,
        "num_frames": int(vertices_seq.shape[0]),
        "num_vertices": int(vertices_seq.shape[1]),
        "num_faces": int(shared_faces.shape[0]),
        "vertices_dtype": str(vertices_seq.dtype),
        "faces_dtype": str(shared_faces.dtype),
        "image_storage_layout": "per_camera_dir/frame_png",

        "root_body_motion_removed_by": "per-frame bbox-center translation",

        "sequence_global_scale": float(sequence_scale),
        "canonical_bbox_min": canonical_bbox_min.tolist(),
        "canonical_bbox_max": canonical_bbox_max.tolist(),

        "num_cameras": int(args.num_cameras),
        "camera_seed": int(camera_seed),
        "camera_elev_min_deg": float(args.camera_elev_min_deg),
        "camera_elev_max_deg": float(args.camera_elev_max_deg),
        "camera_frame_padding": float(args.camera_frame_padding),
        "camera_fit_safety": float(args.camera_fit_safety),
        "camera_distance_jitter_scale": float(args.camera_distance_jitter_scale),

        "camera_distance_method": (
            "per-view camera-space sequence bbox + "
            "view-specific bbox normalization + "
            "tight distance estimation"
        ),

        "randomize_camera_intrinsics": bool(args.randomize_camera_intrinsics),
        "camera_fov_min_deg": float(args.camera_fov_min_deg),
        "camera_fov_max_deg": float(args.camera_fov_max_deg),
        "camera_sensor_size_mm": float(args.camera_sensor_size),

        "static_cameras": camera_infos,

        "render_scene_box": bool(args.render_scene_box),
        "scene_box_affect_normal": bool(args.scene_box_affect_normal),
        "scene_box_thickness": float(args.scene_box_thickness),
        "scene_box_color": [float(x) for x in args.scene_box_color],
        "scene_box_emission_strength": float(args.scene_box_emission_strength),
        "rendered_scene_box_space": "shared_world_canonical_bbox",
        "bbox_min": canonical_bbox_min.astype(np.float32).tolist(),
        "bbox_max": canonical_bbox_max.astype(np.float32).tolist(),

        "removed_imported_lights": removed_lights,
        "lighting": lighting_info,
    }

    print(f"Saving final metadata to {output_file}...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as fw:
        fw.write(json.dumps(final_watertight_data, indent=2))
    print("--- Processing Complete ---")


# =================================================================================
#  7. MAIN
# =================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-view rendered frames, shared-topology mesh npz, per-keyframe mesh ply, and optionally a normalized GLB from animated GLB/GLTF files using Blender."
    )
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the source .glb/.gltf file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output metadata .json file.",
    )
    parser.add_argument(
        "--normalized_glb_path",
        type=str,
        default="",
        help="Optional path to export the normalized animated scene and static cameras as a new .glb file.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Render resolution for output PNGs. This script uses square output: resolution x resolution.",
    )
    parser.add_argument(
        "--render_engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["BLENDER_EEVEE", "CYCLES"],
        help="Blender render engine.",
    )
    parser.add_argument(
        "--transparent_bg",
        action="store_true",
        help="Render with transparent background.",
    )

    parser.add_argument(
        "--hdr_dir",
        type=str,
        default="data/hdr",
        help="Directory containing HDR environment maps (*.exr / *.hdr).",
    )
    parser.add_argument(
        "--hdr_strength",
        type=float,
        default=1.0,
        help="Strength of the HDR environment lighting.",
    )
    parser.add_argument(
        "--hdr_seed",
        type=int,
        default=None,
        help="Optional lighting random seed. If not provided, it is derived from traj_seed and traj_id.",
    )
    parser.add_argument(
        "--sunlight_prob",
        type=float,
        default=0.05,
        help="Probability of using sunlight instead of HDR environment lighting.",
    )
    parser.add_argument(
        "--sunlight_energy",
        type=float,
        default=3.0,
        help="Energy of the sunlight when sunlight lighting mode is selected.",
    )

    parser.add_argument(
        "--traj_id",
        type=int,
        default=0,
        help="Used only to derive a reproducible camera seed.",
    )
    parser.add_argument(
        "--traj_seed",
        type=int,
        default=0,
        help="Base random seed for static multi-view camera generation.",
    )

    parser.add_argument(
        "--num_cameras",
        type=int,
        default=16,
        help="Number of static cameras per frame.",
    )
    parser.add_argument(
        "--camera_elev_min_deg",
        type=float,
        default=0.0,
        help="Minimum elevation in degrees for static cameras.",
    )
    parser.add_argument(
        "--camera_elev_max_deg",
        type=float,
        default=80.0,
        help="Maximum elevation in degrees for static cameras.",
    )
    parser.add_argument(
        "--camera_frame_padding",
        type=float,
        default=0.03,
        help="Fractional padding to leave around the object in the image. Smaller => object fills more of the frame.",
    )
    parser.add_argument(
        "--camera_fit_safety",
        type=float,
        default=1.02,
        help="Small multiplicative safety factor for tight-fit camera distance.",
    )
    parser.add_argument(
        "--camera_distance_jitter_scale",
        type=float,
        default=1.04,
        help="Extra outward distance jitter relative to the tight-fit distance. 1.0 means no jitter.",
    )

    parser.add_argument(
        "--randomize_camera_intrinsics",
        action="store_true",
        help="If set, each camera samples its own common FOV while keeping fovx = fovy.",
    )
    parser.add_argument(
        "--camera_fov_min_deg",
        type=float,
        default=35.0,
        help="Minimum sampled common FOV in degrees when randomizing camera intrinsics.",
    )
    parser.add_argument(
        "--camera_fov_max_deg",
        type=float,
        default=70.0,
        help="Maximum sampled common FOV in degrees when randomizing camera intrinsics.",
    )
    parser.add_argument(
        "--camera_sensor_size",
        type=float,
        default=36.0,
        help="Square sensor size in mm. sensor_width = sensor_height = this value.",
    )

    parser.add_argument(
        "--export_keyframe_ply",
        action="store_true",
        help="If set, export one normalized mesh PLY for each detected keyframe.",
    )

    parser.add_argument(
        "--cycles_backend",
        type=str,
        default="OPTIX",
        choices=["CUDA", "OPTIX"],
        help="Cycles GPU backend.",
    )

    parser.add_argument(
        "--render_scene_box",
        action="store_true",
        help="If set, render the canonical normalized scene box as a wireframe cube.",
    )
    parser.add_argument(
        "--scene_box_affect_normal",
        action="store_true",
        help="If set, the rendered scene box will also appear in normal maps. Default is False.",
    )
    parser.add_argument(
        "--scene_box_thickness",
        type=float,
        default=0.004,
        help="Wireframe thickness of the rendered scene box in canonical scene units.",
    )
    parser.add_argument(
        "--scene_box_color",
        type=float,
        nargs=3,
        default=[1.0, 0.2, 0.2],
        metavar=("R", "G", "B"),
        help="RGB color of the rendered scene box.",
    )
    parser.add_argument(
        "--scene_box_emission_strength",
        type=float,
        default=1.5,
        help="Emission strength of the scene box material.",
    )

    parser.add_argument(
        "--debug_camera_projection",
        action="store_true",
        help="If set, print projected camera-space sequence bbox corners for each camera on the first frame.",
    )

    args = parser.parse_args(get_cli_argv())
    process_geometry(args)