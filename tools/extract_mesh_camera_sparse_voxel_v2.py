#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: process_geometry_with_bpy.py
# Description:
#   1) Load an animated .glb / .gltf in Blender
#   2) Remove root-body translation per frame by subtracting each frame's bbox center
#   3) Rescale the whole sequence so all shapes fit inside a unit bounding box
#   4) Render static multi-view cameras per frame
#   5) Save one shared-topology mesh npz: faces once + per-frame vertices
#   6) Optionally export the normalized animated scene + static cameras as a new GLB
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
# Example:
# blender --background --python process_geometry_with_bpy.py -- \
#   --object_path /path/to/model.glb \
#   --output_file /path/to/result.json \
#   --normalized_glb_path /path/to/result_normalized.glb \
#   --resolution 512 \
#   --transparent_bg \
#   --camera_frame_padding 0.01 \
#   --camera_fit_safety 1.01 \
#   --camera_distance_jitter_scale 1.0

import argparse
import sys
import os
import math
import json
from typing import Dict, Callable

import bpy
import numpy as np
from mathutils import Matrix


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


def load_object(object_path: str) -> None:
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    IMPORT_FUNCTIONS[file_extension](filepath=object_path)


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


def create_camera(name="TrackingCamera", lens=50.0, sensor_width=36.0):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.sensor_width = sensor_width

    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def set_camera_from_cam2world(cam_obj, cam2world: np.ndarray):
    cam_obj.matrix_world = Matrix(cam2world.tolist())


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


# =================================================================================
#  3. SEQUENCE NORMALIZATION
# =================================================================================

def create_sequence_normalizer():
    """
    创建一个空物体作为额外父节点。
    后续逐帧设置其 scale / translation，实现：
        x' = s * (x - center_t)
    即：
      - 去掉每帧的 root-body translation（这里用 bbox center 近似）
      - 全序列统一 scale 到 unit bounding box
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

    Returns:
        shared_faces: [F, 3]
        centers_seq: [T, 3]
        sequence_scale: float
        canonical_bbox_min: [3]   # 已乘 scale 后
        canonical_bbox_max: [3]   # 已乘 scale 后
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


# =================================================================================
#  4. RENDERING
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
    light_data.energy = energy
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    light_obj.location = (2.0, -2.0, 3.0)
    light_obj.rotation_euler = (math.radians(35), 0.0, math.radians(45))
    return light_obj


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


# =================================================================================
#  5. STATIC MULTI-VIEW CAMERAS (TIGHT FIT)
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


def compute_tight_camera_distance_for_bbox(
    cam_obj,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    azim: float,
    elev: float,
    frame_padding: float = 0.03,
    fit_safety: float = 1.02,
):
    """
    对当前视角，计算“刚好能把 canonical bbox 塞进画面”的相机距离。
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    corners = get_bbox_corners(bbox_min, bbox_max)
    print(bbox_min, bbox_max)
    print(corners)

    right, up, forward = compute_camera_axes_from_angles(azim, elev)

    half_fov_x = 0.5 * float(cam_obj.data.angle_x)
    half_fov_y = 0.5 * float(cam_obj.data.angle_y)

    tan_half_fov_x = max(math.tan(half_fov_x), 1e-6)
    tan_half_fov_y = max(math.tan(half_fov_y), 1e-6)

    fill_ratio = max(1e-3, 1.0 - float(frame_padding))

    d_required = 0.0

    for p in corners:
        px = abs(float(np.dot(p, right)))
        py = abs(float(np.dot(p, up)))
        pz = float(np.dot(p, forward))

        req_x = px / (fill_ratio * tan_half_fov_x) - pz
        req_y = py / (fill_ratio * tan_half_fov_y) - pz
        d_required = max(d_required, req_x, req_y)

    min_pz = float(np.min(corners @ forward))
    d_required = max(d_required, -min_pz + 1e-4)

    d_required = max(d_required * float(fit_safety), 1e-4)
    return float(d_required)


def create_static_multiview_cameras(
    num_cameras: int,
    seed: int,
    canonical_bbox_min: np.ndarray,
    canonical_bbox_max: np.ndarray,
    elev_min_deg: float = 0.0,
    elev_max_deg: float = 80.0,
    frame_padding: float = 0.03,
    fit_safety: float = 1.02,
    distance_jitter_scale: float = 1.04,
):
    """
    固定多视角相机：
      - 第 0 个 azimuth 随机
      - 后续 azimuth 均匀分布
      - 每个 camera 的 elevation ~ U([elev_min_deg, elev_max_deg])
      - 每个 camera 的 distance 基于 canonical bbox 做 tight fit
      - distance 只允许少量向外抖动
      - 全部看向原点 (0,0,0)
    """
    rng = np.random.default_rng(seed)
    distance_jitter_scale = max(1.0, float(distance_jitter_scale))

    camera_objs = [create_camera(name=f"TrackingCamera_{k:02d}") for k in range(num_cameras)]

    azim0 = float(rng.uniform(0.0, 2.0 * math.pi))
    azims = azim0 + np.arange(num_cameras, dtype=np.float32) * (2.0 * math.pi / num_cameras)
    elevs_deg = rng.uniform(elev_min_deg, elev_max_deg, size=num_cameras).astype(np.float32)

    target = np.zeros(3, dtype=np.float32)
    camera_infos = []

    for k, cam_obj in enumerate(camera_objs):
        azim = float(azims[k] % (2.0 * math.pi))
        elev_deg = float(elevs_deg[k])
        elev = math.radians(elev_deg)

        tight_distance = compute_tight_camera_distance_for_bbox(
            cam_obj=cam_obj,
            bbox_min=canonical_bbox_min,
            bbox_max=canonical_bbox_max,
            azim=azim,
            elev=elev,
            frame_padding=frame_padding,
            fit_safety=fit_safety,
        )

        distance = tight_distance * float(rng.uniform(1.0, distance_jitter_scale))
        distance = tight_distance

        cam_pos = orbit_offset(distance, azim, elev)
        cam2world = look_at(cam_pos, target)
        set_camera_from_cam2world(cam_obj, cam2world)

        camera_infos.append({
            "camera_name": cam_obj.name,
            "view_index": int(k),
            "azimuth_deg": float(np.degrees(azim)),
            "elevation_deg": float(elev_deg),
            "tight_fit_distance": float(tight_distance),
            "distance": float(distance),
            "distance_scale_from_tight_fit": float(distance / max(tight_distance, 1e-8)),
            "camera_c2w": cam2world.astype(np.float32).tolist(),
            "camera_pos": cam_pos.astype(np.float32).tolist(),
            "camera_target": target.astype(np.float32).tolist(),
        })

    return camera_objs, camera_infos


# =================================================================================
#  6. CORE WORKER
# =================================================================================

def process_geometry(args):
    print("--- Starting Geometry Processing ---")

    output_file = args.output_file
    normalized_glb_path = args.normalized_glb_path.strip()

    json_exists = os.path.exists(output_file)
    glb_exists = (normalized_glb_path == "") or os.path.exists(normalized_glb_path)

    if json_exists and glb_exists:
        print(f"✅ Skipped (requested outputs already exist): json={output_file}, glb={normalized_glb_path or 'N/A'}")
        return

    rgb_dir = os.path.splitext(output_file)[0] + "_rgb"
    normal_dir = os.path.splitext(output_file)[0] + "_normal"
    mesh_npz_path = os.path.splitext(output_file)[0] + "_mesh.npz"

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    print(f"Loading object from: {args.object_path}")
    init_scene()
    load_object(args.object_path)

    sequence_normalizer = create_sequence_normalizer()
    print("Sequence normalizer created.")

    # if args.render_engine == "CYCLES":
    #     enable_cycles_acceleration(args.cycles_backend)

    setup_renderer(
        resolution=args.resolution,
        engine=args.render_engine,
        transparent_bg=args.transparent_bg,
    )
    normal_output_node = setup_normal_output(normal_dir)
    create_sun_light()

    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after loading.")

    print("Found mesh objects:")
    for obj in mesh_objs:
        print("  ", obj.name)

    actions = bpy.data.actions
    frame_start, frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        frame_start = int(min(r[0] for r in ranges))
        frame_end = int(max(r[1] for r in ranges))

    frame_indices = np.arange(frame_start, frame_end + 1, dtype=np.int32)

    # -------------------------------------------------------------------------
    # Pass 1: sequence normalization params + shared topology check
    # -------------------------------------------------------------------------
    shared_faces, centers_seq, sequence_scale, canonical_bbox_min, canonical_bbox_max = \
        compute_sequence_normalization_params(mesh_objs, frame_indices)

    print("Sequence normalization:")
    print(f"  sequence_scale = {sequence_scale}")
    print(f"  canonical_bbox_min = {canonical_bbox_min.tolist()}")
    print(f"  canonical_bbox_max = {canonical_bbox_max.tolist()}")

    # -------------------------------------------------------------------------
    # Static cameras
    # -------------------------------------------------------------------------
    camera_seed = int(args.traj_seed + args.traj_id * 9973)
    camera_objs, camera_infos = create_static_multiview_cameras(
        num_cameras=args.num_cameras,
        seed=camera_seed,
        canonical_bbox_min=canonical_bbox_min,
        canonical_bbox_max=canonical_bbox_max,
        elev_min_deg=args.camera_elev_min_deg,
        elev_max_deg=args.camera_elev_max_deg,
        frame_padding=args.camera_frame_padding,
        fit_safety=args.camera_fit_safety,
        distance_jitter_scale=args.camera_distance_jitter_scale,
    )

    bpy.context.scene.camera = camera_objs[0]

    print("Static multi-view camera setup:")
    for info in camera_infos:
        print(
            f"  view {info['view_index']:02d}: "
            f"azim={info['azimuth_deg']:.2f} deg, "
            f"elev={info['elevation_deg']:.2f} deg, "
            f"tight_dist={info['tight_fit_distance']:.4f}, "
            f"dist={info['distance']:.4f}"
        )

    for view_idx in range(args.num_cameras):
        os.makedirs(os.path.join(rgb_dir, f"view_{view_idx:02d}"), exist_ok=True)
        os.makedirs(os.path.join(normal_dir, f"view_{view_idx:02d}"), exist_ok=True)

    # -------------------------------------------------------------------------
    # Export normalized animated scene + static cameras as a new GLB
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Pass 2: apply normalization, extract normalized vertices, render all views
    # -------------------------------------------------------------------------
    vertices_seq = []
    final_watertight_data = {}

    for local_frame_idx, source_frame in enumerate(frame_indices):
        frame_key = f"frame_{int(source_frame):04d}"
        print(f"Processing {frame_key} ({local_frame_idx + 1}/{len(frame_indices)})")

        bpy.context.scene.frame_set(int(source_frame))

        apply_sequence_normalization(
            sequence_normalizer,
            frame_center=centers_seq[local_frame_idx],
            global_scale=sequence_scale,
        )
        bpy.context.view_layer.update()

        norm_vertices, norm_faces = extract_merged_mesh_world(mesh_objs)
        if norm_faces.shape != shared_faces.shape or not np.array_equal(norm_faces, shared_faces):
            raise RuntimeError(
                f"Topology unexpectedly changed after normalization at frame {int(source_frame)}."
            )

        vertices_seq.append(norm_vertices.astype(np.float16))

        views = []

        for view_idx, (camera_obj, camera_info) in enumerate(zip(camera_objs, camera_infos)):
            bpy.context.scene.camera = camera_obj

            view_rgb_dir = os.path.join(rgb_dir, f"view_{view_idx:02d}")
            view_normal_dir = os.path.join(normal_dir, f"view_{view_idx:02d}")

            rgb_path = os.path.join(view_rgb_dir, f"frame_{int(source_frame):04d}.png")

            normal_prefix = "frame_"
            update_normal_output_path(
                normal_output_node,
                base_dir=view_normal_dir,
                prefix=normal_prefix,
            )
            normal_path = os.path.join(view_normal_dir, f"frame_{int(source_frame):04d}.png")

            render_frame(rgb_path)

            views.append({
                "view_index": int(view_idx),
                "rgb_path": rgb_path,
                "normal_path": normal_path,
                "camera_c2w": camera_info["camera_c2w"],
                "camera_pos": camera_info["camera_pos"],
                "camera_target": camera_info["camera_target"],
                "azimuth_deg": camera_info["azimuth_deg"],
                "elevation_deg": camera_info["elevation_deg"],
                "tight_fit_distance": camera_info["tight_fit_distance"],
                "distance": camera_info["distance"],
                "distance_scale_from_tight_fit": camera_info["distance_scale_from_tight_fit"],
            })

        final_watertight_data[frame_key] = {
            "mesh_npz_path": mesh_npz_path,
            "mesh_frame_index": int(local_frame_idx),
            "frame_index": int(source_frame),
            "frame_bbox_center_before_normalization": centers_seq[local_frame_idx].astype(np.float32).tolist(),
            "canonical_bbox_center": [0.0, 0.0, 0.0],
            "views": views,
        }

    # -------------------------------------------------------------------------
    # Save mesh npz
    # -------------------------------------------------------------------------
    vertices_seq = np.stack(vertices_seq, axis=0)

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

    # -------------------------------------------------------------------------
    # Save json metadata
    # -------------------------------------------------------------------------
    final_watertight_data["_global"] = {
        "mesh_npz_path": mesh_npz_path,
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
        "static_cameras": camera_infos,
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
        description="Generate multi-view rendered frames, shared-topology mesh npz, and optionally a normalized GLB from animated GLB/GLTF files using Blender."
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
        help="Render resolution for output PNGs.",
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
        "--cycles_backend",
        type=str,
        default="OPTIX",
        choices=["CUDA", "OPTIX"],
        help="Cycles GPU backend.",
    )

    args = parser.parse_args(get_cli_argv())
    process_geometry(args)