# File: process_geometry_with_bpy.py
# Description: This script uses Blender (bpy) for scene setup and animation,
# and Trimesh for geometric processing to generate consistent, watertight point clouds.
# It is designed to be run from the command line with Blender.
# Example: blender --background --python process_geometry_with_bpy.py -- --object_path /path/to/model.glb ...

import argparse
import sys
import os
import math
import glob
from typing import *

# Blender-specific imports
import bpy
from mathutils import Vector

# Geometry processing imports
import numpy as np
import trimesh
import torch
# import igl
# import cubvh
from tqdm import tqdm

# =================================================================================
#  1. GEOMETRIC PROCESSING HELPERS (Trimesh, PyTorch, etc.)
# =================================================================================

def extract_merged_mesh_world(mesh_objs):
    """
    提取当前帧所有 mesh，并合并成一个 world-space mesh。

    Returns:
        merged_vertices: np.ndarray [N, 3]
        merged_faces: np.ndarray [F, 3]
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    all_vertices = []
    all_faces = []
    vert_offset = 0

    for obj in mesh_objs:
        if obj.type != 'MESH':
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

def sample_surface_with_barycentric(mesh, count):
    """
    Samples points on the mesh surface and returns their face indices and barycentric coordinates.
    """
    points, face_indices = mesh.sample(count, return_index=True)
    triangles = mesh.vertices[mesh.faces[face_indices]]
    barycentric_coords = trimesh.triangles.points_to_barycentric(triangles, points)
    return points, face_indices, barycentric_coords

# =================================================================================
#  2. BLENDER SCENE SETUP FUNCTIONS (from render4d_32.py)
# =================================================================================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    # Add other formats if needed
}

def init_scene() -> None:
    """Resets the Blender scene to a clean state."""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path: str) -> None:
    """Loads a model into the scene."""
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    import_function = IMPORT_FUNCTIONS[file_extension]
    import_function(filepath=object_path)

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_bbox_robust() -> Tuple[Vector, Vector]:
    """Calculates the precise world-space bounding box for all visible meshes."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found_mesh = False
    
    bpy.context.view_layer.update() # Ensure matrices are up to date
    
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() 
                    if isinstance(obj.data, bpy.types.Mesh) and obj.visible_get()]
    if not scene_meshes:
        raise RuntimeError("No visible mesh objects found to compute bounding box.")

    for obj in scene_meshes:
        found_mesh = True
        for vert in obj.data.vertices:
            world_coord = obj.matrix_world @ vert.co
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, world_coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, world_coord))
            
    if not found_mesh:
        raise RuntimeError("No objects in scene to compute bounding box for")
        
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene to fit in a unit cube, exactly like the render script."""
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    
    if len(scene_root_objects) > 1:
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        for obj in scene_root_objects:
            obj.parent = parent_empty
        scene_root = parent_empty
    elif len(scene_root_objects) == 1:
        scene_root = scene_root_objects[0]
    else:
        raise RuntimeError("No root objects found in the scene.")

    bbox_min, bbox_max = scene_bbox()
    
    box_size = max(bbox_max - bbox_min)
    scale = 1.0 / box_size if box_size > 1e-6 else 1.0

    scene_root.scale = scene_root.scale * scale

    bpy.context.view_layer.update()
    
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene_root.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def compute_target_and_radius(vertices_world: np.ndarray):
    """
    vertices_world: [N, 3] in world coordinates
    Returns:
        center: [3]
        radius: float
    """
    bbox_min = vertices_world.min(axis=0)
    bbox_max = vertices_world.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    extent = bbox_max - bbox_min
    diag = np.linalg.norm(extent)
    radius = max(1.5 * diag, 1e-3)  # camera distance
    return center, radius

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def look_at(cam_pos: np.ndarray, target: np.ndarray, up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    """
    Return Blender-style camera-to-world matrix.
    Blender camera looks along local -Z, with local Y as up.
    """
    forward = normalize(target - cam_pos)   # viewing direction in world
    right = normalize(np.cross(forward, up))
    if np.linalg.norm(right) < 1e-6:
        # fallback if forward almost parallel to up
        up_alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, up_alt))
    true_up = normalize(np.cross(right, forward))

    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = true_up
    cam2world[:3, 2] = -forward   # Blender camera local -Z points forward
    cam2world[:3, 3] = cam_pos
    return cam2world

def orbit_offset(radius: float, azim: float, elev: float):
    """
    azim/elev in radians
    z-up convention
    """
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
    bpy.context.scene.camera = cam_obj
    return cam_obj

from mathutils import Matrix

def set_camera_from_cam2world(cam_obj, cam2world: np.ndarray):
    cam_obj.matrix_world = Matrix(cam2world.tolist())

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

    if engine == "CYCLES":
        scene.cycles.device = "GPU"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True

        # 可选：加速设置
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True

    elif engine == "BLENDER_EEVEE":
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 64

def create_sun_light(name="SunLight", energy=3.0):
    light_data = bpy.data.lights.new(name=name, type='SUN')
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

def keep_largest_mesh_only(use_evaluated=False, delete_non_mesh=False):
    """
    保留场景中 face 数最多的 MESH object，删除其他 MESH object。
    
    Args:
        use_evaluated: 
            False -> 用原始 mesh.data.polygons 计数
            True  -> 用 modifier 后的 evaluated mesh 计数
        delete_non_mesh:
            是否顺便删除非 mesh 对象（如 EMPTY / ARMATURE 等）
    Returns:
        kept_obj: 被保留下来的 mesh object
    """
    scene = bpy.context.scene
    mesh_objs = [obj for obj in scene.objects if obj.type == 'MESH']

    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found in the scene.")

    if use_evaluated:
        depsgraph = bpy.context.evaluated_depsgraph_get()

        def get_face_count(obj):
            obj_eval = obj.evaluated_get(depsgraph)
            mesh_eval = obj_eval.to_mesh()
            n_faces = len(mesh_eval.polygons)
            obj_eval.to_mesh_clear()
            return n_faces
    else:
        def get_face_count(obj):
            return len(obj.data.polygons)

    kept_obj = max(mesh_objs, key=get_face_count)

    # 删除其他 mesh
    for obj in mesh_objs:
        if obj != kept_obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    # 可选：顺便删掉非 mesh 对象
    if delete_non_mesh:
        for obj in list(scene.objects):
            if obj != kept_obj and obj.type != 'MESH':
                bpy.data.objects.remove(obj, do_unlink=True)

    # 设为 active / selected
    bpy.ops.object.select_all(action='DESELECT')
    kept_obj.select_set(True)
    bpy.context.view_layer.objects.active = kept_obj

    return kept_obj

# camera motion
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def get_elev_start_bin(
    traj_id: int,
    num_trajs: int = 5,
    elev_start_max_deg: float = 80.0,
    elev_start_min_deg: float = -60.0,
):
    """
    将 [elev_start_max_deg, elev_start_min_deg] 均匀分成 num_trajs 段，
    根据 traj_id 返回该轨迹对应的起始 elevation 采样区间（单位：度）。
    """
    bucket = traj_id % num_trajs
    edges = np.linspace(elev_start_max_deg, elev_start_min_deg, num_trajs + 1)

    high_deg = float(edges[bucket])
    low_deg = float(edges[bucket + 1])

    # 返回时保证 low <= high，便于 rng.uniform(low, high)
    lo = min(low_deg, high_deg)
    hi = max(low_deg, high_deg)
    return lo, hi

def get_elev_start_bin(
    traj_id: int,
    num_trajs: int = 5,
    elev_start_max_deg: float = 80.0,
    elev_start_min_deg: float = -60.0,
):
    """
    将 [elev_start_max_deg, elev_start_min_deg] 均匀分成 num_trajs 段，
    根据 traj_id 返回该轨迹对应的起始 elevation 采样区间（单位：度）。
    """
    bucket = traj_id % num_trajs
    edges = np.linspace(elev_start_max_deg, elev_start_min_deg, num_trajs + 1)

    high_deg = float(edges[bucket])
    low_deg = float(edges[bucket + 1])

    lo = min(low_deg, high_deg)
    hi = max(low_deg, high_deg)
    return lo, hi


def sample_trajectory_params(
    traj_id: int,
    base_seed: int = 0,
):
    """
    traj_id:
      0~4: 原来的 5 条分桶起始轨迹
      5  : 从高 elev 平滑下降到低 elev
      6  : 从低 elev 平滑上升到高 elev
    """
    rng = np.random.default_rng(base_seed + traj_id * 9973)

    params = {
        # azimuth
        "azim0": math.radians(rng.uniform(0.0, 360.0)),
        "azim_speed": math.radians(rng.uniform(2.0, 10.0)),
        "azim_wobble_amp": math.radians(rng.uniform(2.0, 12.0)),
        "azim_wobble_freq": rng.uniform((1 / 105) * math.pi, (1 / 18) * math.pi),
        "azim_wobble_phase": rng.uniform(0.0, 2.0 * math.pi),

        # elevation oscillation
        "elev_amp1": math.radians(rng.uniform(6.0, 18.0)),
        "elev_freq1": rng.uniform((1 / 80) * math.pi, (1 / 18) * math.pi),
        "elev_phase1": rng.uniform(0.0, 2.0 * math.pi),

        "elev_amp2": math.radians(rng.uniform(3.0, 10.0)),
        "elev_freq2": rng.uniform((1 / 210) * math.pi, (1 / 40) * math.pi),
        "elev_phase2": rng.uniform(0.0, 2.0 * math.pi),

        "elev_amp3": math.radians(rng.uniform(1.0, 6.0)),
        "elev_freq3": rng.uniform((1 / 600) * math.pi, (1 / 80) * math.pi),
        "elev_phase3": rng.uniform(0.0, 2.0 * math.pi),

        "elev_drift_amp": math.radians(rng.uniform(2.0, 8.0)),
        "elev_drift_freq": rng.uniform((1 / 1600) * math.pi, (1 / 160) * math.pi),
        "elev_drift_phase": rng.uniform(0.0, 2.0 * math.pi),

        "elev_min": math.radians(-70.0),
        "elev_max": math.radians(85.0),

        # radius
        "radius_scale": rng.uniform(0.9, 1.1),
        "radius_wobble_amp": rng.uniform(0.0, 0.12),
        "radius_wobble_freq": rng.uniform(0.01, 0.08),
        "radius_wobble_phase": rng.uniform(0.0, 2.0 * math.pi),

        # look-at jitter
        "target_jitter_scale_xy": rng.uniform(0.005, 0.02),
        "target_jitter_scale_z": rng.uniform(0.003, 0.012),

        "target_jitter_freq_x1": rng.uniform(0.03, 0.12),
        "target_jitter_freq_x2": rng.uniform(0.01, 0.05),
        "target_jitter_phase_x1": rng.uniform(0.0, 2.0 * math.pi),
        "target_jitter_phase_x2": rng.uniform(0.0, 2.0 * math.pi),

        "target_jitter_freq_y1": rng.uniform(0.03, 0.12),
        "target_jitter_freq_y2": rng.uniform(0.01, 0.05),
        "target_jitter_phase_y1": rng.uniform(0.0, 2.0 * math.pi),
        "target_jitter_phase_y2": rng.uniform(0.0, 2.0 * math.pi),

        "target_jitter_freq_z1": rng.uniform(0.02, 0.08),
        "target_jitter_freq_z2": rng.uniform(0.008, 0.03),
        "target_jitter_phase_z1": rng.uniform(0.0, 2.0 * math.pi),
        "target_jitter_phase_z2": rng.uniform(0.0, 2.0 * math.pi),
    }

    # 前 5 条：维持原来的“分桶起始”逻辑
    if traj_id in [0, 1, 2, 3, 4]:
        elev0_lo_deg, elev0_hi_deg = get_elev_start_bin(
            traj_id=traj_id,
            num_trajs=5,
            elev_start_max_deg=80.0,
            elev_start_min_deg=-60.0,
        )
        start_elev_deg = rng.uniform(elev0_lo_deg, elev0_hi_deg)
        start_elev = math.radians(start_elev_deg)

        elev_offset_at_t0 = (
            params["elev_amp1"] * math.sin(params["elev_phase1"])
            + params["elev_amp2"] * math.sin(params["elev_phase2"])
            + params["elev_amp3"] * math.sin(params["elev_phase3"])
            + params["elev_drift_amp"] * math.sin(params["elev_drift_phase"])
        )

        params["traj_mode"] = "bucket"
        params["start_elev_deg"] = start_elev_deg
        params["start_elev_bin_deg"] = [elev0_lo_deg, elev0_hi_deg]
        params["elev_bias"] = start_elev - elev_offset_at_t0

    # 第 6 条：高 -> 低
    elif traj_id == 5:
        start_elev_deg = rng.uniform(52.0, 80.0)
        end_elev_deg = rng.uniform(-60.0, -32.0)

        params["traj_mode"] = "sweep_down"
        params["start_elev_deg"] = start_elev_deg
        params["end_elev_deg"] = end_elev_deg
        params["start_elev_bin_deg"] = [52.0, 80.0]
        params["end_elev_bin_deg"] = [-60.0, -32.0]

    # 第 7 条：低 -> 高
    elif traj_id == 6:
        start_elev_deg = rng.uniform(-60.0, -32.0)
        end_elev_deg = rng.uniform(52.0, 80.0)

        params["traj_mode"] = "sweep_up"
        params["start_elev_deg"] = start_elev_deg
        params["end_elev_deg"] = end_elev_deg
        params["start_elev_bin_deg"] = [-60.0, -32.0]
        params["end_elev_bin_deg"] = [52.0, 80.0]

    else:
        raise ValueError(f"traj_id must be in [0, 6], but got {traj_id}")

    return params

def evaluate_trajectory(frame_idx: int, num_frames: int, radius_base: float, params: dict):
    """
    给定 frame_idx，返回当前帧的 azim / elev / radius
    """
    t = float(frame_idx)
    denom = max(num_frames - 1, 1)
    alpha = t / denom

    # 平滑插值系数：0 -> 1，首尾更平滑
    s = alpha * alpha * (3.0 - 2.0 * alpha)

    azim = (
        params["azim0"]
        + t * params["azim_speed"]
        + params["azim_wobble_amp"] * math.sin(
            t * params["azim_wobble_freq"] + params["azim_wobble_phase"]
        )
    )

    # multi-frequency oscillation
    elev_osc = (
        params["elev_amp1"] * math.sin(t * params["elev_freq1"] + params["elev_phase1"])
        + params["elev_amp2"] * math.sin(t * params["elev_freq2"] + params["elev_phase2"])
        + params["elev_amp3"] * math.sin(t * params["elev_freq3"] + params["elev_phase3"])
        + params["elev_drift_amp"] * math.sin(t * params["elev_drift_freq"] + params["elev_drift_phase"])
    )

    traj_mode = params["traj_mode"]

    if traj_mode == "bucket":
        elev = params["elev_bias"] + elev_osc

    elif traj_mode in ["sweep_down", "sweep_up"]:
        start_elev = math.radians(params["start_elev_deg"])
        end_elev = math.radians(params["end_elev_deg"])

        # 趋势项：从 start 平滑过渡到 end
        trend = (1.0 - s) * start_elev + s * end_elev

        # 震荡项：首尾为 0，中间更明显
        envelope = math.sin(math.pi * s)
        elev = trend + 0.45 * envelope * elev_osc

    else:
        raise ValueError(f"Unknown traj_mode: {traj_mode}")

    elev = clamp(elev, params["elev_min"], params["elev_max"])

    radius = radius_base * params["radius_scale"] * (
        1.0
        + params["radius_wobble_amp"] * math.sin(
            t * params["radius_wobble_freq"] + params["radius_wobble_phase"]
        )
    )

    return azim, elev, radius

def evaluate_target_jitter(frame_idx: int, radius_base: float, params: dict):
    """
    返回 look-at point 的平滑小抖动（world space）。
    """
    t = float(frame_idx)

    amp_xy = radius_base * params["target_jitter_scale_xy"]
    amp_z = radius_base * params["target_jitter_scale_z"]

    dx = amp_xy * (
        0.65 * math.sin(t * params["target_jitter_freq_x1"] + params["target_jitter_phase_x1"])
        + 0.35 * math.sin(t * params["target_jitter_freq_x2"] + params["target_jitter_phase_x2"])
    )

    dy = amp_xy * (
        0.65 * math.sin(t * params["target_jitter_freq_y1"] + params["target_jitter_phase_y1"])
        + 0.35 * math.sin(t * params["target_jitter_freq_y2"] + params["target_jitter_phase_y2"])
    )

    dz = amp_z * (
        0.65 * math.sin(t * params["target_jitter_freq_z1"] + params["target_jitter_phase_z1"])
        + 0.35 * math.sin(t * params["target_jitter_freq_z2"] + params["target_jitter_phase_z2"])
    )

    return np.array([dx, dy, dz], dtype=np.float32)

def enable_cycles_acceleration(backend: str = "CUDA"):
    """
    backend: CUDA / OPTIX
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences

    backend = backend.upper()
    prefs.compute_device_type = backend

    # Blender 不同版本接口略有差异
    try:
        prefs.refresh_devices()
    except AttributeError:
        prefs.get_devices()

    found_gpu = False
    print("=== Cycles device discovery ===")
    print("Requested backend:", backend)

    for d in prefs.devices:
        # 仅启用 GPU，禁用 CPU
        d.use = (d.type != "CPU")
        print(f"  name={d.name}, type={d.type}, use={d.use}")
        if d.type != "CPU" and d.use:
            found_gpu = True

    print("scene.render.engine =", scene.render.engine)
    print("scene.cycles.device =", scene.cycles.device)
    print("prefs.compute_device_type =", prefs.compute_device_type)

    if not found_gpu:
        raise RuntimeError(
            f"No usable GPU found for Cycles backend={backend}. "
            f"Try backend='CUDA', check nvidia-smi, and verify Blender can see the GPU."
        )

def setup_normal_output(normal_dir: str):
    """
    开启 Normal pass，并通过 compositor 把 normal map 输出到 normal_dir。
    输出文件名形如: frame_0001.png
    """

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    # 开启 normal pass
    view_layer.use_pass_normal = True

    # 启用 compositor
    scene.use_nodes = True
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True

    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    # Render Layers
    rlayers = nodes.new(type="CompositorNodeRLayers")
    rlayers.location = (-500, 0)

    # 保留正常 RGB 输出
    composite = nodes.new(type="CompositorNodeComposite")
    composite.location = (350, 120)
    links.new(rlayers.outputs["Image"], composite.inputs["Image"])

    # 把 normal 从 [-1, 1] 映射到 [0, 1]，方便存成 PNG 可视化/训练
    normal_mul = nodes.new(type="CompositorNodeMixRGB")
    normal_mul.blend_type = 'MULTIPLY'
    normal_mul.inputs[0].default_value = 1.0
    normal_mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    normal_mul.location = (-150, -80)
    links.new(rlayers.outputs["Normal"], normal_mul.inputs[1])

    normal_add = nodes.new(type="CompositorNodeMixRGB")
    normal_add.blend_type = 'ADD'
    normal_add.inputs[0].default_value = 1.0
    normal_add.inputs[2].default_value = (0.5, 0.5, 0.5, 0.0)
    normal_add.location = (80, -80)
    links.new(normal_mul.outputs["Image"], normal_add.inputs[1])

    # 用原始 alpha 作为 normal 的 alpha
    set_alpha = nodes.new(type="CompositorNodeSetAlpha")
    set_alpha.location = (300, -80)
    links.new(normal_add.outputs["Image"], set_alpha.inputs["Image"])
    links.new(rlayers.outputs["Alpha"], set_alpha.inputs["Alpha"])

    # 输出 normal 到文件
    file_output = nodes.new(type="CompositorNodeOutputFile")
    file_output.location = (550, -80)
    file_output.base_path = normal_dir

    slot = file_output.file_slots[0]
    slot.path = "frame_"
    slot.use_node_format = True
    slot.save_as_render = False

    file_output.format.file_format = "PNG"
    file_output.format.color_mode = "RGBA"
    file_output.format.color_depth = "16"

    links.new(set_alpha.outputs["Image"], file_output.inputs[0])

    print(f"Normal output is enabled. Files will be written to: {normal_dir}")
# =================================================================================
#  3. CORE WORKER FUNCTION (Orchestrator)
# =================================================================================
def process_geometry(args):
    """
    The main function that orchestrates bpy scene setup and trimesh geometry processing.
    """
    print("--- Starting Geometry Processing ---")
    
    # --- Setup Paths ---
    output_file = args.output_file
    if os.path.exists(output_file):
        print(f"✅ Skipped (output file already exists): {output_file}")
        return
    
    rgb_dir = os.path.splitext(output_file)[0] + "_rgb"
    os.makedirs(rgb_dir, exist_ok=True)

    normal_dir = os.path.splitext(output_file)[0] + "_normal"
    os.makedirs(normal_dir, exist_ok=True)

    mesh_dir = os.path.splitext(output_file)[0] + "_mesh"
    os.makedirs(mesh_dir, exist_ok=True)

    # --- 1. Scene Setup using bpy (Identical to render script) ---
    print(f"Loading object from: {args.object_path}")
    init_scene()
    load_object(args.object_path)

    # 仅保留最大的mesh
    # largest_mesh_obj = keep_largest_mesh_only(
    #     use_evaluated=False,   # 一般先这样就够了
    #     delete_non_mesh=False  # 如果你不想留 armature/empty，可改成 True
    # )

    print("Normalizing scene...")
    # The scale and offset are calculated here by bpy, ensuring consistency.
    normalize_scene() 
    print("Scene setup and normalization complete.")
    
    # render
    # if args.render_engine == "CYCLES":
    #     enable_cycles_acceleration(args.cycles_backend)

    setup_renderer(
        resolution=args.resolution,
        engine=args.render_engine,
        transparent_bg=args.transparent_bg
    )
    setup_normal_output(normal_dir)

    # light
    create_sun_light()

    camera_obj = create_camera()
    
    
    
    # --- 2. Get all mesh objects from the Blender scene ---
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after loading.")

    print("Found mesh objects:")
    for obj in mesh_objs:
        print("  ", obj.name)

    # --- 3. Timeline Remapping and Per-Frame Processing ---
    final_watertight_data = {}
    
    # Get original animation frame range from bpy actions
    actions = bpy.data.actions
    frame_start, frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        frame_start = int(min(r[0] for r in ranges))
        frame_end = int(max(r[1] for r in ranges))
    source_duration = float(frame_end - frame_start)

    prev_target = None
    prev_radius = None

    # 轨迹参数：你可以调
    # azim0 = math.radians(30.0)         # 初始水平角
    # elev0 = math.radians(15.0)         # 初始俯仰角
    # azim_speed = math.radians(8.0)     # 每帧绕转 2 度
    # elev_amp = math.radians(15.0)       # 俯仰小幅波动
    # elev_freq =  0.15                   # 波动频率

    traj_params = sample_trajectory_params(
        traj_id=args.traj_id,
        base_seed=args.traj_seed,
        # num_trajs=5,
        # elev_start_max_deg=80.0,
        # elev_start_min_deg=-60.0,
    )

    target_smooth = 0.8   # 越小越稳，越大越跟得紧
    radius_smooth = 0.2

    print("Trajectory params:")
    for k, v in traj_params.items():
        print(f"  {k}: {v}")

    for i in tqdm(range(frame_start, frame_end + 1), desc="Processing Frames"):
        source_frame = i
        frame_key = f"frame_{i:04d}"

        # Apply motion using bpy
        bpy.context.scene.frame_set(source_frame)

        # Get merged deformed mesh data from all bpy mesh objects
        bpy.context.view_layer.update()
        deformed_vertices, deformed_faces = extract_merged_mesh_world(mesh_objs)

        # ----- compute tracking target -----
        target_raw, radius_raw = compute_target_and_radius(deformed_vertices)

        if prev_target is None:
            target = target_raw
            radius = radius_raw
        else:
            target = (1.0 - target_smooth) * prev_target + target_smooth * target_raw
            radius = (1.0 - radius_smooth) * prev_radius + radius_smooth * radius_raw

        prev_target = target
        prev_radius = radius

        # ----- continuous relative pose change -----
        if False:
            frame_idx = i - frame_start
            azim = azim0 + frame_idx * azim_speed
            elev = elev0 + elev_amp * math.sin(frame_idx * elev_freq)

            cam_offset = orbit_offset(radius, azim, elev)
            cam_pos = target + cam_offset

            cam2world = look_at(cam_pos, target)
        else:
            # ----- continuous relative pose change -----
            frame_idx = i - frame_start

            azim, elev, radius_cam = evaluate_trajectory(
                frame_idx=frame_idx,
                num_frames=(frame_end - frame_start + 1),
                radius_base=radius,
                params=traj_params,
            )

            cam_offset = orbit_offset(radius_cam, azim, elev)
            cam_pos = target + cam_offset

            target_jitter = evaluate_target_jitter(
                frame_idx=frame_idx,
                radius_base=radius,
                params=traj_params,
            )
            look_target = target + target_jitter

            cam2world = look_at(cam_pos, look_target)

        # Optional: place a real Blender camera
        set_camera_from_cam2world(camera_obj, cam2world)

        # ----- render RGB -----
        bpy.context.scene.camera = camera_obj
        rgb_path = os.path.join(rgb_dir, f"frame_{i:04d}.png")
        normal_path = os.path.join(normal_dir, f"frame_{i:04d}.png")
        render_frame(rgb_path)
        
        # Create watertight version of the current frame's mesh
        if False:
            mc_verts, mc_faces = Watertight_cubvh(deformed_vertices, deformed_faces, grid_res=args.grid_res)
            if mc_verts.shape[0] < 3:
                print(f"Warning: Watertight failed for frame {i}. Skipping.")
                continue
            watertight_mesh = trimesh.Trimesh(vertices=mc_verts, faces=mc_faces, process=False)
            
            mesh_path = os.path.join(mesh_dir, f"{frame_key}.ply")
            watertight_mesh.export(mesh_path)
        
        current_mesh = trimesh.Trimesh(
            vertices=deformed_vertices,
            faces=deformed_faces,
            process=False
        )
        mesh_path = os.path.join(mesh_dir, f"{frame_key}.ply")
        current_mesh.export(mesh_path)
                        

        final_watertight_data[frame_key] = {
            "mesh_path": mesh_path,
            "rgb_path": rgb_path,
            "normal_path": normal_path,
            "camera_c2w": cam2world.astype(np.float32).tolist(),
            "camera_pos": cam_pos.astype(np.float32).tolist(),
            "camera_target": look_target.astype(np.float32).tolist(),
            "camera_target_center": target.astype(np.float32).tolist(),
        }
        # final_watertight_data[frame_key] = {"random_surface": surface_data}
        # final_watertight_data[frame_key] = {
        #     "mesh_path": mesh_path,
        #     "camera_c2w": cam2world.astype(np.float32).tolist(),
        #     "camera_pos": cam_pos.astype(np.float32).tolist(),
        #     "camera_target": target.astype(np.float32).tolist(),
        # }
        # final_watertight_data[f"{frame_key}_mesh_path"] = np.array(mesh_path)
        # final_watertight_data[f"{frame_key}_camera_c2w"] = cam2world.astype(np.float32)
        # final_watertight_data[f"{frame_key}_camera_pos"] = cam_pos.astype(np.float32)
        # final_watertight_data[f"{frame_key}_camera_target"] = target.astype(np.float32)
    
    # --- 4. Save Final Data ---
    print(f"Saving final data to {output_file}...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    # np.savez(output_file, **final_watertight_data)
    import json
    with open(output_file, "w") as fw:
        fw.write(json.dumps(final_watertight_data, indent=2))
    print("--- Processing Complete ---")

# =================================================================================
#  4. MAIN EXECUTION BLOCK
# =================================================================================

if __name__ == "__main__":
    # Remove Blender's arguments to parse our own
    argv = sys.argv[sys.argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser(description='Generate consistent watertight point clouds from animated GLB files using Blender for scene setup.')
    parser.add_argument('--object_path', type=str, required=True,
                        help='Path to the source .glb file.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output .npz file.')
    parser.add_argument('--grid_res', type=int, default=256,
                        help='Resolution for the watertight grid.')
    parser.add_argument('--sample_count', type=int, default=32768,
                        help='Number of points to sample on the surface.')
    # parser.add_argument('--target_frames', type=int, default=32,
    #                     help='The number of frames to resample the animation to.')
    # parser.add_argument('--rgb_dir', type=str, required=True,
    #                 help='Directory to save rendered RGB PNG frames.')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Render resolution for output PNGs.')
    parser.add_argument('--render_engine', type=str, default='BLENDER_EEVEE',
                        choices=['BLENDER_EEVEE', 'CYCLES'],
                        help='Blender render engine.')
    parser.add_argument('--transparent_bg', action='store_true',
                        help='Render with transparent background.')
    parser.add_argument('--traj_id', type=int, default=0,
                    help='Trajectory id for reproducible camera motion.')
    parser.add_argument('--traj_seed', type=int, default=0,
                        help='Base random seed for trajectory generation.')
    parser.add_argument(
        '--cycles_backend',
        type=str,
        default='CUDA',
        choices=['CUDA', 'OPTIX'],
        help='Cycles GPU backend. For H200, CUDA is the safest default.'
    )
    args = parser.parse_args(argv)
    
    process_geometry(args)