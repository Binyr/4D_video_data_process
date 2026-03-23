#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized process_geometry_with_bpy.py

Main optimizations compared with the previous version:
1. Cache raw vertices during the first topology / normalization scan, so we do not
   call Blender mesh extraction twice for the whole sequence.
2. Use foreach_get + numpy for much faster mesh extraction.
3. Build the HDR world node tree only once and reuse it by only swapping
   image / strength, instead of clearing and rebuilding nodes every render.
4. Avoid redundant lighting state updates by memoizing the active lighting signature.
5. Add faster-by-default Cycles knobs:
   - denoising disabled by default (enable with --cycles_use_denoising)
   - compressed mesh saving optional (enable with --save_compressed_mesh)
6. Add coarse timing logs for the main stages.

CLI remains largely compatible with the earlier script.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


# =====================================================================================
# 0. UTILS / TIMING / DEBUG
# =====================================================================================


def get_cli_argv() -> List[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


class StageTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0

    def log(self, name: str):
        now = time.perf_counter()
        print(f"[timing] {name}: +{now - self.last:.3f}s | total={now - self.t0:.3f}s")
        self.last = now


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


# Optional debug helpers.
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


# =====================================================================================
# 1. SCENE / CAMERA HELPERS
# =====================================================================================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
}


def init_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        if texture.users == 0:
            bpy.data.textures.remove(texture, do_unlink=True)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh, do_unlink=True)
    for world in list(bpy.data.worlds):
        if world.users == 0:
            bpy.data.worlds.remove(world, do_unlink=True)
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def load_object(object_path: str) -> None:
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    IMPORT_FUNCTIONS[file_extension](filepath=object_path)


def look_at(
    cam_pos: np.ndarray,
    target: np.ndarray,
    up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
):
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


# =====================================================================================
# 2. FAST GEOMETRY EXTRACTION
# =====================================================================================


def extract_merged_mesh_world_fast(mesh_objs, depsgraph=None):
    """
    Faster version using foreach_get + numpy. Returns merged world-space vertices / faces.
    """
    if depsgraph is None:
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

        try:
            temp_mesh.calc_loop_triangles()
            num_verts = len(temp_mesh.vertices)
            num_tris = len(temp_mesh.loop_triangles)
            if num_verts == 0 or num_tris == 0:
                continue

            co = np.empty(num_verts * 3, dtype=np.float32)
            temp_mesh.vertices.foreach_get("co", co)
            co = co.reshape(num_verts, 3)

            tri = np.empty(num_tris * 3, dtype=np.int32)
            temp_mesh.loop_triangles.foreach_get("vertices", tri)
            tri = tri.reshape(num_tris, 3)

            world_mat = obj_eval.matrix_world.copy()
            R = np.array(world_mat.to_3x3(), dtype=np.float32)
            t = np.array(world_mat.translation[:], dtype=np.float32)
            verts_world = co @ R.T + t[None, :]

            all_vertices.append(verts_world)
            all_faces.append(tri + vert_offset)
            vert_offset += num_verts
        finally:
            obj_eval.to_mesh_clear()

    if len(all_vertices) == 0:
        raise RuntimeError("No valid mesh found in current frame.")

    merged_vertices = np.concatenate(all_vertices, axis=0)
    merged_faces = np.concatenate(all_faces, axis=0)
    return merged_vertices.astype(np.float32, copy=False), merged_faces.astype(np.int32, copy=False)


def compute_bbox_center(vertices_world: np.ndarray):
    bbox_min = vertices_world.min(axis=0)
    bbox_max = vertices_world.max(axis=0)
    return 0.5 * (bbox_min + bbox_max)


def collect_keyframe_frames(frame_start: int, frame_end: int) -> List[int]:
    keyframes = set()
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            for kp in fcurve.keyframe_points:
                frame = int(round(float(kp.co.x)))
                if frame_start <= frame <= frame_end:
                    keyframes.add(frame)
    return sorted(keyframes)


def save_mesh_as_ply(vertices: np.ndarray, faces: np.ndarray, ply_path: str):
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
        face_dtype = np.dtype([("count", "u1"), ("idx", "<i4", (3,))])
        face_data = np.empty(len(faces), dtype=face_dtype)
        face_data["count"] = 3
        face_data["idx"] = faces.astype("<i4", copy=False)
        face_data.tofile(f)


# =====================================================================================
# 3. SEQUENCE NORMALIZATION + RAW CACHE
# =====================================================================================


def create_sequence_normalizer():
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


def compute_sequence_normalization_params_and_cache(
    mesh_objs,
    frame_indices: np.ndarray,
    cache_raw_vertices: bool = True,
    raw_cache_dtype=np.float16,
):
    """
    First pass only:
      1) check shared topology
      2) compute per-frame bbox centers
      3) compute canonical bbox after removing per-frame translation
      4) optionally cache raw vertices so we never extract them again
    """
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    shared_faces = None
    centers_seq = []
    raw_vertices_cache = [] if cache_raw_vertices else None

    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    for i in frame_indices:
        scene.frame_set(int(i))
        bpy.context.view_layer.update()

        raw_vertices, raw_faces = extract_merged_mesh_world_fast(mesh_objs, depsgraph=depsgraph)

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
                    f"Reference faces shape={shared_faces.shape}, current faces shape={raw_faces.shape}."
                )

        center = compute_bbox_center(raw_vertices).astype(np.float32)
        centers_seq.append(center)

        centered = raw_vertices - center[None, :]
        global_min = np.minimum(global_min, centered.min(axis=0))
        global_max = np.maximum(global_max, centered.max(axis=0))

        if cache_raw_vertices:
            raw_vertices_cache.append(raw_vertices.astype(raw_cache_dtype))

    centers_seq = np.stack(centers_seq, axis=0)
    shared_faces = np.asarray(shared_faces, dtype=np.int32)

    extent = global_max - global_min
    box_size = float(np.max(extent))
    sequence_scale = 1.0 / box_size if box_size > 1e-6 else 1.0

    canonical_bbox_min = (global_min * sequence_scale).astype(np.float32)
    canonical_bbox_max = (global_max * sequence_scale).astype(np.float32)

    return (
        shared_faces,
        centers_seq,
        sequence_scale,
        canonical_bbox_min,
        canonical_bbox_max,
        raw_vertices_cache,
    )


def apply_sequence_normalization(normalizer_obj, frame_center: np.ndarray, global_scale: float):
    normalizer_obj.scale = (global_scale, global_scale, global_scale)
    normalizer_obj.location = tuple((-global_scale * frame_center).tolist())


def bake_sequence_normalization_to_keyframes(
    normalizer_obj,
    frame_indices: np.ndarray,
    centers_seq: np.ndarray,
    global_scale: float,
):
    scene = bpy.context.scene
    current_frame = scene.frame_current

    scene.frame_start = int(frame_indices[0])
    scene.frame_end = int(frame_indices[-1])

    if normalizer_obj.animation_data is not None:
        normalizer_obj.animation_data_clear()

    for local_frame_idx, source_frame in enumerate(frame_indices):
        scene.frame_set(int(source_frame))
        normalizer_obj.scale = (global_scale, global_scale, global_scale)
        normalizer_obj.location = tuple((-global_scale * centers_seq[local_frame_idx]).tolist())
        normalizer_obj.keyframe_insert(data_path="location", frame=int(source_frame))
        normalizer_obj.keyframe_insert(data_path="scale", frame=int(source_frame))

    if normalizer_obj.animation_data is not None and normalizer_obj.animation_data.action is not None:
        for fcurve in normalizer_obj.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = "LINEAR"

    scene.frame_set(current_frame)
    bpy.context.view_layer.update()


def precompute_normalized_mesh_sequence_from_cache(
    raw_vertices_cache: List[np.ndarray],
    centers_seq: np.ndarray,
    global_scale: float,
    shared_faces: np.ndarray,
    frame_indices: np.ndarray,
    export_keyframe_ply: bool = False,
    keyframe_frame_set=None,
    mesh_ply_dir: str = "",
):
    from tqdm import tqdm

    if raw_vertices_cache is None:
        raise RuntimeError("raw_vertices_cache is None. Enable cache_raw_vertices to use optimized precompute path.")
    if keyframe_frame_set is None:
        keyframe_frame_set = set()

    num_frames = len(raw_vertices_cache)
    num_vertices = raw_vertices_cache[0].shape[0]
    vertices_seq = np.empty((num_frames, num_vertices, 3), dtype=np.float16)
    keyframe_ply_records = {}

    for local_frame_idx, raw_vertices in enumerate(tqdm(raw_vertices_cache, desc="Normalizing cached mesh sequence")):
        raw32 = np.asarray(raw_vertices, dtype=np.float32)
        norm_vertices = ((raw32 - centers_seq[local_frame_idx][None, :]) * global_scale).astype(np.float32)
        vertices_seq[local_frame_idx] = norm_vertices.astype(np.float16)

        frame_int = int(frame_indices[local_frame_idx])
        if export_keyframe_ply and (frame_int in keyframe_frame_set):
            mesh_ply_path = os.path.join(mesh_ply_dir, f"frame_{frame_int:04d}.ply")
            save_mesh_as_ply(norm_vertices, shared_faces, mesh_ply_path)
            keyframe_ply_records[frame_int] = mesh_ply_path

    return vertices_seq, keyframe_ply_records


# =====================================================================================
# 4. RENDERING / LIGHTING
# =====================================================================================


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
            f"Try backend='CUDA' or 'OPTIX', check nvidia-smi, and verify Blender can see the GPU."
        )


def setup_renderer(
    resolution=512,
    engine="BLENDER_EEVEE",
    transparent_bg=True,
    cycles_samples: int = 64,
    cycles_use_denoising: bool = False,
    cycles_use_adaptive_sampling: bool = True,
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
        scene.cycles.samples = int(cycles_samples)
        scene.cycles.use_denoising = bool(cycles_use_denoising)
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = bool(cycles_use_adaptive_sampling)
    elif engine == "BLENDER_EEVEE":
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = int(cycles_samples)


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


def resolve_hdr_strength_range(args):
    if args.hdr_strength is not None:
        v = float(args.hdr_strength)
        return v, v
    lo = float(args.hdr_strength_min)
    hi = float(args.hdr_strength_max)
    if lo > hi:
        raise ValueError("hdr_strength_min must be <= hdr_strength_max")
    return lo, hi


def list_hdr_files(hdr_dir: str):
    hdr_dir = Path(hdr_dir)
    if not hdr_dir.exists():
        raise FileNotFoundError(f"HDR directory not found: {hdr_dir}")

    hdr_files = []
    for ext in ("*.exr", "*.hdr", "*.EXR", "*.HDR"):
        for p in hdr_dir.glob(ext):
            if not p.is_file():
                continue
            name = p.name
            if name.startswith("._") or name.startswith("."):
                continue
            hdr_files.append(p.resolve())

    hdr_files = sorted(set(hdr_files))
    if len(hdr_files) == 0:
        raise RuntimeError(f"No valid HDR files (*.exr / *.hdr) found in directory: {hdr_dir}")
    return hdr_files


def duplicate_world_or_none(world, new_name: str):
    if world is None:
        return None
    world_copy = world.copy()
    world_copy.name = new_name
    return world_copy


def build_reusable_hdr_world(name: str = "PerCameraHDRWorld"):
    hdr_world = bpy.data.worlds.new(name)
    hdr_world.use_nodes = True
    nt = hdr_world.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

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

    links.new(node_texcoord.outputs["Generated"], node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_env.inputs["Vector"])
    links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
    links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])

    return hdr_world, node_env, node_bg


def preload_hdr_images_for_configs(per_camera_lighting: List[dict]):
    unique_paths = sorted({cfg["selected_hdr_map"] for cfg in per_camera_lighting if cfg["selected_hdr_map"] is not None})
    image_cache = {}
    for p in unique_paths:
        image_cache[p] = bpy.data.images.load(str(p), check_existing=True)
    return image_cache


def create_per_camera_lighting_controller(scene, sunlight_energy: float, per_camera_lighting: List[dict]):
    imported_lights = []
    for obj in scene.objects:
        if obj.type != "LIGHT":
            continue
        imported_lights.append(
            {
                "obj": obj,
                "name": obj.name,
                "light_type": str(getattr(obj.data, "type", None)) if getattr(obj, "data", None) is not None else None,
                "energy": float(getattr(obj.data, "energy", 0.0)) if getattr(obj, "data", None) is not None else None,
                "hide_render": bool(obj.hide_render),
                "hide_viewport": bool(obj.hide_viewport),
            }
        )

    legacy_world = duplicate_world_or_none(scene.world, "LegacyLightingWorldCopy")
    hdr_world, hdr_env_node, hdr_bg_node = build_reusable_hdr_world()
    hdr_image_cache = preload_hdr_images_for_configs(per_camera_lighting)

    dedicated_sun = create_sun_light(name="PerCameraSunLight", energy=float(sunlight_energy))
    dedicated_sun.hide_render = True
    dedicated_sun.hide_viewport = True

    controller = {
        "legacy_world": legacy_world,
        "hdr_world": hdr_world,
        "hdr_env_node": hdr_env_node,
        "hdr_bg_node": hdr_bg_node,
        "hdr_image_cache": hdr_image_cache,
        "dedicated_sun": dedicated_sun,
        "imported_lights": imported_lights,
        "imported_lights_summary": [
            {
                "name": x["name"],
                "light_type": x["light_type"],
                "energy": x["energy"],
                "hide_render": x["hide_render"],
                "hide_viewport": x["hide_viewport"],
            }
            for x in imported_lights
        ],
        "active_signature": None,
    }
    return controller


def sample_per_camera_lighting_configs(
    num_cameras: int,
    seed: int,
    hdr_dir: str,
    sunlight_prob: float = 0.05,
    sunlight_energy: float = 3.0,
    hdr_strength_min: float = 0.2,
    hdr_strength_max: float = 0.5,
):
    sunlight_prob = float(np.clip(sunlight_prob, 0.0, 1.0))
    hdr_strength_min = float(hdr_strength_min)
    hdr_strength_max = float(hdr_strength_max)
    if hdr_strength_min > hdr_strength_max:
        raise ValueError("hdr_strength_min must be <= hdr_strength_max")

    rng = np.random.default_rng(seed)
    hdr_files = list_hdr_files(hdr_dir)

    configs = []
    for view_idx in range(num_cameras):
        use_sunlight = bool(rng.random() < sunlight_prob)
        if use_sunlight:
            config = {
                "view_index": int(view_idx),
                "lighting_type": "sunlight",
                "lighting_seed": int(seed),
                "lighting_compatibility_mode": "legacy_sunlight_exact_per_camera",
                "sunlight_prob": float(sunlight_prob),
                "sunlight_energy": float(sunlight_energy),
                "sunlight_name": "PerCameraSunLight",
                "hdr_dir": str(Path(hdr_dir).resolve()),
                "selected_hdr_index": None,
                "selected_hdr_map": None,
                "selected_hdr_basename": None,
                "num_available_hdr_maps": int(len(hdr_files)),
                "hdr_strength": None,
                "imported_lights_enabled": True,
                "world_mode": "legacy_world_copy",
            }
        else:
            chosen_idx = int(rng.integers(low=0, high=len(hdr_files)))
            chosen_hdr = hdr_files[chosen_idx]
            sampled_strength = float(rng.uniform(hdr_strength_min, hdr_strength_max))
            config = {
                "view_index": int(view_idx),
                "lighting_type": "random_hdr_environment_map",
                "lighting_seed": int(seed),
                "lighting_compatibility_mode": "new_hdr_exact_per_camera",
                "sunlight_prob": float(sunlight_prob),
                "sunlight_energy": None,
                "sunlight_name": None,
                "hdr_dir": str(Path(hdr_dir).resolve()),
                "selected_hdr_index": int(chosen_idx),
                "selected_hdr_map": str(chosen_hdr),
                "selected_hdr_basename": chosen_hdr.name,
                "num_available_hdr_maps": int(len(hdr_files)),
                "hdr_strength": float(sampled_strength),
                "imported_lights_enabled": False,
                "world_mode": "per_camera_hdr_world",
            }
        configs.append(config)

    return configs


def lighting_signature(lighting_config: dict):
    if lighting_config["lighting_type"] == "sunlight":
        return ("sunlight", float(lighting_config["sunlight_energy"]))
    return (
        "hdr",
        str(lighting_config["selected_hdr_map"]),
        round(float(lighting_config["hdr_strength"]), 6),
    )


def apply_per_camera_lighting(scene, lighting_controller, lighting_config):
    sig = lighting_signature(lighting_config)
    if lighting_controller["active_signature"] == sig:
        return

    dedicated_sun = lighting_controller["dedicated_sun"]
    imported_lights = lighting_controller["imported_lights"]
    legacy_world = lighting_controller["legacy_world"]
    hdr_world = lighting_controller["hdr_world"]
    hdr_env_node = lighting_controller["hdr_env_node"]
    hdr_bg_node = lighting_controller["hdr_bg_node"]
    hdr_image_cache = lighting_controller["hdr_image_cache"]

    lighting_type = lighting_config["lighting_type"]
    if lighting_type == "sunlight":
        scene.world = legacy_world
        for info in imported_lights:
            obj = info["obj"]
            if obj.name not in bpy.data.objects:
                continue
            obj.hide_render = bool(info["hide_render"])
            obj.hide_viewport = bool(info["hide_viewport"])

        dedicated_sun.data.energy = float(lighting_config["sunlight_energy"])
        dedicated_sun.hide_render = False
        dedicated_sun.hide_viewport = False

    elif lighting_type == "random_hdr_environment_map":
        for info in imported_lights:
            obj = info["obj"]
            if obj.name not in bpy.data.objects:
                continue
            obj.hide_render = True
            obj.hide_viewport = True

        dedicated_sun.hide_render = True
        dedicated_sun.hide_viewport = True

        hdr_path = str(lighting_config["selected_hdr_map"])
        hdr_env_node.image = hdr_image_cache[hdr_path]
        hdr_bg_node.inputs["Strength"].default_value = float(lighting_config["hdr_strength"])
        scene.world = hdr_world
    else:
        raise ValueError(f"Unknown lighting_type: {lighting_type}")

    lighting_controller["active_signature"] = sig


def render_frame(output_path: str):
    scene = bpy.context.scene
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def setup_normal_output(normal_root_dir: str):
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


def get_bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray):
    x0, y0, z0 = bbox_min.tolist()
    x1, y1, z1 = bbox_max.tolist()
    return np.array(
        [
            [x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
            [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1],
        ],
        dtype=np.float32,
    )


def create_scene_box_object(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    name: str = "CanonicalSceneBox",
    thickness: float = 0.004,
    color=(1.0, 0.2, 0.2),
    emission_strength: float = 1.5,
):
    corners = get_bbox_corners(np.asarray(bbox_min, dtype=np.float32), np.asarray(bbox_max, dtype=np.float32))
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]

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
    mat = create_emission_material(name=name + "Mat", color=rgba, strength=float(emission_strength))
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    obj.hide_render = True
    obj.hide_viewport = False
    return obj


def cleanup_temp_render_file(base_path_no_ext: str):
    for p in (base_path_no_ext, base_path_no_ext + ".png", base_path_no_ext + ".PNG"):
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


# =====================================================================================
# 5. CAMERA FITTING
# =====================================================================================


def compute_camera_axes_from_angles(azim: float, elev: float):
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
    right, up, forward = compute_camera_axes_from_angles(azim, elev)
    R = np.stack([right, up, forward], axis=0).astype(np.float32)
    return R, right, up, forward


def camera_aligned_bbox_corners_to_world(bbox_min_cam: np.ndarray, bbox_max_cam: np.ndarray, azim: float, elev: float):
    corners_cam = get_bbox_corners(np.asarray(bbox_min_cam, dtype=np.float32), np.asarray(bbox_max_cam, dtype=np.float32))
    right, up, forward = compute_camera_axes_from_angles(azim, elev)
    basis_cols = np.stack([right, up, forward], axis=1).astype(np.float32)
    corners_world = corners_cam @ basis_cols.T
    return corners_world.astype(np.float32)


def debug_project_camera_space_bbox(cam_obj, bbox_min_cam, bbox_max_cam, azim: float, elev: float):
    corners_world = camera_aligned_bbox_corners_to_world(
        np.asarray(bbox_min_cam, dtype=np.float32),
        np.asarray(bbox_max_cam, dtype=np.float32),
        azim=azim,
        elev=elev,
    )
    debug_project_world_points(cam_obj, corners_world, title="camera-space sequence bbox mapped to world")


def compute_camera_space_scene_box_from_vertices_seq(vertices_seq: np.ndarray, rot_world_to_cam_aligned: np.ndarray):
    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    rot_t = rot_world_to_cam_aligned.T.astype(np.float32)

    for f in range(vertices_seq.shape[0]):
        verts32 = np.asarray(vertices_seq[f], dtype=np.float32)
        verts_cam = verts32 @ rot_t
        global_min = np.minimum(global_min, verts_cam.min(axis=0))
        global_max = np.maximum(global_max, verts_cam.max(axis=0))

    return global_min.astype(np.float32), global_max.astype(np.float32)


def compute_bbox_unit_normalization_scale(bbox_min: np.ndarray, bbox_max: np.ndarray):
    extent = np.asarray(bbox_max, dtype=np.float32) - np.asarray(bbox_min, dtype=np.float32)
    max_extent = float(np.max(extent))
    scale = 1.0 / max_extent if max_extent > 1e-6 else 1.0
    return float(scale), extent.astype(np.float32)


def compute_tight_camera_distance_for_aligned_bbox(cam_obj, bbox_min_aligned: np.ndarray, bbox_max_aligned: np.ndarray, frame_padding: float = 0.03, fit_safety: float = 1.02):
    corners = get_bbox_corners(np.asarray(bbox_min_aligned, dtype=np.float32), np.asarray(bbox_max_aligned, dtype=np.float32))

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
    normalized_vertices_seq: np.ndarray,
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
            set_camera_intrinsics_from_fov(cam_obj, fov_deg=sampled_fov_deg, sensor_size=float(camera_sensor_size))
        else:
            sampled_fov_deg = float(math.degrees(cam_obj.data.angle_x))
            cam_obj.data.sensor_fit = "HORIZONTAL"
            cam_obj.data.sensor_width = float(camera_sensor_size)
            cam_obj.data.sensor_height = float(camera_sensor_size)

        azim = float(azims[k] % (2.0 * math.pi))
        elev_deg = float(elevs_deg[k])
        elev = math.radians(elev_deg)

        rot_world_to_cam_aligned, right, up, forward = compute_world_to_camera_aligned_rotation(azim=azim, elev=elev)
        camera_space_bbox_min, camera_space_bbox_max = compute_camera_space_scene_box_from_vertices_seq(
            normalized_vertices_seq,
            rot_world_to_cam_aligned=rot_world_to_cam_aligned,
        )

        view_specific_scale, camera_space_extent = compute_bbox_unit_normalization_scale(camera_space_bbox_min, camera_space_bbox_max)
        normalized_camera_space_bbox_min = (camera_space_bbox_min * view_specific_scale).astype(np.float32)
        normalized_camera_space_bbox_max = (camera_space_bbox_max * view_specific_scale).astype(np.float32)

        tight_distance_in_view_normalized_space = compute_tight_camera_distance_for_aligned_bbox(
            cam_obj=cam_obj,
            bbox_min_aligned=normalized_camera_space_bbox_min,
            bbox_max_aligned=normalized_camera_space_bbox_max,
            frame_padding=frame_padding,
            fit_safety=fit_safety,
        )

        tight_distance = float(tight_distance_in_view_normalized_space / max(view_specific_scale, 1e-8))
        distance = tight_distance * float(rng.uniform(1.0, distance_jitter_scale))
        cam_pos = orbit_offset(distance, azim, elev)
        cam2world = look_at(cam_pos, target)
        set_camera_from_cam2world(cam_obj, cam2world)
        intrinsics = get_camera_intrinsics_dict(cam_obj, resolution=resolution)

        camera_infos.append(
            {
                "camera_name": cam_obj.name,
                "view_index": int(k),
                "azimuth_deg": float(np.degrees(azim)),
                "elevation_deg": float(elev_deg),
                "camera_space_scene_box_min": camera_space_bbox_min.tolist(),
                "camera_space_scene_box_max": camera_space_bbox_max.tolist(),
                "camera_space_scene_box_extent": camera_space_extent.tolist(),
                "camera_space_sequence_scale_for_unit_box": float(view_specific_scale),
                "normalized_camera_space_scene_box_min": normalized_camera_space_bbox_min.tolist(),
                "normalized_camera_space_scene_box_max": normalized_camera_space_bbox_max.tolist(),
                "tight_fit_distance_in_view_normalized_space": float(tight_distance_in_view_normalized_space),
                "tight_fit_distance": float(tight_distance),
                "distance": float(distance),
                "distance_scale_from_tight_fit": float(distance / max(tight_distance, 1e-8)),
                "world_to_camera_aligned_rotation": rot_world_to_cam_aligned.tolist(),
                "camera_right_world": right.tolist(),
                "camera_up_world": up.tolist(),
                "camera_forward_world": forward.tolist(),
                "camera_c2w": cam2world.tolist(),
                "camera_pos": cam_pos.tolist(),
                "camera_target": target.tolist(),
                "sampled_common_fov_deg": float(sampled_fov_deg),
                "intrinsics": intrinsics,
                "distance_estimation_method": (
                    "rotate_normalized_sequence_to_camera_space -> "
                    "aggregate_sequence_bbox_in_camera_space -> "
                    "view_specific_bbox_normalization -> "
                    "tight_distance_on_normalized_camera_space_bbox"
                ),
            }
        )
        camera_objs.append(cam_obj)

    return camera_objs, camera_infos


# =====================================================================================
# 6. CORE WORKER
# =====================================================================================


def process_geometry(args):
    timer = StageTimer()
    print("--- Starting Geometry Processing (optimized) ---")

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
    timer.log("scene init + load object")

    sequence_normalizer = create_sequence_normalizer()
    print("Sequence normalizer created.")
    timer.log("create sequence normalizer")

    if args.render_engine == "CYCLES":
        enable_cycles_acceleration(args.cycles_backend)

    setup_renderer(
        resolution=args.resolution,
        engine=args.render_engine,
        transparent_bg=args.transparent_bg,
        cycles_samples=args.cycles_samples,
        cycles_use_denoising=args.cycles_use_denoising,
        cycles_use_adaptive_sampling=(not args.disable_adaptive_sampling),
    )
    normal_output_node = setup_normal_output(normal_dir)
    timer.log("renderer + normal output setup")

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
            f"  name={obj.name}, type={obj.type}, hide_render={obj.hide_render}, "
            f"hide_viewport={obj.hide_viewport}, visible_get={obj.visible_get(view_layer=bpy.context.view_layer)}"
        )

    actions = bpy.data.actions
    frame_start, frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        frame_start = int(min(r[0] for r in ranges))
        frame_end = int(max(r[1] for r in ranges))

    if frame_end - frame_start > 120:
        frame_start = int(np.random.randint(frame_start, frame_end - 120))
        frame_end = frame_start + 120

    frame_indices = np.arange(frame_start, frame_end + 1, dtype=np.int32)
    keyframe_frames = collect_keyframe_frames(frame_start, frame_end)
    if len(keyframe_frames) == 0:
        print("Warning: no explicit keyframes found in actions/fcurves. No per-keyframe PLY will be exported.")
    else:
        print(f"Detected {len(keyframe_frames)} keyframes: {keyframe_frames}")
    keyframe_frame_set = set(keyframe_frames)

    (
        shared_faces,
        centers_seq,
        sequence_scale,
        canonical_bbox_min,
        canonical_bbox_max,
        raw_vertices_cache,
    ) = compute_sequence_normalization_params_and_cache(
        mesh_objs,
        frame_indices,
        cache_raw_vertices=True,
        raw_cache_dtype=np.float16,
    )
    timer.log("first pass: topology + normalization params + raw cache")

    print("Sequence normalization:")
    print(f"  sequence_scale = {sequence_scale}")
    print(f"  canonical_bbox_min = {canonical_bbox_min.tolist()}")
    print(f"  canonical_bbox_max = {canonical_bbox_max.tolist()}")

    vertices_seq, keyframe_ply_records = precompute_normalized_mesh_sequence_from_cache(
        raw_vertices_cache=raw_vertices_cache,
        centers_seq=centers_seq,
        global_scale=sequence_scale,
        shared_faces=shared_faces,
        frame_indices=frame_indices,
        export_keyframe_ply=args.export_keyframe_ply,
        keyframe_frame_set=keyframe_frame_set,
        mesh_ply_dir=mesh_ply_dir,
    )
    timer.log("normalize cached mesh sequence + optional keyframe ply")

    camera_seed = int(args.traj_seed + args.traj_id * 9973)
    camera_objs, camera_infos = create_static_multiview_cameras(
        num_cameras=args.num_cameras,
        seed=camera_seed,
        normalized_vertices_seq=vertices_seq,
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
    timer.log("create static cameras")

    lighting_seed = resolve_lighting_seed(args)
    hdr_strength_min, hdr_strength_max = resolve_hdr_strength_range(args)
    per_camera_lighting = sample_per_camera_lighting_configs(
        num_cameras=args.num_cameras,
        seed=lighting_seed,
        hdr_dir=args.hdr_dir,
        sunlight_prob=args.sunlight_prob,
        sunlight_energy=args.sunlight_energy,
        hdr_strength_min=hdr_strength_min,
        hdr_strength_max=hdr_strength_max,
    )
    lighting_controller = create_per_camera_lighting_controller(
        scene=bpy.context.scene,
        sunlight_energy=args.sunlight_energy,
        per_camera_lighting=per_camera_lighting,
    )
    for view_idx in range(args.num_cameras):
        camera_infos[view_idx]["lighting"] = per_camera_lighting[view_idx]
    bpy.context.scene.camera = camera_objs[0]
    timer.log("lighting configs + reusable HDR setup")

    print("Static multi-view camera setup:")
    for info in camera_infos:
        intr = info["intrinsics"]
        lighting = info["lighting"]
        print(
            f"  view {info['view_index']:02d}: azim={info['azimuth_deg']:.2f} deg, "
            f"elev={info['elevation_deg']:.2f} deg, tight_dist={info['tight_fit_distance']:.4f}, "
            f"dist={info['distance']:.4f}, view_scale={info['camera_space_sequence_scale_for_unit_box']:.4f}, "
            f"fovx={intr['fov_x_deg']:.2f}, fovy={intr['fov_y_deg']:.2f}, "
            f"fx={intr['fx_px']:.2f}, fy={intr['fy_px']:.2f}, "
            f"lighting={lighting['lighting_type']}, hdr_strength={lighting['hdr_strength']}"
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
        timer.log("optional normalized glb export")

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

    final_watertight_data = {}
    from tqdm import tqdm

    render_stage_t0 = time.perf_counter()
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
            lighting_config = per_camera_lighting[view_idx]
            apply_per_camera_lighting(
                scene=bpy.context.scene,
                lighting_controller=lighting_controller,
                lighting_config=lighting_config,
            )

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
            update_normal_output_path(normal_output_node, base_dir=view_normal_dir, prefix="frame_")
            normal_path = os.path.join(view_normal_dir, f"frame_{frame_int:04d}.png")

            render_rgb_and_normal(
                rgb_path=rgb_path,
                normal_output_node=normal_output_node,
                scene_box_obj=scene_box_obj,
                render_scene_box=args.render_scene_box,
                scene_box_affect_normal=args.scene_box_affect_normal,
            )

            views.append(
                {
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
                    "lighting": lighting_config,
                }
            )

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

    print(f"[timing] rendering loop: {time.perf_counter() - render_stage_t0:.3f}s")
    timer.log("render all frames")

    print(f"Saving mesh sequence to {mesh_npz_path} ...")
    os.makedirs(os.path.dirname(mesh_npz_path) or ".", exist_ok=True)
    if args.save_compressed_mesh:
        np.savez_compressed(mesh_npz_path, vertices=vertices_seq, faces=shared_faces, frame_indices=frame_indices)
    else:
        np.savez(mesh_npz_path, vertices=vertices_seq, faces=shared_faces, frame_indices=frame_indices)
    print(
        f"Mesh npz saved. vertices.shape={vertices_seq.shape}, dtype={vertices_seq.dtype}; "
        f"faces.shape={shared_faces.shape}, dtype={shared_faces.dtype}"
    )
    timer.log("save mesh npz")

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
            "per-view camera-space sequence bbox + view-specific bbox normalization + tight distance estimation"
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
        "lighting_mode": "per_camera_fixed_lighting_over_time",
        "lighting_seed": int(lighting_seed),
        "sunlight_prob": float(args.sunlight_prob),
        "sunlight_energy": float(args.sunlight_energy),
        "hdr_strength_min": float(hdr_strength_min),
        "hdr_strength_max": float(hdr_strength_max),
        "per_camera_lighting": per_camera_lighting,
        "imported_light_objects": lighting_controller["imported_lights_summary"],
        "removed_imported_lights": [],
        "optimization_notes": {
            "raw_vertices_cached_in_first_pass": True,
            "mesh_extraction_uses_foreach_get": True,
            "hdr_world_reused": True,
            "save_compressed_mesh": bool(args.save_compressed_mesh),
            "cycles_use_denoising": bool(args.cycles_use_denoising),
        },
    }

    print(f"Saving final metadata to {output_file}...")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(final_watertight_data, indent=2))
    timer.log("save metadata json")
    print("--- Processing Complete ---")


# =====================================================================================
# 7. MAIN
# =====================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-view rendered frames, shared-topology mesh npz, per-keyframe mesh ply, and optionally a normalized GLB from animated GLB/GLTF files using Blender."
    )
    parser.add_argument("--object_path", type=str, required=True, help="Path to the source .glb/.gltf file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output metadata .json file.")
    parser.add_argument(
        "--normalized_glb_path",
        type=str,
        default="",
        help="Optional path to export the normalized animated scene and static cameras as a new .glb file.",
    )

    parser.add_argument("--resolution", type=int, default=512, help="Square render resolution.")
    parser.add_argument(
        "--render_engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["BLENDER_EEVEE", "CYCLES"],
        help="Blender render engine.",
    )
    parser.add_argument("--transparent_bg", action="store_true", help="Render with transparent background.")

    parser.add_argument("--hdr_dir", type=str, default="data/hdr", help="Directory containing HDR maps.")
    parser.add_argument("--hdr_strength", type=float, default=None)
    parser.add_argument("--hdr_strength_min", type=float, default=0.2)
    parser.add_argument("--hdr_strength_max", type=float, default=0.5)
    parser.add_argument("--hdr_seed", type=int, default=None)
    parser.add_argument("--sunlight_prob", type=float, default=0.05)
    parser.add_argument("--sunlight_energy", type=float, default=3.0)

    parser.add_argument("--traj_id", type=int, default=0)
    parser.add_argument("--traj_seed", type=int, default=0)

    parser.add_argument("--num_cameras", type=int, default=16)
    parser.add_argument("--camera_elev_min_deg", type=float, default=0.0)
    parser.add_argument("--camera_elev_max_deg", type=float, default=80.0)
    parser.add_argument("--camera_frame_padding", type=float, default=0.03)
    parser.add_argument("--camera_fit_safety", type=float, default=1.02)
    parser.add_argument("--camera_distance_jitter_scale", type=float, default=1.04)

    parser.add_argument("--randomize_camera_intrinsics", action="store_true")
    parser.add_argument("--camera_fov_min_deg", type=float, default=35.0)
    parser.add_argument("--camera_fov_max_deg", type=float, default=70.0)
    parser.add_argument("--camera_sensor_size", type=float, default=36.0)

    parser.add_argument("--export_keyframe_ply", action="store_true")

    parser.add_argument(
        "--cycles_backend",
        type=str,
        default="OPTIX",
        choices=["CUDA", "OPTIX"],
        help="Cycles GPU backend.",
    )
    parser.add_argument("--cycles_samples", type=int, default=256)
    parser.add_argument(
        "--cycles_use_denoising",
        action="store_true",
        help="Enable Cycles denoising. Default is off in this optimized version.",
    )
    parser.add_argument(
        "--disable_adaptive_sampling",
        action="store_true",
        help="Disable Cycles adaptive sampling.",
    )
    parser.add_argument(
        "--save_compressed_mesh",
        action="store_true",
        help="Use np.savez_compressed for mesh npz. Default is faster uncompressed np.savez.",
    )

    parser.add_argument("--render_scene_box", action="store_true")
    parser.add_argument("--scene_box_affect_normal", action="store_true")
    parser.add_argument("--scene_box_thickness", type=float, default=0.004)
    parser.add_argument("--scene_box_color", type=float, nargs=3, default=[1.0, 0.2, 0.2], metavar=("R", "G", "B"))
    parser.add_argument("--scene_box_emission_strength", type=float, default=1.5)
    parser.add_argument("--debug_camera_projection", action="store_true")

    args = parser.parse_args(get_cli_argv())
    process_geometry(args)
