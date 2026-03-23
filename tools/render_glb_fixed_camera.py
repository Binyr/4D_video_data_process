#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import argparse
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--fps", type=int, default=None)

    parser.add_argument("--transparent_bg", action="store_true")

    # camera
    parser.add_argument("--cam_az_deg", type=float, default=45.0, help="camera azimuth in degrees")
    parser.add_argument("--cam_el_deg", type=float, default=20.0, help="camera elevation in degrees")
    parser.add_argument("--fov_deg", type=float, default=50.0, help="camera horizontal FOV in degrees")
    parser.add_argument("--margin", type=float, default=1.15, help="extra margin for fitting")

    # fit mode
    parser.add_argument(
        "--fit_mode",
        type=str,
        default="all_frames",
        choices=["all_frames", "first_frame"],
        help="Use all animation frames or only first frame to place the static camera.",
    )

    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(glb_path: str):
    ext = os.path.splitext(glb_path)[1].lower()
    if ext not in [".glb", ".gltf"]:
        raise ValueError(f"Unsupported file extension: {ext}")

    bpy.ops.import_scene.gltf(filepath=glb_path)

    scene = bpy.context.scene
    mesh_objs = [obj for obj in scene.objects if obj.type == "MESH"]

    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after importing the GLB.")

    return scene, mesh_objs


def get_frame_range(scene, start_frame=None, end_frame=None):
    # s = scene.frame_start if start_frame is None else start_frame
    # e = scene.frame_end if end_frame is None else end_frame
    # if e < s:
    #     raise ValueError(f"end_frame ({e}) < start_frame ({s})")
    # Get original animation frame range from bpy actions
    actions = bpy.data.actions
    frame_start, frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        frame_start = int(min(r[0] for r in ranges))
        frame_end = int(max(r[1] for r in ranges))
    source_duration = float(frame_end - frame_start)
    return frame_start, frame_end


def get_object_world_bbox(obj_eval):
    """
    Return 8 bbox corners in world coordinates, shape [8, 3].
    """
    mat = obj_eval.matrix_world
    corners = [mat @ Vector(corner) for corner in obj_eval.bound_box]
    return corners


def get_scene_bbox_at_frame(scene, mesh_objs, frame_idx):
    scene.frame_set(frame_idx)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    pts = []
    for obj in mesh_objs:
        obj_eval = obj.evaluated_get(depsgraph)
        corners = get_object_world_bbox(obj_eval)
        pts.extend(corners)

    if len(pts) == 0:
        raise RuntimeError(f"No bbox points found at frame {frame_idx}.")

    pts = [Vector(p) for p in pts]
    xyz = [(p.x, p.y, p.z) for p in pts]

    min_xyz = Vector((min(v[0] for v in xyz), min(v[1] for v in xyz), min(v[2] for v in xyz)))
    max_xyz = Vector((max(v[0] for v in xyz), max(v[1] for v in xyz), max(v[2] for v in xyz)))
    return min_xyz, max_xyz


def merge_bbox(bbox_list):
    mins = [b[0] for b in bbox_list]
    maxs = [b[1] for b in bbox_list]

    min_xyz = Vector((
        min(v.x for v in mins),
        min(v.y for v in mins),
        min(v.z for v in mins),
    ))
    max_xyz = Vector((
        max(v.x for v in maxs),
        max(v.y for v in maxs),
        max(v.z for v in maxs),
    ))
    return min_xyz, max_xyz


def compute_global_bbox(scene, mesh_objs, start_frame, end_frame, fit_mode="all_frames"):
    if fit_mode == "first_frame":
        return get_scene_bbox_at_frame(scene, mesh_objs, start_frame)

    bboxes = []
    for f in range(start_frame, end_frame + 1):
        bboxes.append(get_scene_bbox_at_frame(scene, mesh_objs, f))
    return merge_bbox(bboxes)


def look_at(obj, target: Vector):
    direction = target - obj.location
    quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = quat.to_euler()


def setup_camera(scene, bbox_min, bbox_max, resolution, cam_az_deg, cam_el_deg, fov_deg, margin):
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    radius = max(extent.length * 0.5, 1e-6) * margin

    # Create camera
    cam_data = bpy.data.cameras.new(name="StaticCamera")
    cam = bpy.data.objects.new("StaticCamera", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    # Camera settings
    cam_data.type = "PERSP"
    cam_data.angle = math.radians(fov_deg)

    aspect = 1.0  # because resolution_x == resolution_y below
    fov_x = math.radians(fov_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) / aspect)
    fit_fov = min(fov_x, fov_y)

    distance = radius / math.tan(fit_fov * 0.5)

    az = math.radians(cam_az_deg)
    el = math.radians(cam_el_deg)

    # spherical direction
    dir_x = math.cos(el) * math.cos(az)
    dir_y = math.cos(el) * math.sin(az)
    dir_z = math.sin(el)

    cam.location = center + Vector((dir_x, dir_y, dir_z)) * distance
    look_at(cam, center)

    return cam, center, radius, distance


def setup_lighting(transparent_bg=False):
    # World light
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[1].default_value = 0.8  # strength

    # Key sun
    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light_data.energy = 2.5
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(35), 0.0, math.radians(45))

    # Fill area
    area_data = bpy.data.lights.new(name="Area", type="AREA")
    area_data.energy = 2000
    area_data.shape = "RECTANGLE"
    area_data.size = 6.0
    area_data.size_y = 6.0
    area_obj = bpy.data.objects.new(name="Area", object_data=area_data)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (4.0, -4.0, 4.0)
    area_obj.rotation_euler = (math.radians(50), 0.0, math.radians(35))


def setup_render(scene, output_dir, resolution, engine="CYCLES", samples=64, transparent_bg=False, fps=None):
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    if fps is not None:
        scene.render.fps = fps

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent_bg else "RGB"
    scene.render.film_transparent = transparent_bg
    scene.render.use_file_extension = True

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    elif engine == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = samples

    Path(output_dir).mkdir(parents=True, exist_ok=True)


def render_frames(scene, output_dir, start_frame, end_frame):
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        out_path = os.path.join(output_dir, f"frame_{frame:04d}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"[Rendered] frame={frame} -> {out_path}")


def main():
    args = parse_args()

    clear_scene()
    scene, mesh_objs = import_glb(args.glb_path)

    start_frame, end_frame = get_frame_range(scene, args.start_frame, args.end_frame)

    print("=" * 80)
    print(f"GLB:         {args.glb_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Frames:      [{start_frame}, {end_frame}]")
    print(f"Fit mode:    {args.fit_mode}")
    print(f"Resolution:  {args.resolution}")
    print(f"Engine:      {args.engine}")
    print("=" * 80)

    bbox_min, bbox_max = compute_global_bbox(
        scene=scene,
        mesh_objs=mesh_objs,
        start_frame=start_frame,
        end_frame=end_frame,
        fit_mode=args.fit_mode,
    )

    print(f"BBox min: {bbox_min}")
    print(f"BBox max: {bbox_max}")

    cam, center, radius, distance = setup_camera(
        scene=scene,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=args.resolution,
        cam_az_deg=args.cam_az_deg,
        cam_el_deg=args.cam_el_deg,
        fov_deg=args.fov_deg,
        margin=args.margin,
    )

    print(f"Camera center target: {center}")
    print(f"Camera distance:      {distance:.6f}")
    print(f"Camera location:      {cam.location}")

    setup_lighting(transparent_bg=args.transparent_bg)

    setup_render(
        scene=scene,
        output_dir=args.output_dir,
        resolution=args.resolution,
        engine=args.engine,
        samples=args.samples,
        transparent_bg=args.transparent_bg,
        fps=args.fps,
    )

    render_frames(
        scene=scene,
        output_dir=args.output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
/efs/yanruibin/projects/blender-4.2.1-linux-x64/blender --background --python tools/render_glb_fixed_camera.py -- \
    --glb_path data/objverse_minghao_4d/glbs/000-033/5dd2ce713485413a84bceacf15e40b9f.glb \
    --output_dir vis/fixed_cam_render/000-033/5dd2ce713485413a84bceacf15e40b9f \
    --resolution 512 \
    --engine CYCLES \
    --samples 64 \
    --fit_mode all_frames \
    --cam_az_deg 45 \
    --cam_el_deg 20 \
    --fov_deg 50 \
    --transparent_bg

python tools/folder_to_mp4.py \
    --input_dir vis/fixed_cam_render/000-033/5dd2ce713485413a84bceacf15e40b9f/ \
    --output_path vis/fixed_cam_render/000-033/5dd2ce713485413a84bceacf15e40b9f.mp4 \
    --fps 30
"""