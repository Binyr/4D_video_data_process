#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Print object->world transforms for all animated objects in a GLB/GLTF.

Usage:
    blender --background --python print_glb_object_world_transforms.py -- \
        --object_path path/to/xxx.glb

Optional:
    --frame_start 1
    --frame_end 120
    --tol 1e-6
    --print_static
"""

import os
import sys
import argparse
import numpy as np
import bpy


# ----------------------------
# Helpers
# ----------------------------
def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=str, required=True)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--print_static",
        action="store_true",
        help="If set, also print static objects once.",
    )
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".glb", ".gltf"]:
        raise ValueError(f"Unsupported file type: {ext}")
    bpy.ops.import_scene.gltf(filepath=filepath)


def get_candidate_objects():
    # Exclude obvious non-object-scene helpers by default
    exclude_types = {"CAMERA", "LIGHT", "SPEAKER", "LIGHT_PROBE"}
    objs = [obj for obj in bpy.context.scene.objects if obj.type not in exclude_types]
    return objs


def matrix_to_numpy(mat):
    # Blender Matrix -> numpy (4,4)
    return np.array([[mat[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def matrix_changed(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return not np.allclose(a, b, atol=tol, rtol=0.0)


def format_matrix(mat_np: np.ndarray) -> str:
    return np.array2string(
        mat_np,
        precision=6,
        suppress_small=False,
        separator=", ",
        max_line_width=200,
    )


def print_trs_from_matrix(mat_world):
    loc, rot, scale = mat_world.decompose()
    print(f"    location   : ({loc.x:.6f}, {loc.y:.6f}, {loc.z:.6f})")
    print(f"    quaternion : ({rot.w:.6f}, {rot.x:.6f}, {rot.y:.6f}, {rot.z:.6f})")
    print(f"    scale      : ({scale.x:.6f}, {scale.y:.6f}, {scale.z:.6f})")


# ----------------------------
# Main scan
# ----------------------------
def main():
    args = parse_args()

    if not os.path.isfile(args.object_path):
        raise FileNotFoundError(f"File not found: {args.object_path}")

    clear_scene()
    import_glb(args.object_path)

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    frame_start = args.frame_start if args.frame_start is not None else scene.frame_start
    frame_end = args.frame_end if args.frame_end is not None else scene.frame_end

    if frame_end < frame_start:
        raise ValueError(f"frame_end ({frame_end}) < frame_start ({frame_start})")

    objects = get_candidate_objects()
    if len(objects) == 0:
        print("No candidate objects found.")
        return

    print("=" * 100)
    print(f"GLB: {args.object_path}")
    print(f"Scene frame range: [{scene.frame_start}, {scene.frame_end}]")
    print(f"Scan frame range : [{frame_start}, {frame_end}]")
    print(f"Tolerance        : {args.tol}")
    print(f"Candidate objects: {len(objects)}")
    print("=" * 100)

    # Store per-object per-frame matrix_world
    per_obj_mats = {obj.name: {} for obj in objects}

    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        view_layer.update()

        for obj in objects:
            mat_np = matrix_to_numpy(obj.matrix_world)
            per_obj_mats[obj.name][f] = mat_np

    dynamic_objects = []
    static_objects = []

    # Decide which objects are dynamic in object->world sense
    for obj in objects:
        mats = per_obj_mats[obj.name]
        ref = mats[frame_start]
        changed_frames = []

        for f in range(frame_start + 1, frame_end + 1):
            if matrix_changed(ref, mats[f], args.tol):
                changed_frames.append(f)

        if len(changed_frames) > 0:
            dynamic_objects.append((obj, changed_frames))
        else:
            static_objects.append(obj)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Dynamic objects (matrix_world changes over time): {len(dynamic_objects)}")
    print(f"Static objects  (matrix_world constant)         : {len(static_objects)}")

    if len(dynamic_objects) == 0:
        print("\nNo dynamic objects found under the definition: matrix_world changes across frames.")
        print("This can happen if the animation is only bone deformation / shape deformation,")
        print("while the object-level transform itself stays constant.")
        if args.print_static and len(static_objects) > 0:
            print("\nStatic objects:")
            for obj in static_objects:
                print(f"  - {obj.name} [{obj.type}]")
        return

    print("\nDynamic objects list:")
    for obj, changed_frames in dynamic_objects:
        print(f"  - {obj.name} [{obj.type}]")
        print(f"    changed frames (vs frame {frame_start}): {changed_frames}")

    if args.print_static and len(static_objects) > 0:
        print("\nStatic objects:")
        for obj in static_objects:
            print(f"  - {obj.name} [{obj.type}]")

    # Print every frame's object->world transform for dynamic objects
    print("\n" + "=" * 100)
    print("PER-FRAME OBJECT -> WORLD TRANSFORMS")
    print("=" * 100)

    for obj, changed_frames in dynamic_objects:
        print("\n" + "#" * 100)
        print(f"Object: {obj.name}")
        print(f"Type  : {obj.type}")
        print(f"Changed frames (vs first scanned frame {frame_start}): {changed_frames}")
        print("#" * 100)

        for f in range(frame_start, frame_end + 1):
            scene.frame_set(f)
            view_layer.update()

            mat_world = obj.matrix_world.copy()
            mat_np = per_obj_mats[obj.name][f]

            print(f"\n[Frame {f}]")
            print_trs_from_matrix(mat_world)
            print("    matrix_world:")
            print(format_matrix(mat_np))

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Note:")
    print("  - This script detects object-level motion via matrix_world.")
    print("  - Pure mesh deformation / armature deformation with fixed object transform")
    print("    will NOT be marked dynamic here.")


if __name__ == "__main__":
    main()