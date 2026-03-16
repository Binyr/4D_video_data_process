#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check whether mesh topology changes across frames in a GLB/GLTF animation.

Usage:
blender --background --python check_glb_topology_change.py -- \
    --glb_path path/to/model.glb \
    --json_out result.json

Optional:
    --start_frame 1
    --end_frame 120
    --ignore_hidden
    --use_scene_frame_range
"""

import os
import sys
import json
import argparse
import hashlib
from typing import Dict, Any, List, Tuple

import bpy
import numpy as np


IMPORT_FUNCTIONS = {
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
}


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_path", type=str, required=True)
    parser.add_argument("--json_out", type=str, default="")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument(
        "--ignore_hidden",
        action="store_true",
        help="Ignore hidden mesh objects.",
    )
    parser.add_argument(
        "--use_scene_frame_range",
        action="store_true",
        help="Use scene.frame_start/frame_end directly instead of action ranges.",
    )
    return parser.parse_args(argv)


def init_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image, do_unlink=True)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)
    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature, do_unlink=True)
    for action in list(bpy.data.actions):
        # 不强制删除 action；这里只是避免旧场景残留影响
        pass


def load_object(object_path: str):
    ext = object_path.split(".")[-1].lower()
    if ext not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    IMPORT_FUNCTIONS[ext](filepath=object_path)


def get_frame_range(use_scene_frame_range: bool) -> Tuple[int, int]:
    scene = bpy.context.scene
    if use_scene_frame_range:
        return int(scene.frame_start), int(scene.frame_end)

    # 优先从 actions 推断
    actions = bpy.data.actions
    if actions:
        ranges = [a.frame_range for a in actions if a is not None]
        if len(ranges) > 0:
            frame_start = int(min(r[0] for r in ranges))
            frame_end = int(max(r[1] for r in ranges))
            return frame_start, frame_end

    return int(scene.frame_start), int(scene.frame_end)


def mesh_triangle_hash(mesh) -> str:
    """
    对 triangulated connectivity 做哈希。
    这里只比较 loop_triangles 对应的 vertex indices。
    """
    mesh.calc_loop_triangles()
    if len(mesh.loop_triangles) == 0:
        tri = np.zeros((0, 3), dtype=np.int32)
    else:
        tri = np.array([lt.vertices[:] for lt in mesh.loop_triangles], dtype=np.int32)

    # 直接按当前顺序 hash。
    # 对大多数 GLB 骨骼/形变动画，这已经足够判断 connectivity 是否变了。
    h = hashlib.sha1()
    h.update(tri.tobytes())
    return h.hexdigest()


def collect_topology_signature(ignore_hidden: bool = False) -> Dict[str, Any]:
    """
    收集当前帧所有 mesh object 的拓扑签名。
    返回 dict:
    {
        "object_names": [...],
        "objects": {
            obj_name: {
                "num_vertices": ...,
                "num_triangles": ...,
                "triangle_hash": ...
            }
        }
    }
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    mesh_objs = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if ignore_hidden and (obj.hide_get() or not obj.visible_get()):
            continue
        mesh_objs.append(obj)

    mesh_objs = sorted(mesh_objs, key=lambda x: x.name)

    result = {
        "object_names": [obj.name for obj in mesh_objs],
        "objects": {}
    }

    for obj in mesh_objs:
        obj_eval = obj.evaluated_get(depsgraph)
        temp_mesh = obj_eval.to_mesh()

        if temp_mesh is None:
            result["objects"][obj.name] = {
                "num_vertices": 0,
                "num_triangles": 0,
                "triangle_hash": hashlib.sha1(b"").hexdigest(),
            }
            continue

        try:
            temp_mesh.calc_loop_triangles()
            num_vertices = len(temp_mesh.vertices)
            num_triangles = len(temp_mesh.loop_triangles)
            tri_hash = mesh_triangle_hash(temp_mesh)

            result["objects"][obj.name] = {
                "num_vertices": int(num_vertices),
                "num_triangles": int(num_triangles),
                "triangle_hash": tri_hash,
            }
        finally:
            obj_eval.to_mesh_clear()

    return result


def compare_signatures(ref_sig: Dict[str, Any], cur_sig: Dict[str, Any]) -> Dict[str, Any]:
    """
    比较两帧签名，返回差异信息。
    """
    ref_names = ref_sig["object_names"]
    cur_names = cur_sig["object_names"]

    if ref_names != cur_names:
        return {
            "same": False,
            "reason": "object_set_changed",
            "ref_object_names": ref_names,
            "cur_object_names": cur_names,
        }

    for name in ref_names:
        ref_info = ref_sig["objects"][name]
        cur_info = cur_sig["objects"][name]

        if ref_info["num_vertices"] != cur_info["num_vertices"]:
            return {
                "same": False,
                "reason": "vertex_count_changed",
                "object_name": name,
                "ref_num_vertices": ref_info["num_vertices"],
                "cur_num_vertices": cur_info["num_vertices"],
            }

        if ref_info["num_triangles"] != cur_info["num_triangles"]:
            return {
                "same": False,
                "reason": "triangle_count_changed",
                "object_name": name,
                "ref_num_triangles": ref_info["num_triangles"],
                "cur_num_triangles": cur_info["num_triangles"],
            }

        if ref_info["triangle_hash"] != cur_info["triangle_hash"]:
            return {
                "same": False,
                "reason": "triangle_connectivity_changed",
                "object_name": name,
                "ref_triangle_hash": ref_info["triangle_hash"],
                "cur_triangle_hash": cur_info["triangle_hash"],
            }

    return {"same": True, "reason": "all_same"}


def main():
    args = parse_args()

    if not os.path.isfile(args.glb_path):
        raise FileNotFoundError(f"GLB file not found: {args.glb_path}")

    print(f"[INFO] Loading: {args.glb_path}")
    init_scene()
    load_object(args.glb_path)

    scene_start, scene_end = get_frame_range(args.use_scene_frame_range)

    start_frame = scene_start if args.start_frame is None else args.start_frame
    end_frame = scene_end if args.end_frame is None else args.end_frame

    if end_frame < start_frame:
        raise ValueError(f"end_frame ({end_frame}) < start_frame ({start_frame})")

    print(f"[INFO] Frame range: {start_frame} -> {end_frame}")

    # 参考帧
    bpy.context.scene.frame_set(start_frame)
    bpy.context.view_layer.update()
    ref_sig = collect_topology_signature(ignore_hidden=args.ignore_hidden)

    if len(ref_sig["object_names"]) == 0:
        raise RuntimeError("No mesh objects found in the loaded scene.")

    result = {
        "glb_path": args.glb_path,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "reference_frame": int(start_frame),
        "reference_object_names": ref_sig["object_names"],
        "topology_changed": False,
        "first_changed_frame": None,
        "change_detail": None,
        "checked_frames": [],
    }

    print("[INFO] Reference objects:")
    for name in ref_sig["object_names"]:
        info = ref_sig["objects"][name]
        print(
            f"  - {name}: V={info['num_vertices']}, T={info['num_triangles']}"
        )

    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        cur_sig = collect_topology_signature(ignore_hidden=args.ignore_hidden)
        cmp_result = compare_signatures(ref_sig, cur_sig)

        result["checked_frames"].append({
            "frame": int(frame),
            "same_as_reference": bool(cmp_result["same"]),
            "reason": cmp_result["reason"],
        })

        if not cmp_result["same"]:
            result["topology_changed"] = True
            result["first_changed_frame"] = int(frame)
            result["change_detail"] = cmp_result
            print(f"[WARN] Topology changed at frame {frame}: {cmp_result}")
            break

        print(f"[OK] Frame {frame}: topology same as reference")

    if not result["topology_changed"]:
        print("[INFO] No topology change detected across all checked frames.")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Result written to: {args.json_out}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
/efs/yanruibin/projects/blender-4.2.1-linux-x64/blender --background --python tools/check_glb_topology_change.py -- \
    --glb_path data/objverse_minghao_4d/glbs/000-033/5dd2ce713485413a84bceacf15e40b9f.glb
"""