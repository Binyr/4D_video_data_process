#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch check whether topology changes across frames for all GLBs in a directory.

This file supports 3 modes:

1) launcher
   Run by regular Python. It scans all .glb files, launches multiple Blender
   worker processes in parallel, and incrementally writes a single JSON dict.

2) worker
   Run inside Blender. It loads one .glb, checks topology frame by frame,
   and writes one JSON result.

3) summary
   Run by regular Python. It reads the final JSON dict and prints statistics.

Example:

python batch_check_glb_topology.py \
    --mode launcher \
    --root_glb_dir data/objverse_minghao_4d/glbs \
    --result_json results/topology_check.json \
    --blender_path /efs/yanruibin/projects/blender-4.2.1-linux-x64/blender \
    --num_workers 8

python batch_check_glb_topology.py \
    --mode summary \
    --result_json results/topology_check.json
"""

import os
import sys
import json
import time
import hashlib
import argparse
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================
# Common utils
# ============================================================

def get_user_argv():
    """
    Works both for normal python and blender --python xxx.py -- ...
    """
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return argv[1:]


def load_json_safe(path: str, default):
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def atomic_write_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def iter_glb_files(root_glb_dir: str) -> List[str]:
    root = Path(root_glb_dir)
    glb_paths = sorted([str(p) for p in root.rglob("*.glb") if p.is_file()])
    return glb_paths


def make_rel_path(glb_path: str, root_glb_dir: str) -> str:
    return os.path.relpath(os.path.abspath(glb_path), os.path.abspath(root_glb_dir))


def make_part_file(parts_dir: str, rel_path: str) -> str:
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    base = os.path.basename(rel_path)
    safe_name = f"{h}__{base}.json"
    return os.path.join(parts_dir, safe_name)


def merge_part_files_into_results(parts_dir: str, results: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.isdir(parts_dir):
        return results

    for p in sorted(Path(parts_dir).glob("*.json")):
        item = load_json_safe(str(p), None)
        if not isinstance(item, dict):
            continue
        rel_path = item.get("rel_path", None)
        if rel_path is None:
            continue
        results[rel_path] = item
    return results


def summarize_results(result_json: str, verbose: bool = True) -> Dict[str, int]:
    data = load_json_safe(result_json, {})
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid result json: {result_json}")

    changed = 0
    unchanged = 0
    failed = 0

    for _, item in data.items():
        if not isinstance(item, dict):
            failed += 1
            continue

        status = item.get("status", "unknown")
        if status != "ok":
            failed += 1
            continue

        topo_changed = bool(item.get("topology_changed", False))
        if topo_changed:
            changed += 1
        else:
            unchanged += 1

    total = changed + unchanged + failed

    summary = {
        "total": total,
        "changed": changed,
        "unchanged": unchanged,
        "failed": failed,
    }

    if verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


# ============================================================
# Worker mode (runs inside Blender)
# ============================================================

def worker_main(args):
    """
    This part is executed inside Blender.
    We import bpy only here, so launcher mode can run with normal Python.
    """
    import bpy
    import numpy as np

    IMPORT_FUNCTIONS = {
        "glb": bpy.ops.import_scene.gltf,
        "gltf": bpy.ops.import_scene.gltf,
    }

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

    def load_object(object_path: str):
        ext = object_path.split(".")[-1].lower()
        if ext not in IMPORT_FUNCTIONS:
            raise ValueError(f"Unsupported file type: {object_path}")
        IMPORT_FUNCTIONS[ext](filepath=object_path)

    def get_frame_range(use_scene_frame_range: bool) -> Tuple[int, int]:
        scene = bpy.context.scene
        if use_scene_frame_range:
            return int(scene.frame_start), int(scene.frame_end)

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
        Hash triangle connectivity in a relatively order-robust way:
        - sort each triangle's 3 vertex ids
        - lexicographically sort all triangles
        - hash the final array

        This ignores triangle winding / raw face order changes,
        but still detects connectivity changes.
        """
        mesh.calc_loop_triangles()
        if len(mesh.loop_triangles) == 0:
            tri = np.zeros((0, 3), dtype=np.int32)
        else:
            tri = np.array([lt.vertices[:] for lt in mesh.loop_triangles], dtype=np.int32)

        if tri.shape[0] > 0:
            tri = np.sort(tri, axis=1)
            order = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
            tri = tri[order]

        h = hashlib.sha1()
        h.update(tri.tobytes())
        return h.hexdigest()

    def collect_topology_signature(ignore_hidden: bool = False) -> Dict[str, Any]:
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

    try:
        if not os.path.isfile(args.glb_path):
            raise FileNotFoundError(f"GLB file not found: {args.glb_path}")

        init_scene()
        load_object(args.glb_path)

        scene_start, scene_end = get_frame_range(args.use_scene_frame_range)
        start_frame = scene_start if args.start_frame is None else args.start_frame
        end_frame = scene_end if args.end_frame is None else args.end_frame

        if end_frame < start_frame:
            raise ValueError(f"end_frame ({end_frame}) < start_frame ({start_frame})")

        bpy.context.scene.frame_set(start_frame)
        bpy.context.view_layer.update()

        ref_sig = collect_topology_signature(ignore_hidden=args.ignore_hidden)
        if len(ref_sig["object_names"]) == 0:
            raise RuntimeError("No mesh objects found in the loaded scene.")

        result = {
            "rel_path": args.rel_path,
            "glb_path": os.path.abspath(args.glb_path),
            "status": "ok",
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "reference_frame": int(start_frame),
            "reference_object_names": ref_sig["object_names"],
            "topology_changed": False,
            "first_changed_frame": None,
            "change_detail": None,
            "checked_frame_count": 0,
        }

        for frame in range(start_frame, end_frame + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()

            cur_sig = collect_topology_signature(ignore_hidden=args.ignore_hidden)
            cmp_result = compare_signatures(ref_sig, cur_sig)
            result["checked_frame_count"] += 1

            if not cmp_result["same"]:
                result["topology_changed"] = True
                result["first_changed_frame"] = int(frame)
                result["change_detail"] = cmp_result
                break

        if args.worker_json_out:
            atomic_write_json(args.worker_json_out, result)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        err = {
            "rel_path": args.rel_path,
            "glb_path": os.path.abspath(args.glb_path) if args.glb_path else "",
            "status": "failed",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(limit=20),
        }
        if args.worker_json_out:
            atomic_write_json(args.worker_json_out, err)
        else:
            print(json.dumps(err, indent=2, ensure_ascii=False))


# ============================================================
# Launcher mode (runs in normal Python)
# ============================================================

def run_single_blender_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    One launcher-side task.
    It starts one Blender process to check one GLB.
    """
    blender_path = job["blender_path"]
    script_path = job["script_path"]
    glb_path = job["glb_path"]
    rel_path = job["rel_path"]
    part_file = job["part_file"]

    # If part file already exists and is OK, skip.
    old = load_json_safe(part_file, None)
    if isinstance(old, dict) and old.get("status") == "ok":
        return {
            "rel_path": rel_path,
            "status": "skipped_existing_part",
            "part_file": part_file,
        }

    cmd = [
        blender_path,
        "--background",
        "--python",
        script_path,
        "--",
        "--mode", "worker",
        "--glb_path", glb_path,
        "--rel_path", rel_path,
        "--worker_json_out", part_file,
    ]

    if job.get("start_frame") is not None:
        cmd += ["--start_frame", str(job["start_frame"])]
    if job.get("end_frame") is not None:
        cmd += ["--end_frame", str(job["end_frame"])]
    if job.get("ignore_hidden", False):
        cmd += ["--ignore_hidden"]
    if job.get("use_scene_frame_range", False):
        cmd += ["--use_scene_frame_range"]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Prefer part_file content as the source of truth.
    item = load_json_safe(part_file, None)

    if isinstance(item, dict):
        return {
            "rel_path": rel_path,
            "status": item.get("status", "unknown"),
            "part_file": part_file,
            "returncode": proc.returncode,
        }

    # Blender failed and produced no usable json.
    fail_item = {
        "rel_path": rel_path,
        "glb_path": os.path.abspath(glb_path),
        "status": "failed",
        "error_type": "BlenderSubprocessError",
        "error": f"Blender exited with code {proc.returncode}",
        "stdout_tail": proc.stdout[-3000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-3000:] if proc.stderr else "",
    }
    atomic_write_json(part_file, fail_item)

    return {
        "rel_path": rel_path,
        "status": "failed",
        "part_file": part_file,
        "returncode": proc.returncode,
    }


def launcher_main(args):
    root_glb_dir = os.path.abspath(args.root_glb_dir)
    result_json = os.path.abspath(args.result_json)
    parts_dir = result_json + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    script_path = os.path.abspath(__file__)
    blender_path = args.blender_path

    if not os.path.isdir(root_glb_dir):
        raise FileNotFoundError(f"root_glb_dir not found: {root_glb_dir}")

    print(f"[INFO] Scanning GLBs under: {root_glb_dir}")
    glb_paths = iter_glb_files(root_glb_dir)
    print(f"[INFO] Found {len(glb_paths)} GLBs")

    # Load old master result and merge existing part files.
    results = load_json_safe(result_json, {})
    if not isinstance(results, dict):
        results = {}
    results = merge_part_files_into_results(parts_dir, results)
    atomic_write_json(result_json, results)

    jobs = []
    skipped_master = 0

    for glb_path in glb_paths:
        rel_path = make_rel_path(glb_path, root_glb_dir)

        # Skip if master result already has a successful record.
        old = results.get(rel_path, None)
        if isinstance(old, dict) and old.get("status") == "ok":
            skipped_master += 1
            continue

        part_file = make_part_file(parts_dir, rel_path)
        part_item = load_json_safe(part_file, None)
        if isinstance(part_item, dict) and part_item.get("status") == "ok":
            results[rel_path] = part_item
            continue

        jobs.append({
            "blender_path": blender_path,
            "script_path": script_path,
            "glb_path": glb_path,
            "rel_path": rel_path,
            "part_file": part_file,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "ignore_hidden": args.ignore_hidden,
            "use_scene_frame_range": args.use_scene_frame_range,
        })

    atomic_write_json(result_json, results)

    print(f"[INFO] Already completed in master json: {skipped_master}")
    print(f"[INFO] Pending jobs: {len(jobs)}")
    print(f"[INFO] num_workers = {args.num_workers}")

    if len(jobs) == 0:
        print("[INFO] Nothing to do.")
        summarize_results(result_json, verbose=True)
        return

    done_count = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(run_single_blender_job, job) for job in jobs]

        for fut in as_completed(futures):
            done_count += 1
            out = fut.result()

            # Reload the part file and merge it into master json.
            rel_path = out["rel_path"]
            part_file = out["part_file"]
            item = load_json_safe(part_file, None)

            if isinstance(item, dict):
                results[rel_path] = item
            else:
                results[rel_path] = {
                    "rel_path": rel_path,
                    "status": "failed",
                    "error_type": "MissingPartFile",
                    "error": f"Part file missing or unreadable: {part_file}",
                }

            atomic_write_json(result_json, results)

            status = results[rel_path].get("status", "unknown")
            topo_changed = results[rel_path].get("topology_changed", None)
            print(
                f"[{done_count}/{len(jobs)}] {rel_path} | status={status} | "
                f"topology_changed={topo_changed}"
            )

    elapsed = time.time() - t0
    print(f"[INFO] All jobs finished. Elapsed: {elapsed:.1f}s")
    summarize_results(result_json, verbose=True)


# ============================================================
# Argument parsing / main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="launcher",
        choices=["launcher", "worker", "summary"],
        help="launcher / worker / summary"
    )

    # launcher / summary
    parser.add_argument("--root_glb_dir", type=str, default="")
    parser.add_argument("--result_json", type=str, default="")
    parser.add_argument("--blender_path", type=str, default="blender")
    parser.add_argument("--num_workers", type=int, default=8)

    # shared frame-range options
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--ignore_hidden", action="store_true")
    parser.add_argument("--use_scene_frame_range", action="store_true")

    # worker-only
    parser.add_argument("--glb_path", type=str, default="")
    parser.add_argument("--rel_path", type=str, default="")
    parser.add_argument("--worker_json_out", type=str, default="")

    return parser.parse_args(get_user_argv())


def main():
    args = parse_args()

    if args.mode == "worker":
        worker_main(args)
        return

    if args.mode == "summary":
        if not args.result_json:
            raise ValueError("--result_json is required for summary mode")
        summarize_results(args.result_json, verbose=True)
        return

    # launcher
    if not args.root_glb_dir:
        raise ValueError("--root_glb_dir is required for launcher mode")
    if not args.result_json:
        raise ValueError("--result_json is required for launcher mode")

    launcher_main(args)


if __name__ == "__main__":
    main()

"""
python tools/check_glb_topology_change_mp.py \
    --mode launcher \
    --root_glb_dir data/objverse_minghao_4d/glbs \
    --result_json data/objverse_minghao_4d/topology_check.json \
    --blender_path /efs/yanruibin/projects/blender-4.2.1-linux-x64/blender \
    --num_workers 8
"""