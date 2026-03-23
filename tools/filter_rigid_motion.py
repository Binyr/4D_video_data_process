#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use Blender to:
1) import a .glb / .gltf
2) extract evaluated mesh vertices for each frame
3) check whether the mesh sequence differs only by a global similarity transform:
      y ~= s * R * x + t
   where R is rotation, s is a single global scale, t is translation.

Usage:
blender --background --python check_glb_similarity_motion.py -- \
    --glb_path path/to/xxx.glb \
    --tol_rel 1e-4 \
    --max_verts 50000 \
    --json_out result.json
"""

import os
import sys
import json
import argparse
import numpy as np
import bpy


def parse_args():
    argv = sys.argv
    # print(argv)
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_path", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument(
        "--tol_rel",
        type=float,
        default=1e-4,
        help="Relative RMS error threshold, normalized by reference bbox diagonal.",
    )
    parser.add_argument(
        "--max_verts",
        type=int,
        default=50000,
        help="Randomly subsample vertices for fitting/checking. 0 or negative means use all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_out", type=str, default=None)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(glb_path: str):
    ext = os.path.splitext(glb_path)[1].lower()
    if ext not in [".glb", ".gltf"]:
        raise ValueError(f"Unsupported file: {glb_path}")

    bpy.ops.import_scene.gltf(filepath=glb_path)

    scene = bpy.context.scene
    mesh_objs = sorted(
        [obj for obj in scene.objects if obj.type == "MESH"],
        key=lambda x: x.name_full,
    )
    if len(mesh_objs) == 0:
        raise RuntimeError("No mesh objects found after import.")

    return scene, mesh_objs


def get_mesh_vertices_world(obj_eval) -> np.ndarray:
    """
    Return evaluated mesh vertices in world coordinates, shape [N, 3].
    """
    mesh = obj_eval.to_mesh()
    try:
        n = len(mesh.vertices)
        verts = np.empty(n * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", verts)
        verts = verts.reshape(n, 3)

        M = np.array(obj_eval.matrix_world, dtype=np.float64)  # [4,4]
        R = M[:3, :3]
        t = M[:3, 3]
        verts_world = verts @ R.T + t[None, :]
        return verts_world
    finally:
        obj_eval.to_mesh_clear()


def get_frame_vertices(scene, mesh_objs):
    """
    Concatenate all mesh vertices in deterministic object-name order.
    Also return topology signature for consistency checks.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    verts_all = []
    topo_sig = []

    for obj in mesh_objs:
        obj_eval = obj.evaluated_get(depsgraph)
        v = get_mesh_vertices_world(obj_eval)
        verts_all.append(v)

        # Use evaluated mesh again for polygon count
        mesh = obj_eval.to_mesh()
        try:
            topo_sig.append((obj.name_full, len(mesh.vertices), len(mesh.polygons)))
        finally:
            obj_eval.to_mesh_clear()

    verts_all = np.concatenate(verts_all, axis=0)
    return verts_all, topo_sig


def bbox_diag(verts: np.ndarray) -> float:
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def umeyama_similarity(X: np.ndarray, Y: np.ndarray):
    """
    Fit Y ~= s * R * X + t
    X, Y: [N, 3], corresponded points
    Returns:
        s: scalar
        R: [3,3], det(R)=+1
        t: [3]
        aligned: [N,3]
        rms: float
        max_err: float
    """
    assert X.shape == Y.shape
    assert X.shape[1] == 3

    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x[None, :]
    Yc = Y - mu_y[None, :]

    cov = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    var_x = np.sum(Xc * Xc) / n
    s = np.trace(np.diag(D) @ S) / max(var_x, 1e-15)

    t = mu_y - s * (R @ mu_x)

    aligned = (s * (R @ X.T)).T + t[None, :]

    err = aligned - Y
    per_v = np.linalg.norm(err, axis=1)
    rms = float(np.sqrt(np.mean(per_v ** 2)))
    max_err = float(np.max(per_v))

    return s, R, t, aligned, rms, max_err


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

def main():
    args = parse_args()
    np.random.seed(args.seed)

    clear_scene()
    scene, mesh_objs = import_glb(args.glb_path)

    # start_frame = scene.frame_start if args.start_frame is None else args.start_frame
    # end_frame = scene.frame_end if args.end_frame is None else args.end_frame
    start_frame, end_frame = get_frame_range(scene, args.start_frame, args.end_frame)

    if end_frame < start_frame:
        raise ValueError(f"end_frame({end_frame}) < start_frame({start_frame})")

    print("=" * 80)
    print(f"GLB: {args.glb_path}")
    print(f"Frames: [{start_frame}, {end_frame}]")
    print(f"Mesh objects: {[obj.name_full for obj in mesh_objs]}")
    print("=" * 80)

    # Reference frame
    scene.frame_set(start_frame)
    ref_verts_full, ref_topo = get_frame_vertices(scene, mesh_objs)

    if ref_verts_full.shape[0] == 0:
        raise RuntimeError("Reference frame has zero vertices.")

    ref_bbox_diag = bbox_diag(ref_verts_full)
    if ref_bbox_diag < 1e-12:
        raise RuntimeError("Reference mesh bbox diagonal is too small.")

    # Optional subsample
    n_total = ref_verts_full.shape[0]
    if args.max_verts is not None and args.max_verts > 0 and n_total > args.max_verts:
        idx = np.random.choice(n_total, size=args.max_verts, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_total)

    ref_verts = ref_verts_full[idx]

    results = []
    all_ok = True

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        cur_verts_full, cur_topo = get_frame_vertices(scene, mesh_objs)

        topo_same = (cur_topo == ref_topo)
        if not topo_same:
            all_ok = False
            result = {
                "frame": frame,
                "ok": False,
                "reason": "topology_changed",
                "topology_reference": ref_topo,
                "topology_current": cur_topo,
            }
            results.append(result)
            print(f"[Frame {frame}] topology changed -> NOT similarity-only")
            continue

        cur_verts = cur_verts_full[idx]

        s, R, t, aligned, rms, max_err = umeyama_similarity(ref_verts, cur_verts)
        rel_rms = rms / ref_bbox_diag
        rel_max = max_err / ref_bbox_diag
        ok = rel_rms <= args.tol_rel

        if not ok:
            all_ok = False

        result = {
            "frame": frame,
            "ok": bool(ok),
            "scale": float(s),
            "rotation_matrix": R.tolist(),
            "translation": t.tolist(),
            "rms_error": float(rms),
            "max_error": float(max_err),
            "rel_rms_error": float(rel_rms),
            "rel_max_error": float(rel_max),
        }
        results.append(result)

        print(
            f"[Frame {frame:04d}] "
            f"scale={s:.8f}, "
            f"rel_rms={rel_rms:.8e}, "
            f"rel_max={rel_max:.8e}, "
            f"ok={ok}"
        )

    summary = {
        "glb_path": args.glb_path,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames": end_frame - start_frame + 1,
        "num_vertices_reference": int(ref_verts_full.shape[0]),
        "num_vertices_used": int(len(idx)),
        "reference_bbox_diagonal": float(ref_bbox_diag),
        "tol_rel": float(args.tol_rel),
        "similarity_only_sequence": bool(all_ok),
        "note": (
            "This checks whether each frame can be explained by a single global "
            "rotation + uniform scale + translation applied to the reference frame."
        ),
        "results": results,
    }

    print("=" * 80)
    print("FINAL RESULT:")
    print(f"similarity_only_sequence = {summary['similarity_only_sequence']}")
    print("=" * 80)

    if args.json_out is not None:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True) if os.path.dirname(args.json_out) else None
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to: {args.json_out}")


if __name__ == "__main__":
    main()

"""
/efs/yanruibin/projects/blender-4.2.1-linux-x64/blender --background --python tools/filter_rigid_motion.py -- \
    --glb_path data/objverse_minghao_4d/glbs/000-033/5dd2ce713485413a84bceacf15e40b9f.glb \
    --tol_rel 1e-4 \
    --max_verts 50000 \
    --json_out data/objverse_minghao_4d/motion_info/000-033/5dd2ce713485413a84bceacf15e40b9f/umeyama_similarity.json
"""