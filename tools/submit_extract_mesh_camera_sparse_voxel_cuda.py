#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch multi-process Blender jobs on a selected chunk of GLB files.

新增功能:
- --cuda_devices 0 1 2 3
- 每个 worker / process 自动绑定一个 CUDA_VISIBLE_DEVICES
- 绑定策略: cuda_devices[worker_rank % len(cuda_devices)]

Example:
python tools/submit_extract_mesh_camera_sparse_voxel.py \
    --root_glb_dir data/objverse_minghao_4d/glbs \
    --output_root vis/rendering_v5 \
    --output_parent_suffix _static_camera_distance_v3 \
    --worker_script tools/extract_mesh_camera_sparse_voxel_v4_interleave_light.py \
    --blender_path /group/40034/yanruibin/projects/blender-4.2.18-linux-x64/blender \
    --num_chunks 4 \
    --chunk_id 1 \
    --num_workers 8 \
    --cuda_devices 0 1 2 3 \
    --resolution 1024 \
    --render_engine CYCLES \
    --transparent_bg \
    --traj_seed 123 \
    --num_cameras 16 \
    --camera_frame_padding 0.0 \
    --camera_fit_safety 1.0 \
    --camera_distance_jitter_scale 1.0 \
    --randomize_camera_intrinsics \
    --camera_fov_min_deg 30 \
    --camera_fov_max_deg 70 \
    --camera_sensor_size 36 \
    --sunlight_prob 0.5
"""

import argparse
import json
import os
import sys
import glob
import subprocess
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

from tqdm import tqdm


@dataclass
class Task:
    glb_path: str
    output_file: str
    log_file: str
    worker_rank: int


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_glb_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--output_parent_suffix", type=str, default="_static_camera_distance_v3")

    parser.add_argument("--blender_path", type=str, required=True)
    parser.add_argument("--worker_script", type=str, required=True)

    parser.add_argument("--num_chunks", type=int, required=True)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)

    # 新增：给 worker 分配 GPU
    parser.add_argument(
        "--cuda_devices",
        type=int,
        nargs="*",
        default=None,
        help=(
            "CUDA device ids used by launcher workers, e.g. --cuda_devices 0 1 2 3. "
            "If omitted, inherit current environment and do not overwrite CUDA_VISIBLE_DEVICES."
        ),
    )

    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry_run", action="store_true")

    # worker args
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--render_engine", type=str, default="CYCLES")
    parser.add_argument("--transparent_bg", action="store_true")
    parser.add_argument("--traj_seed", type=int, default=123)
    parser.add_argument("--num_cameras", type=int, default=10)
    parser.add_argument("--camera_frame_padding", type=float, default=0.0)
    parser.add_argument("--camera_fit_safety", type=float, default=1.0)
    parser.add_argument("--camera_distance_jitter_scale", type=float, default=1.0)
    parser.add_argument("--randomize_camera_intrinsics", action="store_true")
    parser.add_argument("--camera_fov_min_deg", type=float, default=30.0)
    parser.add_argument("--camera_fov_max_deg", type=float, default=70.0)
    parser.add_argument("--camera_sensor_size", type=float, default=36.0)
    parser.add_argument("--sunlight_prob", type=float, default=0.5)
    parser.add_argument("--cycles_device", type=str, default="GPU")

    # optional passthrough after --extra_worker_args
    parser.add_argument("--extra_worker_args", nargs=argparse.REMAINDER, default=[])

    args = parser.parse_args()

    if args.num_chunks <= 0:
        raise ValueError("--num_chunks must be > 0")
    if not (0 <= args.chunk_id < args.num_chunks):
        raise ValueError("--chunk_id must satisfy 0 <= chunk_id < num_chunks")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be > 0")
    if not (0.0 <= args.sunlight_prob <= 1.0):
        raise ValueError("--sunlight_prob must be in [0, 1]")

    if args.cuda_devices is not None:
        if len(args.cuda_devices) == 0:
            args.cuda_devices = None
        else:
            uniq = []
            seen = set()
            for x in args.cuda_devices:
                if x < 0:
                    raise ValueError("--cuda_devices must be non-negative integers")
                if x not in seen:
                    uniq.append(x)
                    seen.add(x)
            args.cuda_devices = uniq

    return args


def scan_glbs(root_glb_dir: str) -> List[str]:
    pattern = os.path.join(root_glb_dir, "**", "*.glb")
    glb_paths = glob.glob(pattern, recursive=True)
    glb_paths = sorted(os.path.abspath(p) for p in glb_paths)
    return glb_paths


def split_contiguous(items: List[str], num_chunks: int, chunk_id: int) -> List[str]:
    n = len(items)
    start = (n * chunk_id) // num_chunks
    end = (n * (chunk_id + 1)) // num_chunks
    return items[start:end]


def safe_rel_parent(glb_path: str, root_glb_dir: str) -> str:
    rel_parent = os.path.relpath(os.path.dirname(glb_path), os.path.abspath(root_glb_dir))
    if rel_parent == ".":
        rel_parent = "root"
    return rel_parent.replace(os.sep, "__")


def make_output_file(
    glb_path: str,
    root_glb_dir: str,
    output_root: str,
    output_parent_suffix: str,
) -> str:
    stem = os.path.splitext(os.path.basename(glb_path))[0]
    parent_token = safe_rel_parent(glb_path, root_glb_dir)
    out_dir = os.path.join(
        output_root,
        f"{parent_token}{output_parent_suffix}",
        stem,
    )
    return os.path.join(out_dir, "result.json")


def make_tasks(glbs_in_chunk: List[str], args) -> Tuple[List[List[Task]], str]:
    launcher_log_root = os.path.join(
        args.output_root,
        "_launcher_logs",
        f"chunk_{args.chunk_id:04d}_of_{args.num_chunks:04d}",
    )
    os.makedirs(launcher_log_root, exist_ok=True)

    worker_buckets: List[List[Task]] = [[] for _ in range(args.num_workers)]

    for task_idx, glb_path in enumerate(glbs_in_chunk):
        worker_rank = task_idx % args.num_workers

        output_file = make_output_file(
            glb_path=glb_path,
            root_glb_dir=args.root_glb_dir,
            output_root=args.output_root,
            output_parent_suffix=args.output_parent_suffix,
        )

        stem = os.path.splitext(os.path.basename(glb_path))[0]
        parent_token = safe_rel_parent(glb_path, args.root_glb_dir)
        log_dir = os.path.join(
            launcher_log_root,
            f"worker_{worker_rank:02d}",
            parent_token,
        )
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{stem}.log")

        worker_buckets[worker_rank].append(
            Task(
                glb_path=os.path.abspath(glb_path),
                output_file=os.path.abspath(output_file),
                log_file=os.path.abspath(log_file),
                worker_rank=worker_rank,
            )
        )

    return worker_buckets, launcher_log_root


def get_assigned_cuda_device(worker_rank: int, cuda_devices: Optional[List[int]]) -> Optional[int]:
    if not cuda_devices:
        return None
    return int(cuda_devices[worker_rank % len(cuda_devices)])


def build_blender_cmd(task: Task, args) -> List[str]:
    cmd = [
        args.blender_path,
        "--background",
        "--python",
        args.worker_script,
        "--",
        "--object_path",
        task.glb_path,
        "--output_file",
        task.output_file,
        "--resolution",
        str(args.resolution),
        "--render_engine",
        str(args.render_engine),
        "--cycles_device",
        str(args.cycles_device),
        "--traj_seed",
        str(args.traj_seed),
        "--num_cameras",
        str(args.num_cameras),
        "--camera_frame_padding",
        str(args.camera_frame_padding),
        "--camera_fit_safety",
        str(args.camera_fit_safety),
        "--camera_distance_jitter_scale",
        str(args.camera_distance_jitter_scale),
        "--camera_fov_min_deg",
        str(args.camera_fov_min_deg),
        "--camera_fov_max_deg",
        str(args.camera_fov_max_deg),
        "--camera_sensor_size",
        str(args.camera_sensor_size),
        "--sunlight_prob",
        str(args.sunlight_prob),
    ]

    if args.transparent_bg:
        cmd.append("--transparent_bg")
    if args.randomize_camera_intrinsics:
        cmd.append("--randomize_camera_intrinsics")
    if args.extra_worker_args:
        cmd.extend(args.extra_worker_args)

    return cmd


def worker_main(worker_rank: int, tasks: List[Task], args, tqdm_lock):
    tqdm.set_lock(tqdm_lock)

    assigned_cuda = get_assigned_cuda_device(worker_rank, args.cuda_devices)

    desc = f"worker {worker_rank:02d}"
    if assigned_cuda is not None:
        desc += f" | gpu {assigned_cuda}"

    pbar = tqdm(
        total=len(tasks),
        desc=desc,
        position=worker_rank,
        dynamic_ncols=True,
        leave=True,
    )

    num_ok = 0
    num_skip = 0
    num_fail = 0

    for task in tasks:
        stem = os.path.splitext(os.path.basename(task.glb_path))[0]
        pbar.set_postfix_str(
            f"{stem} ok={num_ok} skip={num_skip} fail={num_fail}"
        )

        os.makedirs(os.path.dirname(task.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(task.log_file), exist_ok=True)

        if args.skip_existing and os.path.isfile(task.output_file) and os.path.getsize(task.output_file) > 0:
            with open(task.log_file, "a", encoding="utf-8") as f:
                if assigned_cuda is not None:
                    f.write(f"[launcher] worker_rank={worker_rank}, CUDA_VISIBLE_DEVICES={assigned_cuda}\n")
                f.write(f"[launcher] skip existing output: {task.output_file}\n")
            num_skip += 1
            pbar.update(1)
            continue

        cmd = build_blender_cmd(task, args)

        child_env = os.environ.copy()
        if assigned_cuda is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = str(assigned_cuda)

        with open(task.log_file, "w", encoding="utf-8") as log_f:
            log_f.write("[launcher] CMD:\n")
            log_f.write(" ".join(cmd) + "\n\n")
            log_f.write(f"[launcher] worker_rank={worker_rank}\n")
            log_f.write(f"[launcher] assigned_cuda_device={assigned_cuda}\n")
            log_f.write(f"[launcher] CUDA_VISIBLE_DEVICES={child_env.get('CUDA_VISIBLE_DEVICES', '<inherit>')}\n\n")
            log_f.flush()

            try:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    check=False,
                )
                if result.returncode == 0:
                    num_ok += 1
                else:
                    num_fail += 1
                    fail_json = task.log_file + ".fail.json"
                    with open(fail_json, "w", encoding="utf-8") as ff:
                        json.dump(
                            {
                                "returncode": result.returncode,
                                "task": asdict(task),
                                "cmd": cmd,
                                "worker_rank": worker_rank,
                                "assigned_cuda_device": assigned_cuda,
                                "CUDA_VISIBLE_DEVICES": child_env.get("CUDA_VISIBLE_DEVICES", None),
                            },
                            ff,
                            indent=2,
                            ensure_ascii=False,
                        )
            except Exception as e:
                num_fail += 1
                fail_json = task.log_file + ".fail.json"
                with open(fail_json, "w", encoding="utf-8") as ff:
                    json.dump(
                        {
                            "exception": repr(e),
                            "task": asdict(task),
                            "cmd": cmd,
                            "worker_rank": worker_rank,
                            "assigned_cuda_device": assigned_cuda,
                            "CUDA_VISIBLE_DEVICES": child_env.get("CUDA_VISIBLE_DEVICES", None),
                        },
                        ff,
                        indent=2,
                        ensure_ascii=False,
                    )

        pbar.update(1)

    pbar.set_postfix_str(f"done ok={num_ok} skip={num_skip} fail={num_fail}")
    pbar.close()


def main():
    args = parse_args()

    if not os.path.isfile(args.blender_path):
        raise FileNotFoundError(f"blender not found: {args.blender_path}")
    if not os.path.isfile(args.worker_script):
        raise FileNotFoundError(f"worker_script not found: {args.worker_script}")

    all_glbs = scan_glbs(args.root_glb_dir)
    if len(all_glbs) == 0:
        raise RuntimeError(f"no .glb found under: {args.root_glb_dir}")

    glbs_in_chunk = split_contiguous(all_glbs, args.num_chunks, args.chunk_id)

    launcher_log_root = os.path.join(
        args.output_root,
        "_launcher_logs",
        f"chunk_{args.chunk_id:04d}_of_{args.num_chunks:04d}",
    )
    os.makedirs(launcher_log_root, exist_ok=True)

    worker_to_cuda = {
        int(worker_rank): get_assigned_cuda_device(worker_rank, args.cuda_devices)
        for worker_rank in range(args.num_workers)
    }

    manifest = {
        "all_num_glbs": len(all_glbs),
        "selected_num_glbs": len(glbs_in_chunk),
        "selected_num_tasks": len(glbs_in_chunk),
        "args": vars(args),
        "selected_glbs": glbs_in_chunk,
        "worker_to_cuda": worker_to_cuda,
    }
    with open(os.path.join(launcher_log_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(f"total glbs       : {len(all_glbs)}")
    print(f"selected chunk   : {args.chunk_id}/{args.num_chunks - 1}")
    print(f"glbs in chunk    : {len(glbs_in_chunk)}")
    print(f"total tasks      : {len(glbs_in_chunk)}")
    print(f"num workers      : {args.num_workers}")
    print(f"cuda devices     : {args.cuda_devices if args.cuda_devices else '<inherit>'}")
    print(f"worker->gpu      : {worker_to_cuda}")
    print(f"log root         : {launcher_log_root}")
    print("=" * 100)

    if args.dry_run:
        print("[dry_run] exit without launching subprocesses.")
        return

    worker_buckets, launcher_log_root = make_tasks(glbs_in_chunk, args)

    mp.set_start_method("spawn", force=True)
    tqdm_lock = mp.RLock()
    procs = []

    for worker_rank, tasks in enumerate(worker_buckets):
        p = mp.Process(
            target=worker_main,
            args=(worker_rank, tasks, args, tqdm_lock),
            daemon=False,
        )
        p.start()
        procs.append(p)

    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)

    bad_exit = [c for c in exit_codes if c != 0]
    if bad_exit:
        print(f"[launcher] some worker processes exited abnormally: {bad_exit}")
        sys.exit(1)

    print("[launcher] all workers finished.")
    print(f"[launcher] check logs in: {launcher_log_root}")


if __name__ == "__main__":
    main()
"""
python tools/submit_extract_mesh_camera_sparse_voxel_cuda.py \
    --root_glb_dir data/objverse_minghao_4d/glbs \
    --output_root /group/40075/yanruibin/objverse_minghao_4d_mine/rendering_v5 \
    --output_parent_suffix _static_camera_distance_v3 \
    --worker_script tools/extract_mesh_camera_sparse_voxel_v4_interleave_light_optimized_128frame_mp4.py \
    --blender_path /group/40034/yanruibin/projects/blender-4.2.18-linux-x64/blender \
    --num_chunks 8 \
    --chunk_id 1 \
    --num_workers 16 \
    --cuda_devices 0 1 2 3 4 5 6 7\
    --resolution 1024 \
    --render_engine CYCLES \
    --transparent_bg \
    --traj_seed 123 \
    --num_cameras 16 \
    --camera_frame_padding 0.0 \
    --camera_fit_safety 1.0 \
    --camera_distance_jitter_scale 1.0 \
    --randomize_camera_intrinsics \
    --camera_fov_min_deg 30 \
    --camera_fov_max_deg 70 \
    --camera_sensor_size 36 \
    --sunlight_prob 0.5 \
    --extra_worker_args --cycles_backend CUDA
"""