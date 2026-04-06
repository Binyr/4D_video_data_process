#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import os
import json
import gc
import math
import argparse
import subprocess
from typing import List, Dict


# 让脚本能 import 项目内模块
f_path = Path(__file__).absolute()
p_path = f_path.parents[2]
sys.path.insert(0, str(p_path))


DEFAULT_ANNS_JSON = "/group/40075/yanruibin/objverse_minghao_4d_mine/rendering_v5_anns_1.3k.json"
DEFAULT_DATASET_ROOT = "/group/40075/yanruibin/objverse_minghao_4d_mine/rendering_v5"
DEFAULT_OUTPUT_ROOT = "data/train_data_40075_objverse_minghao_4d_mine_rendering_v5_512_4camera"


def load_object_dirs_from_anns(
    anns_json: Path,
    splits: List[str],
) -> List[Path]:
    with open(anns_json, "r", encoding="utf-8") as f:
        anns = json.load(f)

    object_dirs: List[Path] = []
    for split in splits:
        if split not in anns:
            raise KeyError(f"Split '{split}' not found in {anns_json}")
        split_list = anns[split]
        if not isinstance(split_list, list):
            raise TypeError(f"anns['{split}'] should be a list, got {type(split_list)}")
        object_dirs.extend(Path(x) for x in split_list)

    # 去重，保持顺序
    seen = set()
    uniq = []
    for p in object_dirs:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def shard_list_round_robin(items: List[Path], num_shards: int) -> List[List[Path]]:
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(items):
        shards[i % num_shards].append(item)
    return shards


def split_list_into_chunks(items: List[Path], num_chunks: int) -> List[List[Path]]:
    """
    顺序切分成 num_chunks 份，尽量均匀。
    """
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be > 0, got {num_chunks}")

    n = len(items)
    chunk_size = math.ceil(n / num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        chunks.append(items[start:end])
    return chunks


def save_shard_json(shard_paths: List[Path], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump([str(p) for p in shard_paths], f, ensure_ascii=False, indent=2)


def load_camera_c2ws_from_result_json(result_json_path: Path):
    """
    兼容两种 meta 格式，返回按 view_index 排序后的 camera_c2w list
    """
    if not result_json_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        metas = json.load(f)

    camera_c2ws_bl = []

    if "_global" in metas:
        # new meta format
        metas_global = metas["_global"]
        cameras = metas_global["static_cameras"]
        cameras = sorted(cameras, key=lambda x: int(x["view_index"]))
        for camera in cameras:
            camera_c2ws_bl.append(camera["camera_c2w"])
    else:
        # old meta format
        # 旧格式通常是: {frame_xxx: {"views": [...]}, ...}
        first_key = next(iter(metas))
        views = metas[first_key]["views"]
        views = sorted(views, key=lambda x: int(x["view_index"]))
        for view in views:
            camera_c2ws_bl.append(view["camera_c2w"])

    return camera_c2ws_bl


def run_launcher(args):
    anns_json = Path(args.anns_json).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()

    assert anns_json.exists(), f"anns_json not found: {anns_json}"
    assert dataset_root.exists(), f"dataset_root not found: {dataset_root}"

    gpu_ids = args.gpu_ids
    assert len(gpu_ids) > 0, "Please provide at least one gpu id via --gpu_ids"

    object_dirs = load_object_dirs_from_anns(
        anns_json=anns_json,
        splits=args.splits,
    )

    if args.start_idx is not None or args.end_idx is not None:
        s = 0 if args.start_idx is None else args.start_idx
        e = len(object_dirs) if args.end_idx is None else args.end_idx
        object_dirs = object_dirs[s:e]

    if args.max_objects is not None:
        object_dirs = object_dirs[:args.max_objects]

    # ===== 先按大 chunk 切 =====
    if args.num_chunks is not None:
        if args.num_chunks <= 0:
            raise ValueError(f"--num_chunks must be > 0, got {args.num_chunks}")
        if args.chunk_id is None:
            raise ValueError("When --num_chunks is set, --chunk_id must also be set.")
        if not (0 <= args.chunk_id < args.num_chunks):
            raise ValueError(
                f"--chunk_id must satisfy 0 <= chunk_id < num_chunks, "
                f"got chunk_id={args.chunk_id}, num_chunks={args.num_chunks}"
            )

        big_chunks = split_list_into_chunks(object_dirs, args.num_chunks)
        object_dirs = big_chunks[args.chunk_id]

    print(f"[Launcher] anns_json   : {anns_json}")
    print(f"[Launcher] dataset_root: {dataset_root}")
    print(f"[Launcher] output_root : {output_root}")
    print(f"[Launcher] splits      : {args.splits}")
    print(f"[Launcher] gpu_ids     : {gpu_ids}")
    print(f"[Launcher] num_chunks  : {args.num_chunks}")
    print(f"[Launcher] chunk_id    : {args.chunk_id}")
    print(f"[Launcher] total objs to run: {len(object_dirs)}")
    print(f"[Launcher] view_stride : {args.view_stride}")
    print(f"[Launcher] view_offset : {args.view_offset}")

    if len(object_dirs) == 0:
        print("[Launcher] No object dirs to process. Exit.")
        return

    # ===== 再在这个 chunk 内部按 GPU round-robin 分给 worker =====
    shards = shard_list_round_robin(object_dirs, len(gpu_ids))
    shard_dir = output_root / "_submit_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for worker_rank, gpu_id in enumerate(gpu_ids):
        shard = shards[worker_rank]
        if len(shard) == 0:
            print(f"[Launcher] Worker {worker_rank} on GPU {gpu_id}: empty shard, skip")
            continue

        chunk_suffix = ""
        if args.num_chunks is not None:
            chunk_suffix = f"_chunk{args.chunk_id}of{args.num_chunks}"

        shard_json = shard_dir / f"shard_worker{worker_rank}_gpu{gpu_id}{chunk_suffix}.json"
        save_shard_json(shard, shard_json)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker_rank", str(worker_rank),
            "--gpu_id", str(gpu_id),
            "--shard_json", str(shard_json),
            "--dataset_root", str(dataset_root),
            "--output_root", str(output_root),
            "--resolution", str(args.resolution),
            "--view_stride", str(args.view_stride),
            "--view_offset", str(args.view_offset),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.skip_existing_object:
            cmd.append("--skip_existing_object")
        if args.save_failures:
            cmd.append("--save_failures")

        log_path = output_root / f"worker_{worker_rank}_gpu{gpu_id}{chunk_suffix}.log"
        print(f"[Launcher] Start worker {worker_rank} | GPU {gpu_id} | objs {len(shard)}")
        print(f"[Launcher] Log -> {log_path}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            procs.append((worker_rank, gpu_id, proc, log_path))

    exit_codes = []
    for worker_rank, gpu_id, proc, log_path in procs:
        ret = proc.wait()
        exit_codes.append(ret)
        print(f"[Launcher] Worker {worker_rank} | GPU {gpu_id} finished with code {ret}")
        print(f"[Launcher] See log: {log_path}")

    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"Some workers failed. exit_codes={exit_codes}")

    print("[Launcher] All workers finished successfully.")


def run_worker(args):
    import numpy as np
    import torch
    from tqdm import tqdm

    from tools.utils.mesh2watertight_video import cubvh_mesh2watertightsd_vid

    worker_rank = args.worker_rank
    gpu_id = args.gpu_id
    shard_json = Path(args.shard_json).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    resolution = args.resolution
    view_stride = args.view_stride
    view_offset = args.view_offset

    with open(shard_json, "r", encoding="utf-8") as f:
        object_dirs = [Path(x) for x in json.load(f)]

    # launcher 已经设置 CUDA_VISIBLE_DEVICES=<gpu_id>
    # 对 worker 来说可见 GPU 只有 1 张，所以统一用 cuda:0
    device = "cuda:0"
    torch.cuda.set_device(0)

    print(f"[Worker {worker_rank}] visible CUDA = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Worker {worker_rank}] device = {device}")
    print(f"[Worker {worker_rank}] shard_json = {shard_json}")
    print(f"[Worker {worker_rank}] num_objects = {len(object_dirs)}")
    print(f"[Worker {worker_rank}] dataset_root = {dataset_root}")
    print(f"[Worker {worker_rank}] output_root = {output_root}")
    print(f"[Worker {worker_rank}] resolution = {resolution}")
    print(f"[Worker {worker_rank}] view_stride = {view_stride}")
    print(f"[Worker {worker_rank}] view_offset = {view_offset}")

    failures: List[Dict] = []

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def iter_selected_view_indices(num_views: int):
        for ic in range(num_views):
            if ic % view_stride != view_offset:
                continue
            yield ic

    def object_is_fully_done(save_obj_dir: Path, frame_indices, num_views: int) -> bool:
        for ic in iter_selected_view_indices(num_views):
            view_dir = save_obj_dir / f"view_{str(ic).zfill(2)}"
            for f_i in range(len(frame_indices)):
                frame_tag = str(int(f_i)).zfill(3)
                sdf_path = view_dir / f"sparse_sdf_{resolution}_{frame_tag}.npz"
                sign_path = view_dir / f"sdf_sign_{resolution}_{frame_tag}.npz"
                if (not sdf_path.exists()) or (not sign_path.exists()):
                    return False
        return True

    for root in tqdm(object_dirs, desc=f"worker{worker_rank}", dynamic_ncols=True):
        try:
            root = root.resolve()
            if not root.exists():
                raise FileNotFoundError(f"object dir not found: {root}")

            mesh_path = root / "result_mesh.npz"
            if not mesh_path.exists():
                raise FileNotFoundError(f"result_mesh.npz not found: {mesh_path}")

            result_json_path = root / "result.json"
            camera_c2ws_bl = load_camera_c2ws_from_result_json(result_json_path)
            if len(camera_c2ws_bl) == 0:
                raise RuntimeError(f"No cameras found in {result_json_path}")

            # 保持和 input 一致的目录结构:
            # input : dataset_root/xxx-xxx/object_id
            # output: output_root /xxx-xxx/object_id
            rel_obj_dir = root.relative_to(dataset_root)
            save_obj_dir = output_root / rel_obj_dir
            save_obj_dir.mkdir(exist_ok=True, parents=True)

            npz = np.load(mesh_path, mmap_mode="r")
            mesh_v_list = npz["vertices"]      # T N 3
            mesh_f_np = npz["faces"]           # F 3
            f_indices = np.array(npz["frame_indices"])

            if args.skip_existing_object and object_is_fully_done(
                save_obj_dir, f_indices, len(camera_c2ws_bl)
            ):
                print(f"[Worker {worker_rank}] Skip fully done object: {root}")
                del npz
                continue

            mesh_f = torch.from_numpy(mesh_f_np).to(dtype=torch.int32, device=device)

            # 提前把所有 view 的旋转矩阵准备好
            view_rot_mats = []
            for ic, c2w in enumerate(camera_c2ws_bl):
                c2w = np.asarray(c2w, dtype=np.float32)
                if c2w.shape != (4, 4):
                    raise ValueError(f"camera_c2w for view {ic} has invalid shape: {c2w.shape}")
                w2c = np.linalg.inv(c2w)
                w2c_rot = torch.from_numpy(w2c[:3, :3]).to(dtype=torch.float32, device=device)
                view_rot_mats.append(w2c_rot)

            with torch.inference_mode():
                for ic in iter_selected_view_indices(len(view_rot_mats)):
                    w2c_rot = view_rot_mats[ic]
                    view_dir = save_obj_dir / f"view_{str(ic).zfill(2)}"
                    view_dir.mkdir(exist_ok=True, parents=True)

                    for i, f_i in enumerate(f_indices):
                        frame_tag = str(int(i)).zfill(3)
                        sdf_path = view_dir / f"sparse_sdf_{resolution}_{frame_tag}.npz"
                        sign_path = view_dir / f"sdf_sign_{resolution}_{frame_tag}.npz"

                        if (not args.overwrite) and sdf_path.exists() and sign_path.exists():
                            continue

                        mesh_v = None
                        mesh_wt = None
                        sparse_sdf = None
                        sparse_index = None
                        mesh_scale = None
                        sdf_sign = None

                        try:
                            mesh_v_np = mesh_v_list[i]
                            mesh_v = torch.from_numpy(mesh_v_np).to(dtype=torch.float32, device=device)

                            # world -> camera，只做旋转，和你原单进程代码一致
                            mesh_v = torch.matmul(mesh_v, w2c_rot.T)

                            mesh_wt, sparse_sdf, sparse_index, mesh_scale, sdf_sign = \
                                cubvh_mesh2watertightsd_vid(mesh_v, mesh_f, resolution)

                            sparse_sdf_np = to_numpy(sparse_sdf)
                            sparse_index_np = to_numpy(sparse_index)
                            sdf_sign_np = to_numpy(sdf_sign)

                            np.savez_compressed(
                                sdf_path,
                                sparse_sdf=sparse_sdf_np,
                                sparse_index=sparse_index_np,
                            )
                            np.savez_compressed(
                                sign_path,
                                sdf_sign=sdf_sign_np,
                            )

                        finally:
                            del mesh_v, mesh_wt, sparse_sdf, sparse_index, mesh_scale, sdf_sign
                            gc.collect()
                            torch.cuda.empty_cache()

            del view_rot_mats
            del mesh_f
            del npz
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            err = {
                "object_dir": str(root),
                "error": repr(e),
            }
            failures.append(err)
            print(f"[Worker {worker_rank}] ERROR on {root}: {repr(e)}")
            gc.collect()
            torch.cuda.empty_cache()

    if args.save_failures:
        fail_path = output_root / f"worker_{worker_rank}_failures.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"[Worker {worker_rank}] failures saved to {fail_path}")

    if len(failures) > 0:
        print(f"[Worker {worker_rank}] Finished with {len(failures)} failures")
    else:
        print(f"[Worker {worker_rank}] Finished with 0 failures")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit multi-GPU mesh->training-data jobs from anns.json"
    )

    # launcher mode
    parser.add_argument(
        "--anns_json",
        type=str,
        default=DEFAULT_ANNS_JSON,
        help="Path to rendering_v5 anns json",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root of rendering_v5",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root of output training data",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Which splits to process from anns json",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=[0],
        help="GPU ids, one worker per gpu",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="SDF resolution",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Only process the first N objects after loading anns",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Optional start index after loading anns",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Optional end index after loading anns",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Split all objects into N big chunks",
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=None,
        help="Which big chunk to run, in [0, num_chunks-1]",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frame outputs",
    )
    parser.add_argument(
        "--skip_existing_object",
        action="store_true",
        help="If all frames of all selected views for an object already exist, skip the whole object",
    )
    parser.add_argument(
        "--save_failures",
        action="store_true",
        help="Save per-worker failures json",
    )

    # 新增：view 选择逻辑
    parser.add_argument(
        "--view_stride",
        type=int,
        default=1,
        help="Only process views satisfying (view_idx %% view_stride == view_offset). "
             "Use 2 + offset=1 to mimic 'if ic %% 2 == 0: continue'.",
    )
    parser.add_argument(
        "--view_offset",
        type=int,
        default=0,
        help="View offset used with --view_stride",
    )

    # worker mode
    parser.add_argument("--worker", action="store_true", help="Internal worker mode")
    parser.add_argument("--worker_rank", type=int, default=-1, help="Internal")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Internal")
    parser.add_argument("--shard_json", type=str, default="", help="Internal")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.view_stride <= 0:
        raise ValueError(f"--view_stride must be > 0, got {args.view_stride}")
    if not (0 <= args.view_offset < args.view_stride):
        raise ValueError(
            f"--view_offset must satisfy 0 <= view_offset < view_stride, "
            f"got view_offset={args.view_offset}, view_stride={args.view_stride}"
        )

    if args.worker:
        run_worker(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()

"""
python tools/from_mesh_to_training_data/v2_mp.py \
  --splits train test \
  --gpu_ids 0 1 2 3 4 5 6 7\
  --resolution 512 \
  --num_chunks 3 \
  --chunk_id 0 \
  --skip_existing_object \
  --save_failures \
  --view_stride 4
"""