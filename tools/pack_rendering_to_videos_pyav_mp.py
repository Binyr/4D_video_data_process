#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multiprocessing packer with resume + block split support:
- convert result_rgb/view_xx/*.png to rgb_view_xx.mp4
- convert result_normal/view_xx/*.png to normal_view_xx.mp4
- copy result.json and result_mesh.npz
- copy rgb_meta directory if it exists

Features:
- multiprocessing
- one tqdm bar per worker
- suppress noisy libpng / x265 logs
- resume:
    1) skip whole object if all expected outputs already exist
    2) otherwise skip per-view videos that already exist
- block split:
    use --num_blocks and --block_id to process only one block
"""

import argparse
import contextlib
import math
import os
import re
import shutil
from pathlib import Path
from multiprocessing import Pool

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("CV_LOG_STRATEGY", "ERROR")
os.environ.setdefault("X265_LOG_LEVEL", "error")

import av
import cv2
import numpy as np
import torch
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(s: str):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_subdirs(path: Path):
    if not path.exists():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_dir()],
        key=lambda p: natural_key(p.name)
    )


def list_images(path: Path):
    if not path.exists():
        return []
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def copy_if_needed(src: Path, dst: Path, overwrite: bool = False):
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return True
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def copy_dir_if_needed(src_dir: Path, dst_dir: Path, overwrite: bool = False):
    if not src_dir.exists() or not src_dir.is_dir():
        return False
    if dst_dir.exists():
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            return True
    ensure_dir(dst_dir.parent)
    shutil.copytree(src_dir, dst_dir)
    return True


@contextlib.contextmanager
def suppress_stdout_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull)


def write_16bit_depth_video(tensor, save_path, fps=24, modal="rgb"):
    if modal == "rgb":
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = ((tensor * 0.5 + 0.5) * 255.0).clamp(0, 255)
        frames = tensor.cpu().numpy().astype(np.uint8)
    elif modal == "depth":
        tensor = tensor.permute(0, 2, 3, 1)
        max_value = tensor.max().item()
        min_value = tensor.min().item()
        tensor = (tensor - min_value) * (65535.0 / (max_value - min_value + 1e-6))
        frames = tensor.cpu().numpy().astype(np.uint16)
        frames = np.repeat(frames, 3, axis=-1)
    elif modal == "normal":
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = ((tensor * 0.5 + 0.5) * 65535.0).clamp(0, 65535)
        frames = tensor.cpu().numpy().astype(np.uint16)
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    with suppress_stdout_stderr():
        output = av.open(str(save_path), mode="w")

        stream = None
        last_err = None
        for codec_name in ["libx265", "hevc"]:
            try:
                stream = output.add_stream(codec_name, rate=fps)
                break
            except Exception as e:
                last_err = e

        if stream is None:
            output.close()
            raise RuntimeError(f"Failed to create video stream with libx265/hevc: {last_err}")

        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p10le"
        stream.options = {"crf": "10"}

        try:
            stream.codec_context.options = {"log-level": "error"}
        except Exception:
            pass

        for i in range(frames.shape[0]):
            if frames.dtype == np.uint8:
                yuv_frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24").reformat(
                    format="yuv420p10le"
                )
            else:
                yuv_frame = av.VideoFrame.from_ndarray(frames[i], format="rgb48le").reformat(
                    format="yuv420p10le"
                )

            for packet in stream.encode(yuv_frame):
                output.mux(packet)

        for packet in stream.encode():
            output.mux(packet)

        output.close()


def read_rgb_image_to_chw_minus1_1(img_path: Path) -> torch.Tensor:
    with suppress_stdout_stderr():
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"Unsupported image shape {img.shape} for {img_path}")

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        raise RuntimeError(f"Unsupported dtype {img.dtype} for {img_path}")

    img = img * 2.0 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def read_blender_normal_to_chw_minus1_1(img_path: Path) -> torch.Tensor:
    with suppress_stdout_stderr():
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise RuntimeError(
            f"Expected 3/4-channel normal image, got shape {img.shape} for {img_path}"
        )

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        rgb = img[..., :3]
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if rgb.dtype == np.uint8:
        encoded = rgb.astype(np.float32) / 255.0
    elif rgb.dtype == np.uint16:
        encoded = rgb.astype(np.float32) / 65535.0
    else:
        raise RuntimeError(f"Unsupported dtype {rgb.dtype} for {img_path}")

    normal = encoded * 2.0 - 1.0
    return torch.from_numpy(normal).permute(2, 0, 1).contiguous()


def build_video_tensor_from_frame_dir(view_dir: Path, modal: str) -> torch.Tensor:
    frame_paths = list_images(view_dir)
    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {view_dir}")

    if modal == "rgb":
        frames = [read_rgb_image_to_chw_minus1_1(fp) for fp in frame_paths]
    elif modal == "normal":
        frames = [read_blender_normal_to_chw_minus1_1(fp) for fp in frame_paths]
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    h, w = frames[0].shape[1:]
    for i, x in enumerate(frames):
        if x.shape[1:] != (h, w):
            raise RuntimeError(
                f"Frame size mismatch in {view_dir}, "
                f"frame {i} has {tuple(x.shape[1:])}, expected {(h, w)}"
            )

    return torch.stack(frames, dim=0)


def expected_output_video_paths(object_dir: Path, output_object_dir: Path):
    expected = []

    rgb_root = object_dir / "result_rgb"
    normal_root = object_dir / "result_normal"

    for view_dir in list_subdirs(rgb_root):
        expected.append(output_object_dir / f"rgb_{view_dir.name}.mp4")

    for view_dir in list_subdirs(normal_root):
        expected.append(output_object_dir / f"normal_{view_dir.name}.mp4")

    return expected


def is_object_fully_done(object_dir: Path, output_object_dir: Path) -> bool:
    if not (output_object_dir / "result.json").exists():
        return False
    if not (output_object_dir / "result_mesh.npz").exists():
        return False

    in_rgb_meta = object_dir / "rgb_meta"
    out_rgb_meta = output_object_dir / "rgb_meta"
    if in_rgb_meta.exists() and in_rgb_meta.is_dir() and not out_rgb_meta.exists():
        return False

    for p in expected_output_video_paths(object_dir, output_object_dir):
        if not p.exists():
            return False

    return True


def process_view_folder(
    view_dir: Path,
    output_video_path: Path,
    fps: int,
    overwrite: bool,
    modal: str,
):
    frame_paths = list_images(view_dir)
    if len(frame_paths) == 0:
        return False, "no frames"

    if output_video_path.exists() and not overwrite:
        return True, "exists"

    tensor = build_video_tensor_from_frame_dir(view_dir, modal=modal)
    ensure_dir(output_video_path.parent)
    write_16bit_depth_video(tensor, output_video_path, fps=fps, modal=modal)
    return True, "written"


def process_object_dir(
    object_dir: Path,
    output_object_dir: Path,
    fps: int,
    overwrite: bool,
):
    result_json = object_dir / "result.json"
    result_mesh = object_dir / "result_mesh.npz"

    if not (result_json.exists() and result_mesh.exists()):
        return {
            "processed": False,
            "skipped_done": False,
            "reason": "missing result.json or result_mesh.npz",
            "rgb_videos": 0,
            "normal_videos": 0,
            "object_id": object_dir.name,
            "chunk_name": object_dir.parent.name,
        }

    if not overwrite and is_object_fully_done(object_dir, output_object_dir):
        return {
            "processed": False,
            "skipped_done": True,
            "reason": "already done",
            "rgb_videos": 0,
            "normal_videos": 0,
            "object_id": object_dir.name,
            "chunk_name": object_dir.parent.name,
        }

    ensure_dir(output_object_dir)

    copy_if_needed(result_json, output_object_dir / "result.json", overwrite=overwrite)
    copy_if_needed(result_mesh, output_object_dir / "result_mesh.npz", overwrite=overwrite)

    rgb_meta_dir = object_dir / "rgb_meta"
    if rgb_meta_dir.exists() and rgb_meta_dir.is_dir():
        copy_dir_if_needed(
            rgb_meta_dir,
            output_object_dir / "rgb_meta",
            overwrite=overwrite,
        )

    rgb_count = 0
    normal_count = 0

    rgb_root = object_dir / "result_rgb"
    normal_root = object_dir / "result_normal"

    for view_dir in list_subdirs(rgb_root):
        out_video = output_object_dir / f"rgb_{view_dir.name}.mp4"
        ok, _ = process_view_folder(
            view_dir=view_dir,
            output_video_path=out_video,
            fps=fps,
            overwrite=overwrite,
            modal="rgb",
        )
        if ok:
            rgb_count += 1

    for view_dir in list_subdirs(normal_root):
        out_video = output_object_dir / f"normal_{view_dir.name}.mp4"
        ok, _ = process_view_folder(
            view_dir=view_dir,
            output_video_path=out_video,
            fps=fps,
            overwrite=overwrite,
            modal="normal",
        )
        if ok:
            normal_count += 1

    return {
        "processed": True,
        "skipped_done": False,
        "reason": "ok",
        "rgb_videos": rgb_count,
        "normal_videos": normal_count,
        "object_id": object_dir.name,
        "chunk_name": object_dir.parent.name,
    }


def find_object_dirs(input_root: Path):
    object_dirs = []
    for chunk_dir in sorted(input_root.iterdir(), key=lambda p: natural_key(p.name)):
        if not chunk_dir.is_dir():
            continue
        if chunk_dir.name.startswith("_"):
            continue
        if chunk_dir.name == "_launcher_logs":
            continue

        for object_dir in sorted(chunk_dir.iterdir(), key=lambda p: natural_key(p.name)):
            if not object_dir.is_dir():
                continue
            object_dirs.append((chunk_dir.name, object_dir))
    return object_dirs


def split_list_round_robin(items, num_splits):
    splits = [[] for _ in range(num_splits)]
    for i, x in enumerate(items):
        splits[i % num_splits].append(x)
    return splits


def slice_block(items, num_blocks: int, block_id: int):
    if num_blocks <= 1:
        return items
    n = len(items)
    block_size = math.ceil(n / num_blocks)
    start = block_id * block_size
    end = min(start + block_size, n)
    return items[start:end]


def worker_main(worker_id, entries, output_root_str, fps, overwrite):
    output_root = Path(output_root_str)

    processed = 0
    skipped = 0
    skipped_done = 0
    total_rgb_videos = 0
    total_normal_videos = 0
    errors = []

    bar = tqdm(
        entries,
        desc=f"worker-{worker_id}",
        position=worker_id,
        leave=True,
        dynamic_ncols=True,
    )

    for chunk_name, object_dir in bar:
        object_id = object_dir.name
        output_object_dir = output_root / chunk_name / object_id

        try:
            stats = process_object_dir(
                object_dir=object_dir,
                output_object_dir=output_object_dir,
                fps=fps,
                overwrite=overwrite,
            )
        except Exception as e:
            skipped += 1
            errors.append(f"{chunk_name}/{object_id}: {e}")
            bar.set_postfix_str(
                f"done={processed} resume={skipped_done} skip={skipped}"
            )
            continue

        if stats["skipped_done"]:
            skipped_done += 1
        elif not stats["processed"]:
            skipped += 1
        else:
            processed += 1
            total_rgb_videos += stats["rgb_videos"]
            total_normal_videos += stats["normal_videos"]

        bar.set_postfix_str(
            f"done={processed} resume={skipped_done} skip={skipped} rgb={total_rgb_videos} normal={total_normal_videos}"
        )

    return {
        "worker_id": worker_id,
        "processed": processed,
        "skipped": skipped,
        "skipped_done": skipped_done,
        "rgb_videos": total_rgb_videos,
        "normal_videos": total_normal_videos,
        "num_errors": len(errors),
        "errors": errors[:20],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk_filter", type=str, default=None)
    parser.add_argument("--max_objects", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_blocks", type=int, default=1,
                        help="Split all objects into num_blocks contiguous blocks.")
    parser.add_argument("--block_id", type=int, default=0,
                        help="Which block to process, in [0, num_blocks-1].")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_blocks < 1:
        raise ValueError(f"--num_blocks must be >= 1, got {args.num_blocks}")
    if not (0 <= args.block_id < args.num_blocks):
        raise ValueError(
            f"--block_id must satisfy 0 <= block_id < num_blocks, "
            f"got block_id={args.block_id}, num_blocks={args.num_blocks}"
        )

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")

    object_entries = find_object_dirs(input_root)

    if args.chunk_filter is not None:
        object_entries = [
            (chunk_name, obj_dir)
            for chunk_name, obj_dir in object_entries
            if chunk_name == args.chunk_filter
        ]

    if args.max_objects > 0:
        object_entries = object_entries[:args.max_objects]

    total_before_block = len(object_entries)
    object_entries = slice_block(object_entries, args.num_blocks, args.block_id)
    total = len(object_entries)

    print(f"Found {total_before_block} object dirs under {input_root}")
    print(f"Using block {args.block_id}/{args.num_blocks}, block size = {total}")

    if total == 0:
        print("Nothing to process.")
        return

    num_workers = max(1, min(args.num_workers, total))
    entry_splits = split_list_round_robin(object_entries, num_workers)

    worker_args = [
        (worker_id, entry_splits[worker_id], str(output_root), args.fps, args.overwrite)
        for worker_id in range(num_workers)
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(worker_main, worker_args)

    results = sorted(results, key=lambda x: x["worker_id"])

    total_processed = sum(x["processed"] for x in results)
    total_skipped = sum(x["skipped"] for x in results)
    total_skipped_done = sum(x["skipped_done"] for x in results)
    total_rgb_videos = sum(x["rgb_videos"] for x in results)
    total_normal_videos = sum(x["normal_videos"] for x in results)
    total_errors = sum(x["num_errors"] for x in results)

    print("\n===== Per-worker Summary =====")
    for x in results:
        print(
            f"worker-{x['worker_id']}: "
            f"processed={x['processed']} "
            f"resumed_skip={x['skipped_done']} "
            f"skipped={x['skipped']} "
            f"rgb={x['rgb_videos']} "
            f"normal={x['normal_videos']} "
            f"errors={x['num_errors']}"
        )

    print("\n===== Total Summary =====")
    print(f"Block id:            {args.block_id}/{args.num_blocks}")
    print(f"Processed objects:   {total_processed}")
    print(f"Resumed skipped:     {total_skipped_done}")
    print(f"Other skipped:       {total_skipped}")
    print(f"RGB videos:          {total_rgb_videos}")
    print(f"Normal videos:       {total_normal_videos}")
    print(f"Errors:              {total_errors}")
    print(f"Output root:         {output_root}")

    if total_errors > 0:
        print("\n===== Example Errors =====")
        shown = 0
        for x in results:
            for err in x["errors"]:
                print(err)
                shown += 1
                if shown >= 20:
                    return


if __name__ == "__main__":
    main()

"""
python tools/pack_rendering_to_videos_pyav_mp.py \
    --input_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
    --output_root /group/30098/yanruibin/objverse_minghao_4d_mine/rendering_v5_video \
    --fps 8 \
    --num_workers 16 \
    --num_blocks 2 \
    --block_id 1
"""