#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_ANNS_JSON = "data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json"
DEFAULT_DATASET_ROOT = "data/objverse_minghao_4d_mine_40075/rendering_v5"
DEFAULT_OUTPUT_ROOT = "data/train_data_40075_objverse_minghao_4d_mine_rendering_v5_512_8camera"


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


def load_object_dirs_from_anns(anns_json: Path, splits: List[str]) -> List[Path]:
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

    seen = set()
    uniq = []
    for p in object_dirs:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def infer_rel_obj_dir_mode(object_dirs: List[Path], dataset_root: Path) -> str:
    if not dataset_root.is_absolute():
        return "last_two_parts"

    for root in object_dirs:
        root = Path(root)
        if not root.is_absolute():
            return "last_two_parts"
        try:
            root.relative_to(dataset_root)
            return "relative_to_dataset_root"
        except ValueError:
            return "last_two_parts"

    return "relative_to_dataset_root"


def rel_obj_dir_from_root(root: Path, dataset_root: Path, rel_obj_dir_mode: str) -> Path:
    root = Path(root)
    if rel_obj_dir_mode == "relative_to_dataset_root":
        return root.relative_to(dataset_root)
    if rel_obj_dir_mode == "last_two_parts":
        parts = root.parts
        if len(parts) >= 2:
            return Path(parts[-2]) / parts[-1]
        return Path(root.name)
    raise ValueError(f"Unknown rel_obj_dir_mode: {rel_obj_dir_mode}")


def object_key_from_rel_obj_dir(rel_obj_dir: Path) -> str:
    parts = rel_obj_dir.parts
    if len(parts) >= 2:
        return f"{parts[-2]}:{parts[-1]}"
    return str(rel_obj_dir).replace(os.sep, ":")


NUM_FRAMES_PATTERN = re.compile(r'"num_frames"\s*:\s*(\d+)')


def load_camera_count_and_num_frames_from_result_json(result_json_path: Path) -> tuple[int, int]:
    if not result_json_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        metas = json.load(f)

    if "_global" in metas:
        cameras = metas["_global"]["static_cameras"]
        cameras = sorted(cameras, key=lambda x: int(x["view_index"]))
        num_frames = metas["_global"].get("num_frames")
        if num_frames is None:
            frame_keys = [k for k in metas.keys() if k != "_global"]
            num_frames = len(frame_keys)
        return int(len(cameras)), int(num_frames)

    frame_metas = [v for v in metas.values() if isinstance(v, dict) and "views" in v]
    if len(frame_metas) == 0:
        raise RuntimeError(f"Cannot parse frame metadata from {result_json_path}")

    views = frame_metas[0]["views"]
    views = sorted(views, key=lambda x: int(x["view_index"]))
    return int(len(views)), int(len(frame_metas))


def load_num_frames_fast(result_json_path: Path) -> int:
    """
    Fast path:
    1) regex scan for "_global.num_frames"
    2) fallback to full json parsing for compatibility
    """
    if not result_json_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_json_path}")

    with open(result_json_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = NUM_FRAMES_PATTERN.search(text)
    if m is not None:
        return int(m.group(1))

    _, num_frames = load_camera_count_and_num_frames_from_result_json(result_json_path)
    return int(num_frames)


def iter_selected_view_indices(num_views: int, view_stride: int, view_offset: int) -> Iterable[int]:
    for ic in range(num_views):
        if ic % view_stride != view_offset:
            continue
        yield ic


def view_is_done(
    view_dir: Path,
    resolution: int,
    num_frames: int,
    done_check_mode: str,
) -> bool:
    if not view_dir.exists():
        return False

    if done_check_mode == "last_frame":
        frame_tag = str(num_frames - 1).zfill(3)
        sdf_path = view_dir / f"sparse_sdf_{resolution}_{frame_tag}.npz"
        sign_path = view_dir / f"sdf_sign_{resolution}_{frame_tag}.npz"
        return sdf_path.exists() and sign_path.exists()

    if done_check_mode != "full":
        raise ValueError(f"Unsupported done_check_mode: {done_check_mode}")

    # Fast reject on incomplete tail; avoids expensive directory scan for most unfinished views.
    frame_tag = str(num_frames - 1).zfill(3)
    tail_sdf = view_dir / f"sparse_sdf_{resolution}_{frame_tag}.npz"
    tail_sign = view_dir / f"sdf_sign_{resolution}_{frame_tag}.npz"
    if (not tail_sdf.exists()) or (not tail_sign.exists()):
        return False

    names = {x.name for x in view_dir.iterdir() if x.is_file()}
    for i in range(num_frames):
        frame_tag = str(i).zfill(3)
        sdf_name = f"sparse_sdf_{resolution}_{frame_tag}.npz"
        sign_name = f"sdf_sign_{resolution}_{frame_tag}.npz"
        if sdf_name not in names or sign_name not in names:
            return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check which object/views are finished and save to a JSON file."
    )
    parser.add_argument("--anns_json", type=str, default=DEFAULT_ANNS_JSON)
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--num_views",
        type=int,
        default=16,
        help="Total camera views per object. Use <=0 to infer from result.json (slower).",
    )
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--view_offset", type=int, default=0)
    parser.add_argument(
        "--done_check_mode",
        type=str,
        default="last_frame",
        choices=["full", "last_frame"],
        help="full: all frames exist; last_frame: only check the last frame (same as skip-existing-object fast check).",
    )
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_objects", type=int, default=None)
    parser.add_argument(
        "--include_empty_objects",
        action="store_true",
        help="Include objects with no finished selected views in output JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.view_stride <= 0:
        raise ValueError(f"--view_stride must be > 0, got {args.view_stride}")
    if args.num_views == 0:
        raise ValueError("--num_views cannot be 0. Use a positive integer or -1 for auto-infer.")
    if not (0 <= args.view_offset < args.view_stride):
        raise ValueError(
            f"--view_offset must satisfy 0 <= view_offset < view_stride, "
            f"got view_offset={args.view_offset}, view_stride={args.view_stride}"
        )

    anns_json = Path(args.anns_json)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    if not anns_json.exists():
        raise FileNotFoundError(f"anns_json not found: {anns_json}")

    object_dirs = load_object_dirs_from_anns(anns_json=anns_json, splits=args.splits)

    if args.start_idx is not None or args.end_idx is not None:
        s = 0 if args.start_idx is None else args.start_idx
        e = len(object_dirs) if args.end_idx is None else args.end_idx
        object_dirs = object_dirs[s:e]

    if args.max_objects is not None:
        object_dirs = object_dirs[:args.max_objects]

    rel_obj_dir_mode = infer_rel_obj_dir_mode(object_dirs, dataset_root)

    print(f"[Checker] anns_json   : {anns_json}")
    print(f"[Checker] dataset_root: {dataset_root}")
    print(f"[Checker] output_root : {output_root}")
    print(f"[Checker] total objects: {len(object_dirs)}")
    print(f"[Checker] num_views    : {args.num_views}")
    print(f"[Checker] view_stride : {args.view_stride}")
    print(f"[Checker] view_offset : {args.view_offset}")
    print(f"[Checker] resolution  : {args.resolution}")
    print(f"[Checker] done_check_mode: {args.done_check_mode}")
    print(f"[Checker] rel_obj_dir_mode: {rel_obj_dir_mode}")

    finished_object_views: Dict[str, Dict] = {}
    errors: List[Dict] = []

    total_selected_views = 0
    total_finished_views = 0

    for root in tqdm(object_dirs, desc="checking", dynamic_ncols=True):
        root = Path(root)
        try:
            rel_obj_dir = rel_obj_dir_from_root(root, dataset_root, rel_obj_dir_mode)
            object_key = object_key_from_rel_obj_dir(rel_obj_dir)
            out_obj_dir = output_root / rel_obj_dir
            if (not args.include_empty_objects) and (not out_obj_dir.exists()):
                # Same optimization spirit as get_processed_mesh_anns.py:
                # skip expensive metadata parsing when output object dir does not exist.
                continue

            if args.num_views > 0:
                camera_count = int(args.num_views)
            else:
                camera_count, _ = load_camera_count_and_num_frames_from_result_json(root / "result.json")
                if camera_count <= 0:
                    raise RuntimeError(f"No camera found: {root / 'result.json'}")

            num_frames = load_num_frames_fast(root / "result.json")
            if num_frames <= 0:
                raise RuntimeError(f"No frame found in {root / 'result.json'}")

            selected_view_indices = list(
                iter_selected_view_indices(
                    num_views=camera_count,
                    view_stride=args.view_stride,
                    view_offset=args.view_offset,
                )
            )

            finished_view_indices: List[int] = []
            for ic in selected_view_indices:
                view_dir = out_obj_dir / f"view_{str(ic).zfill(2)}"
                if view_is_done(
                    view_dir=view_dir,
                    resolution=args.resolution,
                    num_frames=num_frames,
                    done_check_mode=args.done_check_mode,
                ):
                    finished_view_indices.append(ic)

            total_selected_views += len(selected_view_indices)
            total_finished_views += len(finished_view_indices)

            if finished_view_indices or args.include_empty_objects:
                finished_object_views[object_key] = {
                    "object_dir": str(root),
                    "output_object_dir": str(out_obj_dir),
                    "num_frames": num_frames,
                    "selected_view_indices": selected_view_indices,
                    "finished_view_indices": finished_view_indices,
                    "finished_view_names": [f"view_{str(x).zfill(2)}" for x in finished_view_indices],
                }

        except Exception as e:
            errors.append(
                {
                    "object_dir": str(root),
                    "error": repr(e),
                }
            )

    if args.output_json:
        output_json = Path(args.output_json)
    else:
        output_json = output_root / (
            f"finished_object_views_res{args.resolution}_"
            f"stride{args.view_stride}_offset{args.view_offset}_{args.done_check_mode}.json"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "anns_json": str(anns_json),
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "splits": args.splits,
            "resolution": args.resolution,
            "num_views": args.num_views,
            "view_stride": args.view_stride,
            "view_offset": args.view_offset,
            "done_check_mode": args.done_check_mode,
            "rel_obj_dir_mode": rel_obj_dir_mode,
            "include_empty_objects": args.include_empty_objects,
        },
        "summary": {
            "total_objects_checked": len(object_dirs),
            "objects_with_finished_views_saved": len(finished_object_views),
            "total_selected_views_checked": total_selected_views,
            "total_finished_views": total_finished_views,
            "errors": len(errors),
        },
        "finished_object_views": finished_object_views,
        "errors": errors,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Checker] Saved json: {output_json}")
    print(
        f"[Checker] summary: objects={len(object_dirs)}, "
        f"saved_objects={len(finished_object_views)}, "
        f"finished_views={total_finished_views}, errors={len(errors)}"
    )


if __name__ == "__main__":
    main()
