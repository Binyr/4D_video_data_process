#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
import tarfile
from pathlib import Path
from typing import List, Set
from tqdm import tqdm


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def has_complete_mp4_views(mp4_dir: Path, num_views: int = 16) -> bool:
    if not mp4_dir.exists() or not mp4_dir.is_dir():
        return False

    for i in range(num_views):
        f = mp4_dir / f"view_{i:02d}.mp4"
        if not f.exists():
            return False
    return True


def is_valid_object_dir(obj_dir: Path, num_views: int = 16) -> bool:
    result_json = obj_dir / "result.json"
    result_mesh = obj_dir / "result_mesh.npz"
    rgb_dir = obj_dir / "result_rgb_mp4"
    normal_dir = obj_dir / "result_normal_mp4"

    if not result_json.exists():
        return False
    if not result_mesh.exists():
        return False
    if not has_complete_mp4_views(rgb_dir, num_views=num_views):
        return False
    if not has_complete_mp4_views(normal_dir, num_views=num_views):
        return False

    return True


def find_valid_object_dirs(input_root: Path, chunk_dirs=None, num_views: int = 16) -> List[Path]:
    valid_object_dirs = []

    if chunk_dirs is None or len(chunk_dirs) == 0:
        candidate_chunks = [p for p in input_root.iterdir() if p.is_dir()]
    else:
        candidate_chunks = [input_root / x for x in chunk_dirs]

    candidate_chunks = sorted(candidate_chunks, key=lambda p: natural_key(p.name))

    print(f"[Info] Total chunks to scan: {len(candidate_chunks)}")

    for chunk_dir in tqdm(candidate_chunks, desc="Scanning chunks", unit="chunk"):
        if not chunk_dir.exists():
            tqdm.write(f"[Warning] chunk dir not found: {chunk_dir}")
            continue
        if not chunk_dir.is_dir():
            continue

        object_dirs = [p for p in chunk_dir.iterdir() if p.is_dir()]
        object_dirs = sorted(object_dirs, key=lambda p: natural_key(p.name))

        for obj_dir in tqdm(
            object_dirs,
            desc=f"{chunk_dir.name}",
            unit="obj",
            leave=False,
        ):
            if is_valid_object_dir(obj_dir, num_views=num_views):
                valid_object_dirs.append(obj_dir)

    return valid_object_dirs


def _normalize_tar_member_name(name: str) -> str:
    """
    Normalize tar member path:
    - remove leading './'
    - strip leading/trailing '/'
    """
    while name.startswith("./"):
        name = name[2:]
    return name.strip("/")


def extract_object_relpaths_from_tar(tar_path: Path) -> Set[Path]:
    """
    Recover object relative paths from tar members.

    Expected tar structure (relative to input_root), based on the original pack script:
      chunk_name/object_id/...       (pack_mode='dir')
    or
      chunk_name/object_id/result.json
      chunk_name/object_id/result_mesh.npz
      chunk_name/object_id/result_rgb_mp4/view_00.mp4
      ...                            (pack_mode='required_files')

    We therefore identify an object by the first two path components:
      chunk_name / object_id
    """
    object_relpaths: Set[Path] = set()

    print(f"[Info] Reading tar members from: {tar_path}")
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()

        for m in tqdm(members, desc="Parsing tar members", unit="member"):
            name = _normalize_tar_member_name(m.name)
            if not name:
                continue

            parts = Path(name).parts
            if len(parts) < 2:
                continue

            # object dir is always chunk/object
            obj_rel = Path(parts[0]) / parts[1]
            object_relpaths.add(obj_rel)

    return object_relpaths


def save_anns_json(train_dirs: List[Path], test_dirs: List[Path], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anns = {
        "train": [str(p) for p in train_dirs],
        "test": [str(p) for p in test_dirs],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(anns, f, ensure_ascii=False, indent=2)
    print(f"[Info] anns.json saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate anns.json from valid rendering_v5 object dirs, "
            "restricted to objects that are actually included in a given tar."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to rendering_v5",
    )
    parser.add_argument(
        "--tar_path",
        type=str,
        required=True,
        help="Path to an existing tar file, e.g. data/rendering_v5_0323.tar",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output anns.json path",
    )
    parser.add_argument(
        "--chunk_dirs",
        nargs="*",
        default=None,
        help="Optional specific chunk dirs to include during filesystem scan",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Only keep the first N filtered object dirs after all filtering",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index after filtering valid object dirs",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index (exclusive) after filtering valid object dirs",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=16,
        help="Expected number of RGB/normal mp4 views, default=16",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=100,
        help="Number of random test samples, default=100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for test split",
    )
    parser.add_argument(
        "--save_relative_to_input_root",
        action="store_true",
        help=(
            "If set, save paths in anns.json relative to input_root. "
            "Otherwise save absolute paths."
        ),
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    tar_path = Path(args.tar_path).resolve()
    output_json = Path(args.output_json).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if not tar_path.exists():
        raise FileNotFoundError(f"tar_path not found: {tar_path}")

    print(f"[Info] Scanning filesystem root: {input_root}")
    print(f"[Info] Restricting to tar content: {tar_path}")
    print("[Info] Valid object condition:")
    print("       - result.json exists")
    print("       - result_mesh.npz exists")
    print(f"       - result_rgb_mp4 has {args.num_views} mp4 views")
    print(f"       - result_normal_mp4 has {args.num_views} mp4 views")

    valid_object_dirs = find_valid_object_dirs(
        input_root=input_root,
        chunk_dirs=args.chunk_dirs,
        num_views=args.num_views,
    )
    print(f"[Info] Found {len(valid_object_dirs)} valid object dirs before slicing")

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(valid_object_dirs)
    valid_object_dirs = valid_object_dirs[start_idx:end_idx]

    if args.max_objects is not None:
        valid_object_dirs = valid_object_dirs[:args.max_objects]

    print(f"[Info] Valid object dirs after slicing: {len(valid_object_dirs)}")

    tar_object_relpaths = extract_object_relpaths_from_tar(tar_path)
    print(f"[Info] Unique object dirs found in tar: {len(tar_object_relpaths)}")

    selected_object_dirs = []
    for obj_dir in valid_object_dirs:
        rel_obj_dir = obj_dir.relative_to(input_root)
        if rel_obj_dir in tar_object_relpaths:
            selected_object_dirs.append(obj_dir)

    selected_object_dirs = sorted(selected_object_dirs, key=lambda p: natural_key(str(p.relative_to(input_root))))
    print(f"[Info] Final selected object dirs (valid AND in tar): {len(selected_object_dirs)}")

    if len(selected_object_dirs) == 0:
        print("[Warning] No matched object dirs found. Nothing to save.")
        anns = {"train": [], "test": []}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(anns, f, ensure_ascii=False, indent=2)
        print(f"[Info] Empty anns.json saved to: {output_json}")
        return

    num_test = min(args.num_test, len(selected_object_dirs))
    rng = random.Random(args.seed)
    test_dirs = rng.sample(selected_object_dirs, num_test)
    test_set = set(test_dirs)
    train_dirs = [p for p in selected_object_dirs if p not in test_set]

    if args.save_relative_to_input_root:
        train_dirs_to_save = [p.relative_to(input_root) for p in train_dirs]
        test_dirs_to_save = [p.relative_to(input_root) for p in test_dirs]
    else:
        train_dirs_to_save = train_dirs
        test_dirs_to_save = test_dirs

    print(f"[Info] Train size: {len(train_dirs_to_save)}")
    print(f"[Info] Test size : {len(test_dirs_to_save)}")

    save_anns_json(train_dirs_to_save, test_dirs_to_save, output_json)

    print("[Done]")


if __name__ == "__main__":
    main()

"""
python tools/from_mesh_to_training_data/get_anns.py \
  --input_root data/objverse_minghao_4d_mine_40075/rendering_v5/ \
  --output_json data/objverse_minghao_4d_mine_40075/rendering_v5_anns_1.3k.json \
  --tar_path data/rendering_v5_0323.tar \
  --num_test 100 \
  --seed 42
"""