#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import tarfile
from pathlib import Path
from tqdm import tqdm


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def has_complete_mp4_views(mp4_dir: Path, num_views: int = 16) -> bool:
    if not mp4_dir.exists() or not mp4_dir.is_dir():
        return False

    for i in range(0, num_views, 2):
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
    # if not has_complete_mp4_views(normal_dir, num_views=num_views):
    #     return False

    return True


def find_valid_object_dirs(input_root: Path, chunk_dirs=None, num_views: int = 16):
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


def add_dir_to_tar(tar: tarfile.TarFile, dir_path: Path, arcname: str):
    tar.add(str(dir_path), arcname=arcname, recursive=True)


def add_required_files_to_tar(
    tar: tarfile.TarFile,
    obj_dir: Path,
    input_root: Path,
    num_views: int = 16,
):
    required_files = [
        obj_dir / "result.json",
        obj_dir / "result_mesh.npz",
    ]

    for i in range(0, num_views, 2):
        required_files.append(obj_dir / "result_rgb_mp4" / f"view_{i:02d}.mp4")
        # required_files.append(obj_dir / "result_normal_mp4" / f"view_{i:02d}.mp4")

    for fpath in required_files:
        relpath = fpath.relative_to(input_root)
        tar.add(str(fpath), arcname=str(relpath))
    
    required_dirs = [
        # obj_dir / "result_meta",
        # obj_dir / "result_normal_mp4",
    ]
    for dpath in required_dirs:
        if dpath.is_dir():
            relpath = dpath.relative_to(input_root)
            tar.add(str(dpath), arcname=str(relpath))


def save_manifest(valid_object_dirs, input_root: Path, manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for obj_dir in valid_object_dirs:
            f.write(str(obj_dir.relative_to(input_root)) + "\n")
    print(f"[Info] Manifest saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pack valid rendering_v5 object dirs into an uncompressed tar."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to rendering_v5",
    )
    parser.add_argument(
        "--output_tar",
        type=str,
        required=True,
        help="Output tar path, e.g. xxx.tar",
    )
    parser.add_argument(
        "--chunk_dirs",
        nargs="*",
        default=None,
        help="Optional specific chunk dirs to include",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Only pack the first N valid object dirs",
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
        "--pack_mode",
        type=str,
        default="dir",
        choices=["dir", "required_files"],
        help=(
            "dir: pack the whole object directory; "
            "required_files: only pack checked required files"
        ),
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional path to save the packed object list",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=16,
        help="Expected number of RGB/normal mp4 views, default=16",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_tar = Path(args.output_tar).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    print(f"[Info] Scanning: {input_root}")
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

    print(f"[Info] Final selected object dirs: {len(valid_object_dirs)}")

    if len(valid_object_dirs) == 0:
        print("[Warning] No valid object dirs found. Nothing to pack.")
        return

    output_tar.parent.mkdir(parents=True, exist_ok=True)

    if args.manifest_path is not None:
        save_manifest(valid_object_dirs, input_root, Path(args.manifest_path).resolve())

    print(f"[Info] Writing tar: {output_tar}")
    with tarfile.open(output_tar, mode="w") as tar:
        for obj_dir in tqdm(valid_object_dirs, desc="Packing tar", unit="obj"):
            rel_obj_dir = obj_dir.relative_to(input_root)

            if args.pack_mode == "dir":
                add_dir_to_tar(tar, obj_dir, arcname=str(rel_obj_dir))
            elif args.pack_mode == "required_files":
                add_required_files_to_tar(
                    tar=tar,
                    obj_dir=obj_dir,
                    input_root=input_root,
                    num_views=args.num_views,
                )
            else:
                raise ValueError(f"Unknown pack_mode: {args.pack_mode}")

    print("[Done]")
    print(f"[Done] Tar saved to: {output_tar}")


if __name__ == "__main__":
    main()
