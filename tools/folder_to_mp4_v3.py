#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把如下目录结构中的 view_xx.png 沿时间维拼成 mp4:

result_rgb/
  frame_0001/view_00.png
  frame_0002/view_00.png
  ...
  frame_0120/view_00.png

输出:
result_rgb_mp4/
  view_00.mp4
  view_01.mp4
  ...

用法:
python frames_to_mp4_by_view.py \
    --rgb_root /efs/yanruibin/projects/Direct3D-S2/vis/rendering_v3/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_0/result_rgb \
    --out_dir /efs/yanruibin/projects/Direct3D-S2/vis/rendering_v3/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_0/result_rgb_mp4 \
    --fps 24
"""

import os
import re
import glob
import argparse
from typing import List, Dict

import imageio.v2 as imageio


FRAME_RE = re.compile(r"frame_(\d+)$")
VIEW_RE = re.compile(r"view_(\d+)\.png$")


def sorted_frame_dirs(rgb_root: str) -> List[str]:
    frame_dirs = [
        p for p in glob.glob(os.path.join(rgb_root, "frame_*"))
        if os.path.isdir(p)
    ]

    def key_fn(path: str):
        name = os.path.basename(path)
        m = FRAME_RE.match(name)
        if m is None:
            return 10**18
        return int(m.group(1))

    frame_dirs.sort(key=key_fn)
    return frame_dirs


def discover_views(frame_dirs: List[str]) -> List[str]:
    """
    从所有 frame 目录中收集所有出现过的 view_xx.png 名字。
    """
    view_names = set()
    for frame_dir in frame_dirs:
        for p in glob.glob(os.path.join(frame_dir, "view_*.png")):
            view_names.add(os.path.basename(p))

    def key_fn(name: str):
        m = VIEW_RE.match(name)
        if m is None:
            return 10**18
        return int(m.group(1))

    return sorted(view_names, key=key_fn)


def collect_images_for_view(frame_dirs: List[str], view_name: str) -> List[str]:
    """
    对某个 view，例如 view_00.png，按 frame 顺序收集图片路径。
    只保留实际存在的文件。
    """
    paths = []
    for frame_dir in frame_dirs:
        img_path = os.path.join(frame_dir, view_name)
        if os.path.isfile(img_path):
            paths.append(img_path)
    return paths


def make_mp4(image_paths: List[str], out_path: str, fps: int = 24):
    if len(image_paths) == 0:
        raise ValueError(f"No images found for video: {out_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    try:
        for img_path in image_paths:
            frame = imageio.imread(img_path)
            writer.append_data(frame)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_root", type=str, required=True,
                        help="Path to result_rgb directory.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save mp4 videos.")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output video FPS.")
    parser.add_argument("--views", type=str, nargs="*", default=None,
                        help="Optional subset of view names, e.g. view_00.png view_03.png")
    args = parser.parse_args()

    frame_dirs = sorted_frame_dirs(args.rgb_root)
    if len(frame_dirs) == 0:
        raise RuntimeError(f"No frame_* directories found under: {args.rgb_root}")

    all_views = discover_views(frame_dirs)
    if len(all_views) == 0:
        raise RuntimeError(f"No view_*.png found under: {args.rgb_root}")

    if args.views is not None and len(args.views) > 0:
        target_views = args.views
    else:
        target_views = all_views

    print(f"Found {len(frame_dirs)} frame directories.")
    print(f"Found views: {target_views}")

    for view_name in target_views:
        image_paths = collect_images_for_view(frame_dirs, view_name)
        if len(image_paths) == 0:
            print(f"[Skip] {view_name}: no images found")
            continue

        stem = os.path.splitext(view_name)[0]
        out_path = os.path.join(args.out_dir, f"{stem}.mp4")

        print(f"[Video] {view_name}: {len(image_paths)} frames -> {out_path}")
        make_mp4(image_paths, out_path, fps=args.fps)

    print("Done.")


if __name__ == "__main__":
    main()

"""
python tools/folder_to_mp4_v3.py \
  --rgb_root /efs/yanruibin/projects/Direct3D-S2/vis/rendering_v3/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_0/result_rgb \
  --out_dir /efs/yanruibin/projects/Direct3D-S2/vis/rendering_v3/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_0/result_rgb_mp4 \
  --fps 10 \
  --views view_00.png
"""