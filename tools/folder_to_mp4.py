from pathlib import Path
f_path = Path(__file__).absolute()
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))

import os
import re
import av
import argparse
import numpy as np
from PIL import Image


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]


def list_images(input_dir: str):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
            files.append(path)
    files.sort(key=natural_key)
    return files


def load_image(path: str, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img)


def folder_to_mp4(input_dir: str, output_path: str, fps: int = 24, crf: int = 18):
    image_paths = list_images(input_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {input_dir}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    first = Image.open(image_paths[0]).convert("RGB")
    width, height = first.size

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    # 可选：控制画质，越小越清晰，常用 18~23
    stream.options = {"crf": str(crf)}

    for path in image_paths:
        frame_rgb = load_image(path, target_size=(width, height))
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    # flush
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    print(f"Saved video to: {output_path}")
    print(f"Frames: {len(image_paths)}, Resolution: {width}x{height}, FPS: {fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality, lower is better")
    args = parser.parse_args()

    folder_to_mp4(
        input_dir=args.input_dir,
        output_path=args.output_path,
        fps=args.fps,
        crf=args.crf,
    )