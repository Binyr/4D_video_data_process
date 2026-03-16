import os
import av
import cv2
import argparse
import numpy as np
from fractions import Fraction


def get_video_stream(container):
    return next(s for s in container.streams if s.type == "video")


def resize_keep_aspect(img, target_height):
    h, w = img.shape[:2]
    if h == target_height:
        return img
    scale = target_height / h
    new_w = int(round(w * scale))
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return resized


def concat_videos_width(input_paths, output_path, fps=None):
    if len(input_paths) == 0:
        raise ValueError("No input videos provided.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 打开所有输入视频
    containers = [av.open(p) for p in input_paths]
    streams = [get_video_stream(c) for c in containers]

    # 目标高度：用第一个视频的高度
    target_height = streams[0].codec_context.height

    # 输出 fps
    if fps is not None:
        target_fps = Fraction(str(fps))
    else:
        target_fps = streams[0].average_rate
        if target_fps is None:
            target_fps = Fraction(30, 1)

    # 为每个视频建立 decoder
    decoders = [c.decode(s) for c, s in zip(containers, streams)]

    # 先根据原始尺寸估算输出宽度
    resized_widths = []
    for s in streams:
        h = s.codec_context.height
        w = s.codec_context.width
        scale = target_height / h
        resized_w = int(round(w * scale))
        resized_widths.append(resized_w)

    output_width = int(sum(resized_widths))
    output_height = int(target_height)

    print("Input videos:")
    for path, s, rw in zip(input_paths, streams, resized_widths):
        print(f"  {path}: {s.codec_context.width}x{s.codec_context.height} -> {rw}x{output_height}")
    print(f"Output video: {output_width}x{output_height}, fps={target_fps}")

    out_container = av.open(output_path, mode="w")
    out_stream = out_container.add_stream("h264", rate=target_fps)
    out_stream.width = output_width
    out_stream.height = output_height
    out_stream.pix_fmt = "yuv420p"

    frame_idx = 0
    while True:
        imgs = []

        # 每个视频取一帧；只要有一个视频结束，就整体结束
        for decoder in decoders:
            try:
                frame = next(decoder)
            except StopIteration:
                print(f"Finished at frame {frame_idx} (one input video ended).")
                for packet in out_stream.encode():
                    out_container.mux(packet)
                out_container.close()
                for c in containers:
                    c.close()
                print(f"Saved to: {output_path}")
                return

            img = frame.to_ndarray(format="rgb24")
            img = resize_keep_aspect(img, target_height)
            imgs.append(img)

        # 在宽度维拼接
        concat_img = np.concatenate(imgs, axis=1)  # H x W_total x 3

        out_frame = av.VideoFrame.from_ndarray(concat_img, format="rgb24")
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input video paths, concatenated along width in the given order"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output mp4 path"
    )
    parser.add_argument(
        "--fps",
        type=str,
        default=None,
        help="Optional output FPS, e.g. 30 or 29.97"
    )
    args = parser.parse_args()

    concat_videos_width(args.inputs, args.out, args.fps)


if __name__ == "__main__":
    main()