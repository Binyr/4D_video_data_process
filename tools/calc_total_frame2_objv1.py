#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

def main():
    root = "data/objverse_minghao_4d/motion_info"
    threshold = 1e-3
    cap_for_stats = 128

    valid_json_count = 0
    invalid_json_count = 0
    missing_json_count = 0
    no_motion_count = 0

    total_kept_frames = 0          # 实际保留帧数总和
    total_kept_frames_capped = 0   # 统计时每个序列最多按128帧算

    for shard in sorted(os.listdir(root)):
        shard_dir = os.path.join(root, shard)
        if not os.path.isdir(shard_dir):
            continue

        for obj_id in sorted(os.listdir(shard_dir)):
            obj_dir = os.path.join(shard_dir, obj_id)
            if not os.path.isdir(obj_dir):
                continue

            json_path = os.path.join(obj_dir, "umeyama_similarity.json")
            if not os.path.isfile(json_path):
                missing_json_count += 1
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                num_frames = data.get("num_frames", None)
                results = data.get("results", None)

                if num_frames is None or not isinstance(results, list):
                    print(f"[WARN] invalid json structure: {json_path}")
                    invalid_json_count += 1
                    continue

                num_frames = int(num_frames)

                motion_flags = {}  # frame_idx -> bool
                for item in results:
                    if not isinstance(item, dict):
                        continue

                    frame_idx = item.get("frame", None)
                    rms_error = item.get("rms_error", None)
                    if frame_idx is None or rms_error is None:
                        continue

                    frame_idx = int(frame_idx)
                    if frame_idx < 1 or frame_idx > num_frames:
                        continue

                    motion_flags[frame_idx] = (float(rms_error) > threshold)

                motion_frame_list = [
                    frame_idx
                    for frame_idx in range(1, num_frames + 1)
                    if motion_flags.get(frame_idx, False)
                ]

                if len(motion_frame_list) == 0:
                    data["motion_frame_indices"] = []
                    kept_len = 0
                    no_motion_count += 1
                else:
                    first_motion = motion_frame_list[0]
                    last_motion = motion_frame_list[-1]
                    data["motion_frame_indices"] = list(range(first_motion, last_motion + 1))
                    kept_len = last_motion - first_motion + 1

                total_kept_frames += kept_len
                total_kept_frames_capped += min(kept_len, cap_for_stats)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                valid_json_count += 1

            except Exception as e:
                print(f"[ERROR] Failed to process {json_path}: {e}")
                invalid_json_count += 1

    print("Done.")
    print(f"Valid json count: {valid_json_count}")
    print(f"Missing json count: {missing_json_count}")
    print(f"Invalid json count: {invalid_json_count}")
    print(f"No-motion sequence count: {no_motion_count}")
    print(f"Total kept frames: {total_kept_frames}")
    print(f"Total kept frames (cap 128 per sequence): {total_kept_frames_capped}")

if __name__ == "__main__":
    main()