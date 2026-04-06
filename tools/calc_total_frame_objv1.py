#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
def main():
    root = "data/objverse_minghao_4d/motion_info"
    threshold = 1e-3
    max_frames_per_seq = 128

    total_frames_capped = 0
    total_motion_frames = 0
    total_valid_json = 0
    total_missing_json = 0
    total_invalid_json = 0

    for shard in tqdm(sorted(os.listdir(root))):
        shard_dir = os.path.join(root, shard)
        if not os.path.isdir(shard_dir):
            continue

        for obj_id in sorted(os.listdir(shard_dir)):
            obj_dir = os.path.join(shard_dir, obj_id)
            if not os.path.isdir(obj_dir):
                continue

            json_path = os.path.join(obj_dir, "umeyama_similarity.json")
            if not os.path.isfile(json_path):
                total_missing_json += 1
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                num_frames = data.get("num_frames", None)
                results = data.get("results", None)

                if num_frames is None or results is None or not isinstance(results, list):
                    print(f"[WARN] invalid json structure: {json_path}")
                    total_invalid_json += 1
                    continue

                capped_num_frames = min(int(num_frames), max_frames_per_seq)
                total_frames_capped += capped_num_frames

                motion_frames = 0
                for item in results:
                    if not isinstance(item, dict):
                        continue

                    frame_idx = item.get("frame", None)
                    rms_error = item.get("rms_error", None)

                    if frame_idx is None or rms_error is None:
                        continue

                    if int(frame_idx) > capped_num_frames:
                        continue

                    if float(rms_error) > threshold:
                        motion_frames += 1

                total_motion_frames += motion_frames
                total_valid_json += 1

            except Exception as e:
                print(f"[ERROR] Failed to read {json_path}: {e}")
                total_invalid_json += 1

    ratio = total_motion_frames / total_frames_capped if total_frames_capped > 0 else 0.0

    print(f"Total capped frames: {total_frames_capped}")
    print(f"Total motion frames (rms_error > {threshold}): {total_motion_frames}")
    print(f"Motion frame ratio: {ratio:.6f}")
    print(f"Valid json count: {total_valid_json}")
    print(f"Missing json count: {total_missing_json}")
    print(f"Invalid json count: {total_invalid_json}")

if __name__ == "__main__":
    main()