#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, List

OLD_ROOT = "/group/40075/yanruibin/objverse_minghao_4d_mine/rendering_v5"


def replace_root(path: str, old_root: str, new_root: str) -> str:
    """
    Replace old_root in path with new_root.
    """
    old_root = os.path.normpath(old_root)
    new_root = os.path.normpath(new_root)
    path_norm = os.path.normpath(path)

    if path_norm == old_root:
        return new_root

    if path_norm.startswith(old_root + os.sep):
        rel = os.path.relpath(path_norm, old_root)
        return os.path.join(new_root, rel)

    # 如果这个路径本身不以 old_root 开头，就原样返回
    return path


def check_paths(path_dict: Dict[str, List[str]], input_root: str, old_root: str):
    summary = {}

    for split, paths in path_dict.items():
        if not isinstance(paths, list):
            print(f"[Warning] key '{split}' is not a list, skip.")
            continue

        missing = []
        exists_cnt = 0

        for p in paths:
            new_path = replace_root(p, old_root, input_root)
            if os.path.exists(new_path):
                exists_cnt += 1
            else:
                missing.append({
                    "original_path": p,
                    "checked_path": new_path,
                })

        total = len(paths)
        summary[split] = {
            "total": total,
            "exists": exists_cnt,
            "missing": len(missing),
            "missing_items": missing,
        }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the input json file"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="New root path to replace OLD_ROOT"
    )
    parser.add_argument(
        "--save_missing_json",
        type=str,
        default=None,
        help="Optional path to save missing paths as json"
    )
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = check_paths(data, args.input_root, OLD_ROOT)

    for split, info in summary.items():
        print(f"\n===== {split} =====")
        print(f"total   : {info['total']}")
        print(f"exists  : {info['exists']}")
        print(f"missing : {info['missing']}")

        if info["missing"] > 0:
            print("Missing paths:")
            for item in info["missing_items"]:
                print(item["checked_path"])

    if args.save_missing_json is not None:
        save_data = {
            split: info["missing_items"]
            for split, info in summary.items()
        }
        with open(args.save_missing_json, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved missing paths to: {args.save_missing_json}")


if __name__ == "__main__":
    main()

"""
python tools/tmp/validate_anns.py \
  --json_path data/objverse_minghao_4d_mine_40075/rendering_v5_anns_1.3k.json \
  --input_root /efs/yanruibin/projects/Direct3D-S2/data/objverse_minghao_4d_mine_40075/rendering_v5 \
  --save_missing_json missing_paths.json
"""