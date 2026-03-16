#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import argparse
import pandas as pd
import objaverse.xl as oxl


def parse_meta_save_uid(meta_str):
    """
    从 meta 字段中解析 save_uid。
    meta_str 形如:
    "{'animation_count': 0, ..., 'save_uid': 'xxxx', ...}"
    """
    if pd.isna(meta_str):
        return None
    try:
        meta_dict = ast.literal_eval(meta_str)
        if isinstance(meta_dict, dict):
            return meta_dict.get("save_uid", None)
    except Exception:
        return None
    return None


def load_uuid_set(uuid_txt_path):
    with open(uuid_txt_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_and_filter_metadata(csv_path, uuid_txt_path):
    df = pd.read_csv(csv_path)

    # 统一列名，避免后续使用时不一致
    if "fileIdentifier" in df.columns and "file_identifier" not in df.columns:
        df = df.rename(columns={"fileIdentifier": "file_identifier"})

    required_cols = ["file_identifier", "sha256", "meta"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列: {col}")

    # 从 meta 中提取 save_uid
    df["save_uid"] = df["meta"].apply(parse_meta_save_uid)

    uuid_set = load_uuid_set(uuid_txt_path)

    before = len(df)
    filtered_df = df[df["save_uid"].isin(uuid_set)].copy()
    after = len(filtered_df)

    # 去重，避免重复下载
    filtered_df = filtered_df.drop_duplicates(subset=["file_identifier", "sha256"]).copy()

    print(f"[INFO] original rows: {before}")
    print(f"[INFO] rows after save_uid filtering: {after}")
    print(f"[INFO] rows after dedup: {len(filtered_df)}")

    return filtered_df


def download_filtered_objects(metadata, output_dir, processes=32):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)

    # 下载 annotations
    annotations = oxl.get_annotations()
    print(f"[INFO] total annotations: {len(annotations)}")

    annotations = annotations[annotations["sha256"].isin(metadata["sha256"].values)].copy()
    print(f"[INFO] matched annotations after sha256 filtering: {len(annotations)}")

    # 直接调用你指定的下载方式
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
        processes=processes,
    )

    return annotations, file_paths


def save_file_paths_csv(file_paths, metadata, save_path):
    """
    把 download_objects 的返回结果保存成 csv。
    一般 file_paths 是:
        {file_identifier: downloaded_path}
    """
    meta_small = metadata[["file_identifier", "sha256", "save_uid"]].drop_duplicates().copy()

    rows = []
    for file_identifier, local_path in file_paths.items():
        matched = meta_small[meta_small["file_identifier"] == file_identifier]
        if len(matched) == 0:
            rows.append({
                "file_identifier": file_identifier,
                "sha256": None,
                "save_uid": None,
                "local_path": local_path,
            })
        else:
            for _, row in matched.iterrows():
                rows.append({
                    "file_identifier": file_identifier,
                    "sha256": row["sha256"],
                    "save_uid": row["save_uid"],
                    "local_path": local_path,
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(save_path, index=False)
    print(f"[INFO] file_paths saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/diffusion4d/meta_xl_tot.csv",
    )
    parser.add_argument(
        "--uuid_txt_path",
        type=str,
        default="data/diffusion4d/objaverseXL_curated_uuid_list.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--save_filtered_csv",
        type=str,
        default=None,
        help="可选：保存过滤后的 metadata",
    )
    parser.add_argument(
        "--save_file_paths_csv",
        type=str,
        default=None,
        help="可选：保存 download_objects 返回结果",
    )
    args = parser.parse_args()

    metadata = load_and_filter_metadata(args.csv_path, args.uuid_txt_path)

    if len(metadata) == 0:
        print("[WARN] 过滤后没有可下载的数据，退出。")
        return

    if args.save_filtered_csv is not None:
        metadata.to_csv(args.save_filtered_csv, index=False)
        print(f"[INFO] filtered metadata saved to: {args.save_filtered_csv}")

    annotations, file_paths = download_filtered_objects(
        metadata=metadata,
        output_dir=args.output_dir,
        processes=args.processes,
    )

    print(f"[INFO] downloaded objects: {len(file_paths)}")

    if args.save_file_paths_csv is not None:
        save_file_paths_csv(file_paths, metadata, args.save_file_paths_csv)


if __name__ == "__main__":
    main()

"""
python tools/download_diffusion4d.py \
    --csv_path data/diffusion4d/meta_xl_animation_tot.csv \
    --uuid_txt_path data/diffusion4d/objaverseXL_curated_uuid_list.txt \
    --output_dir data/diffusion4d/objaverse_xl_downloaded \
    --processes 32 \
    --save_filtered_csv data/diffusion4d/objaverse_xl_downloaded/meta_xl_filtered.csv \
    --save_file_paths_csv data/diffusion4d/objaverse_xl_downloaded/file_paths.csv
"""