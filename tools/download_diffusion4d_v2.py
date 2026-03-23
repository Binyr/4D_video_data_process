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


# def normalize_file_identifier_column(df):
#     """
#     统一列名:
#     fileIdentifier -> file_identifier
#     """
#     if "fileIdentifier" in df.columns and "file_identifier" not in df.columns:
#         df = df.rename(columns={"fileIdentifier": "file_identifier"})
#     return df

def normalize_file_identifier_column(df):
    """
    同时保留:
    - fileIdentifier  (给 objaverse.xl 用)
    - file_identifier (给我们自己的逻辑用)
    """
    df = df.copy()

    if "fileIdentifier" in df.columns and "file_identifier" not in df.columns:
        df["file_identifier"] = df["fileIdentifier"]

    if "file_identifier" in df.columns and "fileIdentifier" not in df.columns:
        df["fileIdentifier"] = df["file_identifier"]

    return df


def load_uuid_set(uuid_txt_path):
    with open(uuid_txt_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_and_filter_metadata(csv_path, uuid_txt_path):
    df = pd.read_csv(csv_path)
    df = normalize_file_identifier_column(df)

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


def build_rows_from_file_paths(file_paths, metadata):
    """
    把 download_objects 的返回结果转成 dataframe rows。
    一般 file_paths 是:
        {file_identifier: downloaded_path}
    """
    meta_small = metadata[["file_identifier", "sha256", "save_uid"]].drop_duplicates().copy()

    rows = []
    for file_identifier, local_path in file_paths.items():
        matched = meta_small[meta_small["file_identifier"] == file_identifier]

        # 标记是否真的下载成功
        success = isinstance(local_path, str) and len(local_path) > 0
        if success and os.path.exists(local_path):
            status = "success"
        elif success:
            status = "returned_but_not_found"
        else:
            status = "missing"

        if len(matched) == 0:
            rows.append({
                "file_identifier": file_identifier,
                "sha256": None,
                "save_uid": None,
                "local_path": local_path,
                "status": status,
            })
        else:
            for _, row in matched.iterrows():
                rows.append({
                    "file_identifier": file_identifier,
                    "sha256": row["sha256"],
                    "save_uid": row["save_uid"],
                    "local_path": local_path,
                    "status": status,
                })

    return pd.DataFrame(rows)


def load_checkpoint_df(checkpoint_csv):
    """
    读取已有 checkpoint。
    只保留必要列；如果文件不存在，返回空 dataframe。
    """
    if checkpoint_csv is None or (not os.path.exists(checkpoint_csv)):
        return pd.DataFrame(columns=["file_identifier", "sha256", "save_uid", "local_path", "status"])

    ckpt = pd.read_csv(checkpoint_csv)
    ckpt = normalize_file_identifier_column(ckpt)

    for col in ["file_identifier", "sha256", "save_uid", "local_path", "status"]:
        if col not in ckpt.columns:
            ckpt[col] = None

    ckpt = ckpt[["file_identifier", "sha256", "save_uid", "local_path", "status"]].copy()
    return ckpt


def get_completed_file_identifiers(checkpoint_csv):
    """
    从 checkpoint 中找出已经成功下载且本地文件仍存在的 file_identifier。
    """
    ckpt = load_checkpoint_df(checkpoint_csv)
    if len(ckpt) == 0:
        return set()

    def _is_valid_done(row):
        local_path = row["local_path"]
        status = row["status"]
        return (
            isinstance(local_path, str)
            and len(local_path) > 0
            and os.path.exists(local_path)
            and str(status) == "success"
        )

    done_mask = ckpt.apply(_is_valid_done, axis=1)
    done_ids = set(ckpt.loc[done_mask, "file_identifier"].dropna().astype(str).tolist())
    return done_ids


def merge_and_save_checkpoint(new_rows_df, checkpoint_csv):
    """
    把新结果 merge 到 checkpoint 并保存。
    对同一个 file_identifier，保留最后一次记录。
    """
    os.makedirs(os.path.dirname(checkpoint_csv), exist_ok=True)

    old_ckpt = load_checkpoint_df(checkpoint_csv)

    merged = pd.concat([old_ckpt, new_rows_df], ignore_index=True)

    # 同一个 file_identifier 保留最后一次
    merged = merged.drop_duplicates(subset=["file_identifier"], keep="last").copy()

    merged.to_csv(checkpoint_csv, index=False)
    print(f"[INFO] checkpoint updated: {checkpoint_csv}")


def save_file_paths_csv_from_checkpoint(checkpoint_csv, save_path):
    """
    把 checkpoint 复制/导出成最终 file_paths csv。
    """
    ckpt = load_checkpoint_df(checkpoint_csv)
    ckpt.to_csv(save_path, index=False)
    print(f"[INFO] file_paths saved to: {save_path}")


def prepare_annotations(metadata):
    """
    从 Objaverse XL annotations 中筛出需要下载的对象。
    用 (file_identifier, sha256) 精确匹配，避免误匹配。
    """
    annotations = oxl.get_annotations()
    annotations = normalize_file_identifier_column(annotations)

    required_cols = ["file_identifier", "sha256"]
    for col in required_cols:
        if col not in annotations.columns:
            raise ValueError(f"Annotations 缺少必要列: {col}")

    print(f"[INFO] total annotations: {len(annotations)}")

    meta_keys = metadata[["file_identifier", "sha256"]].drop_duplicates().copy()
    annotations = annotations.merge(
        meta_keys,
        on=["file_identifier", "sha256"],
        how="inner",
    )

    print(f"[INFO] matched annotations after filtering: {len(annotations)}")
    return annotations


def download_filtered_objects(
    metadata,
    output_dir,
    processes=32,
    batch_size=128,
    checkpoint_csv=None,
    resume=True,
):
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    annotations = prepare_annotations(metadata)

    # 断点续下：跳过 checkpoint 中已经成功的 file_identifier
    if resume and checkpoint_csv is not None:
        done_ids = get_completed_file_identifiers(checkpoint_csv)
        if len(done_ids) > 0:
            before = len(annotations)
            annotations = annotations[~annotations["file_identifier"].astype(str).isin(done_ids)].copy()
            print(f"[INFO] resume enabled, skip finished objects: {before - len(annotations)}")
            print(f"[INFO] remaining objects to download: {len(annotations)}")
        else:
            print("[INFO] resume enabled, but no valid finished records found in checkpoint.")

    if len(annotations) == 0:
        print("[INFO] nothing left to download.")
        return {}, 0, 0

    all_file_paths = {}
    total = len(annotations)
    num_batches = (total + batch_size - 1) // batch_size
    success_count_total = 0

    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        batch_annotations = annotations.iloc[start:end].copy()

        print(f"[INFO] downloading batch {batch_idx}/{num_batches}, size={len(batch_annotations)}")

        try:
            file_paths = oxl.download_objects(
                batch_annotations,
                download_dir=raw_dir,
                save_repo_format="zip", # "files",
                processes=processes,
            )
        except KeyboardInterrupt:
            print("\n[WARN] interrupted by user. Previous finished batches have already been saved.")
            raise
        except Exception as e:
            print(f"[ERROR] batch {batch_idx} failed: {e}")
            continue

        if not isinstance(file_paths, dict):
            print(f"[WARN] unexpected file_paths type: {type(file_paths)}")
            continue

        all_file_paths.update(file_paths)

        batch_meta = metadata[
            metadata["file_identifier"].astype(str).isin(
                batch_annotations["file_identifier"].astype(str)
            )
        ].copy()

        batch_rows_df = build_rows_from_file_paths(file_paths, batch_meta)

        success_count = 0
        if len(batch_rows_df) > 0:
            success_count = int((batch_rows_df["status"] == "success").sum())

        success_count_total += success_count
        print(f"[INFO] batch {batch_idx} returned {len(file_paths)} objects, success={success_count}")

        if checkpoint_csv is not None and len(batch_rows_df) > 0:
            merge_and_save_checkpoint(batch_rows_df, checkpoint_csv)

    return all_file_paths, total, success_count_total


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
        "--batch_size",
        type=int,
        default=128,
        help="每个 batch 下载多少个对象。越小越适合断点续下，但总调度开销会更大。",
    )
    parser.add_argument(
        "--save_filtered_csv",
        type=str,
        default=None,
        help="可选：保存过滤后的 metadata；如果该文件已存在，则直接读取它",
    )
    parser.add_argument(
        "--save_file_paths_csv",
        type=str,
        default=None,
        help="保存 file_paths / checkpoint 的 csv。默认保存在 output_dir/file_paths.csv",
    )

    parser.add_argument("--resume", dest="resume", action="store_true", help="开启断点续下")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="关闭断点续下")
    parser.set_defaults(resume=True)

    args = parser.parse_args()

    checkpoint_csv = args.save_file_paths_csv
    if checkpoint_csv is None:
        checkpoint_csv = os.path.join(args.output_dir, "file_paths.csv")

    # -----------------------------
    # 优先复用已保存的 filtered metadata
    # -----------------------------
    if args.save_filtered_csv is not None and os.path.exists(args.save_filtered_csv):
        print(f"[INFO] found existing filtered metadata, loading from: {args.save_filtered_csv}")
        metadata = pd.read_csv(args.save_filtered_csv)
        metadata = normalize_file_identifier_column(metadata)

        required_cols = ["file_identifier", "sha256", "save_uid"]
        for col in required_cols:
            if col not in metadata.columns:
                raise ValueError(
                    f"Existing filtered csv 缺少必要列: {col}. "
                    f"请删除该文件后重新生成: {args.save_filtered_csv}"
                )

        metadata = metadata.drop_duplicates(subset=["file_identifier", "sha256"]).copy()
        print(f"[INFO] loaded filtered metadata rows: {len(metadata)}")
    else:
        metadata = load_and_filter_metadata(args.csv_path, args.uuid_txt_path)

        if args.save_filtered_csv is not None:
            os.makedirs(os.path.dirname(args.save_filtered_csv), exist_ok=True)
            metadata.to_csv(args.save_filtered_csv, index=False)
            print(f"[INFO] filtered metadata saved to: {args.save_filtered_csv}")

    if len(metadata) == 0:
        print("[WARN] 过滤后没有可下载的数据，退出。")
        return

    try:
        file_paths, total_to_download, success_count = download_filtered_objects(
            metadata=metadata,
            output_dir=args.output_dir,
            processes=args.processes,
            batch_size=args.batch_size,
            checkpoint_csv=checkpoint_csv,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        print("[INFO] stopped. You can rerun the same command to resume.")
        return

    print(f"[INFO] attempted objects in this run: {total_to_download}")
    print(f"[INFO] download_objects returned: {len(file_paths)}")
    print(f"[INFO] successful objects in this run: {success_count}")

    save_file_paths_csv_from_checkpoint(checkpoint_csv, checkpoint_csv)


if __name__ == "__main__":
    main()

"""
python tools/download_diffusion4d_v2.py \
    --csv_path data/diffusion4d/meta_xl_animation_tot.csv \
    --uuid_txt_path data/diffusion4d/objaverseXL_curated_uuid_list.txt \
    --output_dir data/diffusion4d/objaverse_xl_downloaded_v2 \
    --processes 32 \
    --batch_size 512 \
    --save_filtered_csv data/diffusion4d/objaverse_xl_downloaded_v2/meta_xl_filtered.csv \
    --save_file_paths_csv data/diffusion4d/objaverse_xl_downloaded_v2/file_paths.csv
"""