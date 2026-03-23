#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Optional, Set

import pandas as pd


UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def read_id_list(path: str) -> Set[str]:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip().lower()
            if x:
                ids.add(x)
    return ids


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.json_normalize(data)
        elif isinstance(data, dict):
            return pd.json_normalize(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def extract_save_uid(meta_val) -> Optional[str]:
    """
    从 target_meta 的 meta 列中提取 save_uid
    """
    if pd.isna(meta_val):
        return None

    if isinstance(meta_val, dict):
        uid = meta_val.get("save_uid", None)
        return str(uid).lower() if uid is not None else None

    s = str(meta_val).strip()
    if not s:
        return None

    # 先尝试把 python dict 字符串解析出来
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            uid = parsed.get("save_uid", None)
            if uid is not None:
                return str(uid).lower()
    except Exception:
        pass

    # 再直接正则找 save_uid
    m = re.search(
        r"""['"]save_uid['"]\s*:\s*['"]([0-9a-fA-F\-]{36})['"]""",
        s,
    )
    if m:
        return m.group(1).lower()

    # 兜底：抓第一个 UUID
    m2 = UUID_RE.search(s)
    if m2:
        return m2.group(1).lower()

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 统一 sha256
    if "sha256" in df.columns:
        df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()

    # 统一 file_identifier / fileIdentifier
    if "file_identifier" in df.columns:
        df["file_identifier_norm"] = df["file_identifier"].astype(str).str.strip()
    elif "fileIdentifier" in df.columns:
        df["file_identifier_norm"] = df["fileIdentifier"].astype(str).str.strip()

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to source metadata.csv")
    parser.add_argument("--target_uid_list", type=str, required=True, help="Path to target uid list txt")
    parser.add_argument("--target_meta", type=str, required=True, help="Path to target meta csv/parquet")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--meta_column", type=str, default="meta", help="Column in target_meta storing dict-like string")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading source")
    source_df = load_table(args.source)
    source_df = normalize_columns(source_df)
    print(f"Source shape: {source_df.shape}")
    print(f"Source columns: {list(source_df.columns)}")

    print("[2/6] Loading target uid list")
    target_uids = read_id_list(args.target_uid_list)
    print(f"Target uid list size: {len(target_uids)}")

    print("[3/6] Loading target meta")
    target_meta_df = load_table(args.target_meta)
    target_meta_df = normalize_columns(target_meta_df)
    print(f"Target meta shape: {target_meta_df.shape}")
    print(f"Target meta columns: {list(target_meta_df.columns)}")

    if args.meta_column not in target_meta_df.columns:
        raise ValueError(
            f"Column '{args.meta_column}' not found in target_meta. "
            f"Available columns: {list(target_meta_df.columns)}"
        )

    print("[4/6] Extracting save_uid from target meta")
    target_meta_df = target_meta_df.copy()
    target_meta_df["save_uid"] = target_meta_df[args.meta_column].map(extract_save_uid)
    extracted_cnt = int(target_meta_df["save_uid"].notna().sum())
    print(f"Rows with extracted save_uid: {extracted_cnt}")

    print("[5/6] Intersecting target_uid_list with target_meta via save_uid")
    target_filtered = target_meta_df[target_meta_df["save_uid"].isin(target_uids)].copy()
    print(f"Filtered target_meta rows: {len(target_filtered)}")
    print(f"Filtered unique save_uid: {target_filtered['save_uid'].nunique()}")

    # 保存第一步结果
    target_filtered_csv = out_dir / "target_uid_meta_intersection.csv"
    target_filtered_parquet = out_dir / "target_uid_meta_intersection.parquet"
    target_filtered.to_csv(target_filtered_csv, index=False)
    target_filtered.to_parquet(target_filtered_parquet, index=False)

    # 第二步：和 source 求交集，优先用 sha256
    print("[6/6] Intersecting filtered target_meta with source via sha256")

    if "sha256" not in source_df.columns:
        raise ValueError("source file does not contain 'sha256' column")
    if "sha256" not in target_filtered.columns:
        raise ValueError("filtered target_meta does not contain 'sha256' column")

    source_sha_set = set(source_df["sha256"].dropna().astype(str).str.lower())
    target_sha_set = set(target_filtered["sha256"].dropna().astype(str).str.lower())
    common_sha = source_sha_set & target_sha_set

    print(f"Common sha256 count: {len(common_sha)}")

    source_matched = source_df[source_df["sha256"].isin(common_sha)].copy()
    target_matched = target_filtered[target_filtered["sha256"].isin(common_sha)].copy()

    # 做一个 join，便于直接看 source 和 target_meta 的对应关系
    joined = pd.merge(
        source_matched,
        target_matched,
        on="sha256",
        how="inner",
        suffixes=("_source", "_target"),
    )

    print(f"Matched source rows: {len(source_matched)}")
    print(f"Matched target rows: {len(target_matched)}")
    print(f"Joined rows: {len(joined)}")

    # 输出文件
    common_sha_txt = out_dir / "common_sha256.txt"
    with open(common_sha_txt, "w", encoding="utf-8") as f:
        for x in sorted(common_sha):
            f.write(x + "\n")

    source_matched_csv = out_dir / "source_intersection.csv"
    target_matched_csv = out_dir / "target_meta_intersection.csv"
    joined_csv = out_dir / "source_target_joined.csv"

    source_matched.to_csv(source_matched_csv, index=False)
    target_matched.to_csv(target_matched_csv, index=False)
    joined.to_csv(joined_csv, index=False)

    # 也存 parquet
    source_matched.to_parquet(out_dir / "source_intersection.parquet", index=False)
    target_matched.to_parquet(out_dir / "target_meta_intersection.parquet", index=False)
    joined.to_parquet(out_dir / "source_target_joined.parquet", index=False)

    # 如果 joined 里同时有 file_identifier_norm_source / target，可顺手做一致性检查
    source_url_col = "file_identifier_norm_source" if "file_identifier_norm_source" in joined.columns else None
    target_url_col = "file_identifier_norm_target" if "file_identifier_norm_target" in joined.columns else None
    if source_url_col and target_url_col:
        url_equal = (joined[source_url_col] == joined[target_url_col]).sum()
        print(f"Rows with identical file identifier after join: {url_equal}/{len(joined)}")

    print("\n========== Done ==========")
    print(f"Saved:")
    print(f"  {target_filtered_csv}")
    print(f"  {target_filtered_parquet}")
    print(f"  {common_sha_txt}")
    print(f"  {source_matched_csv}")
    print(f"  {target_matched_csv}")
    print(f"  {joined_csv}")

    # 打印预览
    preview_cols = [c for c in [
        "sha256",
        "file_identifier_source",
        "fileIdentifier_target",
        "save_uid",
    ] if c in joined.columns]

    if preview_cols:
        print("\nPreview:")
        print(joined[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

"""
python tools/intersect_source_with_target_uid_and_meta.py \
  --source /efs/yanruibin/projects/TRELLIS/datasets/ObjaverseXL_sketchfab/metadata.csv \
  --target_uid_list /efs/yanruibin/projects/Direct3D-S2/data/diffusion4d/objaverseXL_curated_uuid_list.txt \
  --target_meta /efs/yanruibin/projects/Direct3D-S2/data/diffusion4d/meta_xl_tot.csv \
  --out_dir /efs/yanruibin/projects/Direct3D-S2/data/diffusion4d/intersection_results_sketchfab

  github: 16337
"""
