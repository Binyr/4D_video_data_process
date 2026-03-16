#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Set, Tuple, List, Dict

import pandas as pd


# 36-char UUID, e.g. 00017df4-0856-556a-bbfe-3e5fff54f17f
OBJAVERSE_XL_PATTERN = r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"

# 32-char hex id, e.g. 00052d0d9a304d308058260a4a9b7c20
OBJAVERSE_PATTERN = r"(?<![0-9a-fA-F])([0-9a-fA-F]{32})(?![0-9a-fA-F])"


def read_id_list(path: str) -> Set[str]:
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip().lower()
            if x:
                ids.add(x)
    return ids


def load_metadata(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.json_normalize(data)
        elif isinstance(data, dict):
            # 尝试展开 dict；如果顶层不是 list，也尽量转成一行
            return pd.json_normalize(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")

    raise ValueError(f"Unsupported metadata format: {suffix}")


def get_string_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        try:
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
                cols.append(c)
        except Exception:
            pass
    return cols


def scan_intersections(
    df: pd.DataFrame,
    target_ids: Set[str],
    regex_pattern: str,
    label: str,
) -> Tuple[Set[str], pd.DataFrame, Dict[str, int]]:
    """
    在 metadata 的所有字符串列中提取符合 regex_pattern 的 id，
    然后和 target_ids 求交集。
    """
    matched_ids = set()
    matched_rows = []
    per_column_count = {}

    df = df.copy()
    df["__row_idx__"] = df.index

    string_cols = get_string_columns(df)

    for col in string_cols:
        s = df[col].astype("string")

        # 从字符串中提取 id（即使该列里是 URL/路径，也能抓出里面的 id）
        extracted = s.str.extract(regex_pattern, expand=False)

        if extracted is None:
            continue

        extracted = extracted.str.lower()
        mask = extracted.isin(target_ids)

        count = int(mask.sum())
        if count == 0:
            continue

        per_column_count[col] = count
        matched_ids.update(extracted[mask].dropna().tolist())

        tmp = df.loc[mask].copy()
        tmp["matched_id"] = extracted[mask].values
        tmp["matched_column"] = col
        tmp["matched_type"] = label
        matched_rows.append(tmp)

    if len(matched_rows) > 0:
        matched_rows_df = pd.concat(matched_rows, axis=0, ignore_index=True)
    else:
        matched_rows_df = pd.DataFrame(columns=list(df.columns) + ["matched_id", "matched_column", "matched_type"])

    return matched_ids, matched_rows_df, per_column_count


def save_ids(ids: Set[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for x in sorted(ids):
            f.write(x + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata file")
    parser.add_argument("--objaverse_xl_list", type=str, required=True, help="Path to Objaverse-XL UUID list")
    parser.add_argument("--objaverse_list", type=str, required=True, help="Path to Objaverse 32-char ID list")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading metadata: {args.metadata}")
    df = load_metadata(args.metadata)
    print(f"Metadata shape: {df.shape}")

    print(f"[2/4] Loading id lists")
    objxl_ids = read_id_list(args.objaverse_xl_list)
    obj_ids = read_id_list(args.objaverse_list)
    print(f"Objaverse-XL id list size: {len(objxl_ids)}")
    print(f"Objaverse     id list size: {len(obj_ids)}")

    print(f"[3/4] Scanning metadata for Objaverse-XL UUID intersection")
    objxl_intersection, objxl_rows, objxl_col_stats = scan_intersections(
        df=df,
        target_ids=objxl_ids,
        regex_pattern=OBJAVERSE_XL_PATTERN,
        label="objaverse_xl",
    )

    print(f"[4/4] Scanning metadata for Objaverse 32-char ID intersection")
    obj_intersection, obj_rows, obj_col_stats = scan_intersections(
        df=df,
        target_ids=obj_ids,
        regex_pattern=OBJAVERSE_PATTERN,
        label="objaverse",
    )

    # 保存 id
    objxl_ids_out = out_dir / "objaversexl_intersection_ids.txt"
    obj_ids_out = out_dir / "objaverse_intersection_ids.txt"
    save_ids(objxl_intersection, str(objxl_ids_out))
    save_ids(obj_intersection, str(obj_ids_out))

    # 保存匹配到的 metadata 行
    objxl_rows_out = out_dir / "objaversexl_intersection_rows.csv"
    obj_rows_out = out_dir / "objaverse_intersection_rows.csv"
    objxl_rows.to_csv(objxl_rows_out, index=False)
    obj_rows.to_csv(obj_rows_out, index=False)

    print("\n========== Summary ==========")
    print(f"Objaverse-XL intersection size: {len(objxl_intersection)}")
    print(f"Objaverse     intersection size: {len(obj_intersection)}")

    print("\nObjaverse-XL matched columns:")
    if len(objxl_col_stats) == 0:
        print("  None")
    else:
        for k, v in sorted(objxl_col_stats.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    print("\nObjaverse matched columns:")
    if len(obj_col_stats) == 0:
        print("  None")
    else:
        for k, v in sorted(obj_col_stats.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    print("\nSaved files:")
    print(f"  {objxl_ids_out}")
    print(f"  {obj_ids_out}")
    print(f"  {objxl_rows_out}")
    print(f"  {obj_rows_out}")


if __name__ == "__main__":
    main()

"""
python intersect_metadata_ids.py \
  --metadata trellis500k_objaverse_xl_github_metadata.parquet \
  --objaverse_xl_list objaverseXL_curated_uuid_list.txt \
  --objaverse_list objaverse_curated.txt \
  --out_dir intersection_results
"""