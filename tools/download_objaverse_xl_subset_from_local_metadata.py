#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def read_id_list(path: str) -> set[str]:
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
        raise ValueError(f"Unsupported metadata format: {suffix}")


def to_string_series(s: pd.Series) -> pd.Series:
    def _conv(x):
        if pd.isna(x):
            return ""
        if isinstance(x, (dict, list)):
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)
        return str(x)
    return s.map(_conv)


def direct_match_ids(series: pd.Series, target_ids: set[str]) -> pd.Series:
    s = to_string_series(series).str.strip().str.lower()
    return s.where(s.isin(target_ids), other=pd.NA)


def extracted_match_ids(series: pd.Series, target_ids: set[str]) -> pd.Series:
    s = to_string_series(series)
    extracted = s.str.extract(UUID_RE, expand=False)
    if extracted is None:
        return pd.Series([pd.NA] * len(series), index=series.index)
    extracted = extracted.str.lower()
    return extracted.where(extracted.isin(target_ids), other=pd.NA)


def score_columns(df: pd.DataFrame, target_ids: set[str]) -> List[Dict]:
    results = []

    for col in df.columns:
        try:
            dmatch = direct_match_ids(df[col], target_ids)
            ematch = extracted_match_ids(df[col], target_ids)

            dcount = int(dmatch.notna().sum())
            ecount = int(ematch.notna().sum())

            union_ids = set(dmatch.dropna().tolist()) | set(ematch.dropna().tolist())
            total_unique = len(union_ids)

            sample_ids = sorted(list(union_ids))[:5]

            results.append(
                {
                    "column": col,
                    "direct_matches": dcount,
                    "extracted_matches": ecount,
                    "unique_matched_ids": total_unique,
                    "sample_ids": sample_ids,
                }
            )
        except Exception as e:
            results.append(
                {
                    "column": col,
                    "direct_matches": -1,
                    "extracted_matches": -1,
                    "unique_matched_ids": -1,
                    "sample_ids": [f"ERROR: {e}"],
                }
            )

    results = sorted(
        results,
        key=lambda x: (
            x["unique_matched_ids"],
            x["direct_matches"],
            x["extracted_matches"],
        ),
        reverse=True,
    )
    return results


def choose_best_column(scores: List[Dict]) -> Optional[str]:
    if not scores:
        return None
    best = scores[0]
    if best["unique_matched_ids"] <= 0:
        return None
    return best["column"]


def build_subset(df: pd.DataFrame, id_col: str, target_ids: set[str]) -> pd.DataFrame:
    dmatch = direct_match_ids(df[id_col], target_ids)
    ematch = extracted_match_ids(df[id_col], target_ids)

    matched_id = dmatch.fillna(ematch)
    mask = matched_id.notna()

    subset = df.loc[mask].copy()
    subset["matched_uuid"] = matched_id.loc[mask].values
    subset["matched_from_column"] = id_col
    return subset


def maybe_download_subset(subset: pd.DataFrame, download_dir: str, processes: Optional[int]) -> None:
    # lazy import so that column-scanning can still work even if objaverse is not installed
    try:
        import objaverse.xl as oxl
    except Exception as e:
        print(f"[WARN] objaverse.xl import failed, skip downloading: {e}")
        return

    cols = set(subset.columns)

    # unified Objaverse-XL downloader
    if {"fileIdentifier", "sha256"}.issubset(cols):
        objects = subset[["fileIdentifier", "sha256"]].copy()

        # keep extra columns if present
        for c in ["source", "license", "fileType", "metadata"]:
            if c in subset.columns:
                objects[c] = subset[c]

        print("[INFO] Using oxl.download_objects with columns: fileIdentifier + sha256")
        downloaded = oxl.download_objects(
            objects=objects,
            download_dir=download_dir,
            processes=processes,
        )
        try:
            print(f"[INFO] download_objects returned {len(downloaded)} entries")
        except Exception:
            print("[INFO] download_objects finished")
        return
    
    print("enter github download")
    raise
    # GitHub-specific downloader
    github_url_col = None
    for cand in ["githubUrl", "github_url", "url"]:
        if cand in cols:
            github_url_col = cand
            break

    if github_url_col is not None and "sha256" in cols:
        try:
            from objaverse.xl.github import download_github_objects
        except Exception as e:
            print(f"[WARN] Cannot import GitHub downloader, skip downloading: {e}")
            return

        objects = subset[[github_url_col, "sha256"]].copy()
        if github_url_col != "githubUrl":
            objects = objects.rename(columns={github_url_col: "githubUrl"})

        print("[INFO] Using objaverse.xl.github.download_github_objects with columns: githubUrl + sha256")
        downloaded = download_github_objects(
            objects=objects,
            download_dir=download_dir,
            processes=processes,
            save_repo_format="files",
        )
        try:
            print(f"[INFO] download_github_objects returned {len(downloaded)} entries")
        except Exception:
            print("[INFO] download_github_objects finished")
        return

    print("[WARN] Subset metadata does not contain a known downloadable key pair.")
    print("[WARN] Need either:")
    print("       - fileIdentifier + sha256")
    print("       - githubUrl + sha256")
    print("[WARN] So I only saved the filtered subset metadata.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_list", type=str, required=True, help="36-char UUID list txt")
    parser.add_argument("--metadata", type=str, required=True, help="local metadata file: csv/tsv/parquet/jsonl/json")
    parser.add_argument("--download_dir", type=str, default=None, help="download directory")
    parser.add_argument("--processes", type=int, default=None, help="number of processes")
    parser.add_argument("--save_subset_csv", type=str, default=None, help="save filtered subset csv")
    parser.add_argument("--save_subset_parquet", type=str, default=None, help="save filtered subset parquet")
    parser.add_argument("--topk_report", type=int, default=10, help="how many top columns to print")
    args = parser.parse_args()

    target_ids = read_id_list(args.id_list)
    print(f"Loaded {len(target_ids)} UUIDs from {args.id_list}")

    df = load_table(args.metadata)
    print(f"Loaded metadata from {args.metadata}, shape={df.shape}")
    print(f"Columns: {list(df.columns)}")

    scores = score_columns(df, target_ids)

    print("\nTop candidate columns:")
    for i, item in enumerate(scores[: args.topk_report], 1):
        print(
            f"[{i}] {item['column']}: "
            f"unique_matched_ids={item['unique_matched_ids']}, "
            f"direct_matches={item['direct_matches']}, "
            f"extracted_matches={item['extracted_matches']}, "
            f"sample_ids={item['sample_ids']}"
        )

    best_col = choose_best_column(scores)
    if best_col is None:
        print("\n[ERROR] No column in metadata matches the UUID list.")
        print("This means your UUID mapping is probably not stored as a plain column/string in this metadata file.")
        return

    print(f"\nUsing best matched column: {best_col}")

    subset = build_subset(df, best_col, target_ids)
    matched = subset["matched_uuid"].nunique()
    print(f"Matched {matched} unique UUIDs, subset rows = {len(subset)}")

    if args.save_subset_csv is not None:
        out_csv = Path(args.save_subset_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(out_csv, index=False)
        print(f"Saved subset CSV to {out_csv}")

    if args.save_subset_parquet is not None:
        out_pq = Path(args.save_subset_parquet)
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        subset.to_parquet(out_pq, index=False)
        print(f"Saved subset Parquet to {out_pq}")

    if args.download_dir is not None:
        Path(args.download_dir).mkdir(parents=True, exist_ok=True)
        maybe_download_subset(
            subset=subset,
            download_dir=args.download_dir,
            processes=args.processes,
        )


if __name__ == "__main__":
    main()

"""
python tools/download_objaverse_xl_subset_from_local_metadata.py \
  --id_list data/diffusion4d/objaverseXL_curated_uuid_list.txt \
  --metadata data/diffusion4d/meta_xl_animation_tot.csv \
  --save_subset_csv data/diffusion4d/objxl_71k_subset.csv

python tools/download_objaverse_xl_subset_from_local_metadata.py \
  --id_list data/diffusion4d/objaverseXL_curated_uuid_list.txt \
  --metadata data/diffusion4d/meta_xl_tot.csv \
  --save_subset_csv data/diffusion4d/objxl_71k_subset.csv

python tools/download_objaverse_xl_subset_from_local_metadata.py \
  --id_list data/diffusion4d/objaverseXL_curated_uuid_list.txt \
  --metadata data/diffusion4d/meta_xl_animation_tot.csv \
  --download_dir data/diffusion4d/objxl_71k \
  --processes 32 \
  --save_subset_csv data/diffusion4d/objxl_71k_subset.csv
"""