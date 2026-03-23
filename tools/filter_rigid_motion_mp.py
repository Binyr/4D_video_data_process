#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--glb_root",
        type=str,
        default="data/objverse_minghao_4d/glbs",
        help="Root directory containing shard folders and .glb files",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/objverse_minghao_4d/motion_info",
        help="Root directory to save motion analysis outputs",
    )
    parser.add_argument(
        "--blender_path",
        type=str,
        default="/efs/yanruibin/projects/blender-4.2.1-linux-x64/blender",
    )
    parser.add_argument(
        "--tool_script",
        type=str,
        default="tools/filter_rigid_motion.py",
        help="Blender Python script to run for each GLB",
    )

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--tol_rel", type=float, default=1e-3)
    parser.add_argument("--max_verts", type=int, default=50000)
    parser.add_argument("--timeout_sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument(
        "--log_root",
        type=str,
        default="data/objverse_minghao_4d/motion_logs",
        help="Directory for failure logs",
    )

    return parser.parse_args()


def is_valid_result(json_path: Path) -> bool:
    """
    Consider a sample done only when:
    1) file exists
    2) file is valid json
    3) contains some expected fields
    """
    if not json_path.is_file():
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    if "results" not in data:
        return False

    return True


def build_output_json_path(glb_path: Path, glb_root: Path, out_root: Path) -> Path:
    """
    Example:
      glb_path = data/.../glbs/000-033/xxxx.glb
      out_root = data/.../motion_info
    ->
      data/.../motion_info/000-033/xxxx/umeyama_similarity.json
    """
    rel = glb_path.relative_to(glb_root)
    shard_dir = rel.parent
    stem = glb_path.stem
    return out_root / shard_dir / stem / "umeyama_similarity.json"


def build_failure_log_path(glb_path: Path, glb_root: Path, log_root: Path) -> Path:
    rel = glb_path.relative_to(glb_root)
    shard_dir = rel.parent
    stem = glb_path.stem
    return log_root / shard_dir / f"{stem}.log"


def collect_all_glbs(glb_root: Path):
    return sorted(glb_root.rglob("*.glb"))


def split_list_round_robin(items, num_parts):
    chunks = [[] for _ in range(num_parts)]
    for i, x in enumerate(items):
        chunks[i % num_parts].append(x)
    return chunks


def run_one_task(
    glb_path: Path,
    glb_root: Path,
    out_root: Path,
    log_root: Path,
    blender_path: Path,
    tool_script: Path,
    tol_rel: float,
    max_verts: int,
    timeout_sec: int,
    overwrite: bool,
    dry_run: bool,
):
    out_json = build_output_json_path(glb_path, glb_root, out_root)

    if (not overwrite) and is_valid_result(out_json):
        return "skipped", str(glb_path), str(out_json), None

    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(tool_script),
        "--",
        "--glb_path",
        str(glb_path),
        "--tol_rel",
        str(tol_rel),
        "--max_verts",
        str(max_verts),
        "--json_out",
        str(out_json),
    ]

    if dry_run:
        return "dry_run", " ".join(cmd), str(out_json), None

    log_path = build_failure_log_path(glb_path, glb_root, log_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=None if timeout_sec <= 0 else timeout_sec,
                check=False,
            )

        # success only if return code == 0 and output json is valid
        if result.returncode == 0 and is_valid_result(out_json):
            try:
                log_path.unlink()
            except Exception:
                pass
            return "ok", str(glb_path), str(out_json), None
        else:
            return "failed", str(glb_path), str(out_json), str(log_path)

    except subprocess.TimeoutExpired:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[TIMEOUT] timeout_sec={timeout_sec}\n")
        return "timeout", str(glb_path), str(out_json), str(log_path)

    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[EXCEPTION] {repr(e)}\n")
        return "failed", str(glb_path), str(out_json), str(log_path)


def worker_main(
    worker_id: int,
    tasks,
    glb_root: str,
    out_root: str,
    log_root: str,
    blender_path: str,
    tool_script: str,
    tol_rel: float,
    max_verts: int,
    timeout_sec: int,
    overwrite: bool,
    dry_run: bool,
    result_queue,
):
    glb_root = Path(glb_root).resolve()
    out_root = Path(out_root).resolve()
    log_root = Path(log_root).resolve()
    blender_path = Path(blender_path).resolve()
    tool_script = Path(tool_script).resolve()

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0
    timeout_cnt = 0
    dry_cnt = 0
    failures = []

    bar = tqdm(
        total=len(tasks),
        position=worker_id,
        desc=f"worker-{worker_id}",
        dynamic_ncols=True,
        leave=True,
    )

    for glb_path in tasks:
        status, msg1, msg2, msg3 = run_one_task(
            glb_path=Path(glb_path),
            glb_root=glb_root,
            out_root=out_root,
            log_root=log_root,
            blender_path=blender_path,
            tool_script=tool_script,
            tol_rel=tol_rel,
            max_verts=max_verts,
            timeout_sec=timeout_sec,
            overwrite=overwrite,
            dry_run=dry_run,
        )

        if status == "ok":
            ok_cnt += 1
        elif status == "skipped":
            skip_cnt += 1
        elif status == "timeout":
            timeout_cnt += 1
            fail_cnt += 1
            failures.append((msg1, msg3))
        elif status == "dry_run":
            dry_cnt += 1
        else:
            fail_cnt += 1
            failures.append((msg1, msg3))

        bar.set_postfix_str(
            f"ok={ok_cnt} skip={skip_cnt} fail={fail_cnt} timeout={timeout_cnt}"
        )
        bar.update(1)

    bar.close()

    result_queue.put(
        {
            "worker_id": worker_id,
            "num_tasks": len(tasks),
            "ok": ok_cnt,
            "skipped": skip_cnt,
            "failed": fail_cnt,
            "timeout": timeout_cnt,
            "dry_run": dry_cnt,
            "failures": failures,
        }
    )


def main():
    args = parse_args()

    glb_root = Path(args.glb_root).resolve()
    out_root = Path(args.out_root).resolve()
    log_root = Path(args.log_root).resolve()
    blender_path = Path(args.blender_path).resolve()
    tool_script = Path(args.tool_script).resolve()

    if not glb_root.is_dir():
        raise FileNotFoundError(f"glb_root not found: {glb_root}")
    if not blender_path.is_file():
        raise FileNotFoundError(f"blender not found: {blender_path}")
    if not tool_script.is_file():
        raise FileNotFoundError(f"tool_script not found: {tool_script}")

    all_glbs = collect_all_glbs(glb_root)

    if len(all_glbs) == 0:
        print(f"No .glb found under: {glb_root}")
        return

    print("=" * 100)
    print(f"glb_root     : {glb_root}")
    print(f"out_root     : {out_root}")
    print(f"log_root     : {log_root}")
    print(f"blender_path : {blender_path}")
    print(f"tool_script  : {tool_script}")
    print(f"num_workers  : {args.num_workers}")
    print(f"tol_rel      : {args.tol_rel}")
    print(f"max_verts    : {args.max_verts}")
    print(f"timeout_sec  : {args.timeout_sec}")
    print(f"overwrite    : {args.overwrite}")
    print(f"dry_run      : {args.dry_run}")
    print(f"total_glbs   : {len(all_glbs)}")
    print("=" * 100)

    if args.overwrite:
        todo_glbs = all_glbs
        num_already_done = 0
    else:
        todo_glbs = []
        num_already_done = 0
        for glb_path in all_glbs:
            out_json = build_output_json_path(glb_path, glb_root, out_root)
            if is_valid_result(out_json):
                num_already_done += 1
            else:
                todo_glbs.append(glb_path)

    print(f"already_done : {num_already_done}")
    print(f"todo         : {len(todo_glbs)}")

    if len(todo_glbs) == 0:
        print("Nothing to do.")
        return

    num_workers = min(args.num_workers, len(todo_glbs))
    chunks = split_list_round_robin(todo_glbs, num_workers)

    for i, chunk in enumerate(chunks):
        print(f"worker-{i} assigned {len(chunk)} files")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []

    for worker_id, tasks in enumerate(chunks):
        p = ctx.Process(
            target=worker_main,
            args=(
                worker_id,
                [str(x) for x in tasks],
                str(glb_root),
                str(out_root),
                str(log_root),
                str(blender_path),
                str(tool_script),
                args.tol_rel,
                args.max_verts,
                args.timeout_sec,
                args.overwrite,
                args.dry_run,
                result_queue,
            ),
        )
        p.start()
        procs.append(p)

    results = []
    for _ in range(len(procs)):
        results.append(result_queue.get())

    for p in procs:
        p.join()

    results = sorted(results, key=lambda x: x["worker_id"])

    total_ok = sum(r["ok"] for r in results)
    total_skip = sum(r["skipped"] for r in results)
    total_fail = sum(r["failed"] for r in results)
    total_timeout = sum(r["timeout"] for r in results)
    total_dry = sum(r["dry_run"] for r in results)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"all_glbs            : {len(all_glbs)}")
    print(f"already_done_before : {num_already_done}")
    print(f"launched_tasks      : {len(todo_glbs)}")
    print(f"ok                  : {total_ok}")
    print(f"skipped_in_worker   : {total_skip}")
    print(f"failed              : {total_fail}")
    print(f"timeout             : {total_timeout}")
    print(f"dry_run             : {total_dry}")

    all_failures = []
    for r in results:
        all_failures.extend(r["failures"])

    if len(all_failures) > 0:
        print("\nFailed samples:")
        for glb_path, log_path in all_failures[:50]:
            print(f"  GLB: {glb_path}")
            print(f"  LOG: {log_path}")
        if len(all_failures) > 50:
            print(f"  ... and {len(all_failures) - 50} more")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""
python tools/filter_rigid_motion_mp.py \
    --glb_root data/objverse_minghao_4d/glbs \
    --out_root data/objverse_minghao_4d/motion_info \
    --blender_path /efs/yanruibin/projects/blender-4.2.1-linux-x64/blender \
    --tool_script tools/filter_rigid_motion.py \
    --num_workers 8 \
    --tol_rel 1e-3 \
    --max_verts 50000
"""