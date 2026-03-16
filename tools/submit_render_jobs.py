import argparse
import concurrent.futures as cf
import hashlib
import math
import os
import queue
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


FRAME_PROGRESS_RE = re.compile(r"\[FRAME_PROGRESS\]\s+(\d+)\s*/\s*(\d+)")


@dataclass
class Job:
    glb_path: Path
    rel_dir: str
    stem: str
    traj_id: int
    traj_seed: int
    output_file: Path
    log_file: Path
    cmd: List[str]


def stable_seed_from_path(path: Path, base_seed: int = 0) -> int:
    key = str(path).encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    value = int(digest[:8], 16)
    seed = (value + int(base_seed)) % (2**31 - 1)
    return seed


def parse_traj_ids(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def list_glbs(glb_root: Path) -> List[Path]:
    return sorted(glb_root.rglob("*.glb"))


def select_block(items: List[Path], num_blocks: int, block_id: int) -> List[Path]:
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
    if not (0 <= block_id < num_blocks):
        raise ValueError(f"block_id must be in [0, {num_blocks - 1}], got {block_id}")

    n = len(items)
    chunk_size = math.ceil(n / num_blocks)
    start = block_id * chunk_size
    end = min(start + chunk_size, n)
    return items[start:end]


def make_output_file(
    output_root: Path,
    rel_dir: str,
    stem: str,
    traj_id: int,
    output_dir_suffix: str,
) -> Path:
    out_dir = output_root / f"{rel_dir}{output_dir_suffix}" / f"{stem}_traj_{traj_id}"
    return out_dir / "result.json"


def build_jobs(args) -> Tuple[List[Job], List[Path]]:
    glb_root = Path(args.glb_root).resolve()
    output_root = Path(args.output_root).resolve()
    blender_bin = str(Path(args.blender_bin).resolve())
    render_script = str(Path(args.render_script).resolve())

    all_glbs = list_glbs(glb_root)
    if len(all_glbs) == 0:
        raise RuntimeError(f"No .glb files found under: {glb_root}")

    selected_glbs = select_block(all_glbs, args.num_blocks, args.block_id)
    traj_ids = parse_traj_ids(args.traj_ids)
    if len(traj_ids) == 0:
        raise RuntimeError("traj_ids is empty.")

    jobs: List[Job] = []

    for glb_path in selected_glbs:
        rel_parent = glb_path.parent.relative_to(glb_root)
        rel_dir = str(rel_parent)
        stem = glb_path.stem
        traj_seed = stable_seed_from_path(glb_path.relative_to(glb_root), args.base_seed)

        for traj_id in traj_ids:
            output_file = make_output_file(
                output_root=output_root,
                rel_dir=rel_dir,
                stem=stem,
                traj_id=traj_id,
                output_dir_suffix=args.output_dir_suffix,
            )
            job_dir = output_file.parent
            log_file = job_dir / "run.log"

            cmd = [
                blender_bin,
                "--background",
                "--python",
                render_script,
                "--",
                "--object_path", str(glb_path),
                "--output_file", str(output_file),
                "--resolution", str(args.resolution),
                "--render_engine", args.render_engine,
                "--traj_id", str(traj_id),
                "--traj_seed", str(traj_seed),
            ]

            if args.transparent_bg:
                cmd.append("--transparent_bg")

            if args.render_engine == "CYCLES":
                cmd.extend(["--cycles_backend", args.cycles_backend])

            jobs.append(
                Job(
                    glb_path=glb_path,
                    rel_dir=rel_dir,
                    stem=stem,
                    traj_id=traj_id,
                    traj_seed=traj_seed,
                    output_file=output_file,
                    log_file=log_file,
                    cmd=cmd,
                )
            )

    return jobs, selected_glbs


def is_job_done(job: Job) -> bool:
    return job.output_file.exists() and job.output_file.stat().st_size > 0


def stream_subprocess_output(proc, log_f, line_queue):
    """
    子线程：实时读取 stdout，写入日志，同时把文本送给主 worker 更新 tqdm。
    """
    try:
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            line_queue.put(line)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


def run_one_job_with_progress(
    job: Job,
    force: bool,
    dry_run: bool,
    position: int,
    leave_bar: bool = False,
):
    """
    每个任务一条 tqdm 进度条。
    """
    short_name = f"{job.rel_dir}/{job.stem[:10]}.. traj={job.traj_id}"

    if (not force) and is_job_done(job):
        bar = tqdm(
            total=1,
            position=position,
            desc=short_name,
            leave=leave_bar,
            dynamic_ncols=True,
        )
        bar.update(1)
        bar.set_postfix_str("skip")
        bar.close()
        return "skip", 0, f"[skip] {job.output_file}"

    job.output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = " ".join(shlex.quote(x) for x in job.cmd)

    if dry_run:
        bar = tqdm(
            total=1,
            position=position,
            desc=short_name,
            leave=leave_bar,
            dynamic_ncols=True,
        )
        bar.update(1)
        bar.set_postfix_str("dry-run")
        bar.close()
        return "done", 0, f"[dry-run] {cmd_str}"

    with open(job.log_file, "w", encoding="utf-8") as f:
        f.write(f"GLB: {job.glb_path}\n")
        f.write(f"traj_id: {job.traj_id}\n")
        f.write(f"traj_seed: {job.traj_seed}\n")
        f.write(f"output_file: {job.output_file}\n")
        f.write(f"command:\n{cmd_str}\n\n")
        f.flush()

        proc = subprocess.Popen(
            job.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        q = queue.Queue()
        t = threading.Thread(
            target=stream_subprocess_output,
            args=(proc, f, q),
            daemon=True,
        )
        t.start()

        total_frames = None
        current_frames = 0
        last_status = "starting"

        bar = tqdm(
            total=None,
            position=position,
            desc=short_name,
            leave=leave_bar,
            dynamic_ncols=True,
        )
        bar.set_postfix_str(last_status)

        while True:
            try:
                line = q.get(timeout=0.2)
                m = FRAME_PROGRESS_RE.search(line)
                if m:
                    cur = int(m.group(1))
                    tot = int(m.group(2))

                    if total_frames is None:
                        total_frames = tot
                        bar.reset(total=tot)

                    if cur > current_frames:
                        bar.update(cur - current_frames)
                        current_frames = cur
                        last_status = f"{cur}/{tot}"
                        bar.set_postfix_str(last_status)
                else:
                    # 可以按需解析更多状态
                    pass
            except queue.Empty:
                if proc.poll() is not None:
                    break

        t.join(timeout=1.0)
        returncode = proc.wait()

        if total_frames is not None and current_frames < total_frames:
            bar.update(total_frames - current_frames)

        if returncode == 0 and is_job_done(job):
            bar.set_postfix_str("done")
            bar.close()
            return "done", returncode, f"[done] {job.output_file}"
        else:
            bar.set_postfix_str(f"fail:{returncode}")
            bar.close()
            return "fail", returncode, f"[fail:{returncode}] {job.output_file} (log: {job.log_file})"


def main():
    parser = argparse.ArgumentParser(
        description="Submit multi-process Blender rendering jobs for all GLBs."
    )
    parser.add_argument("--glb_root", type=str, default="data/objverse_minghao_4d/glbs")
    parser.add_argument("--output_root", type=str, default="vis/rendering")
    parser.add_argument("--output_dir_suffix", type=str, default="_v4")
    parser.add_argument(
        "--blender_bin",
        type=str,
        default="/efs/yanruibin/projects/blender-4.2.1-linux-x64/blender",
    )
    parser.add_argument(
        "--render_script",
        type=str,
        default="tools/extract_mesh_camera_sparse_voxel.py",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--render_engine",
        type=str,
        default="CYCLES",
        choices=["BLENDER_EEVEE", "CYCLES"],
    )
    parser.add_argument(
        "--cycles_backend",
        type=str,
        default="CUDA",
        choices=["CUDA", "OPTIX"],
    )
    parser.add_argument("--transparent_bg", action="store_true")
    parser.add_argument(
        "--traj_ids",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Your current render script supports 0..6.",
    )
    parser.add_argument("--base_seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--block_id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--leave_bars",
        action="store_true",
        help="Keep finished per-process bars on screen.",
    )
    args = parser.parse_args()

    all_glbs = list_glbs(Path(args.glb_root))
    jobs, selected_glbs = build_jobs(args)

    print(f"Total GLBs under root       : {len(all_glbs)}")
    print(f"Selected GLBs in this block : {len(selected_glbs)}")
    print(f"traj_ids                    : {parse_traj_ids(args.traj_ids)}")
    print(f"Total jobs                  : {len(jobs)}")
    print(f"num_workers                 : {args.num_workers}")
    print(f"block_id / num_blocks       : {args.block_id} / {args.num_blocks}")
    print()

    num_skip = 0
    num_done = 0
    num_fail = 0
    failed_msgs = []

    # 总进度条放在第 0 行
    global_bar = tqdm(
        total=len(jobs),
        desc="All jobs",
        position=0,
        dynamic_ncols=True,
        leave=True,
    )

    # 每个 worker 一条 bar，放在 1..num_workers 行
    slot_queue = queue.Queue()
    for pos in range(1, args.num_workers + 1):
        slot_queue.put(pos)

    def wrapped(job: Job):
        position = slot_queue.get()
        try:
            return run_one_job_with_progress(
                job=job,
                force=args.force,
                dry_run=args.dry_run,
                position=position,
                leave_bar=args.leave_bars,
            )
        finally:
            slot_queue.put(position)

    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(wrapped, job): job for job in jobs}

        for fut in cf.as_completed(futures):
            job = futures[fut]
            try:
                status, code, msg = fut.result()
            except Exception as e:
                status, code, msg = "fail", -1, f"[fail:exception] {job.output_file}: {e}"

            global_bar.update(1)

            if status == "skip":
                num_skip += 1
            elif status == "done":
                num_done += 1
            else:
                num_fail += 1
                failed_msgs.append(msg)

    global_bar.close()

    print("\n========== Summary ==========")
    print(f"done : {num_done}")
    print(f"skip : {num_skip}")
    print(f"fail : {num_fail}")

    if num_fail > 0:
        print("\nFailed jobs:")
        for m in failed_msgs[:50]:
            print("  ", m)
        if len(failed_msgs) > 50:
            print(f"  ... and {len(failed_msgs) - 50} more")


if __name__ == "__main__":
    main()

"""
python tools/submit_render_jobs.py \
  --glb_root data/objverse_minghao_4d/glbs \
  --output_root vis/rendering \
  --output_dir_suffix _v1 \
  --blender_bin /efs/yanruibin/projects/blender-4.2.1-linux-x64/blender \
  --render_script tools/extract_mesh_camera_sparse_voxel.py \
  --resolution 1024 \
  --render_engine CYCLES \
  --cycles_backend CUDA \
  --transparent_bg \
  --traj_ids 0,1,2,3,4,5,6 \
  --base_seed 123 \
  --num_workers 64 \
  --num_blocks 2 \
  --block_id 1 \
"""