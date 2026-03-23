# tools/count_glb_mesh_hist.py
import os
import json
import struct
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


JSON_CHUNK_TYPE = 0x4E4F534A  # b'JSON' little-endian


def load_glb_json(glb_path):
    """
    读取 .glb 文件中的 JSON chunk，并返回解析后的 dict
    """
    with open(glb_path, "rb") as f:
        header = f.read(12)
        if len(header) != 12:
            raise ValueError(f"Invalid GLB header: {glb_path}")

        magic, version, length = struct.unpack("<4sII", header)
        if magic != b"glTF":
            raise ValueError(f"Not a GLB file: {glb_path}")
        if version != 2:
            raise ValueError(f"Unsupported GLB version {version}: {glb_path}")

        json_chunk = None

        while f.tell() < length:
            chunk_header = f.read(8)
            if len(chunk_header) != 8:
                break

            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            chunk_data = f.read(chunk_length)

            if chunk_type == JSON_CHUNK_TYPE:
                json_chunk = chunk_data
                break

        if json_chunk is None:
            raise ValueError(f"No JSON chunk found in: {glb_path}")

        # JSON chunk 末尾可能有 padding
        json_text = json_chunk.decode("utf-8").rstrip("\x00 ").strip()
        return json.loads(json_text)


def count_glb_mesh_info(glb_path):
    """
    统计一个 glb 的 mesh 相关信息
    """
    gltf = load_glb_json(glb_path)

    meshes = gltf.get("meshes", []) or []
    nodes = gltf.get("nodes", []) or []

    num_meshes = len(meshes)
    num_primitives = sum(len(mesh.get("primitives", []) or []) for mesh in meshes)
    num_mesh_nodes = sum(1 for node in nodes if "mesh" in node)

    return {
        "num_meshes": num_meshes,
        "num_primitives": num_primitives,
        "num_mesh_nodes": num_mesh_nodes,
    }


def scan_all_glbs(root_dir):
    """
    遍历 root_dir 下所有 .glb，并返回结果 dict
    """
    root_dir = Path(root_dir)
    glb_paths = sorted(root_dir.rglob("*.glb"))

    results = {}
    errors = {}

    for i, glb_path in enumerate(glb_paths):
        rel_path = str(glb_path.relative_to(root_dir))
        try:
            info = count_glb_mesh_info(glb_path)
            results[rel_path] = info
        except Exception as e:
            errors[rel_path] = str(e)

        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(glb_paths)}] processed")

    return results, errors


def save_json(results, errors, out_json, root_dir):
    payload = {
        "root_dir": str(root_dir),
        "num_files": len(results),
        "num_errors": len(errors),
        "files": results,
        "errors": errors,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved json to: {out_json}")


def plot_hist_from_json(json_path, out_png, key="num_meshes"):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    files = payload["files"]
    values = [v[key] for v in files.values()]

    if len(values) == 0:
        raise ValueError("No valid files found in json.")

    counter = Counter(values)
    xs = sorted(counter.keys())
    ys = [counter[x] for x in xs]

    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys, width=0.8)
    plt.xlabel(key)
    plt.ylabel("Number of GLBs")
    plt.title(f"Histogram of {key}")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved histogram to: {out_png}")
    print("Histogram data:")
    for x, y in zip(xs, ys):
        print(f"  {key}={x}: {y} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/objverse_minghao_4d/glbs",
        help="Root directory containing .glb files",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="vis/objverse_glb_mesh_counts.json",
        help="Output json path",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="vis/objverse_glb_mesh_hist_num_meshes.png",
        help="Output histogram png path",
    )
    parser.add_argument(
        "--hist_key",
        type=str,
        default="num_meshes",
        choices=["num_meshes", "num_primitives", "num_mesh_nodes"],
        help="Which statistic to plot on x-axis",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)

    results, errors = scan_all_glbs(args.root_dir)
    save_json(results, errors, args.out_json, args.root_dir)
    plot_hist_from_json(args.out_json, args.out_png, key=args.hist_key)

    print(f"Valid files: {len(results)}")
    print(f"Error files: {len(errors)}")


if __name__ == "__main__":
    main()