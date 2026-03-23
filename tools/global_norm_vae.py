from pathlib import Path
f_path = Path(__file__).absolute()
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))

import os
import re
import argparse
import numpy as np
import torch
import trimesh

from tools.utils.mesh2watertight import cubvh_mesh2watertightsdf, n_to_xyz

FRAME_RE = re.compile(r"frame_(\d+)$")

def list_frame_meshes(root_dir: str):
    items = []
    for name in os.listdir(root_dir):
        m = FRAME_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        mesh_path = os.path.join(root_dir, name, "mesh.ply")
        if os.path.isfile(mesh_path):
            items.append((idx, name, mesh_path))
    items.sort(key=lambda x: x[0])
    return items

def compute_global_norm(mesh_paths, target=0.95):
    """用所有帧的顶点一起算 bbox，得到统一的 (center, scale)。
       归一化：v' = (v - center) * scale，保证整体落在 [-target, target] 内。
    """
    bb_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bb_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    ok = 0
    for p in mesh_paths:
        mesh = trimesh.load(p, force="mesh", process=False)
        if mesh.vertices is None or len(mesh.vertices) == 0:
            continue
        v = np.asarray(mesh.vertices, dtype=np.float64)
        bb_min = np.minimum(bb_min, v.min(axis=0))
        bb_max = np.maximum(bb_max, v.max(axis=0))
        ok += 1
    if ok == 0:
        raise RuntimeError("All meshes are empty / failed to load.")

    center = (bb_min + bb_max) * 0.5
    extent = (bb_max - bb_min)
    max_extent = float(extent.max())
    if max_extent < 1e-12:
        scale = 1.0
    else:
        scale = (2.0 * target) / max_extent
    return center.astype(np.float32), float(scale), bb_min.astype(np.float32), bb_max.astype(np.float32)

def apply_norm_trimesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float):
    v = np.asarray(mesh.vertices, dtype=np.float32)
    v = (v - center[None, :]) * scale
    mesh.vertices = v
    return mesh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="render_cond 目录（包含 frame_xxxx 子目录）")
    ap.add_argument("--out_dir", required=True, help="输出目录（会创建 frame_xxxx 子目录）")
    ap.add_argument("--resolution", type=int, default=1024, help="watertight+sdf 的 voxel resolution（默认 1024）")
    ap.add_argument("--target", type=float, default=0.95, help="全局归一化到 [-target, target]")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--weight_type", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--save_norm", action="store_true", help="保存每帧归一化后的 mesh_norm.ply")
    ap.add_argument("--save_wt", action="store_true", help="保存每帧 watertight mesh_wt.ply（可能比较大）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frames = list_frame_meshes(args.root_dir)
    if not frames:
        raise FileNotFoundError(f"No frame_xxxx/mesh.ply found under: {args.root_dir}")
    indices, names, mesh_paths = zip(*frames)
    print(f"Found {len(mesh_paths)} meshes: frame {indices[0]} .. {indices[-1]}")

    # -------- 1) 全局归一化参数 --------
    center, scale, bb_min, bb_max = compute_global_norm(mesh_paths, target=args.target)
    print(f"[global norm] center={center.tolist()}  scale={scale:.6g}")
    np.savez(os.path.join(args.out_dir, "global_norm.npz"),
             center=center, scale=scale, bb_min=bb_min, bb_max=bb_max, target=args.target)

    # -------- 2) load pipeline / vae --------
    from direct3d_s2.pipeline import Direct3DS2Pipeline
    device = torch.device(args.device)
    weight_type = torch.float16 if args.weight_type == "float16" else torch.float32

    pipeline = Direct3DS2Pipeline.from_pretrained(
        "wushuang98/Direct3D-S2",
        subfolder="direct3d-s2-v-1-1"
    )
    pipeline.to(device)
    vae = pipeline.sparse_vae_1024
    vae.eval()

    # vae.resolution 用来算 factor；不同版本属性名可能略有不同，这里做个兜底
    vae_res = getattr(vae, "resolution", None)
    if vae_res is None:
        vae_res = getattr(vae, "grid_resolution", None)
    if vae_res is None:
        vae_res = 64  # 常见默认；你也可以 print(dir(vae)) 后改成正确的
        print(f"[warn] vae.resolution not found, fallback vae_res={vae_res}")

    voxel_resolution = int(args.resolution)
    assert voxel_resolution % int(vae_res) == 0, f"resolution={voxel_resolution} must be divisible by vae_res={vae_res}"
    factor = voxel_resolution // int(vae_res)

    # n_to_xyz 要求 resolution 是 2 的幂（1024 ok）
    resolution_bits = int(np.log2(voxel_resolution))
    assert (1 << resolution_bits) == voxel_resolution, "resolution 必须是 2 的幂（如 256/512/1024）"

    # -------- 3) 逐帧处理 --------
    for idx, name, mesh_path in frames:
        if idx < 28:
            continue
        out_frame_dir = os.path.join(args.out_dir, name)
        os.makedirs(out_frame_dir, exist_ok=True)

        print(f"[{name}] loading {mesh_path}")
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if mesh.vertices is None or len(mesh.vertices) == 0:
            print(f"[warn] empty mesh: {mesh_path}")
            continue

        # 全局归一化（所有帧同一个 center/scale）
        mesh = apply_norm_trimesh(mesh, center=center, scale=scale)
        if args.save_norm:
            mesh.export(os.path.join(out_frame_dir, "mesh_norm.ply"))

        # torch tensors
        mesh_v = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device=device)
        mesh_f = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int32)).to(device=device)

        # watertight + sparse sdf
        mesh_wt, sparse_sdf, sparse_index, mesh_scale, sdf_sign = cubvh_mesh2watertightsdf(mesh_v, mesh_f, voxel_resolution)
        if args.save_wt:
            mesh_wt.export(os.path.join(out_frame_dir, "mesh_wt.ply"))

        sparse_sdf_th = torch.from_numpy(sparse_sdf).to(device=device, dtype=weight_type)
        if sparse_sdf_th.ndim == 1:
            sparse_sdf_th = sparse_sdf_th[:, None]  # (N,1)

        sparse_index_th = torch.from_numpy(sparse_index).to(device=device, dtype=torch.int32)
        xyz = n_to_xyz(sparse_index_th, resolution_bits).to(torch.int32)  # (N,3)

        batch_idx = torch.zeros((xyz.shape[0],), dtype=torch.int32, device=device)

        batch = {
            "sparse_sdf": sparse_sdf_th,   # (N,1)
            "sparse_index": xyz,           # (N,3)  注意：这里传 xyz（而不是原 sparse_index）
            "batch_idx": batch_idx,        # (N,)
            "factor": factor,
        }

        with torch.no_grad():
            out = vae(batch)
            reconst_x = out["reconst_x"]  # sp.SparseTensor
            recon_mesh = vae.sparse2mesh(reconst_x, voxel_resolution=voxel_resolution)[0]
            recon_mesh.export(os.path.join(out_frame_dir, "mesh_recon.ply"))

        # 可选：把重建的 sparse 也存一下（debug 用）
        # np.save(os.path.join(out_frame_dir, "recon_sdf.npy"), reconst_x.feats.detach().cpu().numpy())
        # np.save(os.path.join(out_frame_dir, "recon_xyzb.npy"), reconst_x.coords.detach().cpu().numpy())

        # 防止显存碎片（按需）
        torch.cuda.empty_cache()

    print(f"Done. Outputs in: {args.out_dir}")

if __name__ == "__main__":
    main()