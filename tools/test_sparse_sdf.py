from pathlib import Path
f_path = Path(__file__).absolute()
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))

import numpy as np
import torch
import trimesh

from diso import DiffMC, DiffDMC

from direct3d_s2.utils import normalize_mesh, mesh2index
from direct3d_s2.modules import sparse as sp
from tools.utils.mesh2watertight import cubvh_mesh2watertightsdf, n_to_xyz

weight_type = torch.float16
device = "cuda:0"
# load vae
from direct3d_s2.pipeline import Direct3DS2Pipeline
pipeline = Direct3DS2Pipeline.from_pretrained(
  'wushuang98/Direct3D-S2', 
  subfolder="direct3d-s2-v-1-1"
)
pipeline.to(device)
vae = pipeline.sparse_vae_1024

# load mesh and sdf
mesh = trimesh.load("output.obj", force="mesh")
mesh = normalize_mesh(mesh)                  # 大致归一化到 [-0.95, 0.95]
mesh.export("mesh_norm.ply")
mesh_v = mesh.vertices
mesh_f = mesh.faces
mesh_v = torch.from_numpy(mesh_v).to(dtype=torch.float32,device='cuda')
mesh_f = torch.from_numpy(mesh_f).to(dtype=torch.int32,device='cuda')

resolution = 1024

mesh_wt, sparse_sdf, sparse_index, mesh_scale, sdf_sign = cubvh_mesh2watertightsdf(mesh_v, mesh_f, resolution)
mesh_wt.export("mesh_wt.ply")
# import pdb;pdb.set_trace()
sparse_sdf_th = torch.from_numpy(sparse_sdf).to(weight_type).to(device)
sparse_index_th = torch.from_numpy(sparse_index).to(torch.int32).to(device)

voxel_resolution = resolution
resolution_bits = int(np.log2(voxel_resolution))

xyz = n_to_xyz(sparse_index_th, resolution_bits).to(torch.int32)
xyz_float = xyz.to(torch.float32) / (resolution - 1) * 2 - 1
print(xyz_float.max(), xyz_float.min())
# mesh_intermedia = sp.SparseTensor(sparse_sdf_th, xyz)
# mesh_intermedia = vae.sparse2mesh(mesh_intermedia, voxel_resolution=voxel_resolution)
# mesh_intermedia.export("mesh_inter.ply")
print(f"mesh intermedia exported")
# 3) batch_idx：单个 mesh 就全 0
batch_idx = torch.zeros((xyz.shape[0],), dtype=torch.int32, device=device)

# 4) factor（通常 = voxel_resolution // vae.resolution）
#    比如 sparse_vae_512 的 vae.resolution=64，对应 voxel_resolution=512 => factor=8
factor = voxel_resolution // vae.resolution
assert voxel_resolution % vae.resolution == 0, "voxel_resolution 应能整除 vae.resolution"

batch = {
    "sparse_sdf": sparse_sdf_th,          # (N,1) 或 (N,)
    "sparse_index": xyz,        # (N,3)
    "batch_idx": batch_idx,     # (N,)
    "factor": factor,           # 让 encoder 知道坐标尺度（如果你的 encoder 用得到）
}
# -----------------------------
# 5) forward
# -----------------------------
vae.eval()
with torch.no_grad():
    out = vae(batch)
    reconst_x = out["reconst_x"]   # sp.SparseTensor
    posterior = out["posterior"]
    recon_mesh = vae.sparse2mesh(reconst_x, voxel_resolution=voxel_resolution)[0]
    recon_mesh.export("mesh_recon.ply")

# reconst_x.feats: (M,1) 预测的 sparse sdf
# reconst_x.coords: (M,4) 预测的坐标 [batch, x, y, z]
recon_sdf  = reconst_x.feats.detach().cpu().numpy()
recon_xyzb = reconst_x.coords.detach().cpu().numpy()

print(recon_sdf.shape, recon_xyzb.shape)

import pdb
pdb.set_trace()
# -----------------------------
# 6) extract mesh
# -----------------------------

# def extra_mesh(sdf):
#     diffmc = DiffMC(dtype=torch.float32).cuda()
#     vertices, triangles = diffmc(sdf, isovalue=0, normalize=False)
#     del diffmc  # DiffMC对象使用完立即删除
#     vertices = vertices / (sdf.shape[-1] - 1.0) * 2 - 1
#     vertices = vertices.to(torch.float32)
#     triangles = triangles.to(torch.int32)
#     mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=triangles.cpu().numpy(), process=False)
#     return mesh

# recon_mesh = extra_mesh(recon_sdf)
# recon_mesh.export("mesh_recon.ply")
