#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  code builder: Dora team (https://github.com/Seed3D/Dora)
import cubvh
# from data_process_batch_single import xyz_to_n
from .helper_wt import compute_valid_udf
import torch
import numpy as np
import trimesh
from diso import DiffMC, DiffDMC
# import argparse
from tqdm import tqdm
# import os
# import json
# import point_cloud_utils as pcu
import os
# import zlib
# import glob
# import tqdm
import time
# import mcubes
import trimesh
# import argparse
import numpy as np

import torch
import cubvh

# import kiui
from skimage import measure

"""
Extract watertight mesh from a arbitrary mesh by UDF expansion and floodfill.
"""

# @torch.jit.script
def xyz_to_n(xyz: torch.Tensor, resolution_bits:int) -> torch.Tensor:
    """
    最高性能的XYZ到N序列化
    """
    shift1: int = resolution_bits
    shift2: int = resolution_bits << 1
    return xyz[..., 0] + (xyz[..., 1] << shift1) + (xyz[..., 2] << shift2)

# @torch.jit.script
def n_to_xyz(n: torch.Tensor, resolution_bits:int) -> torch.Tensor:
    """
    最高性能的N到XYZ反序列化
    """
    mask: int = (1 << resolution_bits) - 1
    x: torch.Tensor = n & mask
    y: torch.Tensor = (n >> resolution_bits) & mask
    z: torch.Tensor = n >> (resolution_bits << 1)
    return torch.stack([x, y, z], dim=-1)



def sphere_normalize(vertices):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    radius = np.linalg.norm(vertices - bcenter, axis=-1).max()
    vertices = (vertices - bcenter) / (radius)  # to [-1, 1]
    return vertices,bcenter, radius

def sphere_normalize_torch(vertices):
    """PyTorch version of sphere_normalize function.
    
    Args:
        vertices (torch.Tensor): Tensor of shape (N, 3) containing vertex coordinates.
        
    Returns:
        tuple: Normalized vertices, center, and radius.
    """
    bmin, _ = torch.min(vertices, dim=0)
    bmax, _ = torch.max(vertices, dim=0)
    bcenter = (bmax + bmin) / 2
    radius = torch.norm(vertices - bcenter, dim=-1).max()
    vertices = (vertices - bcenter) / radius  # to [-1, 1]
    return vertices, bcenter, radius

def sphere_unnormalize(vertices, bcenter, radius):
    vertices = vertices * radius + bcenter  # to original scale
    return vertices


def sphere_unnormalize_rescale(vertices, bcenter, radius):
    vertices = vertices * radius  # to original scale
    return vertices
def box_normalize(vertices, bound=0.95):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices

def generate_dense_grid_points(
    bbox_min = np.array((-1.05, -1.05, -1.05)),#array([-1.05, -1.05, -1.05])
    bbox_max= np.array((1.05, 1.05, 1.05)),#array([1.05, 1.05, 1.05])
    resolution = 1024,
    indexing = "ij"
):
    length = bbox_max - bbox_min#array([2.1, 2.1, 2.1])
    num_cells = resolution# 512
    x = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3) # 513*513*513，3
    grid_size = [resolution + 1, resolution + 1, resolution + 1] # 513，513，513

    return xyz, grid_size


# def remesh(grid_xyz, grid_size, mesh_path, remesh_path, resolution, use_pcu):
#     eps = 2 / resolution
#     # eps = 1e-3
#     mesh = trimesh.load(mesh_path, force='mesh')

#     # normalize mesh to [-1,1]
#     vertices = mesh.vertices
#     bbmin = vertices.min(0)
#     bbmax = vertices.max(0)
#     center = (bbmin + bbmax) / 2
#     scale = 2.0 / (bbmax - bbmin).max()
#     vertices = (vertices - center) * scale
#     if use_pcu:
#         grid_sdf, fid, bc = pcu.signed_distance_to_mesh(grid_xyz, vertices.astype(np.float32), mesh.faces)
#         grid_udf = torch.FloatTensor(np.abs(grid_sdf)).cuda().view((grid_size[0], grid_size[1], grid_size[2]))
#     else:
#         f = cubvh.cuBVH(torch.as_tensor(vertices, dtype=torch.float32, device='cuda'), torch.as_tensor(mesh.faces, dtype=torch.float32, device='cuda')) # build with numpy.ndarray/torch.Tensor
#         grid_udf, _,_= f.unsigned_distance(grid_xyz, return_uvw=False)
#         grid_udf = grid_udf.view((grid_size[0], grid_size[1], grid_size[2]))
#     diffdmc = DiffDMC(dtype=torch.float32).cuda()
#     vertices, faces = diffdmc(grid_udf, isovalue=eps, normalize= False)
#     bbox_min = np.array((-1.05, -1.05, -1.05))
#     bbox_max= np.array((1.05, 1.05, 1.05))
#     bbox_size = bbox_max - bbox_min
#     vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]
#     mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
#     # keep the max component of the extracted mesh
#     components = mesh.split(only_watertight=False)
#     bbox = []
#     for c in components:
#         bbmin = c.vertices.min(0)
#         bbmax = c.vertices.max(0)
#         bbox.append((bbmax - bbmin).max())
#     max_component = np.argmax(bbox)
#     mesh = components[max_component]
#     mesh.export(remesh_path)
#     return mesh




def normalize_mesh(mesh, scale=0.95):
    vertices = mesh.vertices
    min_coords, max_coords = vertices.min(axis=0), vertices.max(axis=0)
    dxyz = max_coords - min_coords
    dist = max(dxyz)
    mesh_scale = 2.0 * scale / dist
    mesh_offset = -(min_coords + max_coords) / 2
    vertices = (vertices + mesh_offset) * mesh_scale
    mesh.vertices = vertices
    return mesh


# def cubvh_mesh2watertight(mesh,mesh_path, resolution=1024):
#     # global device
#     device = torch.device('cuda')
#     # breakpoint()
#     # mesh = trimesh.load(mesh_path, process=False, force='mesh')
#     # mesh.vertices,bcenter,radius = sphere_normalize(mesh.vertices)
#     # breakpoint()
#     vertices = torch.from_numpy(mesh.vertices).float().to(device)
#     triangles = torch.from_numpy(mesh.faces).long().to(device)
#     vertices , bcenter, radius = sphere_normalize_torch(vertices)
#     # breakpoint()
#     vertices = vertices*0.95
#     t0 = time.time()
#     # print("开始构建BVH")
#     BVH = cubvh.cuBVH(vertices, triangles)
#     # print(f'BVH build time: {time.time() - t0:.4f}s')
#     # print(f'BVH build time: {time.time() - t0:.4f}s')
#     eps = 2 / resolution
#     # device = torch.device('cuda')
#     x = torch.linspace(-1, 1, resolution, device=device)
#     points = torch.stack(
#         torch.meshgrid(
#             x,
#             x,
#             x,
#             indexing="ij",
#         ), dim=-1,
#     ) # [N, N, N, 3]
#     # naive sdf
#     # sdf, _, _ = BVH.signed_distance(points.view(-1, 3), return_uvw=False, mode='raystab') # some mesh may not be watertight...
#     # sdf = sdf.cpu().numpy()
#     # occ = (sdf < 0)

#     # udf
#     t0 = time.time()
#     # print("开始计算UDF")
#     udf, _, _ = BVH.unsigned_distance(points.view(-1, 3), return_uvw=False)
#     # print(f'UDF time: {time.time() - t0:.4f}s')
#     del BVH
#     del points
#     del vertices
#     del triangles
#     # print(f'UDF time: {time.time() - t0:.4f}s')

#     # floodfill
#     # t0 = time.time()
#     udf = udf.view(resolution,resolution,resolution).contiguous()
#     torch.cuda.empty_cache()
#     occ = udf < eps
#     floodfill_mask = cubvh.floodfill(occ)

#     empty_label = floodfill_mask[0, 0, 0].item()
#     empty_mask = (floodfill_mask == empty_label)
#     # print(f'Floodfill time: {time.time() - t0:.4f}s')
#     del empty_label
#     del occ
#     # binary occupancy
#     occ_mask = ~empty_mask

#     # truncated SDF
#     sdf = udf - eps  # inner is negative
#     del udf
#     inner_mask = occ_mask & (sdf > 0)
#     del occ_mask
#     del empty_mask
#     sdf[inner_mask] *= -1
#     del inner_mask

#     torch.cuda.empty_cache()
#     # breakpoint()
#     t0 = time.time()

#     diffmc = DiffMC(dtype=torch.float32).cuda()
#     vertices, triangles = diffmc(sdf, isovalue=0, normalize=False)
#     vertices = vertices / (sdf.shape[-1] - 1.0) * 2 - 1

 
    
#     vertices = vertices.to(torch.float32)
#     triangles = triangles.to(torch.int32)



#     new_BVH = cubvh.cuBVH(vertices, triangles)
#     del vertices, triangles
   

#     new_x = torch.linspace(-1, 1, 512, device=device)
#     new_points = torch.stack(
#         torch.meshgrid(
#             new_x,
#             new_x,
#             new_x,
#             indexing="ij",
#         ), dim=-1,
#     ) # [N, N, N, 3]
#     new_udf, _, _ = new_BVH.unsigned_distance(new_points.view(-1, 3), return_uvw=False)
#     new_udf = new_udf.view(512,512,512).contiguous()

#     new_sdf = new_udf * (sdf.sign()[0::2,0::2,0::2])
#     del new_udf
#     del new_BVH
#     del new_points
#     del sdf
#     torch.cuda.empty_cache()
#     new_diffmc = DiffMC(dtype=torch.float32).cuda()
#     new_vertices, new_triangles = new_diffmc(new_sdf, isovalue=0, normalize=False)
#     new_vertices = new_vertices / (512 - 1.0) * 2 - 1
#     # new_sdf.div_(2.0)
#     # new_sdf.mul_(128)
 
#     new_sdf = new_sdf.cpu().numpy()
    



#     # vertices = vertices/0.95
#     # watertight_mesh_unnormalized_vertices = sphere_unnormalize(vertices, bcenter, radius)
#     # watertight_mesh_unnormalized_vertices = watertight_mesh_unnormalized_vertices.cpu().numpy()
#     # triangles = triangles.cpu().numpy()
#     # watertight_mesh_unnormalized = trimesh.Trimesh(vertices=watertight_mesh_unnormalized_vertices, faces=triangles, process=False)
    
#     new_vertices = new_vertices/0.95
#     watertight_mesh_unnormalized_new_vertices = sphere_unnormalize(new_vertices, bcenter, radius)
#     watertight_mesh_unnormalized_new_vertices = watertight_mesh_unnormalized_new_vertices.cpu().numpy()
#     new_triangles = new_triangles.cpu().numpy()
#     watertight_mesh_unnormalized_new = trimesh.Trimesh(vertices=watertight_mesh_unnormalized_new_vertices, faces=new_triangles, process=False)

#     return watertight_mesh_unnormalized_new, mesh_path.replace('.ply', f'_cubvh_watertight_{resolution}_512.ply'), new_sdf, sparse_index
   




def sphere_normalize_torch_rescale(vertices):
    """PyTorch version of sphere_normalize function.
    
    Args:
        vertices (torch.Tensor): Tensor of shape (N, 3) containing vertex coordinates.
        
    Returns:
        tuple: Normalized vertices, center, and radius.
    """
    bmin, _ = torch.min(vertices, dim=0)
    bmax, _ = torch.max(vertices, dim=0)
    bcenter = (bmax + bmin) / 2
    radius = torch.norm(vertices - bcenter, dim=-1).max()
    vertices = (vertices) / radius  # to [-1, 1]
    return vertices, bcenter, radius




def cubvh_mesh2watertightsdf(mesh_v,mesh_f, output_resolution=1024):
    if output_resolution ==1024:
        resolution = 1024
    elif output_resolution ==512:
        resolution = 512
    elif output_resolution == 256:
        resolution = 256
    else:
        raise ValueError("Only support output_resolution 1024 or 512")
    device = mesh_v.device
    mesh_scale = 0.5
    # vertices = torch.from_numpy(mesh.vertices).float().to(device)
    # triangles = torch.from_numpy(mesh.faces).long().to(device)
    vertices = mesh_v
    triangles = mesh_f
    
    # vertices , bcenter, radius = sphere_normalize_torch_rescale(vertices)
    bcenter = torch.tensor(0.)
    radius = torch.tensor(1.0)
    # vertices = vertices * 0.95
    # mesh_scale *= 0.95
    # import pdb;pdb.set_trace()
    # mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=mesh_f.cpu().numpy(), process=False)

    # mesh_scale 
    mesh_scale /= radius.item()
    BVH = cubvh.cuBVH(vertices, triangles)
    eps = 2 / resolution
    x = torch.linspace(-1, 1, resolution, device=device)
    points = torch.stack(
        torch.meshgrid(
            x,
            x,
            x,
            indexing="ij",
        ), dim=-1,
    ) # [N, N, N, 3]
    # breakpoint()
    udf, _, _ = BVH.unsigned_distance(points.view(-1, 3), return_uvw=False)
    del BVH
    del points
    del vertices
    del triangles
    del x
    torch.cuda.empty_cache()
    udf = udf.view(resolution,resolution,resolution).contiguous()
    occ = udf < eps
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    del empty_label
    del occ
    del floodfill_mask  # 立即释放，节省~4GB显存
    torch.cuda.empty_cache()
    # binary occupancy
    occ_mask = ~empty_mask

    # truncated SDF
    sdf = udf - eps  # inner is negative
    del udf
    inner_mask = occ_mask & (sdf > 0)
    del occ_mask
    del empty_mask
    sdf[inner_mask] *= -1
    del inner_mask

    torch.cuda.empty_cache()
    diffmc = DiffMC(dtype=torch.float32).cuda()
    vertices, triangles = diffmc(sdf, isovalue=0, normalize=False)
    del diffmc  # DiffMC对象使用完立即删除
    vertices = vertices / (sdf.shape[-1] - 1.0) * 2 - 1
    vertices = vertices.to(torch.float32)
    triangles = triangles.to(torch.int32)
    mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=triangles.cpu().numpy(), process=False)
    sdf_sign = sdf.sign()
    # sdf_sign = sdf.sign()[0::2,0::2,0::2]
    # sdf_sign = sdf_sign.to(torch.float16)
    del sdf
    torch.cuda.empty_cache()
    # new_BVH = cubvh.cuBVH(vertices, triangles)
    # del vertices, triangles
    # new_x = torch.linspace(-1, 1, output_resolution, device=device)
    # new_points = torch.stack(
    #     torch.meshgrid(
    #         new_x,
    #         new_x,

    #         new_x,
    #         indexing="ij",
    #     ), dim=-1,
    # ) # [N, N, N, 3]
    # breakpoint()
    # new_udf, _, _ = new_BVH.unsigned_distance(new_points.view(-1, 3), return_uvw=False)
    new_udf = compute_valid_udf(vertices*0.5, triangles, dim=output_resolution, threshold=8.0)
    del vertices, triangles  # compute_valid_udf执行完立即释放
    torch.cuda.empty_cache()
    
    new_udf = new_udf.view(output_resolution,output_resolution,output_resolution).contiguous()
    new_sdf = new_udf * (sdf_sign)
    sdf_sign = (sdf_sign>0).to(torch.bool).cpu().numpy()
    sdf_sign = sdf_sign.ravel().astype(np.uint8)
    sdf_sign = np.packbits(sdf_sign)
    # del sdf_sign
    sss = resolution/4
    eps  = 1/sss
    # print("eps",eps)
    sparse_mask = new_udf< eps
    del new_udf
    
    sparse_index = sparse_mask.nonzero().int()
    sparse_sdf = new_sdf[sparse_mask].unsqueeze(-1)
    sparse_sdf = sparse_sdf*sss
    del new_sdf
    # breakpoint()
    sparse_index = xyz_to_n(sparse_index, resolution_bits=int(np.log2(resolution)))
    sparse_index = sparse_index.cpu().numpy()
    sparse_sdf = sparse_sdf.to(torch.float16).cpu().numpy()
    # del new_udf
    # del sdf_sign  # 立即释放，节省~1GB显存
    # del new_BVH
    # del new_points
    torch.cuda.empty_cache()

    # new_diffmc = DiffMC(dtype=torch.float32).cuda()
    # new_vertices, new_triangles = new_diffmc(new_sdf, isovalue=0, normalize=False)
    # del new_diffmc  # DiffMC对象使用完立即删除
    # new_vertices = new_vertices / (output_resolution - 1.0) * 2 - 1
    # torch.cuda.empty_cache()
    # sparse_mask = (all_sdf_abs <= eps)
        
    #     # new_sdf.div_(2.0)
    #     # new_sdf.mul_(128)
    # new_sdf.mul_(64)
    # new_sdf.clamp_(-3,3)
    # sparse_index = (new_sdf.abs() < 1).nonzero()
    # 转移到CPU并释放GPU内存
    # sparse_index = sparse_index.cpu()
    # new_sdf = new_sdf.to(torch.float16).cpu().numpy()
    
    # 释放GPU内存
    # torch.cuda.empty_cache()





    # new_vertices = new_vertices/0.95
    # watertight_mesh_unnormalized_new_vertices = sphere_unnormalize_rescale(new_vertices, bcenter, radius)
    # watertight_mesh_unnormalized_new_vertices = watertight_mesh_unnormalized_new_vertices.cpu().numpy()
    # new_triangles = new_triangles.cpu().numpy()
    # watertight_mesh_unnormalized_new = trimesh.Trimesh(vertices=watertight_mesh_unnormalized_new_vertices, faces=new_triangles, process=False)  
    return mesh, sparse_sdf,sparse_index, mesh_scale,sdf_sign
    







if __name__ == '__main__':
    # mesh_dir = '/group/40034/yangdyli/Direct3D-S2/mesh_re/watertight_final/Pylon/Pylon.obj'
    mesh_dir = './walle/walle.obj'
    mesh = trimesh.load(mesh_dir,force='mesh')
    mesh = normalize_mesh(mesh, scale=0.95)
    mesh.export(mesh_dir.replace('.obj', '_normalized.obj'))
    # exit()
    # cubvh_visible_mesh2watertight(mesh_dir, resolution=512)
    # dora_mesh2watertight(mesh_dir, resolution=512)
    # cubvh_mesh2watertight(mesh_dir, resolution=512)
    # exit()
    resolution = 1024
    # mesh = winding_number_mesh2watertight(mesh_dir, resolution=resolution)
    new_mesh,new_mesh_path=cubvh_mesh2watertight(mesh_dir, resolution=resolution)
    new_mesh.export(new_mesh_path)
    print(f"Watertight mesh saved to {new_mesh_path}")
    # resolution = 1024
    # new_mesh,new_mesh_path=cubvh_mesh2watertight(mesh_dir, resolution=resolution)
    # new_mesh.export(new_mesh_path)
    filled_mesh = postprocess_mesh(
        vertices=new_mesh.vertices,
        faces=new_mesh.faces,
        simplify=True,
        simplify_ratio=0.95,
        verbose=True,
    )
    newnew_mesh = trimesh.Trimesh(filled_mesh[0], filled_mesh[1])
    newnew_mesh.export(new_mesh_path.replace('.obj', f'_simplify.obj'))


    resolution = 1024
    new_mesh,new_mesh_path=cubvh_mesh2watertight(mesh_dir, resolution=resolution)
    new_mesh.export(new_mesh_path)
    print(f"Watertight mesh saved to {new_mesh_path}")
    new_filled_mesh = postprocess_mesh(
        vertices=new_mesh.vertices,
        faces=new_mesh.faces,
        simplify=True,
        simplify_ratio=0.95,
        verbose=True,
    )
    newnew_mesh = trimesh.Trimesh(new_filled_mesh[0], new_filled_mesh[1])
    newnew_mesh.export(new_mesh_path.replace('.obj', f'_simplify.obj'))
    # mesh = cubvh_mesh2watertight(mesh_dir, resolution=512
    # mesh = step1x_mesh2watertight(mesh_dir,level=0,res=1024)
    # watertight_mesh_normalized = normalize_mesh(mesh, scale=0.95)
    # watertight_mesh_normalized.export(mesh_dir.replace('.obj', f'_step1x_remesh_1024_wind_normalized.obj'))
    # print(f"Watertight mesh saved to {mesh_dir.replace('.obj', f'_step1x_remesh_1024_wind_normalized.obj')}")