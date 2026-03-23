# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import argparse
import igl
import numpy as np
import os
import time
import torch
from scipy.stats import truncnorm
import trimesh
# import pymeshlab
import cubvh
# from tools.watertight.visibility_check import get_visibility
# from tools.watertight.kaolin_mesh_to_sdf import kaolin_sdf

# from diso import DiffDMC

def random_sample_pointcloud(mesh, num = 30000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng()
    index = rng.choice(num, num, replace=False)
    return points[index], normals[index]

def sharp_sample_pointcloud(mesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:,i]], np.sum(VN[F[:,i]] * N, axis=-1)), axis=-1)
        VN2[F[:,i]] = np.min(dot, axis=-1)

    sharp_mask = VN2<0.985
    # collect edge
    edge_a = np.concatenate((F[:,0],F[:,1],F[:,2]))
    edge_b = np.concatenate((F[:,1],F[:,2],F[:,0]))
    sharp_edge = ((sharp_mask[edge_a] * sharp_mask[edge_b]))
    edge_a = edge_a[sharp_edge>0]
    edge_b = edge_b[sharp_edge>0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num,1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals

def SampleMesh(V, F):
    mesh = trimesh.Trimesh(vertices=V, faces=F)

    area = mesh.area
    sample_num = 499712//4

    random_surface, random_normal = random_sample_pointcloud(mesh, num=sample_num)
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    #save_surface
    surface = np.concatenate((random_surface, random_normal), axis = 1).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(np.float16)

    surface_data = {
        "random_surface": surface,
        "sharp_surface": sharp_surface,
    }

    # sdf_data = sample_sdf(mesh, random_surface, random_sharp_surface)
    return surface_data#, sdf_data

def normalize_to_unit_box(V):
    """
    Normalize the vertices V to fit inside a unit bounding box [0,1]^3.
    V: (n,3) numpy array of vertex positions.
    Returns: normalized V
    """
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    scale = (V_max - V_min).max() * 1.01
    V_normalized = (V - V_min) / scale
    return V_normalized

def Watertight_cubvh(V, F, grid_res = 512, epsilon = None):
    if epsilon is None:
        epsilon = 2.0 / grid_res
    # Compute bounding box
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    # Create a uniform grid
    grid_points = torch.stack(
        torch.meshgrid(
            torch.from_numpy(np.linspace(min_corner[0], max_corner[0], grid_res)).to("cuda"),
            torch.from_numpy(np.linspace(min_corner[1], max_corner[1], grid_res)).to("cuda"),
            torch.from_numpy(np.linspace(min_corner[2], max_corner[2], grid_res)).to("cuda"),
            indexing="ij",
        ), dim=-1,
    ).float()

    BVH = cubvh.cuBVH(V, F)
    udf, _, _ = BVH.unsigned_distance(grid_points.view(-1, 3), return_uvw=False)
    udf = udf.view(grid_res, grid_res, grid_res).contiguous()
    
    # floodfill to get SDF
    eps = 2 / grid_res
    occ = udf < 2 / grid_res # tolerance 2 voxel
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    occ_mask = ~empty_mask
    sdf = udf - eps  # inner is negative
    inner_mask = occ_mask & (sdf > 0)
    sdf[inner_mask] *= -1

    mc_verts, mc_faces = igl.marching_cubes(-sdf.view(-1).cpu().numpy(), grid_points.view(-1, 3).cpu().numpy(), grid_res, grid_res, grid_res, 0.)

    return mc_verts, mc_faces

def watertight_pipeline(input_obj, grid_res, output_dir, enable_rotation=True):
    # V, F = igl.read_triangle_mesh(input_obj)
    mesh = trimesh.load(input_obj, force='mesh')
    if enable_rotation:
        transform_matrix = np.array([
        [1,  0,  0, 0],  # X轴保持不变
        [0,  0,  1, 0],  # Y轴变为Z方向
        [0, -1,  0, 0],  # Z轴变为-Y方向
        [0,  0,  0, 1]   # 齐次坐标
        ], dtype=np.float64)
        mesh.apply_transform(transform_matrix)
    V, F = mesh.vertices, mesh.faces
    # if enable_rotation:
    #     V = rotation(V)
    V = normalize_to_unit_box(V)
    mc_verts, mc_faces = Watertight_cubvh(V, F, grid_res=grid_res)
    
    # normalize to -1~1
    center = (np.max(mc_verts, axis=0) + np.min(mc_verts, axis=0)) / 2.
    mc_verts = (mc_verts - center)
    mc_verts = mc_verts * (1.0 / 1.01 / np.max(np.abs(mc_verts)))
    surface_data = SampleMesh(mc_verts, mc_faces)

    return surface_data
