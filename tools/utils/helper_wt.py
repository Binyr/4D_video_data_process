import torch
import numpy as np  
import udf_ext
import cubvh
import trimesh
import time
import os
def compute_valid_udf(vertices, faces, dim=512, threshold=8.0):
    if not faces.is_cuda or not vertices.is_cuda:
        raise ValueError("Both maze and visited tensors must be CUDA tensors")
    udf = torch.zeros(dim**3,device=vertices.device).int() + 10000000
    n_faces = faces.shape[0]
    udf_ext.compute_valid_udf(vertices, faces, udf, n_faces, dim, threshold)
    torch.cuda.empty_cache()
    udf = udf.float()/10000000.
    return udf
#TODO:sphere norm
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

def normalize_mesh_vertcies_torch(mesh_vertices, scale=0.95):
    #TODO：no offset!
    # min_coords, max_coords = mesh_vertices.min(dim=0).values, mesh_vertices.max(dim=0).values
    # dxyz = max_coords - min_coords

    # dist = dxyz.max()
    dist = mesh_vertices.abs().max()*2
    mesh_scale = 2.0 * scale / dist
    vertices = mesh_vertices * mesh_scale
    # mesh_offset = -(min_coords + max_coords) / 2
    # vertices = (mesh_vertices + mesh_offset) * mesh_scale
    return vertices,mesh_scale

def mesh2index(mesh, size=1024, factor=8):
    vertices = torch.Tensor(mesh.vertices).float().cuda() * 0.5
    faces = torch.Tensor(mesh.faces).int().cuda()
    sdf = compute_valid_udf(vertices, faces, dim=size, threshold=4.0)
    sdf = sdf.reshape(size, size, size).unsqueeze(0)

    sparse_index = (sdf < 4/size).nonzero()
    sparse_index[..., 1:] = sparse_index[..., 1:] // factor
    latent_index = torch.unique(sparse_index, dim=0)
    return latent_index


# def mesh2sdf(mesh,size,factor):
#     # size = int(size / factor)
#     mesh = normalize_mesh(mesh, scale=0.95)
#     vertices = torch.Tensor(mesh.vertices).float().cuda() * 0.5
#     faces = torch.Tensor(mesh.faces).int().cuda()
#     udf = compute_valid_udf(vertices, faces, dim=size, threshold=8.0)
#     udf = udf.reshape(size, size, size).unsqueeze(0)
#     sparse_index = (udf.squeeze() < 4/size).nonzero()
#     del udf
#     torch.cuda.empty_cache()
#     size = int(size / factor)
#     vertices = (vertices + 0.5) * (size - 1) / size + 0.5 / size
#     tris = vertices[faces]
#     band = 8 / size
#     sdf = torchcumesh2sdf.get_sdf(tris, R=size, band=band, B=65536)
#     # breakpoint()
#     # assert sdf[sdf<100].max() <= 1.0 and sdf.min()[sdf>-100] >= -1.0, "SDF values out of range"
#     torch.cuda.empty_cache()
#     return sdf, sparse_index

def mesh2sdf_torch(mesh_vertices,mesh_faces,size,factor):
    # size = int(size / factor)
    # mesh = normalize_mesh(mesh, scale=0.95)
    # breakpoint()
    vertices, mesh_scale = normalize_mesh_vertcies_torch(mesh_vertices, scale=0.95)
    # vertices = torch.Tensor(mesh.vertices).float().cuda() * 0.5
    vertices = vertices.float().cuda()* 0.5
    mesh_scale = mesh_scale*0.5
    faces = mesh_faces.int().cuda()
    # print('v bbox:', vertices.min(dim=0).values, vertices.max(dim=0).values)
    # faces = torch.Tensor(mesh.faces).int().cuda()
    # print(vertices.shape, faces.shape)
    # breakpoint()
    vertices = vertices.contiguous()
    # udf = compute_valid_udf(vertices, faces, dim=size, threshold=8.0)
    # udf = udf.reshape(size, size, size).unsqueeze(0)
    # sparse_index = (udf.squeeze() < 4/size).nonzero()
    # del udf
    # torch.cuda.empty_cache()
    size = int(size / factor)

    method = "cubvh"
    if method =='torchcumesh2sdf':
        import torchcumesh2sdf
        import time 
        t0 = time.time()
        vertices = (vertices + 0.5) * (size - 1) / size + 0.5 / size
        tris = vertices[faces]
        band = 8 / size
        sdf = torchcumesh2sdf.get_sdf(tris, R=size, band=band, B=65536)
        print(sdf.max())
        print(method,time.time()-t0)
    elif method =='pysdf':
        # import pysdf
        from pysdf import SDF
        f = SDF(vertices.cpu().numpy(),faces.long().cpu().numpy())
        DIM = 512
        coords = torch.linspace(0, DIM-1, DIM, dtype=torch.float32)
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        # 转换为query点坐标
        query_torch = torch.stack([
            grid_x / (DIM - 1) - 0.5,
            grid_y / (DIM - 1) - 0.5, 
            grid_z / (DIM - 1) - 0.5
        ], dim=-1).view(-1, 3)
        sdf = f(query_torch.cpu().numpy())
        sdf = sdf.reshape(512,512,512)
        sdf = torch.from_numpy(sdf).cuda()
    else:
        import time
        t0 =time.time()




    
        coords = torch.linspace(0, size-1, size, dtype=torch.float32, device=vertices.device)
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        # 转换为query点坐标
        query_torch = torch.stack([
            grid_x / (size - 1) - 0.5,
            grid_y / (size - 1) - 0.5, 
            grid_z / (size - 1) - 0.5
        ], dim=-1).view(-1, 3)
        # print(query_torch.max())
        # points = (points + 0.5) / size - 0.5
        # breakpoint()
        
        # del points
        # import kaolin
        # from kaolin.ops.mesh import check_sign
        # t0  = time.time()
        # sign = check_sign(vertices.unsqueeze(0), faces.long(), query_torch[:].to(vertices.device).unsqueeze(0))
        # print("sdsds",time.time()-t0)
        # sdf_kaolin = udf*sign
        # torch.save(sdf_kaolin, "sdf_kaolin.pt")
        # breakpoint()
        t0 = time.time()
        BVH = cubvh.cuBVH(vertices, faces.long())
        # print(time.time()-t0)
        # chunk = 65536
        # all_distances = []
        # for i in range(0, query_torch.shape[0], chunk):
            # distances, face_id, uvw = BVH.signed_distance(query_torch[i:i+chunk], return_uvw=False, mode='watertight')
            # all_distances.append(distances)
        # distances = torch.cat(all_distances, dim=0)
        distances, face_id, uvw = BVH.signed_distance(query_torch, return_uvw=False, mode='watertight')

        # udf, face_id, _ = BVH.unsigned_distance(query_torch, return_uvw=False)
        # print((distances.abs()-udf).abs().max())
        # print(time.time()-t0)
        # print(distances.max())
        sdf = distances.view(size, size, size)
        # sparse_index = None
    sparse_index = (sdf.abs() < 4/size).nonzero()
        # print(method,time.time()-t0)
        # breakpoint()
        # index = (udf<1/128).squeeze()
        # print(index.device)
        # print(method,time.time()-t0)
        # diff = (udf.squeeze()-sdf.abs()).abs()[index]
        # # print(diff.device)
        # # print(method,time.time()-t0)
        # diff_len = (diff>1e-4).sum()
        # # print(method,time.time()-t0)
        # diff_max = diff.max() 
        # print(method,time.time()-t0)
        # if diff_max > 1e-4:
        #     print(f"Warning: UDF and SDF mismatch! {diff_max.item()} {diff_len.item()}"+"!"*100)
        #     raise ValueError(f"UDF and SDF mismatch! {diff_max.item()} {diff_len.item()}")
            # breakpoint()
        # print(method,time.time()-t0)
    # del udf
    # torch.cuda.empty_cache()
    # breakpoint()
    # assert sdf[sdf<100].max() <= 1.0 and sdf.min()[sdf>-100] >= -1.0, "SDF values out of range"
    torch.cuda.empty_cache()
    return sdf, sparse_index, mesh_scale


import time
# def get_batch(mesh_dir,size,factor=1,type=torch.float32,batch_index=0):
#     t0 = time.time()
#     mesh = trimesh.load_mesh(mesh_dir)
#     t1 = time.time()
#     print(f"加载网格耗时: {t1 - t0:.4f}秒")
#     sdf, sparse_index = mesh2sdf(mesh,size,factor)
    
#     # 
#     # breakpoint()
#     sparse_index[..., 0:] = sparse_index[..., 0:] // factor
#     sparse_index = torch.unique(sparse_index, dim=0)
#     torch.cuda.empty_cache()
#     # breakpoint()
#     sparse_sdf = sdf[sparse_index[:, 0], sparse_index[:, 1], sparse_index[:, 2]].unsqueeze(-1)
#     sparse_sdf = sparse_sdf * 128
#     sparse_sdf = sparse_sdf + 0.2
#     # breakpoint()
#     if batch_index >=0:
#         batch_idx = torch.full((sparse_index.shape[0],), batch_index, dtype=sparse_index.dtype, device=sparse_index.device)
#     else:
#         batch_idx = None
#     batch = {
#         'sparse_sdf': sparse_sdf.to(type),
#         'sparse_index': sparse_index,
#         'batch_idx': batch_idx,
#         'factor': factor
#     }
#     return batch
from .mesh_info import get_mesh_info
def get_batch_mesh(mesh_v,mesh_f,size,factor=1,type=torch.float32,batch_index=0,debug=-1,debug_path=None):
    '''
    input mesh_v need normalized to [-1, 1]
    '''
    # t0 = time.time()
    # mesh = trimesh.load_mesh(mesh_dir)
    # t1 = time.time()

    # print(f"加载网格耗时: {t1 - t0:.4f}秒")
    sdf, sparse_index, mesh_scale = mesh2sdf_torch(mesh_v, mesh_f, size, factor)
    
    if debug>-1:
        # print(f"mesh2sdf_torch耗时: {time.time() - t1:.4f}秒")s
        # if debug==8:
            # breakpoint()
        
            
            from skimage import measure
            vertices, faces, _, _ = measure.marching_cubes(
                    sdf.cpu().numpy(),
                    0,
                    method="lewiner",
                )
            vertices = vertices / 512 * 2 - 1
            mesh = trimesh.Trimesh(vertices, faces,process=False)
            mesh_info = get_mesh_info(mesh.copy(), "debug")
            
            print("debug mesh info:", mesh_info)
            if mesh_info['stats']['connected_components']>1:
                mesh.export(f"{debug_path}/mesh_sdf_{debug}.obj")
                meshs = mesh.split(only_watertight=False)
                for i, m in enumerate(meshs):
                    m.export(f"{debug_path}/mesh_sdf_{debug}_part{i}.obj")
                print(f"Warning: debug mesh has {mesh_info['stats']['connected_components']} components!"+"!"*100)
                # np.save(f"{debug_path}/sdf_{debug}.npy", sdf.cpu().numpy())
                raise ValueError(f"debug mesh has {mesh_info['stats']['connected_components']} components!")
            # os.makedirs('data_save_cubvh',exist_ok=True)
            else:
                mesh.export(f"{debug_path}/mesh_sdf_{debug}_success.obj")
            
            # print(f"Marching cubes mesh saved to ./data_save_cubvh/mesh_sdf_{debug}.obj")
    #
    # breakpoint()
    sparse_index[..., 0:] = sparse_index[..., 0:] // factor
    sparse_index = torch.unique(sparse_index, dim=0)
    torch.cuda.empty_cache()
    # breakpoint()
    # sparse_sdf = sdf[sparse_index[:, 0], sparse_index[:, 1], sparse_index[:, 2]].unsqueeze(-1)
    # sparse_sdf =  sparse_sdf * 128
    sdf.mul_(128) 
    sdf.clamp_(-2,2)
    # addd = 0.0
    # print("offset sdf by ", addd)
    # sparse_sdf = sparse_sdf + addd
    # breakpoint()
    if batch_index >=0:
        batch_idx = torch.full((sparse_index.shape[0],), batch_index, dtype=sparse_index.dtype, device=sparse_index.device)
    else:
        batch_idx = None
    torch.cuda.empty_cache()

    
    dense_index = sparse_index[...,0:]//8
    dense_index = torch.unique(dense_index, dim=0)
    dense_size = int(size/8)
    dense_voxels = torch.zeros(( dense_size, dense_size, dense_size), dtype=torch.float16, device=sdf.device)
    dense_voxels[dense_index[:, 0], dense_index[:, 1], dense_index[:, 2]] = 1

    batch = {
        # 'sparse_sdf': sparse_sdf.to(type),
        # 'sparse_index': sparse_index,
        "all_sdf": sdf.to(type).cpu().numpy(),
        'batch_idx': batch_idx,
        'factor': factor,
        'mesh_scale': mesh_scale,
        # 'dense_index': dense_index,
        'dense_voxels': dense_voxels.to(type).cpu().numpy()
    }
    del sdf
    torch.cuda.empty_cache()
    return batch
