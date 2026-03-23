import json
import os
from random import shuffle
import shutil
from subprocess import DEVNULL, call
import sys
import trimesh
import torch
from mesh2watertight import cubvh_mesh2watertightsdf
# from direct3d_s2.utils.mesh import normalize_mesh
# from ..watertight_1024.helper import postprocess_mesh
import os
import trimesh
import open3d as o3d
import numpy as np
from config import SAVE_PATH,PLY_MESH_PATH,TRANSFORMER_DATA_PATH
from mesh_info import get_mesh_info
# SAVE_PATH = '/group/40075/yangdyli/TRELLIS_WATERTIGHT/watertight_ply_1024/ObjaverseXL_github'
from data_process_batch_single import transform_mesh, xyz_to_n
def mesh_watertight(mesh,mesh_path, resolution=1024):
    """生成水密网格"""
    # dora_mesh, remesh_path = dora_mesh2watertight(mesh_path, resolution)
    # sparse_vae_reconst(remesh_path)
    mesh, mesh_path = cubvh_mesh2watertight(mesh,mesh_path, resolution)
    # sparse_vae_reconst(cubvh_remesh_path)
    # mesh = winding_number_mesh2watertight(mesh_path, resolution)
    # mesh.export(mesh_path)
    return mesh, mesh_path


def mesh_watertight_sdf(mesh_v,mesh_f, resolution=1024):
    """生成水密网格"""
    # dora_mesh, remesh_path = dora_mesh2watertight(mesh_path, resolution)
    # sparse_vae_reconst(remesh_path)
    mesh, sparse_sdf,sparse_index,mesh_scale,sdf_sign= cubvh_mesh2watertightsdf(mesh_v,mesh_f, resolution)
    # sparse_index[]
    # dense_index = sparse_index[...,0:]//8
    # dense_index = torch.unique(dense_index,dim=0)
    # dense_size = int(resolution//8)
    # dense_voxels = torch.zeros((dense_size,dense_size,dense_size),dtype=torch.bool,device='cuda')
    # dense_voxels[dense_index[:,0],dense_index[:,1],dense_index[:,2]] = True
    # dense_voxels = dense_voxels.cpu().numpy()
    # del sparse_index
    # del dense_index
    
    # 强制垃圾回收和GPU内存清理
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # sparse_vae_reconst(cubvh_remesh_path)
    # mesh = winding_number_mesh2watertight(mesh_path, resolution)
    # mesh.export(mesh_path)
    # breakpoint()
    ret = {
        'mesh': mesh,
        # 'sdf': sdf,
        'sdf_sign': sdf_sign,
        'sparse_sdf': sparse_sdf,
        'sparse_index': sparse_index,
        'mesh_scale': mesh_scale,
        # 'dense_voxels': dense_voxels
    }
    return ret




def simplify_mesh(mesh, mesh_path, simplify_ratio=0.95):
    """简化网格"""
    filled_mesh = postprocess_mesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        simplify=True,
        simplify_ratio=simplify_ratio,
        verbose=False,
    )
    newnew_mesh = trimesh.Trimesh(filled_mesh[0], filled_mesh[1])
    newnew_mesh.export(mesh_path.replace('.ply', f'_simplify.ply'))


def load_mesh_with_fallback(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    print(f"[INFO] 使用trimesh成功加载: {mesh_path}")
    return mesh
   

def preprocess_mesh(mesh_path,subset_sha256):
    """预处理网格：trimesh 支持的走 trimesh，不支持的走 open3d"""
    mesh_format = mesh_path.split('.')[-1].lower()
    # mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    subset = subset_sha256.split('/')[0]
    sha256 = subset_sha256.split('/')[1]
    os.makedirs(f'{SAVE_PATH}/{subset_sha256}', exist_ok=True)
    remesh_path = f'{SAVE_PATH}/{subset_sha256}/{sha256}.ply'
    ply_mesh_path = f'{PLY_MESH_PATH.replace("{subset}",subset)}/{sha256}/mesh.ply'
    if os.path.exists(ply_mesh_path):
        print(f"[INFO] PLY 预处理网格已存在: {ply_mesh_path}")
        try:
            mesh = load_mesh_with_fallback(ply_mesh_path)
            return mesh, ply_mesh_path
        except Exception as e:
            print(f"[ERROR] 加载 PLY 预处理网格失败: {e}, 尝试重新加载原始网格")
    assert False,"PLY 预处理网格不存在"
    watertight_path = f'/group/40075/yangdyli/TRELLIS_WATERTIGHT/watertight_ply_512/{subset}/{sha256}/{sha256}.ply'
    if os.path.exists(watertight_path):
        print(f"[INFO] 水密网格已存在: {watertight_path}")
        mesh = load_mesh_with_fallback(watertight_path)
        mesh.export(remesh_path)
        return mesh, remesh_path
    if mesh_format in ['obj','stl','ply']:
        mesh = load_mesh_with_fallback(mesh_path)
        mesh.export(remesh_path)
        return mesh, remesh_path
    
    # args = [
    #     "blender", '-b', '-P', os.path.join(os.path.dirname(__file__), 'export_mesh_obj.py'),
    #     '--',
    #     '--object', os.path.expanduser(mesh_path),
    #     '--output_mesh', remesh_path,
    # ]
    # if mesh_path.endswith('.blend'):
    #     args.insert(1, mesh_path)
    # print(' '.join(args))
    # call(args)
    # remesh_path = remesh_path.replace('watertight_ply_1024','watertight_ply_512')
    if os.path.exists(remesh_path):
        print(f"[INFO] 预处理网格已存在: {remesh_path}")
        return load_mesh_with_fallback(remesh_path), remesh_path
    assert False,"预处理不存在"
    # mesh = trimesh.load(remesh_path, force='mesh')
    # if mesh_format in trimesh.available_formats():
    #     mesh = trimesh.load(mesh_path, force='mesh')
    #     mesh.export(remesh_path)
    # else:
    #     print(f"[INFO] 非Trimesh格式 {mesh_format}，使用 Open3D 尝试加载: {mesh_path}")
    #     # try:
    #     mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    #     if not mesh_o3d.is_empty():
    #         o3d.io.write_triangle_mesh(remesh_path, mesh_o3d)
    #         mesh = trimesh.load(remesh_path, force='mesh')
    #     else:
    #         raise ValueError(f"Open3D 加载 mesh 为空: {mesh_path}")
        # except Exception as e:
            # print(f"[ERROR] 处理失败: {e}")
            # mesh = None
            # remesh_path = None

    return mesh, remesh_path
import gc
def print_gpu_tensor_sizes():
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size = obj.numel() * obj.element_size() / 1024**2  # MB
                print(type(obj), obj.size(), f"{size:.2f} MB")
                total += size
        except:
            pass
    print(f"Total GPU tensor memory: {total:.2f} MB")


def check_mesh_valid(mesh):
    """检查网格是否有效"""
    mesh_bbox = mesh.bounding_box
    if mesh_bbox.volume < 1e-6:
        print(f"Mesh {mesh} is invalid, bounding box volume is too small.")
        return False
    return True
import time
import datetime
def process_single_mesh(mesh_path,subset_sha256, resolution=256):
    """处理单个网格文件的主函数"""
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始处理网格: {mesh_path}, 分辨率: {resolution}")
    data_save_path = f"{SAVE_PATH}/{subset_sha256}"
    os.makedirs(data_save_path, exist_ok=True)
    # 检查文件是否存在
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh path {mesh_path} does not exist")
    
    save_mesh_path = f"{SAVE_PATH}/{subset_sha256}/cubvh_watertight_{resolution}.ply"
    if os.path.exists(save_mesh_path):
        print(f"[INFO] 水密网格已存在，跳过处理: {save_mesh_path}")
        mesh = trimesh.load(save_mesh_path, force='mesh')
        return mesh, save_mesh_path
    
    save_simplify_path = save_mesh_path.replace('.ply', '_simplify.ply')
    if os.path.exists(save_simplify_path):
        print(f"[INFO] 简化后的水密网格已存在，跳过处理: {save_simplify_path}")
        mesh = trimesh.load(save_simplify_path, force='mesh')
        return mesh, save_simplify_path
    subset = subset_sha256.split('/')[0]
    sha256 = subset_sha256.split('/')[1]
    transform_path = os.path.join(TRANSFORMER_DATA_PATH.replace('{subset}',subset),sha256, f'transforms.json')
    if not os.path.exists(transform_path):
        raise FileNotFoundError(f"Transform path {transform_path} does not exist")

    # 加载变换信息
    with open(transform_path, 'r') as f:
        transforms_json = json.load(f)
    transform_mats = transforms_json['frames']
    # 预处理网格
    mesh, remesh_path = preprocess_mesh(mesh_path,subset_sha256)
    
    # 检查网格有效性
    if not check_mesh_valid(mesh):
        raise ValueError(f"Mesh {mesh_path} is invalid, skipping remeshing")
    
    mesh_v = mesh.vertices
    mesh_f = mesh.faces
    mesh_v = torch.from_numpy(mesh_v).to(dtype=torch.float32,device='cuda')
    mesh_f = torch.from_numpy(mesh_f).to(dtype=torch.int32,device='cuda')
    # 生成水密网格
    # mesh, mesh_path,sdf = mesh_watertight_sdf(mesh,remesh_path, resolution)
    # mesh_normalized_v =  mesh_v @ torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32).cuda()
    torch.cuda.empty_cache()
    # breakpoint()
    batchs = []
    batch_norm = {"mesh_scale":1}
    # batch_norm = mesh_watertight_sdf(mesh_normalized_v,mesh_f, resolution)
    batchs.append(batch_norm)
    # breakpoint()
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 水密网格生成完成，开始处理变换")
    torch.cuda.empty_cache()
    # breakpoint()
    transform_mats = transform_mats[:9]
    for idx, transform in enumerate(transform_mats):
        print(f"处理变换 {idx+1}/{len(transform_mats)}")
        transform_mesh_v = transform_mesh(mesh_v, transform)
        batch = mesh_watertight_sdf(transform_mesh_v,mesh_f, resolution)
        batchs.append(batch)
        # breakpoint()


    mesh_scales = [batch['mesh_scale'] for batch in batchs]
    with open(f"{data_save_path}/batch_scales.json", 'w') as f:
        json.dump({"mesh_scales": mesh_scales}, f)
    # breakpoint()
    for idx ,batch in enumerate(batchs):
        if idx==0:
            continue
        # mesh = batch['mesh']
        # sdf = batch['sdf']
        # all_sdf = torch.tensor(sdf, dtype=torch.float16, device='cuda:0')
        # all_sdf = all_sdf * 2
        # eps = 1
        # all_sdf_abs = all_sdf.abs()
        # del all_sdf
        # sparse_mask = (all_sdf_abs <= eps)
        # sparse_index = sparse_mask.nonzero().int()
        # sparse_sdf = all_sdf[sparse_mask].unsqueeze(-1)
        # sparse_index = xyz_to_n(sparse_index, resolution_bits=10)
        # sparse_index = sparse_index.cpu().numpy()
        # sparse_sdf = sparse_sdf.cpu().numpy()
        # mesh = batch['mesh']
        # input_mesh_result = get_mesh_info(mesh,"")
        # input_mesh_path = f"{data_save_path}/input_mesh_info_{str(idx).zfill(2)}.json"
        # with open(input_mesh_path, 'w') as f:
        #     json.dump(input_mesh_result, f)
        sparse_sdf = batch['sparse_sdf']
        sparse_index = batch['sparse_index']
        sdf_sign = batch['sdf_sign']

        np.savez_compressed(f"{data_save_path}/sparse_sdf_{resolution}_{str(idx).zfill(2)}.npz",
                sparse_sdf=sparse_sdf, sparse_index=sparse_index)
        np.savez_compressed(f"{data_save_path}/sdf_sign_{resolution}_{str(idx).zfill(2)}.npz",
                sdf_sign=sdf_sign)
        del sparse_index, sparse_sdf, sdf_sign
        torch.cuda.empty_cache()
    shutil.copy(remesh_path, f"{data_save_path}/{sha256}.ply")
        # eps = 2
        # sparse_mask = (all_sdf_abs <= eps)
        # sparse_index = sparse_mask.nonzero().int()
        # sparse_index = xyz_to_n(sparse_index, resolution_bits=10)
        # sparse_index = sparse_index.cpu().numpy()
        # sparse_sdf = all_sdf[sparse_mask].unsqueeze(-1)
        # sparse_sdf = sparse_sdf.cpu().numpy()
        # np.savez_compressed(f"{data_save_path}/sparse_sdf_{str(idx).zfill(2)}.npz",
        #         sparse_sdf=sparse_sdf, sparse_index=sparse_index)
    
        # torch.cuda.empty_cache()
        # mesh_scale = batch['mesh_scale']
        # dense_voxels = batch['dense_voxels']
        # mesh_save_path = f"{data_save_path}/cubvh_watertight_{resolution}_{str(idx).zfill(2)}.ply"
        # mesh.export(mesh_save_path)
        # mesh_info_res = get_mesh_info(mesh,mesh_save_path)
        # mesh_info_save_path = mesh_save_path.replace('.ply', f'_info.json')
        # with open(mesh_info_save_path, 'w',encoding='utf-8') as f:
            # json.dump(mesh_info_res, f)
        # sdf_save_path = f"{data_save_path}/all_sdf_{resolution}_{str(idx).zfill(2)}.npz"
        # np.savez(sdf_save_path, all_sdf=sdf)
        # dense_voxels_save_path = f"{data_save_path}/batch_dense_{resolution}_{str(idx).zfill(2)}.npz"
        # np.savez(dense_voxels_save_path, dense_voxels=dense_voxels)
    torch.cuda.empty_cache()
    return None, f"{data_save_path}/cubvh_watertight_{resolution}_00.ply"
    # if not os.path.exists(mesh_path):
    #     mesh.export(mesh_path)
    # np.savez(mesh_path.replace('.ply', f'_sdf_{resolution}.npz'), sdf=sdf)
    # 简化网格
    # try:
    #     print(f"开始简化网格: {mesh_path}")
    #     simplify_mesh(mesh, mesh_path, simplify_ratio=0.95)
    #     print(f"简化网格成功: {mesh_path}")
    # except Exception as e:
    #     print(f"简化网格失败: {e}, 跳过简化步骤")
    #     mesh.export(mesh_path)
    #     # 保存水密网格

    
    # 清理CUDA缓存
    torch.cuda.empty_cache()
    
    # print(f"成功处理网格: {mesh_path}")
    return mesh, mesh_path

def main():
    """命令行入口函数 - 纯处理，不包含try逻辑"""
    if len(sys.argv) < 3:
        print("用法: python single_mesh_processor.py <mesh_path> [resolution]")
        print("示例: python single_mesh_processor.py /path/to/mesh.obj 1024")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    subset_sha256 = sys.argv[2]
    resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 1024   
    batch_id = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(mesh_path)
    local_save_path = f"{SAVE_PATH}/{subset_sha256}"
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(local_save_path, exist_ok=True)
    # 直接处理，不捕获异常，让外部调用者处理
    error_path = "./error_wt_256"
    os.makedirs(error_path,exist_ok=True)
    errors_id = []
    if os.path.exists(f"{error_path}/batch_mesh_{batch_id}.txt"):
        with open(f"{error_path}/batch_mesh_{batch_id}.txt",'r') as f:
            errors_id = f.readlines()
    errors_id = set(errors_id)
    errors_id.add(subset_sha256)
    errors_id_list = list(errors_id)
    errors_id = []
    for error_id in errors_id_list:
        error_id = error_id.strip()
        if error_id !='':
            errors_id.append(error_id)

    with open(f"{error_path}/batch_mesh_{batch_id}.txt",'w') as f:
        # f.writelines(errors_id)
        for error_id in errors_id:
            f.write(f'{error_id}\n')
    import time
    t0 =time.time()
    _, mesh_path=process_single_mesh(mesh_path,subset_sha256, resolution)
    print(time.time()-t0)
    print(f"✅ 成功处理 {mesh_path} (分辨率: {resolution})")
    os.makedirs(f'./process_wt_256', exist_ok=True)
    with open(f'./process_wt_256/processed_mesh_{batch_id}.txt', 'a') as f:
        f.write(f'{subset_sha256}\n')
        
    errors_id.remove(subset_sha256)
    with open(f"{error_path}/batch_mesh_{batch_id}.txt",'w') as f:
        # f.writelines(errors_id)
        for error_id in errors_id:
            f.write(f'{error_id}\n')

    

if __name__ == "__main__":
    main()
