from pathlib import Path
f_path = Path(__file__).absolute()
p_path = f_path.parents[2]
import sys
sys.path.insert(0, str(p_path))

import numpy as np
import torch
import trimesh

from diso import DiffMC, DiffDMC
from tqdm import tqdm

from direct3d_s2.utils import normalize_mesh, mesh2index
from direct3d_s2.modules import sparse as sp
from tools.utils.mesh2watertight_video import cubvh_mesh2watertightsd_vid, n_to_xyz
import gc
import json

weight_type = torch.float16
device = "cuda:0"
# load vae
root = Path("data/objverse_minghao_4d_mine_40075/rendering_v5/000-000_static_camera_distance_v3/0032696f5871429fbd0549d9628f812c")
mesh_path = f"{root}/result_mesh.npz"
npz = np.load(mesh_path)
# load mesh and sdf
mesh_v_list = npz["vertices"] # T N 3
mesh_f = npz["faces"] # F 3
f_indices = np.array(npz["frame_indices"])

# load camera
camera_c2ws_bl = []
result_json_path = f"{root}/result.json"
with open(result_json_path) as f:
    metas = json.load(f)
    if "_global" in metas:
        metas = metas["_global"]
        # new meta format
        cameras = metas["static_cameras"]
        cameras.sort(key=lambda x: int(x["view_index"]))
        for camera in cameras:
            camera_c2ws_bl.append(np.array(camera["camera_c2w"]))
    else:
        # old meta format
        views = list(metas.values())[0]["views"]
        views.sort(lambda x: x["view_index"])
        for view in metas:
            camera_c2ws_bl.append(np.array(camera["camera_c2w"]))


mesh_f = torch.from_numpy(mesh_f).to(dtype=torch.int32,device='cuda')
for ic, c2w in enumerate(camera_c2ws_bl):
    if ic % 2 == 0:
        continue
    # transform mesh vertics to camera space
    w2c = np.linalg.inv(c2w)
    print(w2c.round(3))
    continue
    w2c_rot_np = w2c[:3, :3]
    w2c_rot = torch.from_numpy(w2c_rot_np).float().cuda()
    for i, f_i in tqdm(enumerate(f_indices), desc=root.name):
        with torch.no_grad():
            mesh_v = mesh_v_list[i]
            mesh_v = torch.from_numpy(mesh_v).to(dtype=torch.float32,device='cuda')
            print("before", mesh_v.mean())
            mesh_v = torch.matmul(mesh_v, w2c_rot.T)
            print("after", mesh_v.mean())
            resolution = 512
            mesh_wt, sparse_sdf, sparse_index, mesh_scale, sdf_sign = cubvh_mesh2watertightsd_vid(mesh_v, mesh_f, resolution)
            # save_path = Path(root.replace("rendering_v5", "train_data"))
            save_path = Path("vis/train_data_view/") / root.name / f"view_{str(ic).zfill(2)}"
            save_path.mkdir(exist_ok=True, parents=True)

            np.savez_compressed(f"{save_path}/sparse_sdf_{resolution}_{str(f_i).zfill(3)}.npz",
                    sparse_sdf=sparse_sdf, sparse_index=sparse_index)
            np.savez_compressed(f"{save_path}/sdf_sign_{resolution}_{str(f_i).zfill(3)}.npz",
                    sdf_sign=sdf_sign)
            #     del sparse_index, sparse_sdf, sdf_sign
            del mesh_v, mesh_wt, sparse_sdf, sparse_index, mesh_scale, sdf_sign
            gc.collect()
            torch.cuda.empty_cache()
