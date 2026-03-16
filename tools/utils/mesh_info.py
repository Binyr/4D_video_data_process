
from .check_mesh_single import check_mesh_health
def get_mesh_info(mesh,mesh_path=""):
    result = check_mesh_health(mesh, verbose=False)
        
    # 准备保存的结果
    save_result = {
        "mesh_path": mesh_path,
        "healthy": result["healthy"],
        "warnings": result["warnings"],
        "errors": result["errors"],
        "stats": result["stats"],
        "check_timestamp": __import__('time').time(),
        "check_date": __import__('datetime').datetime.now().isoformat()
    }
    return save_result


if __name__ =='__main__':
    import trimesh
    import os
    mesh_path = '/group/40034/yangdyli/Direct3D-S2/watertight_1024/debug_toys4k/cd135bb7b768d40d0708b4a4a5885b40c28f3ebfa873115b11a295b7c5cbc409/cd135bb7b768d40d0708b4a4a5885b40c28f3ebfa873115b11a295b7c5cbc409_cubvh_watertight_512.ply'
    mesh = trimesh.load(mesh_path,force='mesh')
    mesh_info = get_mesh_info(mesh, mesh_path)
    print(mesh_info)