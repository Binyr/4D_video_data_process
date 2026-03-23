import numpy as np
import trimesh
from scipy.spatial.distance import cdist
from scipy.ndimage import map_coordinates
from collections import defaultdict
import warnings
import signal
from contextlib import contextmanager
import json
import os
import sys
import argparse

@contextmanager
def timeout(duration):
    """Linux下的超时上下文管理器"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"操作超时 ({duration}秒)")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        # 恢复原来的信号处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def quick_connectivity_check(mesh, max_time=5):
    """
    快速连通性预检查，避免在坏mesh上浪费时间
    
    Args:
        mesh: trimesh对象
        max_time: 最大执行时间（秒）
        
    Returns:
        dict: 包含连通性信息或错误信息
    """
    result = {"success": False, "n_components": 0, "component_sizes": [], "timeout": False}
    
    try:
        # 快速预检查：如果面数太多可能会很慢
        n_faces = len(mesh.faces)
        n_vertices = len(mesh.vertices)
        
        # 如果mesh太大，直接跳过详细检查
        # if n_faces > 100000 or n_vertices > 50000:
        #     result["skip_reason"] = f"mesh太大 ({n_vertices} 顶点, {n_faces} 面)，跳过连通性检查"
        #     return result
            
        # 快速检查：计算顶点度数分布
        # 如果大多数顶点度数异常，可能是破碎的mesh
        vertex_degrees = np.bincount(mesh.faces.flatten(), minlength=n_vertices)
        isolated_vertices = np.sum(vertex_degrees == 0)
        high_degree_vertices = np.sum(vertex_degrees > 20)  # 度数异常高
        
        if isolated_vertices > n_vertices * 0.1:
            result["skip_reason"] = f"孤立顶点过多 ({isolated_vertices})，可能是严重破碎的mesh"
            return result
            
        if high_degree_vertices > n_vertices * 0.1:
            result["skip_reason"] = f"高度数顶点过多 ({high_degree_vertices})，可能是异常mesh"
            return result
        
        # 使用超时执行连通性分析
        with timeout(max_time):
            connected_components = mesh.split(only_watertight=False)
            
        result["success"] = True
        result["n_components"] = len(connected_components)
        result["component_sizes"] = [len(comp.vertices) for comp in connected_components]
        
        # 计算每个连通分量的体积
        component_volumes = []
        total_volume = 0.0
        
        for comp in connected_components:
            try:
                if comp.is_watertight:
                    comp_volume = abs(comp.volume)
                    component_volumes.append(comp_volume)
                    total_volume += comp_volume
                else:
                    # 对于非水密的分量，体积设为0或计算convex hull体积作为估计
                    try:
                        convex_volume = abs(comp.convex_hull.volume)
                        component_volumes.append(convex_volume)
                        total_volume += convex_volume
                    except:
                        component_volumes.append(0.0)
            except:
                component_volumes.append(0.0)
        
        result["component_volumes"] = component_volumes
        result["total_volume"] = total_volume
        
    except TimeoutError:
        result["timeout"] = True
        result["skip_reason"] = f"连通性分析超时 (>{max_time}秒)，mesh可能严重破碎"
    except Exception as e:
        result["skip_reason"] = f"连通性分析失败: {e}"
    
    return result

def check_mesh_health(mesh_path_or_vertices_faces, verbose=True):
    """
    检查 mesh 是否正常，检测由于 SDF 异常导致的 marching cubes 生成的 mesh 问题
    
    Args:
        mesh_path_or_vertices_faces: str (mesh文件路径) 或 tuple (vertices, faces)
        verbose: bool, 是否输出详细信息
        
    Returns:
        dict: 包含各种检查结果的字典
    """
    
    # 加载mesh
    if isinstance(mesh_path_or_vertices_faces, trimesh.Trimesh):
        mesh = mesh_path_or_vertices_faces
    elif isinstance(mesh_path_or_vertices_faces, str):
        try:
            mesh = trimesh.load_mesh(mesh_path_or_vertices_faces)
        except Exception as e:
            return {"error": f"无法加载mesh文件: {e}", "healthy": False}
    else:
        vertices, faces = mesh_path_or_vertices_faces
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    results = {
        "healthy": True,
        "warnings": [],
        "errors": [],
        "stats": {}
    }
    
    # 基本统计
    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.faces)
    results["stats"]["n_vertices"] = n_vertices
    results["stats"]["n_faces"] = n_faces
    
    if verbose:
        print(f"Mesh基本信息: {n_vertices} 顶点, {n_faces} 面")
    
    # 1. 检查空mesh
    if n_vertices == 0 or n_faces == 0:
        results["errors"].append("空mesh: 没有顶点或面")
        results["healthy"] = False
        return results
    
    # 2. 检查退化三角形 (面积为0的三角形)
    try:
        face_areas = mesh.area_faces
        degenerate_faces = np.sum(face_areas < 1e-10)
        results["stats"]["degenerate_faces"] = int(degenerate_faces)
        
        if degenerate_faces > 0:
            ratio = degenerate_faces / n_faces
            results["warnings"].append(f"发现 {degenerate_faces} 个退化三角形 ({ratio:.2%})")
            if ratio > 0.1:  # 超过10%的退化三角形
                results["errors"].append("退化三角形过多，可能表示SDF异常")
                results["healthy"] = False
                return results
    except:
        results["warnings"].append("无法计算面积")
    
    # 3. 检查连通性（带超时保护）
    try:
        connectivity_result = quick_connectivity_check(mesh, max_time=30)
        
        if not connectivity_result["success"]:
            if connectivity_result["timeout"]:
                results["errors"].append("连通性分析超时，mesh可能严重破碎")
                results["healthy"] = False
                return results
            else:
                results["warnings"].append(connectivity_result.get("skip_reason", "连通性检查跳过"))
                # 如果是因为mesh太大而跳过，不算错误
                if "mesh太大" in connectivity_result.get("skip_reason", ""):
                    results["stats"]["connectivity_skipped"] = True
                else:
                    # 其他原因跳过可能表示mesh有问题
                    results["errors"].append("连通性分析失败，mesh可能异常")
                    results["healthy"] = False
                    return results
        else:
            n_components = connectivity_result["n_components"]
            component_sizes = connectivity_result["component_sizes"]
            component_volumes = connectivity_result.get("component_volumes", [])
            total_volume = connectivity_result.get("total_volume", 0.0)
            
            results["stats"]["connected_components"] = n_components
            results["stats"]["component_volumes"] = component_volumes
            results["stats"]["total_components_volume"] = total_volume
            
            # 计算每个连通分量的顶点数占比
            component_size_ratios = [size / n_vertices for size in component_sizes]
            results["stats"]["component_size_ratios"] = component_size_ratios
            
            # 计算每个连通分量的体积占比
            if component_volumes and total_volume > 0:
                component_volume_ratios = [vol / total_volume for vol in component_volumes]
                results["stats"]["component_volume_ratios"] = component_volume_ratios
            else:
                results["stats"]["component_volume_ratios"] = []
            
            if n_components > 1:
                largest_component_ratio = max(component_sizes) / n_vertices
                results["stats"]["largest_component_ratio"] = largest_component_ratio
                
                # 计算最大分量体积比例
                if component_volumes and total_volume > 0:
                    largest_volume_ratio = max(component_volumes) / total_volume
                    results["stats"]["largest_volume_ratio"] = largest_volume_ratio
                
                results["warnings"].append(f"mesh有 {n_components} 个连通分量")
                
                # 如果有很多小的碎片，可能是SDF异常
                small_fragments = sum(1 for size in component_sizes if size < n_vertices * 0.01)
                if small_fragments > 100:
                    results["errors"].append(f"检测到 {small_fragments} 个小碎片，可能是SDF异常导致的破碎")
                    results["healthy"] = False
                    return results
                    
    except Exception as e:
        results["warnings"].append(f"连通性分析失败: {e}")
    
    # 4. 检查水密性
    try:
        is_watertight = mesh.is_watertight
        results["stats"]["is_watertight"] = is_watertight
        
        if not is_watertight:
            results["warnings"].append("mesh不是水密的")
            
            # 检查边界边
            boundary_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
            n_boundary_edges = len(boundary_edges)
            results["stats"]["boundary_edges"] = n_boundary_edges
            
            if n_boundary_edges > n_faces * 0.1:  # 边界边太多
                results["errors"].append("边界边过多，可能是SDF不连续导致的空洞")
                results["healthy"] = False
                return results
                
    except Exception as e:
        results["warnings"].append(f"水密性检查失败: {e}")
    
    # 5. 检查法向量一致性
    try:
        # 尝试修复法向量方向
        original_faces = mesh.faces.copy()
        mesh.fix_normals()
        
        # 如果修复后面的顺序改变很多，可能存在法向量问题
        if not np.array_equal(original_faces, mesh.faces):
            results["warnings"].append("法向量方向不一致，已自动修复")
            
    except Exception as e:
        results["warnings"].append(f"法向量检查失败: {e}")
    
    # 6. 检查几何异常
    try:
        # 检查bbox
        bbox_size = mesh.bounding_box.extents
        results["stats"]["bbox_size"] = bbox_size.tolist()
        
        # 检查是否有异常大的bbox (可能是SDF数值异常)
        if np.any(bbox_size > 1000) or np.any(bbox_size < 1e-6):
            results["warnings"].append(f"异常的边界框尺寸: {bbox_size}")
            
        # 检查体积
        if mesh.is_watertight:
            volume = abs(mesh.volume)
            results["stats"]["mesh_volume"] = volume
            
            # 体积异常小可能表示mesh严重破碎
            bbox_volume = np.prod(bbox_size)
            if volume < bbox_volume * 1e-6:
                results["errors"].append("体积异常小，可能是严重破碎的mesh")
                results["healthy"] = False
                return results
        else:
            # 对于非水密mesh，尝试计算convex hull体积作为估计
            try:
                convex_volume = abs(mesh.convex_hull.volume)
                results["stats"]["convex_hull_volume"] = convex_volume
            except:
                results["stats"]["convex_hull_volume"] = 0.0
                
    except Exception as e:
        results["warnings"].append(f"几何检查失败: {e}")
    
    # 7. 检查顶点分布异常
    try:
        # 检查重复顶点
        unique_vertices = np.unique(mesh.vertices, axis=0)
        duplicate_vertices = n_vertices - len(unique_vertices)
        results["stats"]["duplicate_vertices"] = duplicate_vertices
        
        if duplicate_vertices > n_vertices * 0.1:
            results["warnings"].append(f"重复顶点过多: {duplicate_vertices}")
            
        # 检查顶点聚集
        if n_vertices > 100:
            # 随机采样检查顶点间距离
            sample_idx = np.random.choice(n_vertices, min(1000, n_vertices), replace=False)
            sample_vertices = mesh.vertices[sample_idx]
            
            # 计算最近邻距离
            if len(sample_vertices) > 1:
                distances = cdist(sample_vertices, sample_vertices)
                np.fill_diagonal(distances, np.inf)
                min_distances = np.min(distances, axis=1)
                
                # 检查是否有异常近的顶点聚集
                very_close = np.sum(min_distances < 1e-6)
                if very_close > len(sample_vertices) * 0.1:
                    results["warnings"].append("检测到顶点异常聚集")
                    
    except Exception as e:
        results["warnings"].append(f"顶点分布检查失败: {e}")
    
    # 8. 检查面的质量
    try:
        # 检查细长三角形 (每个三角形内部的长宽比)
        vertices = mesh.vertices
        faces = mesh.faces
        
        if len(faces) > 0:
            # 计算每个三角形的三条边长
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            
            edge1_length = np.linalg.norm(v1 - v0, axis=1)
            edge2_length = np.linalg.norm(v2 - v1, axis=1)
            edge3_length = np.linalg.norm(v0 - v2, axis=1)
            
            # 对每个三角形计算最长边和最短边的比例
            max_edge_per_face = np.maximum(np.maximum(edge1_length, edge2_length), edge3_length)
            min_edge_per_face = np.minimum(np.minimum(edge1_length, edge2_length), edge3_length)
            
            # 避免除零
            valid_faces = min_edge_per_face > 1e-12
            if np.sum(valid_faces) > 0:
                edge_ratios = max_edge_per_face[valid_faces] / min_edge_per_face[valid_faces]
                max_edge_ratio = np.max(edge_ratios)
                mean_edge_ratio = np.mean(edge_ratios)
                
                results["stats"]["max_triangle_edge_ratio"] = float(max_edge_ratio)
                results["stats"]["mean_triangle_edge_ratio"] = float(mean_edge_ratio)
                
                # 统计异常细长的三角形
                elongated_faces = np.sum(edge_ratios > 100)
                very_elongated_faces = np.sum(edge_ratios > 1000)
                
                results["stats"]["elongated_triangles"] = int(elongated_faces)
                results["stats"]["very_elongated_triangles"] = int(very_elongated_faces)
                
                if very_elongated_faces > 0:
                    ratio = very_elongated_faces / len(faces)
                    results["warnings"].append(f"发现 {very_elongated_faces} 个极细长三角形 ({ratio:.2%})")
                    if ratio > 0.01:  # 超过1%的极细长三角形
                        results["errors"].append("存在过多极细长三角形，可能是SDF梯度异常")
                        results["healthy"] = False
                        return results
                
                if max_edge_ratio > 10000:
                    results["warnings"].append(f"最大三角形边长比例异常: {max_edge_ratio:.1f}")
                    results["errors"].append("存在极度畸形的三角形，可能是SDF数值异常")
                    results["healthy"] = False
                    return results
                
    except Exception as e:
        results["warnings"].append(f"面质量检查失败: {e}")
    
    # 总结
    if verbose:
        print(f"\n=== Mesh健康检查结果 ===")
        print(f"总体状态: {'健康' if results['healthy'] else '异常'}")
        
        if results["warnings"]:
            print(f"\n警告 ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  - {warning}")
                
        if results["errors"]:
            print(f"\n错误 ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"  - {error}")
                
        print(f"\n统计信息:")
        for key, value in results["stats"].items():
            print(f"  {key}: {value}")
    
    return results


def analyze_sdf_mc_consistency(sdf_volume, mesh_vertices, mesh_faces, iso_value=0.0, voxel_size=1.0, verbose=True):
    """
    分析SDF和由其生成的mesh之间的一致性
    
    Args:
        sdf_volume: np.array, SDF体素网格
        mesh_vertices: np.array, mesh顶点
        mesh_faces: np.array, mesh面
        iso_value: float, marching cubes的等值面值
        voxel_size: float, 体素尺寸
        verbose: bool, 是否输出详细信息
        
    Returns:
        dict: 一致性分析结果
    """
    
    results = {
        "consistent": True,
        "issues": [],
        "stats": {}
    }
    
    if len(mesh_vertices) == 0:
        results["issues"].append("mesh为空")
        results["consistent"] = False
        return results
    
    try:
        # 将mesh顶点坐标转换为SDF体素坐标
        mesh_coords = mesh_vertices / voxel_size
        
        # 检查mesh是否在SDF范围内
        sdf_shape = np.array(sdf_volume.shape)
        
        out_of_bounds = np.any(mesh_coords < 0, axis=1) | np.any(mesh_coords >= sdf_shape, axis=1)
        oob_ratio = np.sum(out_of_bounds) / len(mesh_vertices)
        results["stats"]["out_of_bounds_ratio"] = float(oob_ratio)
        
        if oob_ratio > 0.1:
            results["issues"].append(f"{oob_ratio:.2%} 的mesh顶点超出SDF范围")
            results["consistent"] = False
        
        # 对在范围内的顶点进行采样检查
        valid_mask = ~out_of_bounds
        if np.sum(valid_mask) > 0:
            valid_coords = mesh_coords[valid_mask]
            
            # 三线性插值获取SDF值
            sdf_values_at_vertices = map_coordinates(
                sdf_volume, 
                valid_coords.T, 
                order=1, 
                mode='nearest'
            )
            
            # 检查SDF值是否接近等值面
            distance_to_iso = np.abs(sdf_values_at_vertices - iso_value)
            mean_distance = np.mean(distance_to_iso)
            max_distance = np.max(distance_to_iso)
            
            results["stats"]["mean_distance_to_iso"] = float(mean_distance)
            results["stats"]["max_distance_to_iso"] = float(max_distance)
            
            # 如果顶点SDF值偏离等值面太远，说明SDF或MC有问题
            far_from_iso = distance_to_iso > voxel_size * 2
            far_ratio = np.sum(far_from_iso) / len(distance_to_iso)
            results["stats"]["far_from_iso_ratio"] = float(far_ratio)
            
            if far_ratio > 0.2:
                results["issues"].append(f"{far_ratio:.2%} 的顶点远离SDF等值面")
                results["consistent"] = False
            
            # 检查SDF梯度的连续性
            gradient = np.gradient(sdf_volume)
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
            
            # 在mesh顶点位置采样梯度
            gradient_at_vertices = map_coordinates(
                gradient_magnitude,
                valid_coords.T,
                order=1,
                mode='nearest'
            )
            
            # 检查梯度异常 (过大或过小)
            normal_gradient_mask = (gradient_at_vertices > 0.1) & (gradient_at_vertices < 10)
            abnormal_gradient_ratio = 1 - np.sum(normal_gradient_mask) / len(gradient_at_vertices)
            results["stats"]["abnormal_gradient_ratio"] = float(abnormal_gradient_ratio)
            
            if abnormal_gradient_ratio > 0.1:
                results["issues"].append(f"{abnormal_gradient_ratio:.2%} 的位置SDF梯度异常")
                results["consistent"] = False
        
        # 检查SDF的整体质量
        sdf_stats = {
            "min": float(np.min(sdf_volume)),
            "max": float(np.max(sdf_volume)),
            "mean": float(np.mean(sdf_volume)),
            "std": float(np.std(sdf_volume))
        }
        results["stats"]["sdf_stats"] = sdf_stats
        
        # 检查是否有NaN或Inf
        nan_count = np.sum(np.isnan(sdf_volume))
        inf_count = np.sum(np.isinf(sdf_volume))
        
        if nan_count > 0:
            results["issues"].append(f"SDF包含 {nan_count} 个NaN值")
            results["consistent"] = False
            
        if inf_count > 0:
            results["issues"].append(f"SDF包含 {inf_count} 个Inf值")
            results["consistent"] = False
        
    except Exception as e:
        results["issues"].append(f"一致性分析失败: {e}")
        results["consistent"] = False
    
    if verbose:
        print(f"\n=== SDF-Mesh一致性分析 ===")
        print(f"一致性: {'正常' if results['consistent'] else '异常'}")
        
        if results["issues"]:
            print(f"\n发现的问题:")
            for issue in results["issues"]:
                print(f"  - {issue}")
        
        print(f"\n统计信息:")
        for key, value in results["stats"].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    return results


def quick_mesh_check(mesh_path_or_vertices_faces):
    """
    快速检查mesh是否明显异常
    
    Returns:
        bool: True表示mesh看起来正常，False表示有明显问题
    """
    try:
        result = check_mesh_health(mesh_path_or_vertices_faces, verbose=True)
        return result["healthy"] and len(result["errors"]) == 0
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh健康检查工具')
    parser.add_argument('mesh_path', type=str, help='mesh文件路径')
    parser.add_argument('--verbose', action='store_true', default=False, help='输出详细信息')
    
    args = parser.parse_args()
    
    mesh_path = args.mesh_path
    verbose = args.verbose
    
    if not os.path.exists(mesh_path):
        print(f"错误: 文件不存在 {mesh_path}")
        sys.exit(1)
    
    print(f"检查mesh文件: {mesh_path}")
    print("=" * 50)
    
    try:
        # 执行mesh健康检查
        result = check_mesh_health(mesh_path, verbose=verbose)
        
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
        
        # 确定保存路径
        mesh_dir = os.path.dirname(mesh_path)
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        info_path = os.path.join(mesh_dir, f"{mesh_name}_info.json")
        
        # 保存结果到info.json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n检查完成!")
        print(f"结果: {'健康' if result['healthy'] else '异常'}")
        print(f"结果已保存到: {info_path}")
        
        # 如果有错误，以非零状态码退出
        if not result["healthy"]:
            print(f"\n发现 {len(result['errors'])} 个错误，{len(result['warnings'])} 个警告")
            sys.exit(1)
        else:
            print(f"\nmesh健康，发现 {len(result['warnings'])} 个警告")
            sys.exit(0)
            
    except Exception as e:
        error_result = {
            "mesh_path": mesh_path,
            "healthy": False,
            "warnings": [],
            "errors": [f"检查过程中发生异常: {str(e)}"],
            "stats": {},
            "check_timestamp": __import__('time').time(),
            "check_date": __import__('datetime').datetime.now().isoformat()
        }
        
        # 保存错误结果
        mesh_dir = os.path.dirname(mesh_path)
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        info_path = os.path.join(mesh_dir, f"{mesh_name}_info.json")
        
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            print(f"\n检查失败: {e}")
            print(f"错误信息已保存到: {info_path}")
        except Exception as save_error:
            print(f"\n检查失败: {e}")
            print(f"保存错误信息也失败: {save_error}")
        
        sys.exit(1)
