# File: process_geometry_with_bpy.py
# Description: This script uses Blender (bpy) for scene setup and animation,
# and Trimesh for geometric processing to generate consistent, watertight point clouds.
# It is designed to be run from the command line with Blender.
# Example: blender --background --python process_geometry_with_bpy.py -- --object_path /path/to/model.glb ...

import argparse
import sys
import os
import math
import glob
from typing import *

# Blender-specific imports
import bpy
from mathutils import Vector

# Geometry processing imports
import numpy as np
import trimesh
import torch
import igl
import cubvh
from tqdm import tqdm

# =================================================================================
#  1. GEOMETRIC PROCESSING HELPERS (Trimesh, PyTorch, etc.)
# =================================================================================

def sample_surface_with_barycentric(mesh, count):
    """
    Samples points on the mesh surface and returns their face indices and barycentric coordinates.
    """
    points, face_indices = mesh.sample(count, return_index=True)
    triangles = mesh.vertices[mesh.faces[face_indices]]
    barycentric_coords = trimesh.triangles.points_to_barycentric(triangles, points)
    return points, face_indices, barycentric_coords

def Watertight_cubvh(V, F, grid_res=256):
    """
    Uses cuBVH and flood fill to create a high-quality watertight mesh.
    Requires a CUDA-enabled GPU to run efficiently.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: CUDA not available, Watertight_cubvh will run on CPU and may be slow.")
    
    V_torch = torch.from_numpy(V).float().to(device)
    F_torch = torch.from_numpy(F).int().to(device)

    min_corner = V_torch.min(axis=0)[0]
    max_corner = V_torch.max(axis=0)[0]
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    grid_points = torch.stack(
        torch.meshgrid(
            torch.linspace(min_corner[0], max_corner[0], grid_res, device=device),
            torch.linspace(min_corner[1], max_corner[1], grid_res, device=device),
            torch.linspace(min_corner[2], max_corner[2], grid_res, device=device),
            indexing="ij",
        ), dim=-1,
    ).float()

    BVH = cubvh.cuBVH(V_torch, F_torch)
    udf, _, _ = BVH.unsigned_distance(grid_points.reshape(-1, 3), return_uvw=False)
    udf = udf.reshape(grid_res, grid_res, grid_res).contiguous()
    
    occ = udf < (2 / grid_res)
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    occ_mask = ~empty_mask
    
    sdf = udf.clone()
    sdf[occ_mask] *= -1

    mc_verts, mc_faces = igl.marching_cubes(
        -sdf.cpu().numpy().ravel(), 
        grid_points.cpu().numpy().reshape(-1, 3), 
        grid_res, grid_res, grid_res, 0.0
    )[:2]
    return mc_verts, mc_faces

# =================================================================================
#  2. BLENDER SCENE SETUP FUNCTIONS (from render4d_32.py)
# =================================================================================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    # Add other formats if needed
}

def init_scene() -> None:
    """Resets the Blender scene to a clean state."""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path: str) -> None:
    """Loads a model into the scene."""
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    import_function = IMPORT_FUNCTIONS[file_extension]
    import_function(filepath=object_path)

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_bbox_robust() -> Tuple[Vector, Vector]:
    """Calculates the precise world-space bounding box for all visible meshes."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found_mesh = False
    
    bpy.context.view_layer.update() # Ensure matrices are up to date
    
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() 
                    if isinstance(obj.data, bpy.types.Mesh) and obj.visible_get()]
    if not scene_meshes:
        raise RuntimeError("No visible mesh objects found to compute bounding box.")

    for obj in scene_meshes:
        found_mesh = True
        for vert in obj.data.vertices:
            world_coord = obj.matrix_world @ vert.co
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, world_coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, world_coord))
            
    if not found_mesh:
        raise RuntimeError("No objects in scene to compute bounding box for")
        
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene to fit in a unit cube, exactly like the render script."""
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    
    if len(scene_root_objects) > 1:
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        for obj in scene_root_objects:
            obj.parent = parent_empty
        scene_root = parent_empty
    elif len(scene_root_objects) == 1:
        scene_root = scene_root_objects[0]
    else:
        raise RuntimeError("No root objects found in the scene.")

    bbox_min, bbox_max = scene_bbox()
    
    box_size = max(bbox_max - bbox_min)
    scale = 1.0 / box_size if box_size > 1e-6 else 1.0

    scene_root.scale = scene_root.scale * scale

    bpy.context.view_layer.update()
    
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene_root.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

# =================================================================================
#  3. CORE WORKER FUNCTION (Orchestrator)
# =================================================================================

def process_geometry(args):
    """
    The main function that orchestrates bpy scene setup and trimesh geometry processing.
    """
    print("--- Starting Geometry Processing ---")
    
    # --- Setup Paths ---
    output_file = args.output_file
    if os.path.exists(output_file):
        print(f"✅ Skipped (output file already exists): {output_file}")
        return

    # --- 1. Scene Setup using bpy (Identical to render script) ---
    print(f"Loading object from: {args.object_path}")
    init_scene()
    load_object(args.object_path)
    print("Normalizing scene...")
    # The scale and offset are calculated here by bpy, ensuring consistency.
    normalize_scene() 
    print("Scene setup and normalization complete.")
    
    # --- 2. One-Time Sampling on Rest-Pose using trimesh ---
    print("Performing one-time sampling on the normalized rest-pose mesh...")
    # Get the main mesh object from the Blender scene
    mesh_obj = next((obj for obj in bpy.context.scene.objects if obj.type == 'MESH'), None)
    if not mesh_obj:
        raise RuntimeError("No mesh object found after loading.")

    # Convert the normalized rest-pose bpy mesh to a trimesh object
    bpy.context.view_layer.update()
    temp_mesh_data = mesh_obj.to_mesh()
    verts = np.array([v.co for v in temp_mesh_data.vertices])
    # Ensure faces are triangulated for trimesh
    temp_mesh_data.calc_loop_triangles()
    faces = np.array([lt.vertices for lt in temp_mesh_data.loop_triangles])
    
    rest_pose_trimesh = trimesh.Trimesh(vertices=verts, faces=faces)
    bpy.data.meshes.remove(temp_mesh_data) # Clean up temporary data

    # Perform barycentric sampling
    _, face_indices, bary_coords = sample_surface_with_barycentric(rest_pose_trimesh, count=args.sample_count)
    print(f"Sampled {args.sample_count} points.")

    # --- 3. Timeline Remapping and Per-Frame Processing ---
    print(f"Remapping animation to {args.target_frames} frames and processing...")
    final_watertight_data = {}
    
    # Get original animation frame range from bpy actions
    actions = bpy.data.actions
    frame_start, frame_end = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
    if actions:
        ranges = [action.frame_range for action in actions]
        frame_start = int(min(r[0] for r in ranges))
        frame_end = int(max(r[1] for r in ranges))
    source_duration = float(frame_end - frame_start)

    for i in tqdm(range(args.target_frames), desc="Processing Frames"):
        # This time remapping logic is IDENTICAL to your render script
        if source_duration > 1e-6:
            progress = i / (args.target_frames - 1) if args.target_frames > 1 else 0.0
            source_frame = frame_start + progress * source_duration
        else:
            source_frame = float(frame_start)
        
        # Apply motion using bpy
        bpy.context.scene.frame_set(int(round(source_frame)))

        # Get deformed (and already normalized) mesh data from bpy
        bpy.context.view_layer.update()
        temp_mesh_data = mesh_obj.to_mesh()
        deformed_vertices = np.array([v.co for v in temp_mesh_data.vertices])
        # Faces remain the same, but we get them again for safety
        temp_mesh_data.calc_loop_triangles()
        deformed_faces = np.array([lt.vertices for lt in temp_mesh_data.loop_triangles])
        bpy.data.meshes.remove(temp_mesh_data)
        
        # Reconstruct consistent "guide" points in their new positions
        # Note: We use the rest-pose topology (face_indices) with deformed vertex positions
        triangles = deformed_vertices[rest_pose_trimesh.faces[face_indices]]
        guide_points = trimesh.triangles.barycentric_to_points(triangles, bary_coords)
        
        # Create watertight version of the current frame's mesh
        mc_verts, mc_faces = Watertight_cubvh(deformed_vertices, deformed_faces, grid_res=args.grid_res)
        
        if mc_verts.shape[0] < 3:
            print(f"Warning: Watertight failed for frame {i}. Skipping.")
            continue
        
        watertight_mesh = trimesh.Trimesh(vertices=mc_verts, faces=mc_faces)

        # Project the guide points onto the watertight surface
        projected_points, _, face_indices_wt = watertight_mesh.nearest.on_surface(guide_points)
        projected_normals = watertight_mesh.face_normals[face_indices_wt]

        surface_data = np.concatenate((projected_points, projected_normals), axis=1).astype(np.float16)
        
        frame_key = f"frame_{i:04d}"
        final_watertight_data[frame_key] = {"random_surface": surface_data}
    
    # --- 4. Save Final Data ---
    print(f"Saving final data to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, **final_watertight_data)
    print("--- Processing Complete ---")

# =================================================================================
#  4. MAIN EXECUTION BLOCK
# =================================================================================

if __name__ == "__main__":
    # Remove Blender's arguments to parse our own
    argv = sys.argv[sys.argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser(description='Generate consistent watertight point clouds from animated GLB files using Blender for scene setup.')
    parser.add_argument('--object_path', type=str, required=True,
                        help='Path to the source .glb file.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output .npz file.')
    parser.add_argument('--grid_res', type=int, default=256,
                        help='Resolution for the watertight grid.')
    parser.add_argument('--sample_count', type=int, default=32768,
                        help='Number of points to sample on the surface.')
    parser.add_argument('--target_frames', type=int, default=32,
                        help='The number of frames to resample the animation to.')
    
    args = parser.parse_args(argv)
    
    process_geometry(args)