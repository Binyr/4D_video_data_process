for traj_id in 0 
do
    CUDA_VISIBLE_DEVICES=0 /group/40034/yanruibin/projects/blender-4.2.18-linux-x64/blender --background --python tools/extract_mesh_camera_sparse_voxel_v4_interleave_light_optimized_128frame.py -- \
    --object_path data/objverse_minghao_4d/glbs/000-033/5dd2ce713485413a84bceacf15e40b9f.glb \
    --output_file vis/rendering_v5/000-033_static_camera_distance_v3/5dd2ce713485413a84bceacf15e40b9f_traj_$traj_id/result.json \
    --resolution 1024 \
    --render_engine CYCLES \
    --cycles_backend CUDA \
    --cycles_device CPU \
    --transparent_bg \
    --traj_id $traj_id \
    --traj_seed 123 \
    --num_cameras 16 \
    --camera_frame_padding 0.0 \
    --camera_fit_safety 1.0 \
    --camera_distance_jitter_scale 1.0 \
    --randomize_camera_intrinsics \
    --camera_fov_min_deg 30 \
    --camera_fov_max_deg 70 \
    --camera_sensor_size 36 \
    --sunlight_prob 0.5
    # --motion_info_root motion_info \
    # --motion_rms_threshold 0.001
    # --hdr_strength 0.2
    # --render_scene_box 
    # --normalized_glb_path out.glb \
    # --export_keyframe_ply \

    # /efs/yanruibin/projects/blender-4.2.1-linux-x64/4.2/python/bin/python3.11

    python tools/folder_to_mp4.py \
        --input_dir vis/rendering_v5/000-033_static_camera_distance_v3/5dd2ce713485413a84bceacf15e40b9f_traj_$traj_id/result_rgb/view_00 \
        --output_path vis/rendering_v5/000-033_static_camera_distance_v3/5dd2ce713485413a84bceacf15e40b9f_traj_${traj_id}/rgb_view_00.mp4 \
        --fps 30
done
# data/objverse_minghao_4d/glbs/000-000/0032696f5871429fbd0549d9628f812c.glb \

# python tools/concat_videos_width.py \
#     --inputs vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_0/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_1/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_2/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_3/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_4/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_5/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_6/rgb.mp4  --out vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_0123456.mp4


CUDA_VISIBLE_DEVICES=0 /group/40034/yanruibin/projects/blender-4.2.18-linux-x64/blender --background --python tools/extract_mesh_camera_sparse_voxel_v4_interleave_light_optimized_128frame_mp4_trans.py -- \
    --object_path data/objverse_minghao_4d/glbs/000-000/023111dd9b9741628094019cd26085e6.glb \
    --output_file data/objverse_minghao_4d_mine_40075/rendering_v5/000-000_static_camera_distance_v3/023111dd9b9741628094019cd26085e6/result.json \
    --resolution 1024 \
    --render_engine CYCLES \
    --cycles_backend CUDA \
    --cycles_device CPU \
    --transparent_bg \
    --traj_seed 6360651 \
    --num_cameras 1 \
    --camera_frame_padding 0.0 \
    --camera_fit_safety 1.0 \
    --camera_distance_jitter_scale 1.0 \
    --randomize_camera_intrinsics \
    --camera_fov_min_deg 30 \
    --camera_fov_max_deg 70 \
    --camera_sensor_size 36 \
    --sunlight_prob 0.5 \
    # --max_frame 5

# id=21fea5ccd76a40d2b8c195b49622b99e
oid=000-000/02959ad818784bac87a52f4f4c966b4d
oid=000-000/02bccd08cc51480c9bc00afb7e17f164
CUDA_VISIBLE_DEVICES=0 /group/40034/yanruibin/projects/blender-4.2.18-linux-x64/blender --background --python tools/extract_mesh_camera_sparse_voxel_v4_interleave_light_optimized_128frame_mp4_trans.py -- \
    --object_path data/objverse_minghao_4d/glbs/${oid}.glb \
    --output_file vis/rendering_v6_denoise/${oid}/result.json \
    --resolution 1024 \
    --render_engine CYCLES \
    --cycles_backend CUDA \
    --cycles_device CPU \
    --transparent_bg \
    --traj_seed 1229612 \
    --num_cameras 1 \
    --camera_frame_padding 0.0 \
    --camera_fit_safety 1.0 \
    --camera_distance_jitter_scale 1.0 \
    --randomize_camera_intrinsics \
    --camera_fov_min_deg 30 \
    --camera_fov_max_deg 70 \
    --camera_sensor_size 36 \
    --sunlight_prob 0.5 \
    --max_frame 48 --cycles_use_denoising

oid=000-000/02959ad818784bac87a52f4f4c966b4d
oid=000-000/02bccd08cc51480c9bc00afb7e17f164 # people
oid=000-000/01d49e77147a4b57ba3f10eccb949803
CUDA_VISIBLE_DEVICES=0 /group/40034/yanruibin/projects/blender-4.2.18-linux-x64/blender --background --python tools/extract_mesh_camera_sparse_voxel_v4_interleave_light_optimized_128frame_mp4_no_perframe_center.py -- \
    --object_path data/objverse_minghao_4d/glbs/${oid}.glb \
    --output_file vis/rendering_v6_denoise_nocenter/${oid}/result.json \
    --resolution 1024 \
    --render_engine CYCLES \
    --cycles_backend CUDA \
    --cycles_device GPU \
    --transparent_bg \
    --traj_seed 1229612 \
    --num_cameras 4 \
    --camera_frame_padding 0.0 \
    --camera_fit_safety 1.0 \
    --camera_distance_jitter_scale 1.0 \
    --randomize_camera_intrinsics \
    --camera_fov_min_deg 30 \
    --camera_fov_max_deg 70 \
    --camera_sensor_size 36 \
    --sunlight_prob 0.5 \
    --max_frame 48 --cycles_samples 256 \
    --no_render_normal_map \
    --camera_stride 2 \
    
