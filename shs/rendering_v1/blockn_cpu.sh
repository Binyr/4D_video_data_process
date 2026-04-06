cd /group/40034/yanruibin/projects/4D_video_data_process

num_chunks=$1
chunk_id=$2
num_workers=$3
python tools/submit_extract_mesh_camera_sparse_voxel_cuda.py \
    --root_glb_dir data/objverse_minghao_4d/glbs \
    --output_root /group/40075/yanruibin/objverse_minghao_4d_mine/rendering_v5 \
    --output_parent_suffix _static_camera_distance_v3 \
    --worker_script tools/extract_mesh_camera_sparse_voxel_v4_interleave_light_optimized_128frame.py \
    --blender_path /group/40034/yanruibin/projects/blender-4.2.18-linux-x64/blender \
    --num_chunks $num_chunks \
    --chunk_id $chunk_id \
    --num_workers $num_workers \
    --cuda_devices 0 1 2 3 4 5 6 7\
    --resolution 1024 \
    --render_engine CYCLES \
    --cycles_device CPU \
    --transparent_bg \
    --traj_seed 123 \
    --num_cameras 16 \
    --camera_frame_padding 0.0 \
    --camera_fit_safety 1.0 \
    --camera_distance_jitter_scale 1.0 \
    --randomize_camera_intrinsics \
    --camera_fov_min_deg 30 \
    --camera_fov_max_deg 70 \
    --camera_sensor_size 36 \
    --sunlight_prob 0.5 \
    --extra_worker_args 