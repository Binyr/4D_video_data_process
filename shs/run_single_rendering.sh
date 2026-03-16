for traj_id in 0 1
do
    /efs/yanruibin/projects/blender-4.2.1-linux-x64/blender --background --python tools/extract_mesh_camera_sparse_voxel.py -- \
    --object_path data/objverse_minghao_4d/glbs/000-033/5dd2ce713485413a84bceacf15e40b9f.glb \
    --output_file vis/rendering_v2/000-033_static_test/5dd2ce713485413a84bceacf15e40b9f_traj_$traj_id/result.json \
    --resolution 1024 \
    --render_engine CYCLES \
    --transparent_bg \
    --traj_id $traj_id \
    --traj_seed 123

    # /efs/yanruibin/projects/blender-4.2.1-linux-x64/4.2/python/bin/python3.11

    python tools/folder_to_mp4.py \
        --input_dir vis/rendering_v2/000-033_static_test/5dd2ce713485413a84bceacf15e40b9f_traj_$traj_id/result_rgb/ \
        --output_path vis/rendering_v2/000-033_static_test/5dd2ce713485413a84bceacf15e40b9f_traj_$traj_id/rgb.mp4 \
        --fps 10
done
# data/objverse_minghao_4d/glbs/000-000/0032696f5871429fbd0549d9628f812c.glb \

# python tools/concat_videos_width.py \
#     --inputs vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_0/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_1/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_2/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_3/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_4/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_5/rgb.mp4 vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_traj_6/rgb.mp4  --out vis/rendering/000-033_static/5dd2ce713485413a84bceacf15e40b9f_0123456.mp4


