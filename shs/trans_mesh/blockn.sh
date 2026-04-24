num_chunks=$1
chunk_id=$2 

/efs/common/conda_envs/yrb_pixal3d/bin/python tools/from_mesh_to_training_data/v5_mp_h200_direct_netdisk.py \
  --splits train test \
  --gpu_ids 0 1 \
  --resolution 512 \
  --num_chunks $num_chunks \
  --chunk_id $chunk_id \
  --save_failures \
  --view_stride 2 \
  --local_scratch_root /tmp/mesh2sdf_netdisk_scratch \
  --only_unfinished_object \