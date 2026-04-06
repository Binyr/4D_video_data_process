name=5dd2ce713485413a84bceacf15e40b9f
view=top
python tools/vis_objverse_mesh.py   \
    --root_dir data/$name/render_cond   \
    --out vis/${name}_${view}.mp4   \
    --fps 10 --width 1024 --height 1024   --view $view   --use_global_bbox

view=side
python tools/vis_objverse_mesh.py   \
    --root_dir data/$name/render_cond   \
    --out vis/${name}_${view}.mp4   \
    --fps 10 --width 1024 --height 1024   --view $view   --use_global_bbox