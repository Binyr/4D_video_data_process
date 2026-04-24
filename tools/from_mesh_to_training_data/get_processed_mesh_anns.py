from pathlib import Path
p_path = Path(__file__).parent.parent.parent.parent
import sys
sys.path.append(str(p_path))

import json
from tqdm import tqdm

object_ann_path = Path("data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json")
latent_root = Path("data/train_data_40075_objverse_minghao_4d_mine_rendering_v5_512_8camera")

view_anns = {}
with open(object_ann_path, "r") as f:
    anns = json.load(f)
    for split in ["train", "test"]:
        for obj_path in tqdm(anns[split], desc=f"Processing {split} objects"):
            bl_meta_path = Path(obj_path) / "result.json"
            with open(bl_meta_path, "r") as f:
                bl_meta = json.load(f)
                num_frames = bl_meta["_global"]["num_frames"]
            
            shard_name = Path(obj_path).parent.name
            obj_name = Path(obj_path).name

            latent_obj_path = latent_root / shard_name / obj_name
            if not latent_obj_path.exists():
                continue
            
            finished_latent_views = [x for x in latent_obj_path.iterdir() if x.is_dir()]
            finished_latent_views.sort(key=lambda x: int(x.name.split("_")[1]))

            # to see if last view is finished
            finished_sparse_sdfs = [x for x in finished_latent_views[-1].iterdir() if x.name.startswith("sparse_sdf")]
            if len(finished_sparse_sdfs) == num_frames:
                pass
            else:
                finished_latent_views = finished_latent_views[:-1]
            
            for latent_view_dir in finished_latent_views:
                view_anns[str(latent_view_dir)] = num_frames

print(f"Total {len(view_anns)} views have been processed.")

with open("data/processed_mesh_anns_0418.json", "w") as f:
    json.dump(view_anns, f, indent=4)

            

