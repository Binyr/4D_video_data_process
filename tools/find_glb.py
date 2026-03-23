from pathlib import Path
f_path = Path(__file__).absolute()
p_path = f_path.parents[1]
import sys
sys.path.insert(0, str(p_path))
from pathlib import Path

root = Path("data/objverse_minghao_4d/glbs")
target = "5dd2ce713485413a84bceacf15e40b9f.glb"

matches = list(root.rglob(target))

if len(matches) == 0:
    print(f"Not found: {target}")
else:
    print(f"Found {len(matches)} match(es):")
    for p in matches:
        print(p)