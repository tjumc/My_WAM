import json
import subprocess
from pathlib import Path

TASKS = [
    "fridge",
    "dishwasher",
    "oven_put",
    "oven_takeout",
    "collect_clothes",
    "dry_clothes",
    "wash_clothes",
]

cmd = r'''
source configs/data/astribot_dataset_catalog.sh
for task in fridge dishwasher oven_put oven_takeout collect_clothes dry_clothes wash_clothes; do
    while IFS= read -r dir; do
        echo "$task|$dir"
    done <<< "${ASTRIBOT_DATASET_CATALOG[$task]}"
done
'''

lines = subprocess.check_output(["bash", "-c", cmd], text=True).splitlines()

stats = {t: {"datasets": 0, "episodes": 0, "frames": 0} for t in TASKS}

for line in lines:
    task, path = line.split("|", 1)

    with open(Path(path) / "meta/info.json") as f:
        info = json.load(f)

    stats[task]["datasets"] += 1
    stats[task]["episodes"] += info["total_episodes"]
    stats[task]["frames"] += info["total_frames"]

total_frames = sum(x["frames"] for x in stats.values())

print(f"{'Task':<18} {'Datasets':>8} {'Episodes':>10} {'Frames':>12} {'Ratio':>8}")

for task, x in stats.items():
    ratio = x["frames"] / total_frames * 100
    print(
        f"{task:<18} "
        f"{x['datasets']:>8} "
        f"{x['episodes']:>10} "
        f"{x['frames']:>12} "
        f"{ratio:>7.2f}%"
    )

print(f"\nTotal frames: {total_frames}")