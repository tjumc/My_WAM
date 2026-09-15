#!/usr/bin/env python3
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

shell = '''
source configs/data/astribot_dataset_catalog.sh
for task in fridge dishwasher oven_put oven_takeout collect_clothes dry_clothes wash_clothes; do
    while IFS= read -r dir; do
        printf "%s|%s\\n" "$task" "$dir"
    done <<< "${ASTRIBOT_DATASET_CATALOG[$task]}"
done
'''

lines = subprocess.check_output(["bash", "-c", shell], text=True).splitlines()

for task in TASKS:
    print(f"\n===== {task} =====")
    rows = []

    for line in lines:
        t, path = line.split("|", 1)
        if t != task or not path:
            continue

        with open(Path(path) / "meta/info.json") as f:
            info = json.load(f)

        episodes = info["total_episodes"]
        frames = info["total_frames"]
        avg_len = frames / episodes
        rows.append((Path(path).name, episodes, frames, avg_len))

    rows.sort(key=lambda x: x[1])

    print(f"{'dataset':<55} {'episodes':>8} {'frames':>10} {'avg_len':>8}")
    for name, episodes, frames, avg_len in rows:
        print(f"{name:<55} {episodes:>8} {frames:>10} {avg_len:>8.1f}")
