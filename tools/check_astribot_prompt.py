#!/usr/bin/env python3
import json
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

TASKS = [
    "oven_put",
    "fridge",
    "dishwasher",
    "oven_takeout",
    "collect_clothes",
    "dry_clothes",
    "wash_clothes",
]

def load_catalog():
    shell = '''
source ../configs/data/astribot_dataset_catalog.sh
for task in oven_put fridge dishwasher oven_takeout collect_clothes dry_clothes wash_clothes; do
    while IFS= read -r dir; do
        printf "%s\\t%s\\n" "$task" "$dir"
    done <<< "${ASTRIBOT_DATASET_CATALOG[$task]}"
done
'''
    out = subprocess.check_output(["bash", "-c", shell], text=True)

    catalog = {task: [] for task in TASKS}
    for line in out.splitlines():
        task, path = line.split("\t", 1)
        if path:
            catalog[task].append(Path(path))
    return catalog

def read_tasks_parquet(path):
    df = pd.read_parquet(path)

    if df.index.name is None and "__index_level_0__" in df.columns:
        df = df.set_index("__index_level_0__")

    return [str(x).strip() for x in df.index if str(x).strip()]

def read_tasks_jsonl(path):
    tasks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task = str(item.get("task", "")).strip()
            if task:
                tasks.append(task)
    return tasks

def read_prompts(dataset_dir):
    meta = dataset_dir / "meta"
    parquet = meta / "tasks.parquet"
    jsonl = meta / "tasks.jsonl"

    if parquet.exists():
        return read_tasks_parquet(parquet)
    if jsonl.exists():
        return read_tasks_jsonl(jsonl)
    return []

def main():
    catalog = load_catalog()
    global_prompts = set()

    for task in TASKS:
        counter = Counter()
        missing = []

        for dataset_dir in catalog[task]:
            prompts = read_prompts(dataset_dir)
            if not prompts:
                missing.append(str(dataset_dir))
                continue

            counter.update(set(prompts))
            global_prompts.update(prompts)

        print(f"\n===== {task} =====")
        print(f"datasets: {len(catalog[task])}")

        for prompt, count in counter.items():
            print(f"[{count:3d} datasets] {prompt}")

        if missing:
            print("MISSING/EMPTY PROMPT:")
            for path in missing:
                print(f"  {path}")

    print("\n===== SUMMARY =====")
    print(f"catalog tasks : {len(TASKS)}")
    print(f"unique prompts: {len(global_prompts)}")

if __name__ == "__main__":
    main()
