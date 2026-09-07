from __future__ import annotations

import argparse
import json
from pathlib import Path

EXCLUDE_DIRS = {"meta", "data", "videos"}


def _find_task_dirs(root: Path) -> list[str]:
    task_dirs = [p.name for p in root.iterdir() if p.is_dir() and p.name not in EXCLUDE_DIRS]
    return sorted(task_dirs)


def _write_tasks(tasks_path: Path, task_name: str, num_tasks: int) -> None:
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w", encoding="utf-8") as f:
        for idx in range(num_tasks):
            record = {"task_index": idx, "task": task_name}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _count_existing_tasks(tasks_path: Path) -> int:
    if not tasks_path.exists():
        return 1
    with tasks_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())





def main() -> None:
    parser = argparse.ArgumentParser(description="Fill tasks.jsonl from task-named subdirectories.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/pick_place_sample"),
        help="Dataset root containing task-named subdirectories.",
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=None,
        help=(
            "Optional tasks.jsonl path override. If omitted, each task directory's meta/tasks.jsonl is filled."
        ),
    )
    args = parser.parse_args()

    root = args.root

    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    task_dirs = _find_task_dirs(root)
    if not task_dirs:
        raise RuntimeError(
            f"No task directories found under {root}. "
            "Expected subdirectories like 'clean_the_table'."
        )

    for task_dir in task_dirs:
        tasks_path = args.tasks_path or (root / task_dir / "meta" / "tasks.jsonl")
        num_tasks = _count_existing_tasks(tasks_path)
        _write_tasks(tasks_path, task_dir, num_tasks)
        print(f"Wrote {num_tasks} tasks to {tasks_path}")


if __name__ == "__main__":
    main()

