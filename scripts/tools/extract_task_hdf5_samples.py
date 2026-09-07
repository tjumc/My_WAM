#!/usr/bin/env python3
"""Copy representative HDF5 files for tasks in the Astribot catalog."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "configs/data/astribot_dataset_catalog.sh"
HDF5_SUFFIXES = {".h5", ".hdf5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy one or more representative HDF5 files for each task."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Directory that will receive the copied HDF5 files and manifest.tsv.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Dataset catalog path (default: {DEFAULT_CATALOG}).",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Only extract this task. Repeat the option to select multiple tasks.",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Number of HDF5 files copied for each task (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination files that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List catalog tasks and directory counts, then exit.",
    )
    args = parser.parse_args()

    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be at least 1")
    if not args.list and args.output_dir is None:
        parser.error("output_dir is required unless --list is used")
    return args


def load_catalog(catalog_path: Path) -> dict[str, list[Path]]:
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Dataset catalog not found: {catalog_path}")

    bash_code = r'''
source "$1"
for task_name in "${ASTRIBOT_DATASET_TASKS[@]}"; do
  mapfile -t task_dirs <<< "${ASTRIBOT_DATASET_CATALOG[${task_name}]}"
  for dataset_dir in "${task_dirs[@]}"; do
    printf '%s\0%s\0' "${task_name}" "${dataset_dir}"
  done
done
'''
    result = subprocess.run(
        ["bash", "-c", bash_code, "catalog-reader", str(catalog_path.resolve())],
        check=True,
        stdout=subprocess.PIPE,
    )
    fields = result.stdout.decode("utf-8").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    if len(fields) % 2:
        raise RuntimeError("Dataset catalog returned an incomplete task/path record")

    catalog: dict[str, list[Path]] = {}
    for index in range(0, len(fields), 2):
        task_name, dataset_dir = fields[index : index + 2]
        catalog.setdefault(task_name, []).append(Path(dataset_dir))
    return catalog


def iter_hdf5_files(dataset_dirs: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            print(f"[WARN] Dataset directory not found: {dataset_dir}", file=sys.stderr)
            continue

        for root, dirnames, filenames in os.walk(dataset_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                source = Path(root) / filename
                if source.suffix.lower() not in HDF5_SUFFIXES:
                    continue
                resolved = source.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield source


def select_tasks(
    catalog: dict[str, list[Path]], requested_tasks: list[str] | None
) -> list[str]:
    if not requested_tasks:
        return list(catalog)

    unknown = [task for task in requested_tasks if task not in catalog]
    if unknown:
        available = ", ".join(catalog)
        raise KeyError(f"Unknown task(s): {', '.join(unknown)}. Available: {available}")
    return list(dict.fromkeys(requested_tasks))


def main() -> int:
    args = parse_args()
    try:
        catalog = load_catalog(args.catalog)
        tasks = select_tasks(catalog, args.tasks)
    except (FileNotFoundError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.list:
        print(f"Tasks: {len(tasks)}")
        for task_name in tasks:
            print(f"{task_name:<40} {len(catalog[task_name]):>3} directories")
        return 0

    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple[str, str, str]] = []
    failed_tasks: list[str] = []
    for task_name in tasks:
        selected = []
        for source in iter_hdf5_files(catalog[task_name]):
            selected.append(source)
            if len(selected) == args.samples_per_task:
                break

        if len(selected) < args.samples_per_task:
            print(
                f"[ERROR] {task_name}: found {len(selected)} HDF5 file(s), "
                f"need {args.samples_per_task}",
                file=sys.stderr,
            )
            failed_tasks.append(task_name)
            continue

        for sample_id, source in enumerate(selected):
            destination = output_dir / f"{task_name}_{sample_id:03d}.hdf5"
            if destination.exists() and not args.overwrite:
                print(f"[ERROR] Destination already exists: {destination}", file=sys.stderr)
                failed_tasks.append(task_name)
                break

            print(f"[{task_name}] {source} -> {destination}")
            if not args.dry_run:
                shutil.copy2(source, destination)
            manifest_rows.append((destination.name, task_name, str(source.resolve())))

    if not args.dry_run:
        manifest_path = output_dir / "manifest.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(("output_file", "task", "source_file"))
            writer.writerows(manifest_rows)
        print(f"Manifest: {manifest_path}")

    copied = len(manifest_rows)
    print(f"Selected {copied} file(s) from {len(tasks) - len(set(failed_tasks))} task(s).")
    if failed_tasks:
        print(
            f"Failed tasks ({len(set(failed_tasks))}): "
            + ", ".join(dict.fromkeys(failed_tasks)),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
