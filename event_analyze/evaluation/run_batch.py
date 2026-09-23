#!/usr/bin/env python3
"""Run one frozen batch split through an existing versioned run.sh pipeline."""
import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def validate_frozen_manifest(manifest, manifest_path):
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid batch manifest: {manifest_path}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        if "hdf5_root" in manifest and "split" in manifest:
            raise SystemExit(
                f"{manifest_path} is a batch discovery CONFIG, not a frozen manifest.\n"
                "Run: python evaluation/discover_batch.py\n"
                "Then pass batches/dishwasher_v1_manifest.json to run_batch.py."
            )
        raise SystemExit(
            f"Frozen manifest has no episodes: {manifest_path}. "
            "Run evaluation/discover_batch.py first."
        )
    if not manifest.get("dataset_fingerprint"):
        raise SystemExit(
            f"Frozen manifest is missing dataset_fingerprint: {manifest_path}. "
            "Regenerate it with evaluation/discover_batch.py."
        )


def selected_episodes(manifest, split):
    if split == "all":
        return list(manifest.get("episodes", []))
    return [x for x in manifest.get("episodes", []) if x.get("split") == split]


def resolve_under_event_root(event_root, value):
    p = Path(value)
    return p if p.is_absolute() else event_root / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--version", default=None)
    ap.add_argument(
        "--split",
        choices=["development", "validation", "heldout", "reserve", "all"],
        default="validation",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    event_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    if not manifest:
        raise FileNotFoundError(manifest_path)
    validate_frozen_manifest(manifest, manifest_path)

    version = args.version or manifest.get("default_version", "v3_4_4")
    version_run = event_root / "versions" / version / "run.sh"
    if not version_run.exists():
        raise FileNotFoundError(f"version runner not found: {version_run}")

    schema = resolve_under_event_root(event_root, manifest["schema"]).resolve()
    episodes = selected_episodes(manifest, args.split)

    log_root = (
        event_root / "output" / "batch_logs" /
        manifest["batch_name"] / version / args.split
    )
    status_path = log_root / "batch_status.json"
    previous = load_json(status_path, {}) or {}
    previous_rows = previous.get("episodes", {})

    if args.retry_failed:
        episodes = [
            x for x in episodes
            if previous_rows.get(x["name"], {}).get("status") == "failed"
        ]

    status = {
        "batch_name": manifest["batch_name"],
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "version": version,
        "split": args.split,
        "manifest": str(manifest_path),
        "started_utc": now(),
        "workers": args.workers,
        "episodes": dict(previous_rows),
    }
    lock = threading.Lock()
    atomic_json(status_path, status)

    def run_one(ep):
        name = ep["name"]
        hdf5 = Path(ep["hdf5"])
        output_dir = Path(ep["output_dir"])
        result_ann = event_root / "results" / name / version / "hierarchical_annotations.json"
        log_path = log_root / f"{name}.log"

        if not hdf5.exists():
            return name, {
                "status": "failed",
                "return_code": None,
                "reason": "missing_hdf5",
                "hdf5": str(hdf5),
                "log": str(log_path),
                "finished_utc": now(),
            }

        if result_ann.exists() and not args.force:
            return name, {
                "status": "skipped_existing",
                "return_code": 0,
                "result": str(result_ann),
                "log": str(log_path),
                "finished_utc": now(),
            }

        lerobot_episode = ep.get("lerobot_episode")
        lerobot_max = ep.get("lerobot_max_raw_frame")
        cmd = [
            "bash",
            str(version_run),
            str(hdf5),
            str(output_dir),
            manifest["task"],
            "" if lerobot_episode is None else str(lerobot_episode),
            "" if lerobot_max is None else str(lerobot_max),
            str(schema),
        ]

        env = os.environ.copy()
        gt = ep.get("gt")
        if gt:
            env["GT"] = str(resolve_under_event_root(event_root, gt).resolve())
        else:
            env.pop("GT", None)

        if args.dry_run:
            return name, {
                "status": "dry_run",
                "return_code": 0,
                "hdf5": str(hdf5),
                "output_dir": str(output_dir),
                "gt": gt,
                "log": str(log_path),
                "command": cmd,
                "finished_utc": now(),
            }

        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = now()
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=event_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        return name, {
            "status": "success" if proc.returncode == 0 else "failed",
            "return_code": proc.returncode,
            "started_utc": started,
            "finished_utc": now(),
            "hdf5": str(hdf5),
            "output_dir": str(output_dir),
            "gt": gt,
            "log": str(log_path),
            "result": str(result_ann) if result_ann.exists() else None,
        }

    def record(name, row):
        with lock:
            status["episodes"][name] = row
            atomic_json(status_path, status)
        print(
            f"[{row['status']:<16}] {name}"
            + (f" rc={row['return_code']}" if row.get("return_code") is not None else "")
        )

    if args.workers == 1:
        for ep in episodes:
            name, row = run_one(ep)
            record(name, row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, ep) for ep in episodes]
            for future in as_completed(futures):
                name, row = future.result()
                record(name, row)

    status["finished_utc"] = now()
    counts = {}
    for ep in selected_episodes(manifest, args.split):
        st = status["episodes"].get(ep["name"], {}).get("status", "not_run")
        counts[st] = counts.get(st, 0) + 1
    status["counts"] = counts
    atomic_json(status_path, status)

    print("batch status:", counts)
    print("status file:", status_path)

    if not args.no_summary and not args.dry_run:
        summary_script = event_root / "evaluation" / "summarize_batch.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(summary_script),
                str(manifest_path),
                "--version",
                version,
                "--split",
                args.split,
                "--status",
                str(status_path),
            ],
            cwd=event_root,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
