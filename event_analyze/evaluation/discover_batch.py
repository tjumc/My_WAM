#!/usr/bin/env python3
"""Scan an HDF5 directory once and freeze a reproducible batch split."""
import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "batches/dishwasher_v1.json"),
    )
    ap.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "batches/dishwasher_v1_manifest.json"),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    event_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.output).resolve()

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite frozen batch manifest: {out_path}\n"
            "Use --force only if you intentionally want to redefine the dataset split."
        )

    cfg = load_json(config_path)
    root = Path(cfg["hdf5_root"]).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"HDF5 root does not exist: {root}")

    pattern = re.compile(cfg.get("episode_regex", r"episode_(\d+)\.hdf5$"))
    files = []
    for path in sorted(root.glob(cfg.get("glob", "*.hdf5"))):
        m = pattern.search(path.name)
        if not m:
            continue
        ep = int(m.group(1))
        files.append((ep, path.resolve()))

    if not files:
        raise RuntimeError(f"No matching HDF5 files found under {root}")

    expected_count = cfg.get("expected_episode_count")
    if expected_count is not None and len(files) != int(expected_count):
        raise RuntimeError(
            f"Expected {int(expected_count)} HDF5 episodes but found {len(files)} under {root}. "
            "Check the dataset directory before freezing the split."
        )

    seen = {}
    for ep, path in files:
        if ep in seen:
            raise RuntimeError(
                f"Duplicate episode id {ep}: {seen[ep]} and {path}"
            )
        seen[ep] = path

    split_cfg = cfg.get("split", {})
    dev = [int(x) for x in split_cfg.get("development_episodes", [26, 27])]
    missing_dev = [x for x in dev if x not in seen]
    if missing_dev:
        raise RuntimeError(
            "Configured development episodes missing from scan: "
            + ", ".join(map(str, missing_dev))
        )

    candidates = sorted(ep for ep in seen if ep not in set(dev))
    rng = random.Random(int(split_cfg.get("seed", 20260923)))
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    n_val = max(0, int(split_cfg.get("validation_count", 6)))
    n_test = max(0, int(split_cfg.get("heldout_count", 6)))
    validation = shuffled[:n_val]
    heldout = shuffled[n_val:n_val + n_test]
    reserve = shuffled[n_val + n_test:]

    split_of = {}
    for ep in dev:
        split_of[ep] = "development"
    for ep in validation:
        split_of[ep] = "validation"
    for ep in heldout:
        split_of[ep] = "heldout"
    for ep in reserve:
        split_of[ep] = "reserve"

    legacy_gt = {int(k): v for k, v in (cfg.get("ground_truth") or {}).items()}
    sequence_gt_map = {
        int(k): v for k, v in (cfg.get("sequence_ground_truth") or legacy_gt).items()
    }
    temporal_cfg = cfg.get("temporal_ground_truth")
    temporal_gt_map = {
        int(k): v for k, v in ((legacy_gt if temporal_cfg is None else temporal_cfg) or {}).items()
    }
    overrides = {int(k): v for k, v in (cfg.get("episode_overrides") or {}).items()}

    episodes = []
    for ep in sorted(seen):
        path = seen[ep]
        stem = path.stem
        row = {
            "episode_id": ep,
            "name": stem,
            "split": split_of[ep],
            "hdf5": str(path),
            "hdf5_bytes": path.stat().st_size,
            "output_dir": str(event_root / "output" / f"{stem}_analysis"),
            "sequence_gt": sequence_gt_map.get(ep),
            "temporal_gt": temporal_gt_map.get(ep),
        }
        row.update(overrides.get(ep, {}))
        episodes.append(row)

    manifest_core = {
        "batch_name": cfg["batch_name"],
        "task_family": cfg["task_family"],
        "task": cfg["task"],
        "default_version": cfg.get("version", "v3_4_4"),
        "schema": cfg["schema"],
        "source": {
            "config": str(config_path),
            "hdf5_root": str(root.resolve()),
            "glob": cfg.get("glob", "*.hdf5"),
            "episode_regex": cfg.get("episode_regex", r"episode_(\d+)\.hdf5$"),
        },
        "split_policy": {
            "development_episodes": dev,
            "validation_count_requested": n_val,
            "heldout_count_requested": n_test,
            "seed": int(split_cfg.get("seed", 20260923)),
            "selection_rule": "deterministic shuffle of all non-development episode ids",
        },
        "splits": {
            "development": sorted(dev),
            "validation": sorted(validation),
            "heldout": sorted(heldout),
            "reserve": sorted(reserve),
        },
        "episodes": episodes,
    }

    canonical = json.dumps(
        manifest_core, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    manifest = {
        **manifest_core,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": sha256_text(canonical),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"scanned HDF5 files: {len(episodes)}")
    for name in ("development", "validation", "heldout", "reserve"):
        ids = manifest["splits"][name]
        print(f"{name:12s} ({len(ids):2d}): {ids}")
    print("dataset_fingerprint:", manifest["dataset_fingerprint"])
    print("saved:", out_path)


if __name__ == "__main__":
    main()
