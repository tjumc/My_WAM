#!/usr/bin/env python3
"""Compact a completed V3.4.8 runtime directory."""
import argparse
import os
import shutil
from pathlib import Path


CACHE_FILES = [
    "entity_observations_base.jsonl",
    "entity_observations.jsonl",
    "targeted_observations.jsonl",
    "dense_cutlery_transition_evidence.jsonl",
    "dense_cutlery_observations.jsonl",
    "boundaries.png",
]
CACHE_DIRS = [
    "proposal",
    "targeted_temporal_strips",
    "vlm_raw_targeted",
]
DELETE_FILES = [
    "tracked_entity_states.json",
    "tracked_entity_states_owned.json",
    "hand_object_ownership.json",
    "skill_candidates.json",
    "skill_candidates_validated.json",
    "interaction_validation_report.json",
    "ambiguity_requests.json",
]
DELETE_DIAGNOSTICS = [
    "pass1_anchors.json",
    "pass2_before_reconcile.json",
    "rejected_candidate_pool.json",
]


def move_replace(src, dst):
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--keep-debug", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    keep_debug = args.keep_debug or os.environ.get("KEEP_DEBUG", "0") == "1"
    if keep_debug:
        print("KEEP_DEBUG=1: preserving full V3.4.8 runtime artifacts")
        return

    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    moved, removed = [], []
    for name in CACHE_FILES:
        src = out / name
        if src.exists():
            move_replace(src, cache / name)
            moved.append(name)
    for name in CACHE_DIRS:
        src = out / name
        if src.exists():
            move_replace(src, cache / name)
            moved.append(name + "/")

    for name in DELETE_FILES:
        p = out / name
        if p.exists():
            p.unlink()
            removed.append(name)

    diag = out / "diagnostics"
    for name in DELETE_DIAGNOSTICS:
        p = diag / name
        if p.exists():
            p.unlink()
            removed.append("diagnostics/" + name)

    print("runtime compacted")
    print("  moved to cache:", ", ".join(moved) if moved else "none")
    print("  removed intermediates:", ", ".join(removed) if removed else "none")


if __name__ == "__main__":
    main()
