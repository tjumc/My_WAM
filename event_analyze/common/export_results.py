#!/usr/bin/env python3
"""Export a compact, Git-friendly experiment result package.

Heavy/reproducible runtime artifacts stay under event_analyze/output/ and should
not be committed. This exporter copies only final annotations/evaluation and a
small reproducibility manifest to event_analyze/results/.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


KEEP_FILES = [
    "hierarchical_annotations.json",
    "evaluation_temporal.json",
]

KEEP_DIAGNOSTICS = [
    "reasoning_trace.json",
    "final_consistency.json",
]


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_rel(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(Path(path).name)


def git_head(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--event-root", required=True)
    ap.add_argument("--results-root", default=None)
    args = ap.parse_args()

    event_root = Path(args.event_root).resolve()
    out = Path(args.output_dir).resolve()
    schema = Path(args.schema).resolve()
    hdf5 = Path(args.hdf5)
    episode_name = Path(args.episode_dir).name
    if episode_name.endswith("_analysis"):
        episode_name = episode_name[:-len("_analysis")]

    results_root = Path(args.results_root).resolve() if args.results_root else event_root / "results"
    dst = results_root / episode_name / args.version
    dst.mkdir(parents=True, exist_ok=True)

    copied = {}
    for name in KEEP_FILES:
        src = out / name
        if src.exists():
            target = dst / name
            shutil.copy2(src, target)
            copied[name] = {
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }

    diagnostics_copied = {}
    diag_src = out / "diagnostics"
    if diag_src.exists():
        diag_dst = dst / "diagnostics"
        for name in KEEP_DIAGNOSTICS:
            src = diag_src / name
            if not src.exists():
                continue
            diag_dst.mkdir(parents=True, exist_ok=True)
            target = diag_dst / name
            shutil.copy2(src, target)
            key = f"diagnostics/{name}"
            diagnostics_copied[key] = {
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
            copied[key] = diagnostics_copied[key]

    ann_path = out / "hierarchical_annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"missing final annotation: {ann_path}")
    ann = load_json(ann_path)
    schema_doc = load_json(schema)
    task_family = ann.get("task_family") or schema_doc.get("task_family")
    phases = ann.get("semantic_phases", [])

    summary = {
        "annotation_version": ann.get("annotation_version", args.version),
        "episode": episode_name,
        "task_goal": ann.get("task_goal", args.task),
        "task_family": task_family,
        "phase_count": len(phases),
        "skill_sequence": [x.get("skill_type") for x in phases],
        "task_end_raw_frame": ann.get("task_end_raw_frame"),
        "phases": [
            {
                "skill_type": x.get("skill_type"),
                "raw_start_frame": x.get("raw_start_frame"),
                "raw_end_frame": x.get("raw_end_frame"),
                "provisional_start_frame": x.get("provisional_start_frame"),
                "provisional_end_frame": x.get("provisional_end_frame"),
                "confidence": x.get("confidence"),
            }
            for x in phases
        ],
    }
    summary_path = dst / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    copied["summary.json"] = {
        "sha256": sha256_file(summary_path),
        "bytes": summary_path.stat().st_size,
    }

    evidence = {}
    for name in [
        "entity_observations_base.jsonl",
        "dense_cutlery_transition_evidence.jsonl",
        "tracked_entity_states.json",
        "tracked_entity_states_owned.json",
        "hand_object_ownership.json",
        "ambiguity_requests.json",
        "targeted_observations.jsonl",
        "tracked_entity_states_pass1.json",
        "hand_object_ownership_pass1.json",
        "skill_candidates_validated_pass1.json",
        "interaction_validation_report_pass1.json",
        "skill_candidates_validated.json",
    ]:
        p = out / name
        if p.exists():
            evidence[name] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }

    closed_loop = None
    ambiguity_path = out / "ambiguity_requests.json"
    targeted_path = out / "targeted_observations.jsonl"
    if ambiguity_path.exists() or targeted_path.exists():
        ambiguity_doc = load_json(ambiguity_path) if ambiguity_path.exists() else {}
        targeted_rows = []
        if targeted_path.exists():
            targeted_rows = [
                json.loads(x)
                for x in targeted_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
        closed_loop = {
            "num_ambiguity_requests": len(ambiguity_doc.get("requests", [])),
            "num_targeted_queries": len(targeted_rows),
            "num_merged_targeted_observations": sum(
                bool(x.get("_merge_accepted")) for x in targeted_rows
            ),
            "ambiguity_kinds": sorted({
                x.get("kind") for x in ambiguity_doc.get("requests", [])
                if x.get("kind")
            }),
        }


    final_consistency = None
    consistency_path = out / "diagnostics" / "final_consistency.json"
    if consistency_path.exists():
        consistency_doc = load_json(consistency_path)
        final_consistency = consistency_doc.get("final_consistency") or {
            "num_dropped": len(consistency_doc.get("dropped_phases", [])),
            "num_clamped": len(consistency_doc.get("clamped_phases", [])),
            "num_lifecycle_violations": len(consistency_doc.get("lifecycle_violations", [])),
            "num_expected_final_state_violations": len(consistency_doc.get("expected_final_state_violations", [])),
        }

    manifest = {
        "format_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "episode": episode_name,
        "version": args.version,
        "task": args.task,
        "task_family": task_family,
        "git_commit_at_export": git_head(event_root.parent),
        "input": {
            "hdf5_basename": hdf5.name,
            "hdf5_bytes": hdf5.stat().st_size if hdf5.exists() else None,
            "schema": safe_rel(schema, event_root),
            "schema_sha256": sha256_file(schema),
        },
        "runtime_output": safe_rel(out, event_root),
        "evidence_fingerprints": evidence,
        "closed_loop_perception": closed_loop,
        "final_consistency": final_consistency,
        "exported_files": copied,
        "storage_policy": {
            "git_keeps": [
                "final annotations",
                "evaluation report when available",
                "summary",
                "manifest",
                "compact reasoning/final-consistency diagnostics",
            ],
            "local_only": [
                "contact sheets",
                "dense temporal strips",
                "overview images",
                "raw VLM responses",
                "proposal visualizations",
                "other reproducible runtime caches",
            ],
        },
    }
    manifest_path = dst / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"exported compact result: {dst}")
    print("files:", ", ".join(sorted(p.name for p in dst.iterdir() if p.is_file())))


if __name__ == "__main__":
    main()
