#!/usr/bin/env python3
"""Aggregate compact per-episode results for a frozen batch manifest."""
import argparse
import json
import math
import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from schema_runtime import load_schema, policy_skill_aliases


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        n = [i]
        for j, y in enumerate(b, 1):
            n.append(min(n[-1] + 1, d[j] + 1, d[j - 1] + (x != y)))
        d = n
    return d[-1]


def lcs(a, b):
    d = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = d[j]
            d[j] = prev + 1 if x == y else max(d[j], d[j - 1])
            prev = old
    return d[-1]


def seq_metrics(pred, gt):
    n = lcs(pred, gt)
    p = n / len(pred) if pred else 0.0
    r = n / len(gt) if gt else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "pred_n": len(pred),
        "gt_n": len(gt),
        "matched": n,
        "precision": p,
        "recall": r,
        "f1": f,
        "edit_distance": edit(pred, gt),
        "exact": pred == gt,
    }


def load_gt_sequence(path):
    doc = load_json(path)
    if not doc:
        return None
    if isinstance(doc.get("sequence"), list):
        return list(doc["sequence"])
    if isinstance(doc.get("steps"), list):
        return [x["skill_type"] for x in doc["steps"] if "skill_type" in x]
    return None


def finite_mean(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def flatten_temporal(eval_doc):
    if not eval_doc:
        return {}
    block = eval_doc.get("policy_normalized") or eval_doc
    sem = block.get("semantic_temporal") or {}
    core = block.get("core_temporal") or {}

    def boundary_f1(section, tol):
        for x in section.get("boundary_f1", []) or []:
            if float(x.get("tolerance_sec", -1)) == float(tol):
                return x.get("f1")
        return None

    return {
        "semantic_miou": sem.get("matched_mean_iou"),
        "semantic_boundary_mae_sec": sem.get("boundary_mae_sec"),
        "semantic_boundary_f1_1s": boundary_f1(sem, 1.0),
        "semantic_boundary_f1_2s": boundary_f1(sem, 2.0),
        "core_miou": core.get("matched_mean_iou"),
        "core_boundary_mae_sec": core.get("boundary_mae_sec"),
    }


def validate_frozen_manifest(manifest, manifest_path):
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list) or not episodes:
        if isinstance(manifest, dict) and "hdf5_root" in manifest and "split" in manifest:
            raise SystemExit(
                f"{manifest_path} is a batch discovery CONFIG, not a frozen manifest.\n"
                "Run evaluation/discover_batch.py first and summarize the generated manifest."
            )
        raise SystemExit(f"Frozen manifest has no episodes: {manifest_path}")
    if not manifest.get("dataset_fingerprint"):
        raise SystemExit(
            f"Frozen manifest is missing dataset_fingerprint: {manifest_path}"
        )


def selected_episodes(manifest, split):
    if split == "all":
        return list(manifest.get("episodes", []))
    return [x for x in manifest.get("episodes", []) if x.get("split") == split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--version", default=None)
    ap.add_argument(
        "--split",
        choices=["development", "validation", "heldout", "reserve", "all"],
        default="validation",
    )
    ap.add_argument("--status", default=None)
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()

    event_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    validate_frozen_manifest(manifest, manifest_path)
    version = args.version or manifest.get("default_version", "v3_4_4")

    schema_path = Path(manifest["schema"])
    if not schema_path.is_absolute():
        schema_path = event_root / schema_path
    aliases = policy_skill_aliases(load_schema(schema_path))

    status_path = (
        Path(args.status).resolve()
        if args.status
        else (
            event_root / "output" / "batch_logs" /
            manifest["batch_name"] / version / args.split / "batch_status.json"
        )
    )
    status_doc = load_json(status_path, {}) or {}
    status_rows = status_doc.get("episodes", {})

    rows = []
    failures = []
    for ep in selected_episodes(manifest, args.split):
        name = ep["name"]
        result_dir = event_root / "results" / name / version
        runtime_dir = Path(ep["output_dir"]) / "versions" / version

        summary = load_json(result_dir / "summary.json")
        compact_manifest = load_json(result_dir / "manifest.json")
        eval_doc = load_json(result_dir / "evaluation_temporal.json")
        if summary is None:
            summary = load_json(runtime_dir / "hierarchical_annotations.json")
            if summary and "semantic_phases" in summary:
                summary = {
                    "skill_sequence": [
                        x.get("skill_type") for x in summary.get("semantic_phases", [])
                    ],
                    "phase_count": len(summary.get("semantic_phases", [])),
                }

        batch_status = status_rows.get(name, {})
        pred = list((summary or {}).get("skill_sequence", []))

        gt_metrics = None
        gt_path = ep.get("gt")
        gt_sequence = None
        if gt_path:
            gp = Path(gt_path)
            if not gp.is_absolute():
                gp = event_root / gp
            gt_sequence = load_gt_sequence(gp)
            if gt_sequence is not None:
                pred_policy = [aliases.get(x, x) for x in pred]
                gt_policy = [aliases.get(x, x) for x in gt_sequence]
                gt_metrics = seq_metrics(pred_policy, gt_policy)

        temporal = flatten_temporal(eval_doc)
        closed = (compact_manifest or {}).get("closed_loop_perception") or {}

        row = {
            "episode_id": ep["episode_id"],
            "episode": name,
            "split": ep["split"],
            "run_status": batch_status.get("status"),
            "result_available": summary is not None,
            "phase_count": (summary or {}).get("phase_count", len(pred)),
            "skill_sequence": pred,
            "gt_available": gt_sequence is not None,
            "policy_sequence": gt_metrics,
            **temporal,
            "num_ambiguity_requests": closed.get("num_ambiguity_requests"),
            "num_targeted_queries": closed.get("num_targeted_queries"),
            "num_merged_targeted_observations": closed.get("num_merged_targeted_observations"),
            "ambiguity_kinds": closed.get("ambiguity_kinds"),
            "result_git_commit": (compact_manifest or {}).get("git_commit_at_export"),
            "schema_sha256": ((compact_manifest or {}).get("input") or {}).get("schema_sha256"),
        }
        rows.append(row)

        reasons = []
        if batch_status.get("status") == "failed":
            reasons.append("run_failed")
        if summary is None:
            reasons.append("missing_result")
        if gt_metrics is not None and not gt_metrics["exact"]:
            reasons.append("non_exact_policy_sequence")
        if reasons:
            failures.append({
                "episode_id": ep["episode_id"],
                "episode": name,
                "split": ep["split"],
                "reasons": reasons,
                "policy_sequence": gt_metrics,
                "skill_sequence": pred,
                "log": batch_status.get("log"),
            })

    gt_rows = [x for x in rows if x["policy_sequence"] is not None]
    exact_rows = [
        x["policy_sequence"]["exact"] for x in gt_rows
        if x["policy_sequence"] is not None
    ]

    aggregate = {
        "num_episodes": len(rows),
        "num_results": sum(bool(x["result_available"]) for x in rows),
        "num_with_gt": len(gt_rows),
        "policy_sequence": {
            "mean_precision": finite_mean([
                x["policy_sequence"]["precision"] for x in gt_rows
            ]),
            "mean_recall": finite_mean([
                x["policy_sequence"]["recall"] for x in gt_rows
            ]),
            "mean_f1": finite_mean([
                x["policy_sequence"]["f1"] for x in gt_rows
            ]),
            "mean_edit_distance": finite_mean([
                x["policy_sequence"]["edit_distance"] for x in gt_rows
            ]),
            "exact_rate": (
                sum(bool(x) for x in exact_rows) / len(exact_rows)
                if exact_rows else None
            ),
        },
        "temporal": {
            key: finite_mean([x.get(key) for x in rows])
            for key in [
                "semantic_miou",
                "semantic_boundary_mae_sec",
                "semantic_boundary_f1_1s",
                "semantic_boundary_f1_2s",
                "core_miou",
                "core_boundary_mae_sec",
            ]
        },
        "closed_loop_perception": {
            "total_ambiguity_requests": sum(
                int(x.get("num_ambiguity_requests") or 0) for x in rows
            ),
            "total_targeted_queries": sum(
                int(x.get("num_targeted_queries") or 0) for x in rows
            ),
            "total_merged_targeted_observations": sum(
                int(x.get("num_merged_targeted_observations") or 0) for x in rows
            ),
            "mean_targeted_queries_per_episode": finite_mean([
                x.get("num_targeted_queries") for x in rows
            ]),
        },
    }

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else event_root / "results" / "batches" / manifest["batch_name"] / version / args.split
    )
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "episode_metrics.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "failure_cases.json").write_text(
        json.dumps({"failures": failures}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    batch_summary = {
        "batch_name": manifest["batch_name"],
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "version": version,
        "split": args.split,
        "aggregate": aggregate,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"batch={manifest['batch_name']} version={version} split={args.split} "
        f"episodes={len(rows)} results={aggregate['num_results']}"
    )
    if gt_rows:
        s = aggregate["policy_sequence"]
        print(
            "policy: "
            f"F1={s['mean_f1']:.3f} "
            f"Edit={s['mean_edit_distance']:.3f} "
            f"ExactRate={s['exact_rate']:.3f}"
        )
    q = aggregate["closed_loop_perception"]
    print(
        f"targeted queries={q['total_targeted_queries']} "
        f"merged={q['total_merged_targeted_observations']}"
    )
    print("saved:", output_root)


if __name__ == "__main__":
    main()
