#!/usr/bin/env python3
"""Sequence-level regression matrix for held-out dishwasher episodes."""
import argparse
import json
import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from schema_runtime import load_schema, policy_skill_aliases


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


def metrics(pred, gt):
    n = lcs(pred, gt)
    p = n / len(pred) if pred else 0.0
    r = n / len(gt) if gt else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"pred_n": len(pred), "matched": n, "precision": p, "recall": r,
            "f1": f, "edit_distance": edit(pred, gt), "exact": pred == gt}


def load_pred(path):
    doc = json.load(open(path, encoding="utf-8"))
    return [x["skill_type"] for x in doc.get("semantic_phases", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--versions", default="v3_3_1,v3_3_2,v3_3_3,v3_4_0,v3_4_1")
    ap.add_argument("--output", default=None)
    ap.add_argument("--schema", default=None)
    args = ap.parse_args()

    root = Path(args.event_root)
    schema_path = Path(args.schema) if args.schema else root / "ontologies/dishwasher_loading.json"
    aliases = policy_skill_aliases(load_schema(schema_path))
    versions = [x.strip() for x in args.versions.split(",") if x.strip()]
    seq26 = json.load(open(root / "regression/dishwasher_episode26_sequence_gt.json", encoding="utf-8"))["sequence"]
    gt27 = json.load(open(root / "regression/dishwasher_episode27_manual_gt.json", encoding="utf-8"))
    seq27 = [x["skill_type"] for x in gt27["steps"]]

    episodes = {
        "episode26": (
            root / "output/dishwasher_2_fx_20260529_episode_26_analysis/versions",
            root / "results/dishwasher_2_fx_20260529_episode_26",
            seq26,
            "manual_sequence_only_provisional",
        ),
        "episode27": (
            root / "output/dishwasher_2_fx_20260529_episode_27_analysis/versions",
            root / "results/dishwasher_2_fx_20260529_episode_27",
            seq27,
            gt27.get("status", "manual_gt"),
        ),
    }

    rows = []
    for ep, (runtime_base, results_base, gt, status) in episodes.items():
        for ver in versions:
            runtime_path = runtime_base / ver / "hierarchical_annotations.json"
            compact_path = results_base / ver / "hierarchical_annotations.json"
            path = runtime_path if runtime_path.exists() else compact_path
            if not path.exists():
                continue
            pred = load_pred(path)
            pred_policy = [aliases.get(x, x) for x in pred]
            gt_policy = [aliases.get(x, x) for x in gt]
            raw = metrics(pred, gt)
            policy = metrics(pred_policy, gt_policy)
            row = {
                "episode": ep, "version": ver, "gt_status": status,
                "predicted": pred, "ground_truth": gt,
                "policy_predicted": pred_policy, "policy_ground_truth": gt_policy,
                "raw": raw, "policy": policy,
            }
            rows.append(row)

    print("episode   version   rawF1 rawEdit  policyF1 policyEdit  policyP policyR")
    for x in rows:
        r = x["raw"]; p = x["policy"]
        print(f"{x['episode']:<9} {x['version']:<9} "
              f"{r['f1']:.3f} {r['edit_distance']:>7}  "
              f"{p['f1']:.3f} {p['edit_distance']:>10}  "
              f"{p['precision']:.3f}   {p['recall']:.3f}")

    if args.output:
        Path(args.output).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
