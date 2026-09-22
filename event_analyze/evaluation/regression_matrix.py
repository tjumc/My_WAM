#!/usr/bin/env python3
"""Sequence-level regression matrix for held-out dishwasher episodes."""
import argparse
import json
from pathlib import Path


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
    ap.add_argument("--versions", default="v3_3_1,v3_3_2,v3_3_3,v3_4_0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.event_root)
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
            row = {"episode": ep, "version": ver, "gt_status": status,
                   "predicted": pred, "ground_truth": gt}
            row.update(metrics(pred, gt))
            rows.append(row)

    print("episode   version   pred  match   P      R      F1     edit exact")
    for x in rows:
        print(f"{x['episode']:<9} {x['version']:<9} {x['pred_n']:>4} {x['matched']:>6} "
              f"{x['precision']:.3f}  {x['recall']:.3f}  {x['f1']:.3f}  "
              f"{x['edit_distance']:>4}  {str(x['exact']):>5}")

    if args.output:
        Path(args.output).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
