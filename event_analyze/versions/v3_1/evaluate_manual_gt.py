#!/usr/bin/env python3
"""Evaluate V3 skill order against a manual regression fixture. Not used by the pipeline."""
import argparse
import json
from pathlib import Path


def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i-1][j-1] + 1 if x == y else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, dp[j] + 1, dp[j-1] + (x != y)))
        dp = new
    return dp[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("fixture")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    pred = json.load(open(args.annotations, encoding="utf-8"))
    gt = json.load(open(args.fixture, encoding="utf-8"))
    p = [x["skill_type"] for x in pred.get("semantic_phases", [])]
    g = [x["skill_type"] for x in gt["steps"]]
    m = lcs(p, g)
    report = {
        "predicted": p,
        "ground_truth": g,
        "lcs": m,
        "sequence_recall": round(m / len(g), 3) if g else 1.0,
        "sequence_precision": round(m / len(p), 3) if p else 0.0,
        "edit_distance": edit_distance(p, g),
        "exact_sequence_match": p == g,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
