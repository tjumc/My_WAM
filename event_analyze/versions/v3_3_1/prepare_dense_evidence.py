#!/usr/bin/env python3
"""Prepare V3.3.1 entity observations from already-computed V3.3 dense results.

This lets V3.3 -> V3.3.1 isolate fusion/validation changes without another VLM call.
Only confident directional dense windows are injected.
"""
import argparse
import json
from pathlib import Path

from dense_targeted_reobserve import pseudo_observation


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.62)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_jsonl(args.base)
    dense = load_jsonl(args.dense)
    pseudos = []
    for x in dense:
        if float(x.get("confidence", 0.0)) < args.min_confidence:
            continue
        if x.get("relative_motion") not in {"moving_out", "moving_in"}:
            continue
        p = pseudo_observation(x, 10000 + int(x["_dense_window_id"]))
        p["_v331_dense_target_entity"] = "cutlery_basket"
        pseudos.append(p)

    (out / "dense_cutlery_observations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in dense) + ("\n" if dense else ""),
        encoding="utf-8",
    )
    (out / "dense_cutlery_pseudo_observations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in pseudos) + ("\n" if pseudos else ""),
        encoding="utf-8",
    )
    merged = sorted(base + pseudos, key=lambda x: (int(x["_raw_frame"]), int(x["_event_id"])))
    (out / "entity_observations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in merged) + "\n",
        encoding="utf-8",
    )
    print(f"V3.3.1 reused dense directional observations: {len(pseudos)}/{len(dense)}")


if __name__ == "__main__":
    main()
