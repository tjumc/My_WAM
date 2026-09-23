#!/usr/bin/env python3
"""Reuse V3.4.5 targeted observations only when request signatures match exactly."""
import argparse
import json
import shutil
import sys
from pathlib import Path


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def signature_request(x):
    return (
        x.get("kind"),
        tuple(x.get("target_entities") or []),
        int(x.get("start_frame", -1)),
        int(x.get("end_frame", -1)),
    )


def signature_obs(x):
    return (
        x.get("_ambiguity_kind"),
        tuple(x.get("_target_entities") or []),
        int(x.get("_window_start_frame", -1)),
        int(x.get("_window_end_frame", -1)),
    )


def source_targeted(source):
    for p in [
        source / "targeted_observations.jsonl",
        source / "cache" / "targeted_observations.jsonl",
    ]:
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    src = Path(args.source_dir)
    reqs = load_json(out / "ambiguity_requests.json", {}).get("requests", [])
    src_path = source_targeted(src)
    if src_path is None:
        print("targeted reuse: no source targeted observations")
        raise SystemExit(2)

    rows = load_jsonl(src_path)
    if len(rows) != len(reqs):
        print(f"targeted reuse: request count mismatch current={len(reqs)} source={len(rows)}")
        raise SystemExit(2)

    for i, (req, row) in enumerate(zip(reqs, rows)):
        if signature_request(req) != signature_obs(row):
            print(
                "targeted reuse: signature mismatch",
                i, signature_request(req), signature_obs(row),
            )
            raise SystemExit(2)

    shutil.copy2(src_path, out / "targeted_observations.jsonl")
    print(f"targeted reuse: exact signature match, reused {len(rows)} observations from {src_path}")
    return 0


if __name__ == "__main__":
    main()
