#!/usr/bin/env python3
"""Capture compact pass-1 state and build a pass1->pass2/final reasoning trace."""
import argparse
import difflib
import json
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
    return [
        json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def skill_summary(x):
    return {
        "skill_type": x.get("skill_type"),
        "entity": x.get("entity"),
        "start_frame": x.get("provisional_start_frame", x.get("raw_start_frame")),
        "end_frame": x.get("provisional_end_frame", x.get("raw_end_frame")),
        "confidence": x.get("confidence"),
    }


def sequence_diff(before, after):
    a = [x.get("skill_type") for x in before]
    b = [x.get("skill_type") for x in after]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.append({
            "op": tag,
            "before_index": [i1, i2],
            "after_index": [j1, j2],
            "before": a[i1:i2],
            "after": b[j1:j2],
        })
    return out


def rejected_summary(report):
    out = []
    for d in report.get("decisions", []):
        if d.get("accepted"):
            continue
        out.append({
            "original_skill_id": d.get("original_skill_id"),
            "skill_type": d.get("skill_type"),
            "reasons": d.get("reasons", []),
            "signal_score": (d.get("signal") or {}).get("score"),
            "contact_count": (d.get("interaction_evidence") or {}).get("contact_count"),
            "direction_only_count": (d.get("interaction_evidence") or {}).get("direction_only_count"),
        })
    return out


def capture(out):
    validated = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []})
    report = load_json(out / "interaction_validation_report.json", {"decisions": []})
    ownership = load_json(out / "hand_object_ownership.json", {"candidates": []})

    snap = {
        "annotation_version": "v3.4.5",
        "pass": 1,
        "skills": [skill_summary(x) for x in validated.get("skill_sequence", [])],
        "rejected_candidates": rejected_summary(report),
        "ownership_ambiguities": [
            {
                "object": x.get("object"),
                "start_frame": x.get("resolved_start_frame", x.get("start_frame")),
                "end_frame": x.get("resolved_end_frame", x.get("end_frame")),
                "resolution_reasons": x.get("resolution_reasons", []),
            }
            for x in ownership.get("candidates", [])
            if x.get("assignment_status") == "ambiguous"
        ],
    }
    diag = out / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    path = diag / "pass1_snapshot.json"
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("captured", path)


def finalize(out):
    diag = out / "diagnostics"
    snap = load_json(diag / "pass1_snapshot.json", {"skills": []})
    validated = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []})
    final_ann = load_json(out / "hierarchical_annotations.json", {"semantic_phases": []})
    ambiguity = load_json(out / "ambiguity_requests.json", {"requests": []})
    targeted = load_jsonl(out / "targeted_observations.jsonl")

    before = snap.get("skills", [])
    pass2 = [skill_summary(x) for x in validated.get("skill_sequence", [])]
    final = [skill_summary(x) for x in final_ann.get("semantic_phases", [])]

    trace = {
        "annotation_version": "v3.4.5",
        "pass1": {
            "skill_sequence": [x.get("skill_type") for x in before],
            "phase_count": len(before),
            "rejected_candidates": snap.get("rejected_candidates", []),
            "ownership_ambiguities": snap.get("ownership_ambiguities", []),
        },
        "targeted_perception": {
            "num_candidate_requests": ambiguity.get("num_candidate_requests"),
            "num_requests": len(ambiguity.get("requests", [])),
            "requests": [
                {
                    "request_id": x.get("request_id"),
                    "kind": x.get("kind"),
                    "semantic_goal": x.get("semantic_goal"),
                    "target_entities": x.get("target_entities"),
                    "start_frame": x.get("start_frame"),
                    "end_frame": x.get("end_frame"),
                    "utility_score": x.get("utility_score"),
                }
                for x in ambiguity.get("requests", [])
            ],
            "num_targeted_queries": len(targeted),
            "num_merged_targeted_observations": sum(
                bool(x.get("_merge_accepted")) for x in targeted
            ),
        },
        "pass2": {
            "skill_sequence": [x.get("skill_type") for x in pass2],
            "phase_count": len(pass2),
            "changes_from_pass1": sequence_diff(before, pass2),
        },
        "final": {
            "skill_sequence": [x.get("skill_type") for x in final],
            "phase_count": len(final),
            "changes_from_pass2": sequence_diff(pass2, final),
            "consistency": final_ann.get("final_consistency"),
        },
    }
    path = diag / "reasoning_trace.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["capture", "finalize"], required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if args.mode == "capture":
        capture(out)
    else:
        finalize(out)


if __name__ == "__main__":
    main()
