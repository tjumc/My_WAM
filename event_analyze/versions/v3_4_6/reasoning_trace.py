#!/usr/bin/env python3
"""Build compact V3.4.6 pass1 -> raw pass2 -> reconciled -> final trace."""
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
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def skill_summary(x):
    return {
        "skill_type": x.get("skill_type"),
        "entity": x.get("entity"),
        "fine_entity": x.get("fine_entity"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    diag = out / "diagnostics"

    p1 = load_json(diag / "pass1_anchors.json", {"skill_sequence": []})
    raw2 = load_json(diag / "pass2_before_reconcile.json", {"skill_sequence": []})
    reconciled_doc = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []})
    final_ann = load_json(out / "hierarchical_annotations.json", {"semantic_phases": []})
    ambiguity = load_json(out / "ambiguity_requests.json", {"requests": []})
    targeted = load_jsonl(out / "targeted_observations.jsonl")
    conservative = load_json(diag / "conservative_update.json", {})

    pass1 = [skill_summary(x) for x in p1.get("skill_sequence", [])]
    pass2_raw = list(raw2.get("skill_sequence", []))
    reconciled = [skill_summary(x) for x in reconciled_doc.get("skill_sequence", [])]
    final = [skill_summary(x) for x in final_ann.get("semantic_phases", [])]

    trace = {
        "annotation_version": "v3.4.6",
        "pass1": {
            "skill_sequence": [x.get("skill_type") for x in pass1],
            "phase_count": len(pass1),
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
        "pass2_raw": {
            "skill_sequence": [x.get("skill_type") for x in pass2_raw],
            "phase_count": len(pass2_raw),
            "changes_from_pass1": sequence_diff(pass1, pass2_raw),
        },
        "conservative_assimilation": {
            "restored_pass1_anchors": conservative.get("restored_pass1_anchors", []),
            "directly_contradicted_pass1_anchors": conservative.get("directly_contradicted_pass1_anchors", []),
            "pass2_additions": conservative.get("pass2_additions", []),
        },
        "reconciled": {
            "skill_sequence": [x.get("skill_type") for x in reconciled],
            "phase_count": len(reconciled),
            "changes_from_raw_pass2": sequence_diff(pass2_raw, reconciled),
            "changes_from_pass1": sequence_diff(pass1, reconciled),
        },
        "final": {
            "skill_sequence": [x.get("skill_type") for x in final],
            "phase_count": len(final),
            "changes_from_reconciled": sequence_diff(reconciled, final),
            "consistency": final_ann.get("final_consistency"),
            "task_completion_frontier": final_ann.get("task_completion_frontier"),
        },
    }

    path = diag / "reasoning_trace.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
