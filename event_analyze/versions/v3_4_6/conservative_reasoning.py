#!/usr/bin/env python3
"""V3.4.6 conservative pass1->pass2 semantic assimilation.

Targeted observations are allowed to recover or refine uncertain hypotheses, but
validated pass1 skills are semantic anchors by default. A pass1 anchor may be
removed only when targeted evidence directly contradicts the same entity
transition with strong confidence.
"""
import argparse
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


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def span(skill):
    s = int(skill.get("provisional_start_frame", skill.get("raw_start_frame", 0)))
    e = int(skill.get("provisional_end_frame", skill.get("raw_end_frame", s)))
    return s, e


def center(skill):
    s, e = span(skill)
    return (s + e) / 2.0


def physical_entity(skill):
    return skill.get("fine_entity") or skill.get("entity")


def semantic_key(skill):
    return (skill.get("skill_type"), physical_entity(skill))


def overlap_or_close(a, b, max_center_gap=180):
    sa, ea = span(a)
    sb, eb = span(b)
    inter = max(0, min(ea, eb) - max(sa, sb) + 1)
    if inter > 0:
        return True
    return abs(center(a) - center(b)) <= max_center_gap


def match_anchors(pass1, pass2):
    """Greedy one-to-one matching of the same semantic event across passes."""
    pairs = []
    for i, a in enumerate(pass1):
        for j, b in enumerate(pass2):
            if semantic_key(a) != semantic_key(b):
                continue
            if not overlap_or_close(a, b):
                continue
            pairs.append((abs(center(a) - center(b)), i, j))
    pairs.sort()
    matched1, matched2, mapping = set(), set(), {}
    for _, i, j in pairs:
        if i in matched1 or j in matched2:
            continue
        matched1.add(i)
        matched2.add(j)
        mapping[i] = j
    return mapping, matched2


def transition_for(schema, entity, skill_type):
    cfg = schema.get("entities", {}).get(entity, {})
    rows = [
        (t.get("from"), t.get("to"))
        for t in cfg.get("transitions", [])
        if t.get("skill_type") == skill_type
    ]
    return rows[0] if len(rows) == 1 else None


def strong_targeted_contradiction(anchor, targeted, schema, min_conf=0.82):
    """Only direct reverse-transition visual evidence may delete an anchor."""
    entity = physical_entity(anchor)
    if entity not in schema.get("entities", {}):
        return None
    tr = transition_for(schema, entity, anchor.get("skill_type"))
    if tr is None:
        return None
    src, dst = tr
    cfg = schema["entities"][entity]
    key = cfg.get("observation_key", entity)
    a0, a1 = span(anchor)

    reverse_pairs = {
        (t.get("from"), t.get("to"))
        for t in cfg.get("transitions", [])
        if t.get("from") == dst and t.get("to") == src
    }
    if not reverse_pairs:
        return None

    for row in targeted:
        if not row.get("_merge_accepted"):
            continue
        if entity not in (row.get("_target_entities") or []):
            continue
        rs = int(row.get("_window_start_frame", row.get("_raw_frame", 0)))
        re = int(row.get("_window_end_frame", row.get("_raw_frame", rs)))
        if re < a0 or rs > a1:
            continue
        sb = (row.get("state_before") or {}).get(key)
        sa = (row.get("state_after") or {}).get(key)
        cb = float((row.get("confidence_before") or {}).get(key, 0.0))
        ca = float((row.get("confidence_after") or {}).get(key, 0.0))
        if min(cb, ca) < min_conf:
            continue
        if (sb, sa) in reverse_pairs:
            return {
                "entity": entity,
                "expected_transition": [src, dst],
                "observed_transition": [sb, sa],
                "confidence": [cb, ca],
                "window": [rs, re],
                "request_kind": row.get("_ambiguity_kind"),
            }
    return None


def compact_skill(x):
    s, e = span(x)
    return {
        "skill_type": x.get("skill_type"),
        "entity": x.get("entity"),
        "fine_entity": x.get("fine_entity"),
        "start_frame": s,
        "end_frame": e,
        "confidence": x.get("confidence"),
    }


def capture(out):
    validated = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []})
    diag = out / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    payload = {
        "annotation_version": "v3.4.6",
        "skill_sequence": validated.get("skill_sequence", []),
        "validation_policy": validated.get("validation_policy", {}),
    }
    write_json(diag / "pass1_anchors.json", payload)
    print("captured conservative anchors:", len(payload["skill_sequence"]))


def reconcile(out, schema_path):
    schema = load_json(schema_path)
    diag = out / "diagnostics"
    pass1_doc = load_json(diag / "pass1_anchors.json", {"skill_sequence": []})
    pass2_doc = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []})
    targeted = load_jsonl(out / "targeted_observations.jsonl")

    pass1 = list(pass1_doc.get("skill_sequence", []))
    pass2 = list(pass2_doc.get("skill_sequence", []))
    mapping, matched2 = match_anchors(pass1, pass2)

    write_json(diag / "pass2_before_reconcile.json", {
        "annotation_version": "v3.4.6",
        "skill_sequence": [compact_skill(x) for x in pass2],
    })

    merged = [dict(x) for x in pass2]
    restored = []
    contradicted = []
    represented = []

    for i, anchor0 in enumerate(pass1):
        anchor = dict(anchor0)
        if i in mapping:
            represented.append({
                "pass1": compact_skill(anchor),
                "pass2": compact_skill(pass2[mapping[i]]),
            })
            continue

        contradiction = strong_targeted_contradiction(anchor, targeted, schema)
        if contradiction is not None:
            contradicted.append({
                "anchor": compact_skill(anchor),
                "contradiction": contradiction,
            })
            continue

        anchor.setdefault("conservative_update", {})
        anchor["conservative_update"].update({
            "restored_from_pass1": True,
            "reason": "validated_pass1_anchor_not_directly_contradicted",
        })
        merged.append(anchor)
        restored.append(compact_skill(anchor))

    # Pass2-only phases are legitimate additions from targeted evidence. Their
    # lifecycle and global task relevance are handled by the final decoder.
    additions = [
        compact_skill(pass2[j])
        for j in range(len(pass2))
        if j not in matched2
    ]

    merged.sort(key=lambda x: (span(x)[0], span(x)[1], x.get("skill_type", "")))
    for new_id, skill in enumerate(merged):
        skill["skill_id"] = new_id

    pass2_doc["annotation_version"] = "v3.4.6"
    pass2_doc["skill_sequence"] = merged
    pass2_doc["num_validated_skills"] = len(merged)
    pass2_doc["conservative_assimilation"] = {
        "pass1_count": len(pass1),
        "raw_pass2_count": len(pass2),
        "reconciled_count": len(merged),
        "restored_pass1_anchors": len(restored),
        "directly_contradicted_pass1_anchors": len(contradicted),
        "pass2_additions": len(additions),
        "policy": "preserve validated pass1 anchors unless strong same-entity targeted reverse-transition evidence contradicts them",
    }
    write_json(out / "skill_candidates_validated.json", pass2_doc)

    report = {
        "annotation_version": "v3.4.6",
        "pass1_count": len(pass1),
        "raw_pass2_count": len(pass2),
        "reconciled_count": len(merged),
        "restored_pass1_anchors": restored,
        "directly_contradicted_pass1_anchors": contradicted,
        "pass2_additions": additions,
        "represented_pass1_anchors": represented,
    }
    write_json(diag / "conservative_update.json", report)
    print(
        "conservative assimilation:",
        f"pass1={len(pass1)} raw_pass2={len(pass2)} final={len(merged)}",
        f"restored={len(restored)} contradicted={len(contradicted)} additions={len(additions)}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["capture", "reconcile"], required=True)
    ap.add_argument("--schema", default=None)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if args.mode == "capture":
        capture(out)
    else:
        if not args.schema:
            raise SystemExit("--schema is required for reconcile")
        reconcile(out, args.schema)


if __name__ == "__main__":
    main()
