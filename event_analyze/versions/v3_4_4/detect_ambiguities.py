#!/usr/bin/env python3
"""Detect trajectory ambiguities that justify targeted visual re-observation.

This stage consumes reasoning outputs, never manual GT. It asks for more
perception only when uncertainty can change the semantic annotation.

Triggers are generic:
- unresolved hand-object ownership conflict;
- candidate rejected because visual interaction evidence is insufficient;
- receptacle is used without a lifecycle that reaches its schema usage state;
- articulated entity fails to reach schema-declared expected final state;
- accepted container action has contact but lacks direction evidence.

Fine identity uncertainty that is already safely handled by semantic backoff is
NOT automatically re-observed.
"""
import argparse
import json
from pathlib import Path

import h5py


VISUAL_REJECTION_REASONS = {
    "no_entity_specific_hand_contact",
    "no_reliable_fine_hold_or_uniquely_owned_coarse_bridge",
    "schema_lifecycle_inconsistent_or_duplicate",
    "schema_lifecycle_ambiguous_from_unknown",
    "receptacle_pull_after_use_started",
    "receptacle_push_before_use_finished",
}


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def total_frames(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        if "time" in f:
            return len(f["time"])
        return len(f["images_dict/head/rgb_size"])


def transition_maps(schema):
    out = {}
    for name, cfg in schema.get("entities", {}).items():
        rules = {}
        for t in cfg.get("transitions", []):
            rules[(t["from"], t["skill_type"])] = t["to"]
        if rules:
            out[name] = rules
    return out


def relevant_use_end(skill, entity):
    if skill.get("entity") == entity:
        return int(skill.get("provisional_end_frame", 0))
    st = skill.get("state_transition") or {}
    if st.get("after") == entity:
        return int(skill.get("provisional_end_frame", 0))
    fine = skill.get("fine_entity")
    if fine and st.get("after") == entity:
        return int(skill.get("provisional_end_frame", 0))
    return None


def windowed_requests(kind, priority, entities, start, end, reason, policy, fps, source):
    start, end = int(start), int(end)
    if end < start:
        return []
    width = max(2, int(round(float(policy.get("window_sec", 4.5)) * fps)))
    stride = max(1, int(round(float(policy.get("stride_sec", 2.0)) * fps)))
    out = []
    cur = start
    while cur <= end:
        wend = min(end, cur + width - 1)
        if wend - cur + 1 >= max(2, int(round(1.0 * fps))):
            out.append({
                "kind": kind,
                "priority": int(priority),
                "target_entities": sorted(set(entities)),
                "start_frame": int(cur),
                "end_frame": int(wend),
                "reason": reason,
                "source": source,
            })
        if wend >= end:
            break
        cur += stride
    return out


def add_single(reqs, kind, priority, entities, start, end, reason, source):
    reqs.append({
        "kind": kind,
        "priority": int(priority),
        "target_entities": sorted(set(entities)),
        "start_frame": int(start),
        "end_frame": int(end),
        "reason": reason,
        "source": source,
    })


def request_iou(a, b):
    if set(a["target_entities"]) != set(b["target_entities"]):
        return 0.0
    lo = max(a["start_frame"], b["start_frame"])
    hi = min(a["end_frame"], b["end_frame"])
    inter = max(0, hi - lo + 1)
    if inter == 0:
        return 0.0
    union = max(a["end_frame"], b["end_frame"]) - min(a["start_frame"], b["start_frame"]) + 1
    return inter / max(1, union)


def dedupe(requests, max_requests):
    ordered = sorted(
        requests,
        key=lambda x: (-int(x["priority"]), int(x["start_frame"]), x["kind"]),
    )
    kept = []
    for r in ordered:
        if any(request_iou(r, k) >= 0.55 for k in kept):
            continue
        kept.append(r)
        if len(kept) >= int(max_requests):
            break
    for i, r in enumerate(sorted(kept, key=lambda x: x["start_frame"])):
        r["request_id"] = i
    return sorted(kept, key=lambda x: x["request_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    schema = load_json(args.schema)
    policy = schema.get("targeted_perception", {})
    max_requests = int(policy.get("max_requests", 6))
    pad = int(round(float(policy.get("padding_sec", 1.25)) * args.fps))
    horizon = int(round(float(policy.get("search_horizon_sec", 10.0)) * args.fps))
    T = total_frames(args.hdf5)

    ownership = load_json(out / "hand_object_ownership.json", {"candidates": []})
    report = load_json(out / "interaction_validation_report.json", {"decisions": []})
    candidates_doc = load_json(out / "skill_candidates.json", {"skill_sequence": []})
    validated = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []})

    candidate_by_id = {
        int(x["skill_id"]): x for x in candidates_doc.get("skill_sequence", [])
        if "skill_id" in x
    }

    requests = []

    # 1) Unresolved joint ownership.
    for c in ownership.get("candidates", []):
        if c.get("assignment_status") != "ambiguous":
            continue
        entities = [c.get("object")]
        for why in c.get("resolution_reasons", []):
            if why.get("competing_object"):
                entities.append(why["competing_object"])
            entities.extend(why.get("objects", []))
        entities = [x for x in entities if x]
        add_single(
            requests,
            "ownership_conflict",
            100,
            entities,
            max(0, int(c.get("resolved_start_frame", c.get("start_frame", 0))) - pad),
            min(T - 1, int(c.get("resolved_end_frame", c.get("end_frame", 0))) + pad),
            "joint ownership remains ambiguous after arbitration",
            {"ownership_candidate": c},
        )

    # 2) Rejected candidates where more visual evidence could change the decision.
    for d in report.get("decisions", []):
        if d.get("accepted"):
            continue
        reasons = set(d.get("reasons", []))
        if not reasons.intersection(VISUAL_REJECTION_REASONS):
            continue
        sid = int(d.get("original_skill_id", -1))
        c = candidate_by_id.get(sid)
        if not c:
            continue
        entity = c.get("entity")
        if not entity:
            continue
        add_single(
            requests,
            "rejected_visual_evidence",
            90,
            [entity],
            max(0, int(c["provisional_start_frame"]) - pad),
            min(T - 1, int(c["provisional_end_frame"]) + pad),
            "candidate rejected for insufficient or inconsistent visual interaction evidence",
            {"skill_type": c.get("skill_type"), "reasons": sorted(reasons)},
        )

    # 3) Receptacle use is observed/candidate-supported but its accepted
    # lifecycle does not reach the schema-declared usage state beforehand.
    tmap = transition_maps(schema)
    initial = (
        validated.get("validation_policy", {}).get("initial_container_state", {})
        or {}
    )
    accepted_skills = validated.get("skill_sequence", [])

    for receptacle, cfg in schema.get("entities", {}).items():
        usage_state = cfg.get("usage_state")
        if usage_state is None or receptacle not in tmap:
            continue

        placements = [
            s for s in candidates_doc.get("skill_sequence", [])
            if (s.get("state_transition") or {}).get("after") == receptacle
        ]
        if not placements:
            continue

        first_place = min(int(s["provisional_start_frame"]) for s in placements)
        state = initial.get(receptacle, "unknown")
        for s in sorted(
            [
                x for x in accepted_skills
                if x.get("entity") == receptacle
                and int(x["provisional_end_frame"]) <= first_place
            ],
            key=lambda x: x["provisional_start_frame"],
        ):
            nxt = tmap[receptacle].get((state, s.get("skill_type")))
            if nxt is not None:
                state = nxt

        if state != usage_state:
            start = max(0, first_place - horizon)
            end = min(T - 1, first_place + pad)
            requests.extend(windowed_requests(
                "receptacle_usage_without_ready_lifecycle",
                98,
                [receptacle],
                start,
                end,
                f"portable-object placement evidence uses receptacle while accepted lifecycle state is {state!r}, not usage state {usage_state!r}",
                policy,
                args.fps,
                {
                    "entity": receptacle,
                    "observed_state_before_use": state,
                    "usage_state": usage_state,
                    "first_placement_frame": first_place,
                },
            ))

    # 4) Expected final lifecycle not reached.
    initial = (
        validated.get("validation_policy", {}).get("initial_container_state", {})
        or {}
    )
    current = dict(initial)
    skills = validated.get("skill_sequence", [])
    for s in sorted(skills, key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"])):
        entity = s.get("entity")
        if entity not in tmap:
            continue
        cur = current.get(entity, "unknown")
        nxt = tmap[entity].get((cur, s.get("skill_type")))
        if nxt is not None:
            current[entity] = nxt

    for entity, cfg in schema.get("entities", {}).items():
        expected = cfg.get("expected_final_state")
        if expected is None or entity not in tmap:
            continue
        observed_final = current.get(entity, "unknown")
        if observed_final == expected:
            continue

        uses = [
            z for z in (relevant_use_end(s, entity) for s in skills)
            if z is not None
        ]
        start = max(0, (max(uses) if uses else 0) - int(round(0.5 * args.fps)))
        end = min(T - 1, start + horizon)
        requests.extend(windowed_requests(
            "unfinished_lifecycle",
            95,
            [entity],
            start,
            end,
            f"schema expected final state {expected!r}, but accepted reasoning ends at {observed_final!r}",
            policy,
            args.fps,
            {
                "entity": entity,
                "observed_final_state": observed_final,
                "expected_final_state": expected,
                "last_relevant_use_end": max(uses) if uses else None,
            },
        ))

    # 5) Accepted container transitions that have contact but no directional evidence.
    for d in report.get("decisions", []):
        if not d.get("accepted"):
            continue
        ev = d.get("interaction_evidence") or {}
        if int(ev.get("contact_count", 0)) <= 0 or int(ev.get("directed_count", 0)) > 0:
            continue
        sid = int(d.get("original_skill_id", -1))
        c = candidate_by_id.get(sid)
        if not c or c.get("entity") not in tmap:
            continue
        center = int(ev.get("nearest_contact_frame", c["provisional_start_frame"]))
        add_single(
            requests,
            "weak_direction_confirmation",
            60,
            [c["entity"]],
            max(0, center - int(round(2.25 * args.fps))),
            min(T - 1, center + int(round(2.25 * args.fps))),
            "accepted articulated transition has contact but no direct directional visual evidence",
            {"skill_type": c.get("skill_type"), "contact_frame": center},
        )

    requests = dedupe(requests, max_requests)
    result = {
        "annotation_version": "v3.4.4",
        "task_family": schema.get("task_family"),
        "num_requests": len(requests),
        "policy": policy,
        "requests": requests,
        "notes": [
            "Manual ground truth is never read.",
            "Reliable semantic backoff alone does not trigger re-observation.",
            "Requests are generated only when additional visual evidence can change semantic or temporal reasoning.",
        ],
    }
    path = out / "ambiguity_requests.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"ambiguity requests: {len(requests)}")
    for r in requests:
        print(
            f"  Q{r['request_id']:02d} p={r['priority']} {r['kind']} "
            f"{r['target_entities']} [{r['start_frame']},{r['end_frame']}]"
        )
    print("saved", path)


if __name__ == "__main__":
    main()
