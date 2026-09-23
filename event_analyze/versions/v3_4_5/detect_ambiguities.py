#!/usr/bin/env python3
"""V3.4.5 ambiguity-triggered targeted re-observation.

Compared with V3.4.4, one semantic inconsistency produces at most one focused
request. Requests are ranked by semantic impact plus existing robot/interaction
support, so the query budget is a ceiling rather than a target.

Manual GT is never read.
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
    "direction_seen_but_contact_bridge_not_supported",
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
    skill_to_transition = {}
    for name, cfg in schema.get("entities", {}).items():
        rules = {}
        for t in cfg.get("transitions", []):
            rules[(t["from"], t["skill_type"])] = t["to"]
            skill_to_transition[(name, t["skill_type"])] = (t["from"], t["to"])
        if rules:
            out[name] = rules
    return out, skill_to_transition


def decision_support(decision):
    if not decision:
        return 0.0
    sig = float((decision.get("signal") or {}).get("score", 0.0))
    ev = decision.get("interaction_evidence") or {}
    contact = min(1.0, 0.25 * float(ev.get("contact_count", 0)))
    direction = min(1.0, 0.35 * float(ev.get("direction_only_count", ev.get("directed_count", 0))))
    return min(1.0, 0.55 * sig + 0.25 * contact + 0.20 * direction)


def add_request(reqs, *, kind, priority, impact, entities, start, end, reason,
                source, goal, robot_support=0.0):
    reqs.append({
        "kind": kind,
        "priority": int(priority),
        "semantic_impact": int(impact),
        "robot_support": round(float(robot_support), 3),
        "utility_score": round(float(impact) * 100.0 + float(priority) + 25.0 * float(robot_support), 3),
        "semantic_goal": str(goal),
        "target_entities": sorted(set(x for x in entities if x)),
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
    if inter <= 0:
        return 0.0
    union = max(a["end_frame"], b["end_frame"]) - min(a["start_frame"], b["start_frame"]) + 1
    return inter / max(1, union)


def select_requests(requests, max_requests):
    # First keep only the highest-value request for the same semantic goal.
    best = {}
    for r in requests:
        key = (tuple(r["target_entities"]), r["semantic_goal"])
        if key not in best or r["utility_score"] > best[key]["utility_score"]:
            best[key] = r

    ordered = sorted(
        best.values(),
        key=lambda x: (-float(x["utility_score"]), int(x["start_frame"]), x["kind"]),
    )
    kept = []
    for r in ordered:
        if any(
            set(r["target_entities"]) == set(k["target_entities"])
            and request_iou(r, k) >= 0.60
            for k in kept
        ):
            continue
        kept.append(r)
        if len(kept) >= int(max_requests):
            break

    kept.sort(key=lambda x: x["start_frame"])
    for i, r in enumerate(kept):
        r["request_id"] = i
    return kept


def candidate_decision_maps(candidates_doc, report):
    candidates = candidates_doc.get("skill_sequence", [])
    by_id = {
        int(x["skill_id"]): x for x in candidates if "skill_id" in x
    }
    decisions = {
        int(x.get("original_skill_id", -1)): x
        for x in report.get("decisions", [])
        if int(x.get("original_skill_id", -1)) >= 0
    }
    return candidates, by_id, decisions


def best_transition_candidate(candidates, decisions, entity, skill_types, lo, hi):
    skill_types = set(skill_types)
    rows = []
    for c in candidates:
        if c.get("entity") != entity or c.get("skill_type") not in skill_types:
            continue
        s = int(c.get("provisional_start_frame", 0))
        e = int(c.get("provisional_end_frame", s))
        if e < lo or s > hi:
            continue
        d = decisions.get(int(c.get("skill_id", -1)), {})
        support = decision_support(d)
        accepted = bool(d.get("accepted"))
        rows.append((accepted, support, -abs((s + e) // 2 - (lo + hi) // 2), c, d))
    if not rows:
        return None, None, 0.0
    # Prefer rejected candidates: accepted ones cannot recover a missing phase.
    rows.sort(key=lambda x: (x[0], -x[1], -x[2]))
    accepted, support, _, cand, dec = rows[0]
    return cand, dec, support


def padded_span(c, pad, T):
    s = max(0, int(c["provisional_start_frame"]) - pad)
    e = min(T - 1, int(c["provisional_end_frame"]) + pad)
    return s, e


def fallback_window(anchor, before, policy, fps, T):
    width = max(2, int(round(float(policy.get("window_sec", 4.5)) * fps)))
    if before:
        end = min(T - 1, max(0, int(anchor)))
        start = max(0, end - width + 1)
    else:
        start = min(T - 1, max(0, int(anchor)))
        end = min(T - 1, start + width - 1)
    return start, end


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
    max_requests = int(policy.get("max_requests", 8))
    pad = int(round(float(policy.get("padding_sec", 1.25)) * args.fps))
    horizon = int(round(float(policy.get("search_horizon_sec", 8.0)) * args.fps))
    min_robot_support = float(policy.get("min_robot_support_to_query", 0.08))
    T = total_frames(args.hdf5)

    ownership = load_json(out / "hand_object_ownership.json", {"candidates": []})
    report = load_json(out / "interaction_validation_report.json", {"decisions": []})
    candidates_doc = load_json(out / "skill_candidates.json", {"skill_sequence": []})
    validated = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []})
    candidates, candidate_by_id, decisions = candidate_decision_maps(candidates_doc, report)

    tmap, skill_transition = transition_maps(schema)
    requests = []

    # 1) Joint ownership conflicts have high semantic impact.
    for c in ownership.get("candidates", []):
        if c.get("assignment_status") != "ambiguous":
            continue
        entities = [c.get("object")]
        for why in c.get("resolution_reasons", []):
            if why.get("competing_object"):
                entities.append(why["competing_object"])
            entities.extend(why.get("objects", []))
        s = max(0, int(c.get("resolved_start_frame", c.get("start_frame", 0))) - pad)
        e = min(T - 1, int(c.get("resolved_end_frame", c.get("end_frame", 0))) + pad)
        add_request(
            requests,
            kind="ownership_conflict",
            priority=100,
            impact=4,
            entities=entities,
            start=s,
            end=e,
            reason="joint ownership remains ambiguous after arbitration",
            source={"ownership_candidate": c},
            goal="resolve_ownership:" + ",".join(sorted(set(x for x in entities if x))),
            robot_support=float(c.get("direct_confidence", 0.0)),
        )

    # 2) Rejected candidates are queried only when existing robot/interaction
    # evidence says that more vision could plausibly change the decision.
    for d in report.get("decisions", []):
        if d.get("accepted"):
            continue
        reasons = set(d.get("reasons", []))
        if not reasons.intersection(VISUAL_REJECTION_REASONS):
            continue
        sid = int(d.get("original_skill_id", -1))
        c = candidate_by_id.get(sid)
        if not c or not c.get("entity"):
            continue
        support = decision_support(d)
        if support < min_robot_support:
            continue
        s, e = padded_span(c, pad, T)
        add_request(
            requests,
            kind="rejected_visual_evidence",
            priority=85,
            impact=2,
            entities=[c["entity"]],
            start=s,
            end=e,
            reason="candidate has robot support but was rejected for missing/inconsistent visual evidence",
            source={"skill_type": c.get("skill_type"), "reasons": sorted(reasons)},
            goal=f"recover_candidate:{c.get('entity')}:{c.get('skill_type')}",
            robot_support=support,
        )

    initial = (
        validated.get("validation_policy", {}).get("initial_container_state", {})
        or {}
    )
    accepted_skills = validated.get("skill_sequence", [])

    # 3) Receptacle usage without the required ready lifecycle. Instead of
    # tiling the whole search horizon, focus on the best existing transition
    # candidate; if none exists, use one fallback window immediately before use.
    for receptacle, cfg in schema.get("entities", {}).items():
        usage_state = cfg.get("usage_state")
        if usage_state is None or receptacle not in tmap:
            continue
        placements = [
            s for s in candidates
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
        if state == usage_state:
            continue

        skills_to_usage = [
            skill for (src, skill), dst in tmap[receptacle].items()
            if dst == usage_state
        ]
        lo = max(0, first_place - horizon)
        hi = min(T - 1, first_place + pad)
        cand, dec, support = best_transition_candidate(
            candidates, decisions, receptacle, skills_to_usage, lo, hi
        )
        if cand is not None:
            s, e = padded_span(cand, pad, T)
            source = {
                "entity": receptacle,
                "observed_state_before_use": state,
                "usage_state": usage_state,
                "first_placement_frame": first_place,
                "candidate_skill": cand.get("skill_type"),
            }
        else:
            s, e = fallback_window(first_place - pad, True, policy, args.fps, T)
            support = 0.0
            source = {
                "entity": receptacle,
                "observed_state_before_use": state,
                "usage_state": usage_state,
                "first_placement_frame": first_place,
                "candidate_skill": None,
            }
        add_request(
            requests,
            kind="receptacle_usage_without_ready_lifecycle",
            priority=98,
            impact=4,
            entities=[receptacle],
            start=s,
            end=e,
            reason=f"receptacle is used while accepted lifecycle state is {state!r}, not {usage_state!r}",
            source=source,
            goal=f"reach_usage_state:{receptacle}:{usage_state}",
            robot_support=support,
        )

    # 4) Schema accessibility prerequisite missing before an entity is used.
    # This catches cascades such as a downstream articulated action being
    # rejected only because its prerequisite gate transition was missed.
    for entity, cfg in schema.get("entities", {}).items():
        requirements = list(cfg.get("requires", []))
        if not requirements:
            continue

        uses = [
            x for x in candidates
            if x.get("entity") == entity
            or (x.get("state_transition") or {}).get("after") == entity
        ]
        if not uses:
            continue
        first_use = min(int(x.get("provisional_start_frame", 0)) for x in uses)

        for req in requirements:
            req_entity = req.get("entity")
            req_state = req.get("state")
            if not req_entity or req_entity not in tmap:
                continue

            state = initial.get(req_entity, "unknown")
            for s0 in sorted(
                [
                    x for x in accepted_skills
                    if x.get("entity") == req_entity
                    and int(x.get("provisional_end_frame", 0)) <= first_use
                ],
                key=lambda x: x.get("provisional_start_frame", 0),
            ):
                nxt = tmap[req_entity].get((state, s0.get("skill_type")))
                if nxt is not None:
                    state = nxt
            if state == req_state:
                continue

            skills_to_required = [
                skill for (src, skill), dst in tmap[req_entity].items()
                if dst == req_state
            ]
            lo = max(0, first_use - horizon)
            hi = min(T - 1, first_use + pad)
            cand, dec, support = best_transition_candidate(
                candidates, decisions, req_entity, skills_to_required, lo, hi
            )
            if cand is not None:
                s0, e0 = padded_span(cand, pad, T)
            else:
                s0, e0 = fallback_window(first_use - pad, True, policy, args.fps, T)
                support = 0.0

            add_request(
                requests,
                kind="accessibility_prerequisite_missing",
                priority=99,
                impact=4,
                entities=[req_entity],
                start=s0,
                end=e0,
                reason=(
                    f"{entity!r} is used while prerequisite {req_entity!r} "
                    f"is at {state!r}, not required state {req_state!r}"
                ),
                source={
                    "used_entity": entity,
                    "first_use_frame": first_use,
                    "required_entity": req_entity,
                    "observed_prerequisite_state": state,
                    "required_state": req_state,
                    "candidate_skill": cand.get("skill_type") if cand else None,
                },
                goal=f"reach_accessibility_state:{req_entity}:{req_state}",
                robot_support=support,
            )

    # 5) Expected final lifecycle not reached. Again, use one focused candidate
    # window rather than a sequence of overlapping windows.
    current = dict(initial)
    for s in sorted(
        accepted_skills,
        key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"]),
    ):
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
            int(s.get("provisional_end_frame", 0))
            for s in accepted_skills
            if s.get("entity") == entity
            or (s.get("state_transition") or {}).get("after") == entity
        ]
        anchor = max(uses) if uses else 0
        skills_to_final = [
            skill for (src, skill), dst in tmap[entity].items()
            if dst == expected
        ]
        lo = max(0, anchor - pad)
        hi = min(T - 1, anchor + horizon)
        cand, dec, support = best_transition_candidate(
            candidates, decisions, entity, skills_to_final, lo, hi
        )
        if cand is not None:
            s, e = padded_span(cand, pad, T)
        else:
            s, e = fallback_window(anchor + 1, False, policy, args.fps, T)
            support = 0.0

        add_request(
            requests,
            kind="unfinished_lifecycle",
            priority=95,
            impact=4,
            entities=[entity],
            start=s,
            end=e,
            reason=f"schema expects final state {expected!r}, but accepted reasoning ends at {observed_final!r}",
            source={
                "entity": entity,
                "observed_final_state": observed_final,
                "expected_final_state": expected,
                "last_relevant_use_end": anchor if uses else None,
                "candidate_skill": cand.get("skill_type") if cand else None,
            },
            goal=f"reach_final_state:{entity}:{expected}",
            robot_support=support,
        )

    selected = select_requests(requests, max_requests)
    result = {
        "annotation_version": "v3.4.5",
        "task_family": schema.get("task_family"),
        "num_candidate_requests": len(requests),
        "num_requests": len(selected),
        "policy": policy,
        "requests": selected,
        "notes": [
            "Manual ground truth is never read.",
            "One semantic inconsistency produces at most one focused request.",
            "Requests are ranked by semantic impact and existing robot/interaction support.",
            "Accepted transitions that only lack extra directional confirmation do not consume targeted-query budget.",
        ],
    }
    path = out / "ambiguity_requests.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"ambiguity requests: {len(selected)} (from {len(requests)} candidates)")
    for r in selected:
        print(
            f"  Q{r['request_id']:02d} value={r['utility_score']:.1f} "
            f"{r['kind']} {r['target_entities']} "
            f"[{r['start_frame']},{r['end_frame']}]"
        )
    print("saved", path)


if __name__ == "__main__":
    main()
