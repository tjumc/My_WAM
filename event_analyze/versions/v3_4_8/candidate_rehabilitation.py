#!/usr/bin/env python3
"""Revalidate existing rejected transition candidates using direct evidence."""
import argparse
import json
from pathlib import Path


VISUAL_REJECTIONS = {
    "no_entity_specific_hand_contact",
    "direction_seen_but_contact_bridge_not_supported",
    "schema_lifecycle_inconsistent_or_duplicate",
    "schema_lifecycle_ambiguous_from_unknown",
    "receptacle_pull_after_use_started",
    "receptacle_push_before_use_finished",
}


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def span(skill):
    start = int(skill.get("provisional_start_frame", skill.get("raw_start_frame", 0)))
    end = int(skill.get("provisional_end_frame", skill.get("raw_end_frame", start)))
    return start, end


def center(skill):
    start, end = span(skill)
    return (start + end) / 2.0


def decision_support(decision):
    signal = float((decision.get("signal") or {}).get("score", 0.0))
    evidence = decision.get("interaction_evidence") or {}
    contact = min(1.0, 0.25 * float(evidence.get("contact_count", 0)))
    direction = min(1.0, 0.35 * float(evidence.get("direction_only_count", evidence.get("directed_count", 0))))
    return min(1.0, 0.55 * signal + 0.25 * contact + 0.20 * direction)


def transition_for(schema, entity, skill_type):
    matches = [
        (row.get("from"), row.get("to"))
        for row in schema.get("entities", {}).get(entity, {}).get("transitions", [])
        if row.get("skill_type") == skill_type
    ]
    return matches[0] if len(matches) == 1 else None


def candidate_pool(out, schema):
    candidates = load_json(out / "skill_candidates.json", {"skill_sequence": []}).get("skill_sequence", [])
    decisions = load_json(out / "interaction_validation_report.json", {"decisions": []}).get("decisions", [])
    by_id = {int(row.get("skill_id", -1)): row for row in candidates}
    threshold = float(schema.get("targeted_perception", {}).get("min_robot_support_to_query", 0.08))
    rows = []
    for decision in decisions:
        if decision.get("accepted"):
            continue
        reasons = sorted(set(decision.get("reasons", [])))
        candidate = by_id.get(int(decision.get("original_skill_id", -1)))
        if not candidate or candidate.get("entity") not in schema.get("entities", {}):
            continue
        if not set(reasons).intersection(VISUAL_REJECTIONS):
            continue
        support = decision_support(decision)
        if support < threshold or not transition_for(schema, candidate["entity"], candidate.get("skill_type")):
            continue
        rows.append({
            "pass1_skill_id": candidate.get("skill_id"),
            "candidate": candidate,
            "decision": decision,
            "robot_support": round(support, 4),
        })
    return rows


def capture(out, schema):
    rows = candidate_pool(out, schema)
    write_json(Path(out) / "diagnostics" / "rejected_candidate_pool.json", {
        "annotation_version": "v3.4.8",
        "min_robot_support_to_query": schema.get("targeted_perception", {}).get("min_robot_support_to_query", 0.08),
        "candidates": rows,
    })
    print("captured robot-supported rejected candidates:", len(rows))


def request_matches(request, candidate):
    entity = candidate.get("entity")
    skill = candidate.get("skill_type")
    source = request.get("source") or {}
    goal = request.get("semantic_goal", "")
    linked = (
        goal == f"recover_candidate:{entity}:{skill}"
        or (request.get("kind") == "rejected_visual_evidence" and source.get("skill_type") == skill)
        or (source.get("entity") == entity and source.get("candidate_skill") == skill)
    )
    if not linked or entity not in request.get("target_entities", []):
        return False
    start, end = span(candidate)
    return int(request.get("end_frame", -1)) >= start and int(request.get("start_frame", 10**9)) <= end


def targeted_transition_evidence(rows, request, candidate, schema, min_conf):
    entity = candidate["entity"]
    cfg = schema["entities"][entity]
    key = cfg.get("observation_key", entity)
    tokens = set(cfg.get("contact_tokens", []))
    transition = transition_for(schema, entity, candidate["skill_type"])
    if transition is None:
        return None
    source, destination = transition
    motions = {
        motion
        for motion in set(cfg.get("motion_source", {})) | set(cfg.get("motion_endpoint", {}))
        if cfg.get("motion_source", {}).get(motion) == source
        and cfg.get("motion_endpoint", {}).get(motion) == destination
    }
    start, end = span(candidate)
    request_id = request.get("request_id")
    for row in rows:
        if not row.get("_merge_accepted") or entity not in (row.get("_target_entities") or []):
            continue
        if request_id is not None and int(row.get("_event_id", -1)) != 20000 + int(request_id):
            continue
        lo = int(row.get("_window_start_frame", row.get("_raw_frame", 0)))
        hi = int(row.get("_window_end_frame", row.get("_raw_frame", lo)))
        if hi < start or lo > end:
            continue
        before = (row.get("state_before") or {}).get(key)
        after = (row.get("state_after") or {}).get(key)
        confidence_before = float((row.get("confidence_before") or {}).get(key, 0.0))
        confidence_after = float((row.get("confidence_after") or {}).get(key, 0.0))
        before_ok = before == source or before in motions
        after_ok = after == destination or after in motions
        if not before_ok or not after_ok or min(confidence_before, confidence_after) < min_conf:
            continue
        if before == after and before in {source, destination}:
            continue
        hand_values = {
            (row.get("state_before") or {}).get("right_hand"),
            (row.get("state_before") or {}).get("left_hand"),
            (row.get("state_after") or {}).get("right_hand"),
            (row.get("state_after") or {}).get("left_hand"),
        }
        return {
            "request_id": request_id,
            "ambiguity_kind": request.get("kind"),
            "window": [lo, hi],
            "transition": [source, destination],
            "observed_before_after": [before, after],
            "confidence_before_after": [confidence_before, confidence_after],
            "entity_contact_seen": bool(tokens.intersection(hand_values)),
            "targeted_observation_accepted": True,
        }
    return None


def accepted_uses(phases, entity):
    return [
        phase for phase in phases
        if (phase.get("state_transition") or {}).get("after") == entity
        and phase.get("entity") != entity
    ]


def state_before(candidate, phases, schema, validated):
    entity = candidate["entity"]
    cfg = schema["entities"][entity]
    initial = (validated.get("validation_policy", {}).get("initial_container_state") or {}).get(
        entity, cfg.get("expected_initial_state", "unknown")
    )
    state = initial
    events = []
    for phase in phases:
        start, end = span(phase)
        if phase.get("entity") == entity:
            tr = transition_for(schema, entity, phase.get("skill_type"))
            if tr:
                events.append((end, "transition", tr[0], tr[1]))
        target = (phase.get("state_transition") or {}).get("after")
        usage = schema.get("entities", {}).get(target, {}).get("usage_state")
        if target == entity and usage is not None:
            events.append((end, "implication", None, usage))
    for frame, kind, source, destination in sorted(events):
        if frame >= center(candidate):
            break
        if kind == "implication":
            state = destination
        elif state == source or state == "unknown":
            state = destination
    return state


def implication_gap(candidate, phases, schema, validated):
    entity = candidate["entity"]
    cfg = schema["entities"][entity]
    uses = accepted_uses(phases, entity)
    if not uses:
        return None
    transition = transition_for(schema, entity, candidate["skill_type"])
    if transition is None:
        return None
    source, destination = transition
    usage = cfg.get("usage_state")
    final = cfg.get("expected_final_state")
    at = center(candidate)
    use_centers = [center(phase) for phase in uses]
    state = state_before(candidate, phases, schema, validated)

    if destination == usage and at < min(use_centers) and state == source:
        return {
            "kind": "establish_usage_state_before_accepted_use",
            "implied_state": usage,
            "supporting_use_frames": [[*span(phase)] for phase in uses if center(phase) > at],
            "state_before_candidate": state,
        }
    if source == usage and destination == final and at > max(use_centers) and state == source:
        return {
            "kind": "restore_expected_final_state_after_last_accepted_use",
            "implied_state": usage,
            "expected_final_state": final,
            "supporting_use_frames": [[*span(phase)] for phase in uses if center(phase) <= at],
            "state_before_candidate": state,
        }
    return None


def same_candidate(a, b):
    if a.get("entity") != b.get("entity") or a.get("skill_type") != b.get("skill_type"):
        return False
    sa, ea = span(a)
    sb, eb = span(b)
    return max(sa, sb) <= min(ea, eb) or abs(center(a) - center(b)) <= 45


def prepare(out, schema):
    diag = Path(out) / "diagnostics"
    pool = load_json(diag / "rejected_candidate_pool.json", {"candidates": []}).get("candidates", [])
    requests = load_json(Path(out) / "ambiguity_requests.json", {"requests": []}).get("requests", [])
    targeted = load_jsonl(Path(out) / "targeted_observations.jsonl")
    validated = load_json(Path(out) / "skill_candidates_validated.json", {"skill_sequence": []})
    anchors = load_json(diag / "pass1_anchors.json", {"skill_sequence": []}).get("skill_sequence", [])
    current_phases = validated.get("skill_sequence", [])
    phases = list(current_phases) + [
        anchor for anchor in anchors
        if not any(same_candidate(anchor, phase) for phase in current_phases)
    ]
    candidates_doc = load_json(Path(out) / "skill_candidates.json", {"skill_sequence": []})
    candidates = list(candidates_doc.get("skill_sequence", []))
    min_conf = float(schema.get("targeted_perception", {}).get("min_confidence_to_merge", 0.65))
    min_robot = float(schema.get("targeted_perception", {}).get("min_robot_support_to_query", 0.08))
    audit = {
        "annotation_version": "v3.4.8",
        "policy": "rehabilitate only a pass1-rejected, robot-supported transition with a selected matching query, direct directional targeted evidence, and a placement-implied lifecycle gap",
        "candidates": [],
    }

    for row in pool:
        candidate = dict(row["candidate"])
        decision = row.get("decision") or {}
        support = float(row.get("robot_support", decision_support(decision)))
        record = {
            "pass1_skill_id": row.get("pass1_skill_id"),
            "entity": candidate.get("entity"),
            "skill_type": candidate.get("skill_type"),
            "candidate_span": list(span(candidate)),
            "robot_support": support,
            "status": "not_rehabilitated",
        }
        matched = [req for req in requests if request_matches(req, candidate)]
        if support < min_robot or not matched:
            record["reason"] = "no_selected_robot_supported_targeted_request_for_candidate"
            audit["candidates"].append(record)
            continue
        evidence = next((
            ev for req in matched
            if (ev := targeted_transition_evidence(targeted, req, candidate, schema, min_conf))
        ), None)
        if evidence is None:
            record["reason"] = "no_high_confidence_directional_targeted_observation"
            audit["candidates"].append(record)
            continue
        gap = implication_gap(candidate, phases, schema, validated)
        if gap is None:
            record["reason"] = "no_accepted_placement_implies_candidate_transition"
            record["targeted_evidence"] = evidence
            audit["candidates"].append(record)
            continue

        record.update({
            "status": "queued_for_revalidation",
            "reason": "robot_support_targeted_direction_and_state_implication_agree",
            "targeted_evidence": evidence,
            "state_implication": gap,
        })
        rehab_id = f"pass1:{row.get('pass1_skill_id')}:{candidate['entity']}:{candidate['skill_type']}:{span(candidate)[0]}"
        candidate["candidate_rehabilitation"] = {
            "rehabilitation_id": rehab_id,
            "pass1_skill_id": row.get("pass1_skill_id"),
            "state_implication": gap,
            "targeted_evidence": evidence,
            "robot_support": support,
        }
        existing = next((x for x in candidates if same_candidate(x, candidate)), None)
        if existing:
            existing["candidate_rehabilitation"] = candidate["candidate_rehabilitation"]
            record["queued_source"] = "matching_pass2_candidate"
        else:
            candidates.append(candidate)
            record["queued_source"] = "preserved_pass1_rejected_candidate"
        record["rehabilitation_id"] = rehab_id
        audit["candidates"].append(record)

    candidates.sort(key=lambda x: (span(x)[0], span(x)[1], x.get("skill_type", "")))
    for skill_id, candidate in enumerate(candidates):
        candidate["skill_id"] = skill_id
    candidates_doc["skill_sequence"] = candidates
    candidates_doc["num_inferred_skills"] = len(candidates)
    write_json(Path(out) / "skill_candidates.json", candidates_doc)
    write_json(diag / "candidate_rehabilitation.json", audit)
    print("candidate rehabilitation queued:", sum(x["status"] == "queued_for_revalidation" for x in audit["candidates"]))


def finalize(out):
    out = Path(out)
    diag = out / "diagnostics"
    audit = load_json(diag / "candidate_rehabilitation.json", {"candidates": []})
    validated = load_json(out / "skill_candidates_validated.json", {"skill_sequence": []}).get("skill_sequence", [])
    decisions = load_json(out / "interaction_validation_report.json", {"decisions": []}).get("decisions", [])
    decision_by_id = {int(x.get("original_skill_id", -1)): x for x in decisions}
    candidate_by_rehab_id = {
        (x.get("candidate_rehabilitation") or {}).get("rehabilitation_id"): x
        for x in validated
    }
    input_candidates = load_json(out / "skill_candidates.json", {"skill_sequence": []}).get("skill_sequence", [])
    input_by_rehab_id = {
        (x.get("candidate_rehabilitation") or {}).get("rehabilitation_id"): x
        for x in input_candidates
    }
    for row in audit.get("candidates", []):
        rehab_id = row.get("rehabilitation_id")
        if row.get("status") != "queued_for_revalidation" or not rehab_id:
            continue
        validated_candidate = candidate_by_rehab_id.get(rehab_id)
        source_candidate = input_by_rehab_id.get(rehab_id)
        current_decision = decision_by_id.get(int((source_candidate or {}).get("skill_id", -1)), {})
        if validated_candidate:
            row["status"] = "accepted_by_interaction_validator"
            row["validator_reasons"] = validated_candidate.get("validation", {}).get("reasons", [])
        else:
            row["status"] = "rejected_by_interaction_validator"
            row["validator_reasons"] = current_decision.get("reasons", [])
        row["validator_signal"] = current_decision.get("signal")
    write_json(diag / "candidate_rehabilitation.json", audit)
    print("candidate rehabilitation accepted:", sum(x["status"] == "accepted_by_interaction_validator" for x in audit.get("candidates", [])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["capture", "prepare", "finalize"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--schema")
    args = ap.parse_args()
    if args.mode == "finalize":
        finalize(args.output_dir)
        return
    if not args.schema:
        ap.error("--schema is required for capture/prepare")
    schema = load_json(args.schema)
    if args.mode == "capture":
        capture(args.output_dir, schema)
    else:
        prepare(args.output_dir, schema)


if __name__ == "__main__":
    main()
