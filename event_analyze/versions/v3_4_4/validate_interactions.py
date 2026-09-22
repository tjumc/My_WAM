#!/usr/bin/env python3
"""V3.4.4 interaction-grounded validation for tracked skill candidates.

Visual state changes are not sufficient. Container transitions must be supported
by robot interaction evidence. Object placements require an actually observed
held state. Short state reversals are treated as visual jitter.
"""
import argparse
import json
import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from schema_runtime import (
    load_schema, container_rules, portable_rules,
    receptacle_placements, accessibility_requirements, stable_motion_rules,
    training_semantic, parent_members, skill_labels,
)

import h5py
import numpy as np

CONTAINER_ACTIONS = {}
CONTACT_TOKEN = {}
CONTAINER_OBS_KEY = {}
CONTAINER_MOTION = {}
OBJECT_TRACK = {}
OBJECT_SKILLS = set()
RECEPTACLE_PLACEMENTS = {}
ACCESS_REQUIREMENTS = {}
CONTAINER_TRANSITIONS = {}
INITIAL_RULES = {}
OBJECT_CONFIG = {}
SKILL_LABELS = {}

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def get_dataset(f, candidates):
    for p in candidates:
        if p in f:
            return f[p]
    raise KeyError("missing HDF5 dataset; tried: " + ", ".join(candidates))


def load_signals(path):
    with h5py.File(path, "r") as f:
        qv = get_dataset(f, ["joints_dict/joints_velocity_state", "joints_velocity_state"])[:].astype(float)
        rw = get_dataset(f, [
            "endpoint_wrench_dict/astribot_arm_right",
            "astribot_arm_right_wrench",
        ])[:].astype(float)
        rg = get_dataset(f, [
            "command_poses_dict/astribot_gripper_right",
            "astribot_gripper_right_command",
        ])[:, 0].astype(float)

    arm = np.linalg.norm(qv[:, 15:22], axis=1) if qv.shape[1] >= 22 else np.linalg.norm(qv[:, 3:], axis=1)
    force = np.linalg.norm(rw[:, :3], axis=1)
    force_change = np.abs(np.diff(force, prepend=force[0]))
    grip_change = np.abs(np.diff(rg, prepend=rg[0]))
    return {
        "arm": arm,
        "force_change": force_change,
        "grip_change": grip_change,
        "arm_thr": float(np.percentile(arm, 60)),
        "force_ref": max(1e-6, float(np.percentile(force_change, 90))),
        "grip_ref": max(1e-6, float(np.percentile(grip_change, 90))),
    }


def signal_support(signals, start, end, pad=20):
    T = len(signals["arm"])
    s = max(0, int(start) - pad)
    e = min(T, int(end) + pad + 1)
    if e <= s:
        e = min(T, s + 1)
    arm = signals["arm"][s:e]
    fc = signals["force_change"][s:e]
    gc = signals["grip_change"][s:e]
    motion = float(np.mean(arm > max(1e-5, signals["arm_thr"])))
    force = min(1.0, float(np.percentile(fc, 90)) / signals["force_ref"])
    grip = min(1.0, float(np.percentile(gc, 90)) / signals["grip_ref"])
    score = 0.55 * motion + 0.25 * force + 0.20 * grip
    return {
        "window": [s, e - 1],
        "motion_fraction": round(motion, 3),
        "force_change_support": round(force, 3),
        "gripper_change_support": round(grip, 3),
        "score": round(score, 3),
    }


def obs_near(obs, start, end, radius=120):
    lo, hi = int(start) - radius, int(end) + radius
    return [o for o in obs if lo <= int(o["_raw_frame"]) <= hi]


def hand_contact(o, tokens):
    if isinstance(tokens, str):
        tokens = {tokens}
    else:
        tokens = set(tokens or [])
    for side in ("before", "after"):
        st = o.get(f"state_{side}") or {}
        if st.get("right_hand") in tokens or st.get("left_hand") in tokens:
            return True
    return False


def expected_transition(entity, skill_type):
    rows = [
        (src, dst)
        for (src, skill), dst in CONTAINER_TRANSITIONS.get(entity, {}).items()
        if skill == skill_type
    ]
    return rows[0] if len(rows) == 1 else None


def expected_motion_states(entity, skill_type):
    tr = expected_transition(entity, skill_type)
    if tr is None:
        return set()
    src, dst = tr
    return {
        motion
        for motion, endpoints in CONTAINER_MOTION.get(entity, {}).items()
        if endpoints == (src, dst)
    }


def direction_match(o, entity, skill_type):
    tr = expected_transition(entity, skill_type)
    if tr is None:
        return False
    src, dst = tr
    motions = expected_motion_states(entity, skill_type)
    key = CONTAINER_OBS_KEY.get(entity, entity)
    sb = (o.get("state_before") or {}).get(key)
    sa = (o.get("state_after") or {}).get(key)

    before_ok = sb == src or sb in motions
    after_ok = sa == dst or sa in motions
    if not (before_ok and after_ok):
        return False
    # stable->same-stable is not directional evidence; motion->motion is.
    if sb == sa and sb in {src, dst}:
        return False
    return True


def dense_direction_match(x, entity, skill_type):
    return x.get("relative_motion") in expected_motion_states(entity, skill_type)


def container_evidence(skill, obs, dense_rows=None):
    entity = skill["entity"]
    tokens = CONTACT_TOKEN[entity]
    nearby = obs_near(obs, skill["provisional_start_frame"], skill["provisional_end_frame"])

    def usable_contact(o):
        if not hand_contact(o, tokens):
            return False
        dense_target = o.get("_v331_dense_target_entity")
        if dense_target == entity:
            # Dense evidence is directional. A moving-out dense window must not
            # be recycled as generic contact evidence for a push-in candidate.
            return direction_match(o, entity, skill["skill_type"])
        return True

    contacts = [o for o in nearby if usable_contact(o)]
    directed = [o for o in nearby if direction_match(o, entity, skill["skill_type"]) and hand_contact(o, tokens)]

    dense_rows = dense_rows or []
    lo = int(skill["provisional_start_frame"]) - 120
    hi = int(skill["provisional_end_frame"]) + 120
    dense_near = [
        x for x in dense_rows
        if x.get("entity") == entity
        and int(x.get("window_end_frame", -1)) >= lo
        and int(x.get("window_start_frame", 10**9)) <= hi
    ]
    dense_contact = [
        x for x in dense_near
        if float(x.get("confidence", 0.0)) >= 0.62
        and x.get("right_hand_relation") in tokens
        and dense_direction_match(x, entity, skill["skill_type"])
    ]
    dense_ids = [int(x.get("event_id", 10000 + int(x.get("dense_window_id", 0)))) for x in dense_contact]
    dense_frames = [int(x.get("center_frame", 0)) for x in dense_contact]
    generic_frames = [int(o["_raw_frame"]) for o in contacts]
    all_frames = generic_frames + dense_frames

    return {
        "nearby_event_ids": [int(o["_event_id"]) for o in nearby],
        "contact_event_ids": [int(o["_event_id"]) for o in contacts] + dense_ids,
        "directed_contact_event_ids": [int(o["_event_id"]) for o in directed] + dense_ids,
        "contact_count": len(contacts) + len(dense_contact),
        "directed_count": len(directed) + len(dense_contact),
        "dense_directional_contact_count": len(dense_contact),
        "nearest_contact_frame": (
            min(all_frames, key=lambda f: abs(f - int(skill["provisional_start_frame"])))
            if all_frames else None
        ),
    }


def held_observed(skill, tracked):
    entity = OBJECT_TRACK.get(skill.get("entity"))
    if not entity:
        return False, []
    segs = tracked["tracks"][entity]["segments"]
    s, e = int(skill["provisional_start_frame"]), int(skill["provisional_end_frame"])
    good = [
        x for x in segs
        if x["state"] in {"right_hand", "left_hand"}
        and x.get("support") == "observed"
        and float(x.get("confidence", 0.0)) >= 0.70
        and x["end_frame"] >= s
        and x["start_frame"] <= e
    ]
    return bool(good), good


def ownership_candidate(skill, ownership):
    obj = skill.get("entity")
    s = int(skill["provisional_start_frame"])
    e = int(skill["provisional_end_frame"])
    rows = [
        x for x in ownership.get("candidates", [])
        if x.get("object") == obj
        and int(x.get("resolved_end_frame", -1)) >= s
        and int(x.get("resolved_start_frame", 10**18)) <= e
    ]
    if not rows:
        return None
    rows.sort(
        key=lambda x: (
            x.get("assignment_status") != "direct",
            -float(x.get("direct_confidence", 0.0)),
            abs(int(x.get("resolved_start_frame", s)) - s),
        )
    )
    return rows[0]


def observed_source_target_bridge(skill, tracked):
    """Require directly observed source and target around an inferred ownership."""
    obj = skill.get("entity")
    key = OBJECT_TRACK.get(obj)
    cfg = OBJECT_CONFIG.get(obj)
    if not key or not cfg or key not in tracked.get("tracks", {}):
        return False, {}

    segs = tracked["tracks"][key].get("segments", [])
    start = int(skill["provisional_start_frame"])
    end = int(skill["provisional_end_frame"])
    source_states = set(cfg.get("source_states", []))
    target = cfg.get("target")

    sources = [
        x for x in segs
        if x.get("state") in source_states
        and x.get("support") == "observed"
        and float(x.get("confidence", 0.0)) >= 0.70
        and int(x.get("start_frame", 0)) <= start
    ]
    targets = [
        x for x in segs
        if x.get("state") == target
        and x.get("support") == "observed"
        and float(x.get("confidence", 0.0)) >= 0.70
        and int(x.get("end_frame", -1)) >= end
    ]
    if not sources or not targets:
        return False, {"sources": sources, "targets": targets}
    return True, {
        "source": max(sources, key=lambda x: int(x.get("end_frame", -1))),
        "target": min(targets, key=lambda x: int(x.get("start_frame", 10**18))),
    }


def ownership_supported_coarse(skill, tracked, ownership, schema):
    """Validate a coarse parent label without borrowing sibling held evidence.

    The candidate must own its own interval after joint arbitration. For an
    inferred ownership, directly observed source and target states are also
    required. This preserves recall under occlusion while preventing one
    sibling object's held evidence from validating another sibling episode.
    """
    sem = training_semantic(schema, skill.get("entity"))
    if not sem.get("backoff"):
        return False, {}
    policy = schema.get("manipulation_policy", {})
    if not policy.get("allow_inferred_parent_ownership", False):
        return False, {}

    row = ownership_candidate(skill, ownership)
    if row is None:
        return False, {"reason": "no_ownership_candidate"}
    status = row.get("assignment_status")
    if status not in {"direct", "inferred_unique"}:
        return False, {"reason": f"ownership_status_{status}", "ownership": row}

    if status == "direct":
        return True, {"ownership": row, "mode": "direct_parent_ownership"}

    ok, bridge = observed_source_target_bridge(skill, tracked)
    if not ok:
        return False, {
            "reason": "inferred_ownership_without_observed_source_target_bridge",
            "ownership": row,
            "bridge": bridge,
        }
    return True, {
        "ownership": row,
        "mode": "inferred_unique_with_observed_source_target_bridge",
        "bridge": bridge,
    }


def placement_has_held_evidence(skill, tracked, schema, ownership):
    ok, _ = held_observed(skill, tracked)
    if ok:
        return True
    coarse_ok, _ = ownership_supported_coarse(skill, tracked, ownership, schema)
    return coarse_ok


def affordance_temporal_rejections(skills, tracked, schema, ownership, fps=30.0, tolerance_sec=1.0, skip_indices=None):
    """Reject container transitions inconsistent with receptacle usage.

    This is role-based rather than a hard-coded action sequence:
    - pull_out(receptacle) should happen before the first place(*, receptacle)
    - push_in(receptacle) should happen after the last place(*, receptacle)
    """
    rejected = {}
    skip_indices = set(skip_indices or [])
    tol = int(round(float(fps) * float(tolerance_sec)))
    for receptacle, placement_types in RECEPTACLE_PLACEMENTS.items():
        placements = [
            s for s in skills
            if s["skill_type"] in placement_types and placement_has_held_evidence(s, tracked, schema, ownership)
        ]
        if not placements:
            continue
        first_place = min(int(s["provisional_start_frame"]) for s in placements)
        last_place = max(int(s["provisional_end_frame"]) for s in placements)
        for i, s in enumerate(skills):
            if i in skip_indices or s.get("entity") != receptacle:
                continue
            typ = s["skill_type"]
            start = int(s["provisional_start_frame"])
            end = int(s["provisional_end_frame"])
            if typ.startswith("pull_out_") and start > first_place + tol:
                rejected[i] = {
                    "reason": "receptacle_pull_after_use_started",
                    "first_placement_frame": first_place,
                    "tolerance_frames": tol,
                }
            elif typ.startswith("push_in_") and end < last_place - tol:
                rejected[i] = {
                    "reason": "receptacle_push_before_use_finished",
                    "last_placement_frame": last_place,
                    "tolerance_frames": tol,
                }
    return rejected


def mark_short_reverse_pairs(skills, max_gap_frames=30, skip_indices=None):
    rejected = set()
    skip_indices = set(skip_indices or [])
    by_entity = {}
    for i, s in enumerate(skills):
        if i in skip_indices:
            continue
        if s.get("entity") in CONTAINER_ACTIONS:
            by_entity.setdefault(s["entity"], []).append((i, s))
    for entity, items in by_entity.items():
        items.sort(key=lambda x: x[1]["provisional_start_frame"])
        for (ia, a), (ib, b) in zip(items, items[1:]):
            pair = CONTAINER_ACTIONS[entity]
            if a["skill_type"] == b["skill_type"]:
                continue
            gap = int(b["provisional_start_frame"]) - int(a["provisional_end_frame"])
            if gap <= max_gap_frames:
                rejected.add(ia)
                rejected.add(ib)
    return rejected


def initial_container_states(obs, max_observations=5, min_conf=0.70):
    """Infer initial lifecycle from schema-defined early direct observations."""
    generic = [
        o for o in sorted(
            obs,
            key=lambda x: (
                int(x.get("_window_start_frame", 0)),
                int(x.get("_event_id", 0)),
            ),
        )
        if not o.get("_v331_dense_target_entity") and not o.get("_v33_dense_cutlery")
    ][:max_observations]

    out = {}
    for logical_entity, rule in INITIAL_RULES.items():
        key = rule["observation_key"]
        stable_states = set(rule["stable"])
        motion_map = dict(rule["motion_source"])
        motion_source = None
        earliest_stable = None
        transition_source = None

        for o in generic:
            sb = (o.get("state_before") or {}).get(key)
            sa = (o.get("state_after") or {}).get(key)
            cb = float((o.get("confidence_before") or {}).get(key, 0.0))
            ca = float((o.get("confidence_after") or {}).get(key, 0.0))

            if motion_source is None and sb in motion_map and cb >= min_conf:
                motion_source = motion_map[sb]
            if earliest_stable is None and sb in stable_states and cb >= min_conf:
                earliest_stable = sb
            if (
                transition_source is None
                and sb in stable_states and sa in stable_states
                and sb != sa and min(cb, ca) >= min_conf
            ):
                transition_source = sb

        out[logical_entity] = motion_source or earliest_stable or transition_source or "unknown"
    return out


def validate(args):
    schema = load_schema(args.schema)
    global CONTAINER_ACTIONS, CONTACT_TOKEN, CONTAINER_OBS_KEY, CONTAINER_MOTION
    global OBJECT_TRACK, OBJECT_SKILLS
    global RECEPTACLE_PLACEMENTS, ACCESS_REQUIREMENTS, CONTAINER_TRANSITIONS, INITIAL_RULES
    global OBJECT_CONFIG, SKILL_LABELS

    cr = container_rules(schema)
    CONTAINER_ACTIONS = {}
    CONTACT_TOKEN = {}
    CONTAINER_TRANSITIONS = {}
    CONTAINER_OBS_KEY = {}
    CONTAINER_MOTION = {}
    for entity, cfg in cr.items():
        skills = tuple(cfg["rules"].values())
        CONTAINER_ACTIONS[entity] = skills
        tokens = cfg.get("contact_tokens", [])
        CONTACT_TOKEN[entity] = set(tokens or [f"contact_{entity}"])
        CONTAINER_OBS_KEY[entity] = cfg.get("observation_key", entity)
        CONTAINER_MOTION[entity] = {
            motion: (
                cfg.get("motion_source", {}).get(motion),
                cfg.get("motion_endpoint", {}).get(motion),
            )
            for motion in set(cfg.get("motion_source", {})) | set(cfg.get("motion_endpoint", {}))
        }
        CONTAINER_TRANSITIONS[entity] = {
            (src, skill): dst for (src, dst), skill in cfg["rules"].items()
        }

    pr = portable_rules(schema)
    OBJECT_TRACK = {name: cfg["observation_key"] for name, cfg in pr.items()}
    OBJECT_CONFIG = pr
    OBJECT_SKILLS = {cfg["skill_type"] for cfg in pr.values()}
    SKILL_LABELS = skill_labels(schema)
    RECEPTACLE_PLACEMENTS = receptacle_placements(schema)
    ACCESS_REQUIREMENTS = accessibility_requirements(schema)
    INITIAL_RULES = stable_motion_rules(schema)

    out = Path(args.output_dir)
    cand = json.load(open(out / "skill_candidates.json", encoding="utf-8"))
    owned_path = out / "tracked_entity_states_owned.json"
    tracked_path = owned_path if owned_path.exists() else out / "tracked_entity_states.json"
    tracked = json.load(open(tracked_path, encoding="utf-8"))
    ownership_path = out / "hand_object_ownership.json"
    ownership = (
        json.load(open(ownership_path, encoding="utf-8"))
        if ownership_path.exists() else {"candidates": []}
    )
    obs = load_jsonl(out / "entity_observations.jsonl")
    dense_path = out / "dense_cutlery_transition_evidence.jsonl"
    dense_rows = load_jsonl(dense_path) if dense_path.exists() else []
    signals = load_signals(args.hdf5)
    skills = cand.get("skill_sequence", [])

    # Resolve local opposite-direction jitter before applying long-horizon
    # affordance constraints; otherwise rejecting one half can leave the other
    # false transition alive.
    short_reverse = mark_short_reverse_pairs(
        skills, int(round(args.short_reverse_sec * args.fps))
    )
    affordance_rejected = affordance_temporal_rejections(
        skills, tracked, schema, ownership, fps=args.fps,
        tolerance_sec=args.affordance_tolerance_sec,
        skip_indices=short_reverse,
    )
    decisions = []
    provisional = []

    for i, s0 in enumerate(skills):
        s = dict(s0)
        sig = signal_support(signals, s["provisional_start_frame"], s["provisional_end_frame"])
        d = {"original_skill_id": s0["skill_id"], "skill_type": s["skill_type"], "signal": sig, "accepted": False, "reasons": []}

        if s["skill_type"] == "navigate_to_station":
            d["accepted"] = True
            d["reasons"].append("hdf5_base_motion")
            provisional.append(s)
            decisions.append(d)
            continue

        if i in short_reverse:
            d["reasons"].append("short_reverse_pair_visual_jitter")
            decisions.append(d)
            continue

        if i in affordance_rejected:
            info = affordance_rejected[i]
            d["reasons"].append(info["reason"])
            d["affordance_evidence"] = info
            decisions.append(d)
            continue

        if s["skill_type"] in OBJECT_SKILLS:
            ok, held = held_observed(s, tracked)
            coarse_ok, coarse_evidence = ownership_supported_coarse(
                s, tracked, ownership, schema
            )
            d["held_segments"] = held
            d["ownership_coarse_evidence"] = coarse_evidence
            if not ok and not coarse_ok:
                d["reasons"].append(
                    "no_reliable_fine_hold_or_uniquely_owned_coarse_bridge"
                )
            elif sig["score"] < args.min_signal_score:
                d["reasons"].append("weak_robot_interaction_signal")
            else:
                d["accepted"] = True
                if ok:
                    d["reasons"].append("observed_held_state")
                else:
                    d["reasons"].append(
                        "joint_ownership_plus_observed_source_target_bridge"
                    )
                    d["semantic_backoff_required"] = True
                provisional.append(s)
            decisions.append(d)
            continue

        if s.get("entity") in CONTAINER_ACTIONS:
            entity = s.get("entity")
            ev = container_evidence(s, obs, dense_rows)
            d["interaction_evidence"] = ev

            if ev["contact_count"] == 0:
                d["reasons"].append("no_entity_specific_hand_contact")
                decisions.append(d)
                continue

            # Boundary relocation changes the hypothesis being validated.
            # Therefore contact anchoring must happen BEFORE the final robot
            # signal decision, and the signal evidence must be recomputed at
            # the relocated interval. A candidate is never allowed to pass on
            # signal evidence measured only at its old location.
            cf = ev.get("nearest_contact_frame")
            if (
                entity != "door"
                and cf is not None
                and abs(cf - int(s["provisional_start_frame"])) > int(round(0.75 * args.fps))
            ):
                old_span = [
                    int(s["provisional_start_frame"]),
                    int(s["provisional_end_frame"]),
                ]
                width = max(1, old_span[1] - old_span[0])
                s["provisional_start_frame"] = int(cf)
                s["provisional_end_frame"] = int(
                    cf + min(width, int(round(1.5 * args.fps)))
                )
                s["interaction_anchor_adjusted"] = True
                s["interaction_anchor_source_span"] = old_span
                d["signal_before_anchor"] = sig
                sig = signal_support(
                    signals,
                    s["provisional_start_frame"],
                    s["provisional_end_frame"],
                )
                d["signal"] = sig
                d["post_anchor_revalidated"] = True
            else:
                d["post_anchor_revalidated"] = False

            if sig["score"] < args.min_signal_score:
                d["reasons"].append("weak_robot_interaction_signal_at_final_anchor")
            else:
                d["accepted"] = True
                d["reasons"].append("entity_contact_plus_robot_signal")
                if s.get("interaction_anchor_adjusted"):
                    d["reasons"].append("robot_signal_revalidated_after_contact_anchor")
                if ev["directed_count"]:
                    d["reasons"].append("direction_consistent_visual_transition")
                provisional.append(s)
            decisions.append(d)
            continue

        d["reasons"].append("unsupported_skill_family")
        decisions.append(d)

    # One physical open/close or pull/push lifecycle per container for this
    # coarse household task. Repeated cycles need direct direction evidence.
    final = []
    container_state = initial_container_states(obs)
    decision_by_orig = {d["original_skill_id"]: d for d in decisions}

    for s in sorted(provisional, key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"])):
        entity = s.get("entity")
        if entity not in container_state:
            final.append(s)
            continue

        d = decision_by_orig[s["skill_id"]]
        typ = s["skill_type"]
        state = container_state[entity]
        directed = int((d.get("interaction_evidence") or {}).get("directed_count", 0))

        unmet = [
            req for req in ACCESS_REQUIREMENTS.get(entity, [])
            if container_state.get(req["entity"]) != req["state"]
        ]
        if unmet:
            d["accepted"] = False
            d["reasons"].append("schema_accessibility_requirement_not_met")
            d["accessibility_requirements"] = unmet
            continue

        transition_map = CONTAINER_TRANSITIONS.get(entity, {})
        next_state = transition_map.get((state, typ))

        if next_state is not None:
            final.append(s)
            container_state[entity] = next_state
        elif state == "unknown":
            candidates = [
                (src, dst) for (src, skill), dst in transition_map.items()
                if skill == typ
            ]
            if len(candidates) == 1:
                final.append(s)
                container_state[entity] = candidates[0][1]
                d["reasons"].append("lifecycle_initialized_from_first_valid_action")
            else:
                d["accepted"] = False
                d["reasons"].append("schema_lifecycle_ambiguous_from_unknown")
        else:
            d["accepted"] = False
            d["reasons"].append("schema_lifecycle_inconsistent_or_duplicate")

    # Convert accepted portable-object skills to the task-preferred
    # training granularity. Fine identity remains attached for diagnostics.
    for s in final:
        if s.get("skill_type") not in OBJECT_SKILLS:
            continue
        fine_entity = s.get("entity")
        sem = training_semantic(schema, fine_entity)
        if not sem.get("backoff"):
            continue
        fine_skill = s["skill_type"]
        s["fine_skill_type"] = fine_skill
        s["fine_entity"] = fine_entity
        s["skill_type"] = sem["skill_type"]
        s["entity"] = sem["entity"]
        label = SKILL_LABELS.get(s["skill_type"], {})
        s["instruction_zh"] = label.get("zh", s["skill_type"])
        s["instruction_en"] = label.get("en", s["skill_type"].replace("_", " "))
        s["semantic_backoff"] = {
            "from_skill_type": fine_skill,
            "from_entity": fine_entity,
            "to_skill_type": s["skill_type"],
            "to_entity": s["entity"],
            "reason": "schema_policy_equivalent_parent_training_label",
        }

    final.sort(key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"]))
    for new_id, s in enumerate(final):
        s["source_skill_id"] = s["skill_id"]
        s["skill_id"] = new_id
        d = decision_by_orig[s["source_skill_id"]]
        s["validation"] = {
            "signal": d["signal"],
            "reasons": d["reasons"],
            "interaction_evidence": d.get("interaction_evidence"),
        }

    result = {
        "annotation_version": "v3.4.4",
        "task_goal": cand.get("task_goal"),
        "task_schema": str(args.schema),
        "task_family": schema.get("task_family"),
        "num_input_skills": len(skills),
        "num_validated_skills": len(final),
        "skill_sequence": final,
        "validation_policy": {
            "short_reverse_sec": args.short_reverse_sec,
            "min_signal_score": args.min_signal_score,
            "one_container_cycle_by_default": not args.allow_extra_cycles,
            "object_placement_requires_reliable_fine_or_uniquely_owned_coarse_bridge": True,
            "sibling_held_evidence_reuse": False,
            "joint_hand_object_ownership": ownership_path.exists(),
            "reliability_first_semantic_backoff": True,
            "receptacle_affordance_temporal_consistency": True,
            "affordance_tolerance_sec": args.affordance_tolerance_sec,
            "initial_container_state": initial_container_states(obs),
            "initial_state_source": "early_direct_generic_observations",
            "short_reverse_before_affordance": True,
            "dense_directional_contact_evidence": True,
            "post_anchor_signal_revalidation": True,
            "schema_driven_accessibility": True,
            "schema_driven_lifecycle": True,
            "schema_driven_contact_tokens": True,
            "schema_driven_direction_matching": True,
        },
    }
    (out / "skill_candidates_validated.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "interaction_validation_report.json").write_text(
        json.dumps({"decisions": decisions}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"validated skills: {len(skills)} -> {len(final)}")
    for s in final:
        print(
            f"  {s['skill_id']:02d} {s['skill_type']} "
            f"[{s['provisional_start_frame']},{s['provisional_end_frame']}] "
            f"conf={s['confidence']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--short-reverse-sec", type=float, default=1.0)
    ap.add_argument("--min-signal-score", type=float, default=0.12)
    ap.add_argument("--allow-extra-cycles", action="store_true")
    ap.add_argument("--affordance-tolerance-sec", type=float, default=1.0)
    args = ap.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
