#!/usr/bin/env python3
"""V3.4.2 skill inference from trajectory-level tracked entity states.

No LLM is used here. Skills are inferred from state trajectories, not from
independent before/after windows.
"""
import argparse
import json
import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from schema_runtime import load_schema, container_rules, portable_rules, skill_labels

import h5py
import numpy as np

SKILL_LABELS = {}
CONTAINER_RULES = {}
MOVING_DIRECTION = {}
OBJECT_RULES = {}
ENTITY_NAME_BY_TRACK = {}

def add_skill(skills, skill_type, entity, start, end, event_ids, confidence, transition, inference_mode, active_arm="right"):
    label = SKILL_LABELS.get(skill_type, {})
    zh = label.get("zh", skill_type)
    en = label.get("en", skill_type.replace("_", " "))
    skills.append({
        "skill_id": -1,
        "skill_type": skill_type,
        "instruction_zh": zh,
        "instruction_en": en,
        "entity": entity,
        "active_arm": active_arm,
        "provisional_start_frame": int(start),
        "provisional_end_frame": int(end),
        "evidence_event_ids": sorted(set(int(x) for x in event_ids)),
        "confidence": round(float(confidence), 3),
        "state_transition": transition,
        "inference_mode": inference_mode,
        "source": "trajectory_level_entity_state_machine",
    })


def _get_hdf5_dataset(f, candidates):
    for path in candidates:
        if path in f:
            return f[path], path
    raise KeyError("None of expected HDF5 datasets exist: " + ", ".join(candidates))


def infer_navigation(hdf5_path, first_manip_frame, fps=30.0):
    with h5py.File(hdf5_path, "r") as f:
        ds, _ = _get_hdf5_dataset(f, ["joints_dict/joints_velocity_state", "joints_velocity_state"])
        v = ds[:, :3].astype(float)
    speed = np.linalg.norm(v[:, :2], axis=1)
    limit = max(1, min(len(speed), int(first_manip_frame)))
    active = speed[:limit] > 0.025
    if active.sum() < int(round(0.5 * fps)):
        return None
    idx = np.flatnonzero(active)
    start = int(max(0, idx[0] - round(0.25 * fps)))
    end = int(min(limit - 1, idx[-1] + round(0.25 * fps)))
    return start, end, min(0.95, 0.65 + float(active.mean()))


def segment_events(segs, i0, i1):
    return sorted({eid for s in segs[i0:i1 + 1] for eid in s.get("event_ids", [])})


def container_skills(track, entity):
    segs = track.get("segments", [])
    rules = CONTAINER_RULES[entity]
    logical_entity = ENTITY_NAME_BY_TRACK.get(entity, entity)
    moving = MOVING_DIRECTION[entity]
    out = []

    # Stable endpoint transitions: moving states may be missing or mislabeled.
    stable_states = set(x for pair in rules for x in pair)
    last_stable_i = None
    for i, seg in enumerate(segs):
        st = seg["state"]
        if st not in stable_states:
            continue
        if last_stable_i is None:
            last_stable_i = i
            continue
        prev = segs[last_stable_i]
        if prev["state"] == st:
            last_stable_i = i
            continue
        pair = (prev["state"], st)
        if pair in rules:
            bridge = segs[last_stable_i + 1:i]
            start = bridge[0]["start_frame"] if bridge else prev["end_frame"]
            end = seg["start_frame"]
            conf = min(prev["confidence"], seg["confidence"])
            if prev.get("support") == "interpolated" or seg.get("support") == "interpolated":
                conf *= 0.82
            add_skill(
                out, rules[pair], logical_entity, start, end,
                segment_events(segs, last_stable_i, i),
                conf, {"before": pair[0], "after": pair[1]},
                "tracked_endpoint_transition",
            )
        last_stable_i = i

    # Partial terminal motion, e.g. open -> closing when the final closed state is occluded.
    if segs:
        for i, seg in enumerate(segs):
            st = seg["state"]
            if st not in moving:
                continue
            a, b = moving[st]
            # Find nearest stable state before this motion.
            prev_i = next((j for j in range(i - 1, -1, -1) if segs[j]["state"] in {a, b}), None)
            if prev_i is None or segs[prev_i]["state"] != a:
                continue
            # If b is later observed, the endpoint transition above already covers it.
            later_b = any(x["state"] == b for x in segs[i + 1:])
            if later_b:
                continue
            # If we return to a, this was only a reversal/noise.
            later_a = any(x["state"] == a for x in segs[i + 1:])
            if later_a:
                continue
            pair = (a, b)
            if pair in rules:
                conf = min(segs[prev_i]["confidence"], seg["confidence"]) * 0.82
                add_skill(
                    out, rules[pair], logical_entity,
                    seg["start_frame"], seg["end_frame"],
                    segment_events(segs, prev_i, i),
                    conf, {"before": a, "motion": st, "after": b, "after_inferred": True},
                    "tracked_terminal_motion",
                )
    return out


def observed_hand_segment(seg, hand_states):
    return (
        seg.get("state") in hand_states
        and seg.get("support") == "observed"
        and float(seg.get("confidence", 0.0)) >= 0.70
    )


def other_object_context_switch(all_tracks, current_entity, start_frame, end_frame):
    """Whether another portable object is clearly manipulated in this gap.

    This is used as a semantic episode boundary. It avoids relying on a fixed
    target dwell-time threshold: target->hand is treated as adjustment/re-grasp
    unless the robot has clearly switched to another object in between.
    """
    if end_frame < start_frame:
        return False
    for key, cfg in OBJECT_RULES.items():
        if key == current_entity or key not in all_tracks:
            continue
        hand_states = set(cfg["hand_states"])
        for seg in all_tracks[key].get("segments", []):
            if not observed_hand_segment(seg, hand_states):
                continue
            if int(seg["end_frame"]) >= int(start_frame) and int(seg["start_frame"]) <= int(end_frame):
                return True
    return False


def object_placement_skills(track, entity, all_tracks, fps=30.0, retry_sec=3.0, stable_target_sec=1.2):
    """Build completion-aware object-centric manipulation episodes.

    A target contact does not finish a semantic placement if the same object is
    re-grasped before a clear object-context switch. This naturally collapses
    hand->target->hand->target adjustment sequences without tuning a target
    dwell-time threshold.

    retry_sec/stable_target_sec are retained only for CLI compatibility with
    older versions; completion no longer depends on them.
    """
    segs = track.get("segments", [])
    cfg = OBJECT_RULES[entity]
    target = cfg["target"]
    skill_type = cfg["skill_type"]
    obj_name = cfg["obj_name"]
    source_states = set(cfg["source_states"])
    hand_states = set(cfg["hand_states"])
    out = []

    i = 0
    while i < len(segs):
        if segs[i]["state"] not in hand_states:
            i += 1
            continue

        prev_source = next(
            (j for j in range(i - 1, -1, -1) if segs[j]["state"] in source_states),
            None,
        )
        if prev_source is None:
            i += 1
            continue

        first_hand_i = i
        final_target_i = None
        retried = False
        k = i + 1

        while k < len(segs):
            st = segs[k]["state"]

            if st == target:
                next_hand = next(
                    (h for h in range(k + 1, len(segs)) if segs[h]["state"] in hand_states),
                    None,
                )
                if next_hand is None:
                    final_target_i = k
                    break

                gap_start = int(segs[k]["end_frame"]) + 1
                gap_end = int(segs[next_hand]["start_frame"]) - 1
                if other_object_context_switch(all_tracks, entity, gap_start, gap_end):
                    final_target_i = k
                    break

                # No semantic context switch: treat the later hand state as an
                # adjustment/re-grasp inside the same manipulation episode.
                retried = True
                k = next_hand + 1
                continue

            # Ownership arbitration may trim an uncertain held tail before
            # the sparse tracker later confirms the destination. This synthetic
            # gap is not evidence that the object returned to an unrelated
            # state; bridge it to the next directly supported target.
            if (
                st == "other"
                and segs[k].get("support") == "ownership_suppressed"
            ):
                next_target = next(
                    (
                        t for t in range(k + 1, len(segs))
                        if segs[t]["state"] == target
                    ),
                    None,
                )
                if next_target is not None:
                    final_target_i = next_target
                    break

            if st in source_states:
                next_hand = next(
                    (h for h in range(k + 1, len(segs)) if segs[h]["state"] in hand_states),
                    None,
                )
                if next_hand is not None:
                    gap_start = int(segs[k]["start_frame"])
                    gap_end = int(segs[next_hand]["start_frame"]) - 1
                    if not other_object_context_switch(all_tracks, entity, gap_start, gap_end):
                        retried = True
                        k = next_hand + 1
                        continue
                # Returned to source and then switched context: this attempt did
                # not complete a placement.
                break

            k += 1

        if final_target_i is None:
            i = max(i + 1, k)
            continue

        target_seg = segs[final_target_i]
        hand_segments = [
            x for x in segs[first_hand_i:final_target_i + 1]
            if x["state"] in hand_states
        ]
        observed_hand_conf = [
            float(x.get("confidence", 0.0))
            for x in hand_segments if x.get("support") == "observed"
        ]
        hand_conf = max(observed_hand_conf) if observed_hand_conf else float(segs[first_hand_i].get("confidence", 0.0))
        target_conf = float(target_seg.get("confidence", 0.0))
        conf = min(hand_conf, target_conf)
        inferred = target_seg.get("support") == "interpolated"
        if inferred:
            conf *= 0.78
        mode = "completion_aware_object_episode_with_affordance_completion" if inferred else "completion_aware_object_episode"

        source_state = segs[prev_source]["state"]

        # Sparse perception may confirm the target after the robot has already
        # switched to the next object. When ownership arbitration provides a
        # physically grounded context-switch boundary, use it as an upper bound
        # on semantic completion while retaining the later target observation
        # as destination evidence.
        context_end = None
        for hs in segs[first_hand_i:final_target_i + 1]:
            if hs.get("state") not in hand_states:
                continue
            if hs.get("ownership_end_reason") == "next_object_direct_ownership":
                f = hs.get("ownership_context_end_frame")
                if f is not None:
                    context_end = int(f) if context_end is None else max(context_end, int(f))

        semantic_end = int(target_seg["start_frame"])
        if context_end is not None:
            semantic_end = min(semantic_end, context_end)

        add_skill(
            out, skill_type, obj_name,
            segs[first_hand_i]["start_frame"], semantic_end,
            segment_events(segs, prev_source, final_target_i),
            conf,
            {
                "before": source_state,
                "held": segs[first_hand_i]["state"],
                "after": target,
                "target_support": target_seg.get("support"),
                "target_evidence_frame": int(target_seg["start_frame"]),
                "ownership_context_end_frame": context_end,
                "retry_collapsed": retried,
                "completion_rule": "target_without_regrasp_before_object_context_switch",
            },
            mode,
        )
        if retried:
            out[-1]["retry_collapsed"] = True
            out[-1]["completion_aware_merge"] = True

        i = final_target_i + 1

    return out


def deduplicate(skills, fps=30.0):
    """Remove only near-identical duplicate skill hypotheses.

    Object-placement retry grouping is handled upstream by the object-centric
    episode builder, so this function no longer depends on list adjacency for
    retry semantics.
    """
    if not skills:
        return []
    skills = sorted(skills, key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"], x["skill_type"]))
    gap = int(round(0.75 * fps))
    grouped = {}
    order = []

    for s in skills:
        key = (s["skill_type"], s["entity"])
        bucket = grouped.setdefault(key, [])
        merged = False
        for p in reversed(bucket):
            if s["provisional_start_frame"] <= p["provisional_end_frame"] + gap:
                p["provisional_start_frame"] = min(p["provisional_start_frame"], s["provisional_start_frame"])
                p["provisional_end_frame"] = max(p["provisional_end_frame"], s["provisional_end_frame"])
                p["evidence_event_ids"] = sorted(set(p["evidence_event_ids"] + s["evidence_event_ids"]))
                p["confidence"] = round(max(p["confidence"], s["confidence"]), 3)
                p.setdefault("merged_transitions", []).append(s["state_transition"])
                merged = True
                break
        if not merged:
            x = dict(s)
            bucket.append(x)
            order.append(x)

    order.sort(key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"], x["skill_type"]))
    for i, s in enumerate(order):
        s["skill_id"] = i
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min-skill-confidence", type=float, default=0.38)
    ap.add_argument("--retry-merge-sec", type=float, default=3.0)
    ap.add_argument("--stable-target-sec", type=float, default=1.2)
    args = ap.parse_args()

    schema = load_schema(args.schema)
    global SKILL_LABELS, CONTAINER_RULES, MOVING_DIRECTION, OBJECT_RULES, ENTITY_NAME_BY_TRACK
    SKILL_LABELS = skill_labels(schema)

    cr = container_rules(schema)
    CONTAINER_RULES = {}
    MOVING_DIRECTION = {}
    ENTITY_NAME_BY_TRACK = {}
    for logical_name, cfg in cr.items():
        key = cfg["observation_key"]
        ENTITY_NAME_BY_TRACK[key] = logical_name
        CONTAINER_RULES[key] = cfg["rules"]
        moving = {}
        for motion, source in cfg.get("motion_source", {}).items():
            endpoint = cfg.get("motion_endpoint", {}).get(motion)
            if endpoint is not None:
                moving[motion] = (source, endpoint)
        MOVING_DIRECTION[key] = moving

    pr = portable_rules(schema)
    OBJECT_RULES = {}
    for logical_name, cfg in pr.items():
        key = cfg["observation_key"]
        OBJECT_RULES[key] = {
            "target": cfg["target"],
            "skill_type": cfg["skill_type"],
            "obj_name": logical_name,
            "source_states": cfg["source_states"],
            "hand_states": cfg["hand_states"],
            "episode_completion": cfg.get("episode_completion"),
        }

    out = Path(args.output_dir)
    owned_path = out / "tracked_entity_states_owned.json"
    tracked_path = owned_path if owned_path.exists() else out / "tracked_entity_states.json"
    tracked = json.load(open(tracked_path, encoding="utf-8"))
    tracks = tracked["tracks"]

    skills = []
    for entity in CONTAINER_RULES:
        if entity in tracks:
            skills.extend(container_skills(tracks[entity], entity))
    for entity in OBJECT_RULES:
        if entity in tracks:
            skills.extend(object_placement_skills(
                tracks[entity], entity, tracks, args.fps, args.retry_merge_sec, args.stable_target_sec
            ))

    manip = [s for s in skills if s["confidence"] >= args.min_skill_confidence]
    first_manip = min((s["provisional_start_frame"] for s in manip), default=10**9)
    nav = infer_navigation(args.hdf5, first_manip, args.fps)
    if nav:
        s, e, c = nav
        add_skill(skills, "navigate_to_station", "robot_base", s, e, [], c, {"base_motion": True}, "hdf5_base_motion", active_arm="none")

    skills = [s for s in skills if s["confidence"] >= args.min_skill_confidence]
    skills = deduplicate(skills, args.fps)

    result = {
        "annotation_version": "v3.4.2",
        "task_goal": args.task,
        "task_schema": str(args.schema),
        "task_family": schema.get("task_family"),
        "num_inferred_skills": len(skills),
        "skill_sequence": skills,
        "notes": [
            "No LLM action/phase generation is used in skill inference.",
            "Skills are inferred from trajectory-level tracked entity states rather than independent windows.",
            "Stable endpoint changes can bridge missing motion states.",
            "Object placements can bridge short occlusions; weak affordance completions are explicitly marked and down-weighted.",
            "Object-centric manipulation episodes close on target only when no re-grasp occurs before an object-context switch; completion does not use a fixed target dwell-time threshold.",
            "Joint hand-object ownership can trim speculative held intervals and provide a context-switch upper bound for sparse target observations.",
            "The manual regression fixture is not read by this script.",
        ],
    }
    path = out / "skill_candidates.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("skill sequence:")
    for s in skills:
        print(f"  {s['skill_id']:02d} {s['skill_type']} [{s['provisional_start_frame']},{s['provisional_end_frame']}] conf={s['confidence']} mode={s['inference_mode']}")
    print("saved", path)


if __name__ == "__main__":
    main()
