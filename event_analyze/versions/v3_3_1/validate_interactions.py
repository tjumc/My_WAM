#!/usr/bin/env python3
"""V3.3.1 interaction-grounded validation for tracked skill candidates.

Visual state changes are not sufficient. Container transitions must be supported
by robot interaction evidence. Object placements require an actually observed
held state. Short state reversals are treated as visual jitter.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

CONTAINER_ACTIONS = {
    "door": ("open_dishwasher_door", "close_dishwasher_door"),
    "dish_rack": ("pull_out_dish_rack", "push_in_dish_rack"),
    "cutlery_basket": ("pull_out_cutlery_basket", "push_in_cutlery_basket"),
}
CONTACT_TOKEN = {
    "door": "contact_door",
    "dish_rack": "contact_dish_rack",
    "cutlery_basket": "contact_cutlery_basket",
}
OBJECT_TRACK = {
    "knife": "knife_location",
    "fork": "fork_location",
    "plate": "plate_location",
}
OBJECT_SKILLS = {
    "place_knife_in_cutlery_basket",
    "place_fork_in_cutlery_basket",
    "place_plate_in_dish_rack",
}

# Generic affordance relation: a receptacle must be made accessible before
# placing objects into it, and can only be returned after the placements finish.
RECEPTACLE_PLACEMENTS = {
    "cutlery_basket": {
        "place_knife_in_cutlery_basket",
        "place_fork_in_cutlery_basket",
    },
    "dish_rack": {
        "place_plate_in_dish_rack",
    },
}


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


def hand_contact(o, token):
    for side in ("before", "after"):
        st = o.get(f"state_{side}") or {}
        if st.get("right_hand") == token or st.get("left_hand") == token:
            return True
    return False


def direction_match(o, entity, skill_type):
    sb = (o.get("state_before") or {}).get(entity)
    sa = (o.get("state_after") or {}).get(entity)
    if entity == "door":
        if skill_type == "open_dishwasher_door":
            return sb in {"closed", "opening"} and sa in {"opening", "open"}
        return sb in {"open", "closing"} and sa in {"closing", "closed"}
    if skill_type.startswith("pull_out_"):
        return sb in {"in", "moving_out"} and sa in {"moving_out", "out"}
    if skill_type.startswith("push_in_"):
        return sb in {"out", "moving_in"} and sa in {"moving_in", "in"}
    return False


def container_evidence(skill, obs):
    entity = skill["entity"]
    token = CONTACT_TOKEN[entity]
    nearby = obs_near(obs, skill["provisional_start_frame"], skill["provisional_end_frame"])

    def usable_contact(o):
        if not hand_contact(o, token):
            return False
        dense_target = o.get("_v331_dense_target_entity")
        if dense_target == entity:
            # Dense evidence is directional. A moving-out dense window must not
            # be recycled as generic contact evidence for a push-in candidate.
            return direction_match(o, entity, skill["skill_type"])
        return True

    contacts = [o for o in nearby if usable_contact(o)]
    directed = [o for o in nearby if direction_match(o, entity, skill["skill_type"]) and hand_contact(o, token)]
    return {
        "nearby_event_ids": [int(o["_event_id"]) for o in nearby],
        "contact_event_ids": [int(o["_event_id"]) for o in contacts],
        "directed_contact_event_ids": [int(o["_event_id"]) for o in directed],
        "contact_count": len(contacts),
        "directed_count": len(directed),
        "nearest_contact_frame": (
            min(
                (int(o["_raw_frame"]) for o in contacts),
                key=lambda f: abs(f - int(skill["provisional_start_frame"])),
            )
            if contacts else None
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


def affordance_temporal_rejections(skills, tracked, fps=30.0, tolerance_sec=1.0):
    """Reject container transitions inconsistent with receptacle usage.

    This is role-based rather than a hard-coded action sequence:
    - pull_out(receptacle) should happen before the first place(*, receptacle)
    - push_in(receptacle) should happen after the last place(*, receptacle)
    """
    rejected = {}
    tol = int(round(float(fps) * float(tolerance_sec)))
    for receptacle, placement_types in RECEPTACLE_PLACEMENTS.items():
        placements = [
            s for s in skills
            if s["skill_type"] in placement_types and held_observed(s, tracked)[0]
        ]
        if not placements:
            continue
        first_place = min(int(s["provisional_start_frame"]) for s in placements)
        last_place = max(int(s["provisional_end_frame"]) for s in placements)
        for i, s in enumerate(skills):
            if s.get("entity") != receptacle:
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


def validate(args):
    out = Path(args.output_dir)
    cand = json.load(open(out / "skill_candidates.json", encoding="utf-8"))
    tracked = json.load(open(out / "tracked_entity_states.json", encoding="utf-8"))
    obs = load_jsonl(out / "entity_observations.jsonl")
    signals = load_signals(args.hdf5)
    skills = cand.get("skill_sequence", [])

    affordance_rejected = affordance_temporal_rejections(
        skills, tracked, fps=args.fps, tolerance_sec=args.affordance_tolerance_sec
    )
    short_reverse = mark_short_reverse_pairs(
        skills,
        int(round(args.short_reverse_sec * args.fps)),
        skip_indices=affordance_rejected,
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

        if i in affordance_rejected:
            info = affordance_rejected[i]
            d["reasons"].append(info["reason"])
            d["affordance_evidence"] = info
            decisions.append(d)
            continue

        if i in short_reverse:
            d["reasons"].append("short_reverse_pair_visual_jitter")
            decisions.append(d)
            continue

        if s["skill_type"] in OBJECT_SKILLS:
            ok, held = held_observed(s, tracked)
            d["held_segments"] = held
            if not ok:
                d["reasons"].append("no_observed_held_state")
            elif sig["score"] < args.min_signal_score:
                d["reasons"].append("weak_robot_interaction_signal")
            else:
                d["accepted"] = True
                d["reasons"].append("observed_held_state")
                provisional.append(s)
            decisions.append(d)
            continue

        if s.get("entity") in CONTAINER_ACTIONS:
            ev = container_evidence(s, obs)
            d["interaction_evidence"] = ev
            if ev["contact_count"] == 0:
                d["reasons"].append("no_entity_specific_hand_contact")
            elif sig["score"] < args.min_signal_score:
                d["reasons"].append("weak_robot_interaction_signal")
            else:
                d["accepted"] = True
                d["reasons"].append("entity_contact_plus_robot_signal")
                if ev["directed_count"]:
                    d["reasons"].append("direction_consistent_visual_transition")
                # If the tracker boundary was driven by sparse state anchors,
                # center it on the nearest actual contact observation.
                cf = ev.get("nearest_contact_frame")
                if cf is not None and abs(cf - int(s["provisional_start_frame"])) > int(round(0.75 * args.fps)):
                    width = max(1, int(s["provisional_end_frame"]) - int(s["provisional_start_frame"]))
                    s["provisional_start_frame"] = int(cf)
                    s["provisional_end_frame"] = int(cf + min(width, int(round(1.5 * args.fps))))
                    s["interaction_anchor_adjusted"] = True
                provisional.append(s)
            decisions.append(d)
            continue

        d["reasons"].append("unsupported_skill_family")
        decisions.append(d)

    # One physical open/close or pull/push lifecycle per container for this
    # coarse household task. Repeated cycles need direct direction evidence.
    final = []
    container_state = {
        "door": "pre_open",
        "dish_rack": "pre_pull",
        "cutlery_basket": "pre_pull",
    }
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

        if entity == "door":
            if state == "pre_open" and typ == "open_dishwasher_door":
                final.append(s); container_state[entity] = "open"
            elif state == "open" and typ == "close_dishwasher_door":
                final.append(s); container_state[entity] = "closed"
            else:
                d["accepted"] = False
                d["reasons"].append("door_lifecycle_inconsistent_or_duplicate")
        else:
            if state == "pre_pull" and typ.startswith("pull_out_"):
                final.append(s); container_state[entity] = "out"
            elif state == "out" and typ.startswith("push_in_"):
                final.append(s); container_state[entity] = "in"
            elif directed >= 1 and state == "in" and typ.startswith("pull_out_") and args.allow_extra_cycles:
                final.append(s); container_state[entity] = "out"
            else:
                d["accepted"] = False
                d["reasons"].append("container_lifecycle_inconsistent_or_duplicate")

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
        "annotation_version": "v3.3.1",
        "task_goal": cand.get("task_goal"),
        "num_input_skills": len(skills),
        "num_validated_skills": len(final),
        "skill_sequence": final,
        "validation_policy": {
            "short_reverse_sec": args.short_reverse_sec,
            "min_signal_score": args.min_signal_score,
            "one_container_cycle_by_default": not args.allow_extra_cycles,
            "object_placement_requires_observed_held_state": True,
            "receptacle_affordance_temporal_consistency": True,
            "affordance_tolerance_sec": args.affordance_tolerance_sec,
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
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--short-reverse-sec", type=float, default=1.0)
    ap.add_argument("--min-signal-score", type=float, default=0.12)
    ap.add_argument("--allow-extra-cycles", action="store_true")
    ap.add_argument("--affordance-tolerance-sec", type=float, default=1.0)
    args = ap.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
