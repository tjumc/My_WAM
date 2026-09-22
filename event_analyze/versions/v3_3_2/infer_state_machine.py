#!/usr/bin/env python3
"""V3.3.2 skill inference from trajectory-level tracked entity states.

No LLM is used here. Skills are inferred from state trajectories, not from
independent before/after windows.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SKILL_LABELS = {
    "navigate_to_station": ("移动到桌面/洗碗机前", "Move to the table/dishwasher station"),
    "open_dishwasher_door": ("右手打开洗碗机门", "Open the dishwasher door with the right hand"),
    "pull_out_dish_rack": ("右手拉出碗篮", "Pull out the dish rack with the right hand"),
    "pull_out_cutlery_basket": ("右手拉出餐具篮", "Pull out the cutlery basket with the right hand"),
    "place_knife_in_cutlery_basket": ("右手夹取刀并放入餐具篮", "Pick up the knife and place it in the cutlery basket with the right hand"),
    "place_fork_in_cutlery_basket": ("右手夹取叉子并放入餐具篮", "Pick up the fork and place it in the cutlery basket with the right hand"),
    "push_in_cutlery_basket": ("右手推入餐具篮", "Push in the cutlery basket with the right hand"),
    "place_plate_in_dish_rack": ("右手夹取盘子并放入碗篮", "Pick up the plate and place it in the dish rack with the right hand"),
    "push_in_dish_rack": ("右手推入碗篮", "Push in the dish rack with the right hand"),
    "close_dishwasher_door": ("右手合上洗碗机门", "Close the dishwasher door with the right hand"),
}

CONTAINER_RULES = {
    "door": {
        ("closed", "open"): "open_dishwasher_door",
        ("open", "closed"): "close_dishwasher_door",
    },
    "dish_rack": {
        ("in", "out"): "pull_out_dish_rack",
        ("out", "in"): "push_in_dish_rack",
    },
    "cutlery_basket": {
        ("in", "out"): "pull_out_cutlery_basket",
        ("out", "in"): "push_in_cutlery_basket",
    },
}

MOVING_DIRECTION = {
    "door": {"opening": ("closed", "open"), "closing": ("open", "closed")},
    "dish_rack": {"moving_out": ("in", "out"), "moving_in": ("out", "in")},
    "cutlery_basket": {"moving_out": ("in", "out"), "moving_in": ("out", "in")},
}

OBJECT_RULES = {
    "knife_location": ("cutlery_basket", "place_knife_in_cutlery_basket", "knife"),
    "fork_location": ("cutlery_basket", "place_fork_in_cutlery_basket", "fork"),
    "plate_location": ("dish_rack", "place_plate_in_dish_rack", "plate"),
}


def add_skill(skills, skill_type, entity, start, end, event_ids, confidence, transition, inference_mode, active_arm="right"):
    zh, en = SKILL_LABELS[skill_type]
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
                out, rules[pair], entity, start, end,
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
                    out, rules[pair], entity,
                    seg["start_frame"], seg["end_frame"],
                    segment_events(segs, prev_i, i),
                    conf, {"before": a, "motion": st, "after": b, "after_inferred": True},
                    "tracked_terminal_motion",
                )
    return out


def object_placement_skills(track, entity):
    segs = track.get("segments", [])
    target, skill_type, obj_name = OBJECT_RULES[entity]
    out = []
    hand_states = {"right_hand", "left_hand"}

    for i, seg in enumerate(segs):
        if seg["state"] not in hand_states:
            continue

        # Require an earlier tabletop observation/interpolation before this grasp episode.
        prev_table = next((j for j in range(i - 1, -1, -1) if segs[j]["state"] == "tabletop"), None)
        if prev_table is None:
            continue

        # Find the first non-hand state after the held run.
        j = i
        while j + 1 < len(segs) and segs[j + 1]["state"] in hand_states:
            j += 1
        target_i = next((k for k in range(j + 1, len(segs)) if segs[k]["state"] not in hand_states), None)
        if target_i is None or segs[target_i]["state"] != target:
            continue

        # Do not duplicate one placement across multiple hand segments.
        if out and seg["start_frame"] <= out[-1]["provisional_end_frame"]:
            continue

        target_seg = segs[target_i]
        conf = min(seg["confidence"], target_seg["confidence"])
        inferred = target_seg.get("support") == "interpolated"
        if inferred:
            conf *= 0.78
        inference_mode = "tracked_object_transition_with_affordance_completion" if inferred else "tracked_object_transition"
        add_skill(
            out, skill_type, obj_name,
            seg["start_frame"], target_seg["start_frame"],
            segment_events(segs, prev_table, target_i),
            conf,
            {
                "before": "tabletop",
                "held": seg["state"],
                "after": target,
                "target_support": target_seg.get("support"),
            },
            inference_mode,
        )
    return out


def deduplicate(skills, fps=30.0, retry_sec=3.0):
    """Collapse repeated grasp/re-grasp attempts into one semantic placement.

    Container reversals keep a short merge window. Object-placement candidates
    for the same object/target use a longer retry window so failed grasp attempts
    remain internal execution details rather than separate skills.
    """
    if not skills:
        return []
    skills = sorted(skills, key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"], x["skill_type"]))
    out = []
    default_gap = int(round(0.75 * fps))
    retry_gap = int(round(retry_sec * fps))
    for s in skills:
        same = (
            out
            and out[-1]["skill_type"] == s["skill_type"]
            and out[-1]["entity"] == s["entity"]
        )
        gap = retry_gap if s["skill_type"].startswith("place_") else default_gap
        if same and s["provisional_start_frame"] <= out[-1]["provisional_end_frame"] + gap:
            p = out[-1]
            p["provisional_start_frame"] = min(p["provisional_start_frame"], s["provisional_start_frame"])
            p["provisional_end_frame"] = max(p["provisional_end_frame"], s["provisional_end_frame"])
            p["evidence_event_ids"] = sorted(set(p["evidence_event_ids"] + s["evidence_event_ids"]))
            p["confidence"] = round(max(p["confidence"], s["confidence"]), 3)
            p.setdefault("merged_transitions", []).append(s["state_transition"])
            if s["skill_type"].startswith("place_"):
                p["retry_collapsed"] = True
                p["retry_merge_window_sec"] = retry_sec
        else:
            out.append(dict(s))
    for i, s in enumerate(out):
        s["skill_id"] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min-skill-confidence", type=float, default=0.38)
    ap.add_argument("--retry-merge-sec", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    tracked = json.load(open(out / "tracked_entity_states.json", encoding="utf-8"))
    tracks = tracked["tracks"]

    skills = []
    for entity in ("door", "dish_rack", "cutlery_basket"):
        skills.extend(container_skills(tracks[entity], entity))
    for entity in ("knife_location", "fork_location", "plate_location"):
        skills.extend(object_placement_skills(tracks[entity], entity))

    manip = [s for s in skills if s["confidence"] >= args.min_skill_confidence]
    first_manip = min((s["provisional_start_frame"] for s in manip), default=10**9)
    nav = infer_navigation(args.hdf5, first_manip, args.fps)
    if nav:
        s, e, c = nav
        add_skill(skills, "navigate_to_station", "robot_base", s, e, [], c, {"base_motion": True}, "hdf5_base_motion", active_arm="none")

    skills = [s for s in skills if s["confidence"] >= args.min_skill_confidence]
    skills = deduplicate(skills, args.fps, args.retry_merge_sec)

    result = {
        "annotation_version": "v3.3.2",
        "task_goal": args.task,
        "num_inferred_skills": len(skills),
        "skill_sequence": skills,
        "notes": [
            "No LLM action/phase generation is used in skill inference.",
            "Skills are inferred from trajectory-level tracked entity states rather than independent windows.",
            "Stable endpoint changes can bridge missing motion states.",
            "Object placements can bridge short occlusions; weak affordance completions are explicitly marked and down-weighted.",
            "Repeated grasp/re-grasp attempts for the same object are collapsed into one semantic placement skill.",
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
