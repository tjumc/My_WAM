#!/usr/bin/env python3
"""V3 deterministic entity-state-machine skill inference.

No LLM is used here. Skills are generated from constrained entity transitions.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

UNKNOWN = {"uncertain", "not_visible", None}

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


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def conf(o, side, key):
    return float((o.get(f"confidence_{side}") or {}).get(key, 0.0))


def val(o, side, key):
    return (o.get(f"state_{side}") or {}).get(key, "uncertain")


def reliable(o, side, key, threshold):
    return val(o, side, key) not in UNKNOWN and conf(o, side, key) >= threshold


def add_skill(skills, skill_type, entity, start, end, event_ids, confidence, transition, active_arm="right"):
    cn, en = SKILL_LABELS[skill_type]
    rec = {
        "skill_id": len(skills),
        "skill_type": skill_type,
        "instruction_zh": cn,
        "instruction_en": en,
        "entity": entity,
        "active_arm": active_arm,
        "provisional_start_frame": int(start),
        "provisional_end_frame": int(end),
        "evidence_event_ids": sorted(set(int(x) for x in event_ids)),
        "confidence": round(float(confidence), 3),
        "state_transition": transition,
        "source": "deterministic_entity_state_machine",
    }
    skills.append(rec)


def direct_container_transitions(obs, threshold):
    skills = []
    specs = [
        ("door", {("closed", "opening"), ("closed", "open"), ("closing", "open")}, "open_dishwasher_door"),
        ("door", {("open", "closing"), ("open", "closed"), ("opening", "closed")}, "close_dishwasher_door"),
        ("dish_rack", {("in", "moving_out"), ("in", "out"), ("moving_in", "moving_out"), ("moving_in", "out")}, "pull_out_dish_rack"),
        ("dish_rack", {("out", "moving_in"), ("out", "in"), ("moving_out", "moving_in"), ("moving_out", "in")}, "push_in_dish_rack"),
        ("cutlery_basket", {("in", "moving_out"), ("in", "out"), ("moving_in", "moving_out"), ("moving_in", "out")}, "pull_out_cutlery_basket"),
        ("cutlery_basket", {("out", "moving_in"), ("out", "in"), ("moving_out", "moving_in"), ("moving_out", "in")}, "push_in_cutlery_basket"),
    ]
    for o in obs:
        for entity, pairs, skill in specs:
            if not (reliable(o, "before", entity, threshold) and reliable(o, "after", entity, threshold)):
                continue
            a, b = val(o, "before", entity), val(o, "after", entity)
            if (a, b) in pairs:
                c = min(conf(o, "before", entity), conf(o, "after", entity))
                add_skill(skills, skill, entity, o["_window_start_frame"], o["_window_end_frame"], [o["_event_id"]], c, {"before": a, "after": b})
    return skills


def infer_object_placement(obs, obj, target, skill_type, threshold):
    key = f"{obj}_location"
    hand_states = {"right_hand", "left_hand"}
    start = None
    start_events = []
    start_conf = []
    candidates = []

    for o in obs:
        b, a = val(o, "before", key), val(o, "after", key)
        cb, ca = conf(o, "before", key), conf(o, "after", key)
        if min(cb, ca) < threshold:
            continue
        eid = o["_event_id"]
        if b == "tabletop" and a in hand_states:
            start = int(o["_window_start_frame"])
            start_events = [eid]
            start_conf = [cb, ca]
        elif b in hand_states and a == target:
            s = start if start is not None else int(o["_window_start_frame"])
            evs = start_events + [eid]
            cs = start_conf + [cb, ca]
            candidates.append((s, int(o["_window_end_frame"]), evs, min(cs), {"before": b, "after": a}))
            start = None; start_events = []; start_conf = []
        elif b == "tabletop" and a == target:
            candidates.append((int(o["_window_start_frame"]), int(o["_window_end_frame"]), [eid], min(cb, ca) * 0.8, {"before": b, "after": a, "direct": True}))
    return candidates


def infer_navigation(hdf5_path, first_manip_frame, fps=30.0):
    with h5py.File(hdf5_path, "r") as f:
        v = f["joints_velocity_state"][:, :3].astype(float)
    speed = np.linalg.norm(v[:, :2], axis=1)
    limit = max(1, min(len(speed), int(first_manip_frame)))
    s = speed[:limit]
    active = s > 0.025
    # Require at least 0.5 s of base motion before manipulation.
    if active.sum() < int(round(0.5 * fps)):
        return None
    idx = np.flatnonzero(active)
    start = int(max(0, idx[0] - round(0.25 * fps)))
    end = int(min(limit - 1, idx[-1] + round(0.25 * fps)))
    return start, end, min(0.95, 0.65 + float(active.mean()))


def deduplicate(skills, fps=30.0):
    if not skills:
        return []
    skills = sorted(skills, key=lambda x: (x["provisional_start_frame"], x["provisional_end_frame"]))
    out = []
    max_gap = int(round(2.0 * fps))
    for s in skills:
        if out and out[-1]["skill_type"] == s["skill_type"] and s["provisional_start_frame"] <= out[-1]["provisional_end_frame"] + max_gap:
            p = out[-1]
            p["provisional_start_frame"] = min(p["provisional_start_frame"], s["provisional_start_frame"])
            p["provisional_end_frame"] = max(p["provisional_end_frame"], s["provisional_end_frame"])
            p["evidence_event_ids"] = sorted(set(p["evidence_event_ids"] + s["evidence_event_ids"]))
            p["confidence"] = round(max(p["confidence"], s["confidence"]), 3)
            p.setdefault("merged_transitions", []).append(s["state_transition"])
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
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min-entity-confidence", type=float, default=0.72)
    args = ap.parse_args()

    out = Path(args.output_dir)
    obs = sorted(load_jsonl(out / "entity_observations.jsonl"), key=lambda x: x["_raw_frame"])
    skills = direct_container_transitions(obs, args.min_entity_confidence)

    for obj, target, skill_type in [
        ("knife", "cutlery_basket", "place_knife_in_cutlery_basket"),
        ("fork", "cutlery_basket", "place_fork_in_cutlery_basket"),
        ("plate", "dish_rack", "place_plate_in_dish_rack"),
    ]:
        for s, e, evs, c, trans in infer_object_placement(obs, obj, target, skill_type, args.min_entity_confidence):
            add_skill(skills, skill_type, obj, s, e, evs, c, trans)

    manip = [s for s in skills if s["skill_type"] != "navigate_to_station"]
    first_manip = min((s["provisional_start_frame"] for s in manip), default=len(obs) and obs[0]["_raw_frame"] or 0)
    nav = infer_navigation(args.hdf5, first_manip, args.fps)
    if nav:
        s, e, c = nav
        add_skill(skills, "navigate_to_station", "robot_base", s, e, [], c, {"base_motion": True}, active_arm="none")

    skills = deduplicate(skills, args.fps)
    result = {
        "annotation_version": "v3",
        "num_entity_windows": len(obs),
        "num_inferred_skills": len(skills),
        "skill_sequence": skills,
        "notes": [
            "No LLM action/phase generation is used in skill inference.",
            "Skills are generated only from constrained entity-state transitions plus base-motion evidence for navigation.",
            "Unknown/occluded entities do not create skills.",
        ],
    }
    (out / "skill_candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("skill sequence:")
    for s in skills:
        print(f"  {s['skill_id']:02d} {s['skill_type']} [{s['provisional_start_frame']},{s['provisional_end_frame']}] conf={s['confidence']}")
    print("saved", out / "skill_candidates.json")


if __name__ == "__main__":
    main()
