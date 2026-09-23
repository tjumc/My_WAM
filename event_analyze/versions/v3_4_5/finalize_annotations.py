#!/usr/bin/env python3
"""V3.4.5 final global consistency gate.

The gate never invents missing semantic actions. It only removes/clamps final
outputs that contradict the trajectory horizon, unresolved duration constraints,
or schema lifecycle, and reports unresolved expected-final-state violations.
"""
import argparse
import json
from pathlib import Path

import h5py


MIN_DURATION_SEC = {
    "navigate_to_station": 1.0,
    "open_dishwasher_door": 0.7,
    "pull_out_dish_rack": 0.6,
    "pull_out_cutlery_basket": 0.6,
    "place_knife_in_cutlery_basket": 0.8,
    "place_fork_in_cutlery_basket": 0.8,
    "place_utensil_in_cutlery_basket": 0.8,
    "push_in_cutlery_basket": 0.6,
    "place_plate_in_dish_rack": 0.8,
    "push_in_dish_rack": 0.6,
    "close_dishwasher_door": 0.7,
}


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def total_frames(path):
    with h5py.File(path, "r") as f:
        if "time" in f:
            return len(f["time"])
        return len(f["images_dict/head/rgb_size"])


def transition_maps(schema):
    out = {}
    for entity, cfg in schema.get("entities", {}).items():
        rows = {}
        for t in cfg.get("transitions", []):
            rows[(t["from"], t["skill_type"])] = t["to"]
        if rows:
            out[entity] = rows
    return out


def minimum_duration_frames(skill_type, fps):
    return max(1, int(round(MIN_DURATION_SEC.get(skill_type, 0.5) * fps)))


def has_duration_conflict(phase):
    return any(
        bool(x.get("duration_conflict"))
        for x in phase.get("boundary_arbitration", []) or []
    )


def complement(spans, T):
    spans = sorted((max(0, int(s)), min(T - 1, int(e))) for s, e in spans if e >= s)
    out, cur = [], 0
    for s, e in spans:
        if s > cur:
            out.append([cur, s - 1])
        cur = max(cur, e + 1)
    if cur < T:
        out.append([cur, T - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    ann_path = out / "hierarchical_annotations.json"
    ann = load_json(ann_path)
    schema = load_json(args.schema)
    validated = load_json(out / "skill_candidates_validated.json", {})

    T = total_frames(args.hdf5)
    original_task_end = int(ann.get("task_end_raw_frame", T - 1))
    task_end = max(0, min(T - 1, original_task_end))
    transitions = transition_maps(schema)
    initial = (
        validated.get("validation_policy", {}).get("initial_container_state", {})
        or {}
    )

    report = {
        "annotation_version": "v3.4.5",
        "trajectory_frames": T,
        "original_task_end_raw_frame": original_task_end,
        "effective_task_end_raw_frame": task_end,
        "dropped_phases": [],
        "clamped_phases": [],
        "evidence_horizon_warnings": [],
        "lifecycle_violations": [],
        "expected_final_state_violations": [],
    }

    candidates = []
    for phase in sorted(
        ann.get("semantic_phases", []),
        key=lambda x: (int(x.get("raw_start_frame", 0)), int(x.get("raw_end_frame", 0))),
    ):
        p = dict(phase)
        s = int(p.get("raw_start_frame", 0))
        e = int(p.get("raw_end_frame", s))

        if s >= T or s > task_end:
            report["dropped_phases"].append({
                "skill_type": p.get("skill_type"),
                "raw_span": [s, e],
                "reason": "phase_starts_after_task_horizon",
            })
            continue
        if e < 0:
            report["dropped_phases"].append({
                "skill_type": p.get("skill_type"),
                "raw_span": [s, e],
                "reason": "phase_ends_before_trajectory",
            })
            continue

        ns = max(0, s)
        ne = min(e, task_end, T - 1)
        if [ns, ne] != [s, e]:
            p.setdefault("consistency_adjustments", []).append({
                "type": "clamp_to_task_horizon",
                "original_raw_span": [s, e],
                "new_raw_span": [ns, ne],
            })
            report["clamped_phases"].append({
                "skill_type": p.get("skill_type"),
                "original_raw_span": [s, e],
                "new_raw_span": [ns, ne],
            })
        p["raw_start_frame"], p["raw_end_frame"] = ns, ne

        need = minimum_duration_frames(p.get("skill_type"), args.fps)
        duration = ne - ns + 1
        if has_duration_conflict(p) and duration < need:
            report["dropped_phases"].append({
                "skill_type": p.get("skill_type"),
                "raw_span": [ns, ne],
                "reason": "unresolved_duration_conflict_below_minimum",
                "duration_frames": duration,
                "minimum_duration_frames": need,
            })
            continue

        st = p.get("state_transition") or {}
        target_frame = st.get("target_evidence_frame")
        if target_frame is not None and int(target_frame) > task_end:
            p.setdefault("consistency_flags", []).append(
                "target_evidence_after_task_horizon"
            )
            report["evidence_horizon_warnings"].append({
                "skill_type": p.get("skill_type"),
                "target_evidence_frame": int(target_frame),
                "task_end_raw_frame": task_end,
            })

        candidates.append(p)

    # Final lifecycle gate. Invalid transitions are removed; unknown initial
    # state may be initialized only when the skill has one unambiguous source.
    state = dict(initial)
    kept = []
    for p in candidates:
        entity = p.get("entity")
        skill = p.get("skill_type")
        tmap = transitions.get(entity)
        if not tmap:
            kept.append(p)
            continue

        cur = state.get(entity, "unknown")
        nxt = tmap.get((cur, skill))
        if nxt is not None:
            state[entity] = nxt
            kept.append(p)
            continue

        if cur == "unknown":
            possible = [(src, dst) for (src, sk), dst in tmap.items() if sk == skill]
            if len(possible) == 1:
                src, dst = possible[0]
                state[entity] = dst
                p.setdefault("consistency_flags", []).append(
                    "lifecycle_initialized_from_unique_transition"
                )
                kept.append(p)
                continue

        report["lifecycle_violations"].append({
            "skill_type": skill,
            "entity": entity,
            "current_state": cur,
            "raw_span": [p["raw_start_frame"], p["raw_end_frame"]],
            "action": "dropped",
        })
        report["dropped_phases"].append({
            "skill_type": skill,
            "entity": entity,
            "raw_span": [p["raw_start_frame"], p["raw_end_frame"]],
            "reason": "schema_lifecycle_inconsistent_final_gate",
        })

    for entity, cfg in schema.get("entities", {}).items():
        expected = cfg.get("expected_final_state")
        if expected is None or entity not in transitions:
            continue
        observed = state.get(entity, "unknown")
        if observed != expected:
            report["expected_final_state_violations"].append({
                "entity": entity,
                "observed_final_state": observed,
                "expected_final_state": expected,
                "action": "reported_unresolved_no_action_invented",
            })

    kept.sort(key=lambda x: (x["raw_start_frame"], x["raw_end_frame"]))
    for i, p in enumerate(kept):
        p["skill_id"] = i

    ann["annotation_version"] = "v3.4.5"
    ann["task_end_raw_frame"] = task_end
    ann["semantic_phases"] = kept
    ann["trajectory_instruction"] = "；".join(
        x.get("instruction_zh", x.get("skill_type", "")) for x in kept
    )
    ann["coarse_only_segments"] = [
        {
            "raw_start_frame": s,
            "raw_end_frame": e,
            "label_policy": "coarse_task_only",
        }
        for s, e in complement(
            [(x["raw_start_frame"], x["raw_end_frame"]) for x in kept],
            T,
        )
    ]
    ann["final_consistency"] = {
        "num_dropped": len(report["dropped_phases"]),
        "num_clamped": len(report["clamped_phases"]),
        "num_lifecycle_violations": len(report["lifecycle_violations"]),
        "num_expected_final_state_violations": len(report["expected_final_state_violations"]),
        "status": (
            "consistent"
            if not report["expected_final_state_violations"]
            and not report["lifecycle_violations"]
            else "unresolved"
        ),
    }
    ann.setdefault("notes", []).append(
        "V3.4.5 final consistency gate never invents missing actions; it removes invalid final phases and reports unresolved schema-final-state violations."
    )

    ann_path.write_text(
        json.dumps(ann, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    diag_dir = out / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    report["final_phase_count"] = len(kept)
    report["final_consistency"] = ann["final_consistency"]
    (diag_dir / "final_consistency.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "final consistency:",
        f"kept={len(kept)}",
        f"dropped={len(report['dropped_phases'])}",
        f"clamped={len(report['clamped_phases'])}",
        f"final_state_violations={len(report['expected_final_state_violations'])}",
    )


if __name__ == "__main__":
    main()
