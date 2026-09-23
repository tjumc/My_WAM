#!/usr/bin/env python3
"""V3.4.8 state-implication-aware final task-graph decoder.

The task-completion frontier is not defined by the last predicted
phase. It is inferred from schema-declared expected final states plus dependency
use, so a late spurious prediction cannot extend the horizon that validates it.
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
            rows[(t.get("from"), t.get("skill_type"))] = t.get("to")
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


def phase_entity_uses(phase):
    out = set()
    ent = phase.get("entity")
    fine = phase.get("fine_entity")
    if ent:
        out.add(ent)
    if fine:
        out.add(fine)
    st = phase.get("state_transition") or {}
    after = st.get("after")
    if after:
        out.add(after)
    return out


def dependency_descendants(schema):
    """Map an entity to all downstream entities whose access depends on it."""
    direct = {}
    for entity, cfg in schema.get("entities", {}).items():
        for req in cfg.get("requires", []) or []:
            parent = req.get("entity")
            if parent:
                direct.setdefault(parent, set()).add(entity)

    out = {}
    for root in schema.get("entities", {}):
        seen = set()
        stack = list(direct.get(root, set()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(direct.get(x, set()))
        out[root] = seen
    return out


def initial_states(schema, validated):
    observed = (
        validated.get("validation_policy", {}).get("initial_container_state", {})
        or {}
    )
    state = dict(observed)
    source = {k: "perception" for k in state}
    for entity, cfg in schema.get("entities", {}).items():
        prior = cfg.get("expected_initial_state")
        if prior is not None:
            state[entity] = prior
            source[entity] = "schema_expected_initial_state"
    return state, source


def phase_state_implications(schema, phase):
    """Return latent entity states logically required by an accepted phase.

    Two schema-driven implication families are supported:

    1. accessibility prerequisite:
       an accepted use of entity X implies every X.requires state held;
    2. receptacle usage state:
       an accepted placement whose target is receptacle R implies
       R=usage_state during that placement.

    These implications update hidden lifecycle state only. They never fabricate
    a missing semantic action.
    """
    rows = []

    # Accessibility / dependency implications.
    for used in phase_entity_uses(phase):
        cfg = schema.get("entities", {}).get(used, {})
        for req in cfg.get("requires", []) or []:
            parent = req.get("entity")
            required = req.get("state")
            if parent and required is not None:
                rows.append({
                    "entity": parent,
                    "state": required,
                    "supporting_entity": used,
                    "reason": "accepted_dependent_use_requires_prerequisite_state",
                })

    # Placement-to-receptacle usage implication. The trajectory-level portable
    # object state machine already records the destination in state_transition.after.
    target = (phase.get("state_transition") or {}).get("after")
    target_cfg = schema.get("entities", {}).get(target, {})
    usage_state = target_cfg.get("usage_state")
    if target and usage_state is not None:
        rows.append({
            "entity": target,
            "state": usage_state,
            "supporting_entity": target,
            "reason": "accepted_placement_requires_receptacle_usage_state",
        })

    # Deduplicate equivalent implications within one phase.
    unique = []
    seen = set()
    for row in rows:
        key = (row["entity"], row["state"], row["reason"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def apply_state_implications(schema, phase, state, report=None, attach=True):
    for imp in phase_state_implications(schema, phase):
        entity = imp["entity"]
        required = imp["state"]
        current = state.get(entity, "unknown")
        if current == required:
            continue
        state[entity] = required
        row = {
            "supporting_skill": phase.get("skill_type"),
            "supporting_entity": imp.get("supporting_entity"),
            "implied_entity": entity,
            "previous_state": current,
            "inferred_state": required,
            "raw_span": [phase.get("raw_start_frame"), phase.get("raw_end_frame")],
            "reason": imp["reason"],
        }
        if attach:
            existing = phase.setdefault("latent_state_inference", [])
            key = (
                row["supporting_skill"], row["implied_entity"],
                row["inferred_state"], row["reason"],
            )
            existing_keys = {
                (
                    x.get("supporting_skill"),
                    x.get("implied_entity", x.get("prerequisite_entity")),
                    x.get("inferred_state"),
                    x.get("reason"),
                )
                for x in existing
            }
            if key not in existing_keys:
                existing.append(row)
        if report is not None:
            report.setdefault("latent_state_inferences", []).append(row)
            if row["reason"] == "accepted_dependent_use_requires_prerequisite_state":
                report.setdefault("latent_prerequisite_state_inferences", []).append(row)
            elif row["reason"] == "accepted_placement_requires_receptacle_usage_state":
                report.setdefault("latent_usage_state_inferences", []).append(row)

def lifecycle_filter(phases, transitions, initial, report, schema, attach_implications=True):
    state = dict(initial)
    kept = []
    for p in sorted(phases, key=lambda x: (x["raw_start_frame"], x["raw_end_frame"])):
        apply_state_implications(
            schema, p, state, report=report, attach=attach_implications
        )

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
            p.setdefault("lifecycle_decode", {})["before"] = cur
            p["lifecycle_decode"]["after"] = nxt
            kept.append(p)
            continue

        if cur == "unknown":
            possible = [(src, dst) for (src, sk), dst in tmap.items() if sk == skill]
            if len(possible) == 1:
                src, dst = possible[0]
                state[entity] = dst
                p.setdefault("lifecycle_decode", {})["before"] = src
                p["lifecycle_decode"]["after"] = dst
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
    return kept, state


def terminal_closure_info(schema, phases, initial, transitions):
    descendants = dependency_descendants(schema)
    info = {}

    # A use of an articulated entity includes direct placement into that entity
    # and any action involving a descendant that requires it to be accessible.
    for entity, cfg in schema.get("entities", {}).items():
        expected = cfg.get("expected_final_state")
        if expected is None or entity not in transitions:
            continue

        downstream = descendants.get(entity, set())
        use_frames = []
        for p in phases:
            uses = phase_entity_uses(p)
            if entity in uses and p.get("entity") != entity:
                use_frames.append(int(p["raw_end_frame"]))
            elif downstream.intersection(uses):
                use_frames.append(int(p["raw_end_frame"]))
        last_use = max(use_frames) if use_frames else -1

        cur = initial.get(entity, "unknown")
        closure = None
        lifecycle = []
        for p in sorted(phases, key=lambda x: (x["raw_start_frame"], x["raw_end_frame"])):
            # Accepted downstream actions may imply latent prerequisite or
            # receptacle usage states even when the explicit transition was missed.
            for imp in phase_state_implications(schema, p):
                if imp.get("entity") == entity:
                    cur = imp.get("state")

            if p.get("entity") != entity:
                continue
            skill = p.get("skill_type")
            nxt = transitions[entity].get((cur, skill))
            if nxt is None and cur == "unknown":
                poss = [(src, dst) for (src, sk), dst in transitions[entity].items() if sk == skill]
                if len(poss) == 1:
                    cur, nxt = poss[0]
            if nxt is None:
                continue
            before = cur
            cur = nxt
            lifecycle.append({
                "skill_type": skill,
                "raw_span": [p["raw_start_frame"], p["raw_end_frame"]],
                "before": before,
                "after": nxt,
            })
            if (
                closure is None
                and nxt == expected
                and int(p["raw_end_frame"]) >= int(last_use)
            ):
                closure = {
                    "skill_type": skill,
                    "raw_start_frame": int(p["raw_start_frame"]),
                    "raw_end_frame": int(p["raw_end_frame"]),
                }

        info[entity] = {
            "expected_final_state": expected,
            "last_dependent_use_end": last_use if last_use >= 0 else None,
            "terminal_closure": closure,
            "lifecycle": lifecycle,
        }
    return info


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
    transitions = transition_maps(schema)
    initial, initial_source = initial_states(schema, validated)
    completion_policy = schema.get("task_completion", {}) or {}

    report = {
        "annotation_version": "v3.4.8",
        "trajectory_frames": T,
        "upstream_predicted_task_end_raw_frame": ann.get("task_end_raw_frame"),
        "initial_container_state": initial,
        "initial_state_source": initial_source,
        "dropped_phases": [],
        "clamped_phases": [],
        "lifecycle_violations": [],
        "latent_state_inferences": [],
        "latent_prerequisite_state_inferences": [],
        "latent_usage_state_inferences": [],
        "gratuitous_lifecycle_cycles": [],
        "expected_final_state_violations": [],
    }

    # Stage 1: only trajectory bounds and hard duration conflicts are enforced.
    candidates = []
    for phase in sorted(
        ann.get("semantic_phases", []),
        key=lambda x: (int(x.get("raw_start_frame", 0)), int(x.get("raw_end_frame", 0))),
    ):
        p = dict(phase)
        s = int(p.get("raw_start_frame", 0))
        e = int(p.get("raw_end_frame", s))
        if s >= T or e < 0:
            report["dropped_phases"].append({
                "skill_type": p.get("skill_type"),
                "raw_span": [s, e],
                "reason": "outside_trajectory_horizon",
            })
            continue
        ns, ne = max(0, s), min(T - 1, e)
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
        candidates.append(p)

    # Stage 2: lifecycle legality from schema initial state / observed fallback.
    kept, _ = lifecycle_filter(candidates, transitions, initial, report, schema)

    # Stage 3: identify the first expected-final transition after the last
    # dependent use. Later same-entity cycles have no task-enabling role.
    closures = terminal_closure_info(schema, kept, initial, transitions)
    if completion_policy.get("prune_gratuitous_post_final_cycles", True):
        drop_ids = set()
        for entity, x in closures.items():
            closure = x.get("terminal_closure")
            if not closure:
                continue
            cutoff = int(closure["raw_end_frame"])
            for idx, p in enumerate(kept):
                if p.get("entity") == entity and int(p["raw_start_frame"]) > cutoff:
                    drop_ids.add(idx)
                    report["gratuitous_lifecycle_cycles"].append({
                        "entity": entity,
                        "skill_type": p.get("skill_type"),
                        "raw_span": [p["raw_start_frame"], p["raw_end_frame"]],
                        "terminal_closure": closure,
                        "reason": "post_expected_final_transition_without_later_dependent_use",
                    })
                    report["dropped_phases"].append({
                        "entity": entity,
                        "skill_type": p.get("skill_type"),
                        "raw_span": [p["raw_start_frame"], p["raw_end_frame"]],
                        "reason": "gratuitous_post_final_lifecycle_cycle",
                    })
        if drop_ids:
            kept = [p for i, p in enumerate(kept) if i not in drop_ids]
            # Recompute closure metadata after pruning.
            closures = terminal_closure_info(schema, kept, initial, transitions)

    # Stage 4: independent task-completion frontier.
    frontier_entities = list(completion_policy.get("frontier_entities") or [])
    if not frontier_entities:
        # Generic fallback: expected-final entities that gate downstream entities.
        descendants = dependency_descendants(schema)
        frontier_entities = [
            e for e, cfg in schema.get("entities", {}).items()
            if cfg.get("expected_final_state") is not None and descendants.get(e)
        ]

    frontier_rows = []
    for entity in frontier_entities:
        row = closures.get(entity, {})
        closure = row.get("terminal_closure")
        frontier_rows.append({
            "entity": entity,
            "expected_final_state": row.get("expected_final_state"),
            "last_dependent_use_end": row.get("last_dependent_use_end"),
            "terminal_closure": closure,
        })

    frontier_resolved = bool(frontier_rows) and all(x.get("terminal_closure") for x in frontier_rows)
    task_end = (
        max(int(x["terminal_closure"]["raw_end_frame"]) for x in frontier_rows)
        if frontier_resolved else T - 1
    )

    # Any semantic phase strictly after a resolved completion frontier is not
    # part of the task, regardless of whether it is visually plausible.
    final = []
    for p in kept:
        s, e = int(p["raw_start_frame"]), int(p["raw_end_frame"])
        if frontier_resolved and s > task_end:
            report["dropped_phases"].append({
                "skill_type": p.get("skill_type"),
                "entity": p.get("entity"),
                "raw_span": [s, e],
                "reason": "phase_after_task_completion_frontier",
            })
            continue
        ne = min(e, task_end) if frontier_resolved else e
        if ne != e:
            report["clamped_phases"].append({
                "skill_type": p.get("skill_type"),
                "original_raw_span": [s, e],
                "new_raw_span": [s, ne],
            })
            p.setdefault("consistency_adjustments", []).append({
                "type": "clamp_to_task_completion_frontier",
                "original_raw_span": [s, e],
                "new_raw_span": [s, ne],
            })
            p["raw_end_frame"] = ne

        ps = int(p.get("provisional_start_frame", p["raw_start_frame"]))
        pe = int(p.get("provisional_end_frame", p["raw_end_frame"]))
        limit = task_end if frontier_resolved else T - 1
        nps = max(0, min(ps, limit))
        npe = max(nps, min(pe, limit))
        if [nps, npe] != [ps, pe]:
            p.setdefault("consistency_adjustments", []).append({
                "type": "clamp_provisional_to_completion_frontier",
                "original_provisional_span": [ps, pe],
                "new_provisional_span": [nps, npe],
            })
            report["clamped_phases"].append({
                "skill_type": p.get("skill_type"),
                "field": "provisional",
                "original_provisional_span": [ps, pe],
                "new_provisional_span": [nps, npe],
            })
        p["provisional_start_frame"], p["provisional_end_frame"] = nps, npe
        final.append(p)

    # Re-evaluate final state after all pruning.
    _, final_state = lifecycle_filter(final, transitions, initial, {
        "lifecycle_violations": [],
        "dropped_phases": [],
        "latent_state_inferences": [],
        "latent_prerequisite_state_inferences": [],
        "latent_usage_state_inferences": [],
    }, schema, attach_implications=False)
    for entity, cfg in schema.get("entities", {}).items():
        expected = cfg.get("expected_final_state")
        if expected is None or entity not in transitions:
            continue
        observed = final_state.get(entity, "unknown")
        if observed != expected:
            report["expected_final_state_violations"].append({
                "entity": entity,
                "observed_final_state": observed,
                "expected_final_state": expected,
                "action": "reported_unresolved_no_action_invented",
            })

    final.sort(key=lambda x: (x["raw_start_frame"], x["raw_end_frame"]))
    for i, p in enumerate(final):
        p["skill_id"] = i

    frontier = {
        "policy": completion_policy.get(
            "frontier_policy",
            "first_expected_final_state_after_last_dependent_use",
        ),
        "frontier_entities": frontier_entities,
        "entities": frontier_rows,
        "resolved": frontier_resolved,
        "task_end_raw_frame": int(task_end),
        "fallback_when_unresolved": "trajectory_end",
    }

    status = (
        "consistent"
        if frontier_resolved
        and not report["lifecycle_violations"]
        and not report["expected_final_state_violations"]
        else "unresolved"
    )

    ann["annotation_version"] = "v3.4.8"
    ann["task_end_raw_frame"] = int(task_end)
    ann["task_completion_frontier"] = frontier
    ann["semantic_phases"] = final
    ann["trajectory_instruction"] = "；".join(
        x.get("instruction_zh", x.get("skill_type", "")) for x in final
    )
    ann["coarse_only_segments"] = [
        {
            "raw_start_frame": s,
            "raw_end_frame": e,
            "label_policy": "coarse_task_only",
        }
        for s, e in complement(
            [(x["raw_start_frame"], x["raw_end_frame"]) for x in final],
            T,
        )
    ]
    ann["final_consistency"] = {
        "num_dropped": len(report["dropped_phases"]),
        "num_clamped": len(report["clamped_phases"]),
        "num_lifecycle_violations": len(report["lifecycle_violations"]),
        "num_latent_state_inferences": len(report["latent_state_inferences"]),
        "num_latent_usage_state_inferences": len(report["latent_usage_state_inferences"]),
        "num_gratuitous_lifecycle_cycles": len(report["gratuitous_lifecycle_cycles"]),
        "num_expected_final_state_violations": len(report["expected_final_state_violations"]),
        "frontier_resolved": frontier_resolved,
        "status": status,
    }
    ann.setdefault("notes", []).append(
        "V3.4.8 additionally rehabilitates only directly evidenced rejected candidates and preserves validated pass1 anchors during boundary arbitration; missing actions are never fabricated."
    )

    ann_path.write_text(json.dumps(ann, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report["task_completion_frontier"] = frontier
    report["final_phase_count"] = len(final)
    report["final_consistency"] = ann["final_consistency"]
    diag = out / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    (diag / "final_consistency.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "final consistency:",
        f"kept={len(final)}",
        f"dropped={len(report['dropped_phases'])}",
        f"frontier_resolved={frontier_resolved}",
        f"task_end={task_end}",
        f"final_state_violations={len(report['expected_final_state_violations'])}",
    )


if __name__ == "__main__":
    main()
