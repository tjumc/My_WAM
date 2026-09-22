#!/usr/bin/env python3
"""V3.4.4 joint temporal ownership for portable-object hand states.

Independent object trackers can assign overlapping hand intervals to different
objects. This stage resolves those conflicts before semantic episode inference.

The resolver uses only:
- task schema;
- tracked entity states;
- direct structured VLM hand/object evidence.

It never reads manual GT.

Key rules:
1. direct hand-object evidence is anchored to its observation window;
2. non-portable hand interactions can block a speculative portable-object hold;
3. when a task schema declares sequential portable manipulation, an earlier
   object completion owns an ambiguous prefix until the next object's direct
   ownership begins;
4. evidence is assigned to one object episode and is not borrowed by a sibling.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from schema_runtime import load_schema, portable_rules


BENIGN_HAND = {"free", "uncertain", "not_visible", None}
BLOCKER_PREFIXES = (
    "contact_", "pulling_", "pushing_", "opening_", "closing_",
    "pressing_", "turning_",
)


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def overlap(a0, a1, b0, b1):
    return int(a1) >= int(b0) and int(b1) >= int(a0)


def clip_interval(start, end):
    start, end = int(start), int(end)
    return (start, end) if end >= start else None


def holding_token_map(schema):
    out = {}
    for name, cfg in portable_rules(schema).items():
        tokens = cfg.get("holding_tokens") or [f"holding_{name}"]
        for token in tokens:
            out[token] = name
    return out


def side_owner_candidates(state, conf, hand, portable, token_map, min_conf):
    """Collect object identities directly supported on one temporal side."""
    cand = {}
    hv = state.get(hand)
    hc = float(conf.get(hand, 0.0))
    owner = token_map.get(hv)
    if owner is not None and hc >= min_conf:
        # Explicit hand identity is the stronger relation. Conflicting
        # object-location classifications are treated as perception ambiguity,
        # not as simultaneous direct ownership.
        cand[owner] = max(cand.get(owner, 0.0), hc)
        return cand

    for name, cfg in portable.items():
        key = cfg["observation_key"]
        if state.get(key) == hand:
            c = float(conf.get(key, 0.0))
            if c >= min_conf:
                cand[name] = max(cand.get(name, 0.0), c)
    return cand


def is_blocker(value, conf, min_conf):
    if value in BENIGN_HAND or float(conf) < min_conf:
        return False
    if isinstance(value, str) and value.startswith("holding_"):
        return False
    return isinstance(value, str) and value.startswith(BLOCKER_PREFIXES)


def build_direct_evidence(obs, schema):
    portable = portable_rules(schema)
    policy = schema.get("manipulation_policy", {})
    min_conf = float(policy.get("direct_ownership_min_confidence", 0.70))
    token_map = holding_token_map(schema)

    direct = {name: [] for name in portable}
    blockers = []

    for o in obs:
        if o.get("_v331_dense_target_entity") or o.get("_v33_dense_cutlery"):
            continue

        start = int(o.get("_window_start_frame", o.get("_raw_frame", 0)))
        end = int(o.get("_window_end_frame", o.get("_raw_frame", start)))
        mid = int(o.get("_raw_frame", (start + end) // 2))

        before = o.get("state_before") or {}
        after = o.get("state_after") or {}
        cb = o.get("confidence_before") or {}
        ca = o.get("confidence_after") or {}

        for hand in ("right_hand", "left_hand"):
            bowners = side_owner_candidates(before, cb, hand, portable, token_map, min_conf)
            aowners = side_owner_candidates(after, ca, hand, portable, token_map, min_conf)
            all_names = set(bowners) | set(aowners)

            for name in all_names:
                bc = bowners.get(name)
                ac = aowners.get(name)
                if bc is not None and ac is not None:
                    direct[name].append({
                        "hand": hand,
                        "start_frame": start,
                        "end_frame": end,
                        "confidence": round(min(bc, ac), 3),
                        "event_id": int(o.get("_event_id", -1)),
                        "support": "direct_window",
                    })
                elif bc is not None:
                    direct[name].append({
                        "hand": hand,
                        "start_frame": start,
                        "end_frame": mid,
                        "confidence": round(bc, 3),
                        "event_id": int(o.get("_event_id", -1)),
                        "support": "direct_before_half",
                    })
                elif ac is not None:
                    direct[name].append({
                        "hand": hand,
                        "start_frame": mid,
                        "end_frame": end,
                        "confidence": round(ac, 3),
                        "event_id": int(o.get("_event_id", -1)),
                        "support": "direct_after_half",
                    })

            bv, av = before.get(hand), after.get(hand)
            bc, ac = float(cb.get(hand, 0.0)), float(ca.get(hand, 0.0))
            bb = is_blocker(bv, bc, min_conf)
            ab = is_blocker(av, ac, min_conf)

            if bb and ab and bv == av:
                blockers.append({
                    "hand": hand, "start_frame": start, "end_frame": end,
                    "token": bv, "confidence": round(min(bc, ac), 3),
                    "event_id": int(o.get("_event_id", -1)),
                })
            else:
                if bb:
                    blockers.append({
                        "hand": hand, "start_frame": start, "end_frame": mid,
                        "token": bv, "confidence": round(bc, 3),
                        "event_id": int(o.get("_event_id", -1)),
                    })
                if ab:
                    blockers.append({
                        "hand": hand, "start_frame": mid, "end_frame": end,
                        "token": av, "confidence": round(ac, 3),
                        "event_id": int(o.get("_event_id", -1)),
                    })

    for name in direct:
        direct[name].sort(key=lambda x: (x["start_frame"], x["end_frame"]))
    blockers.sort(key=lambda x: (x["start_frame"], x["end_frame"]))
    return direct, blockers


def candidate_hand_segments(tracked, schema, direct):
    portable = portable_rules(schema)
    candidates = []

    for name, cfg in portable.items():
        key = cfg["observation_key"]
        if key not in tracked.get("tracks", {}):
            continue
        segs = tracked["tracks"][key].get("segments", [])
        hands = set(cfg.get("hand_states", ["right_hand", "left_hand"]))
        target = cfg["target"]

        for i, seg in enumerate(segs):
            if seg.get("state") not in hands:
                continue
            target_i = next(
                (j for j in range(i + 1, len(segs)) if segs[j].get("state") == target),
                None,
            )
            if target_i is None:
                continue

            start = int(seg["start_frame"])
            end = int(seg["end_frame"])
            hand = seg["state"]
            ds = [
                x for x in direct.get(name, [])
                if x["hand"] == hand
                and overlap(start, end, x["start_frame"], x["end_frame"])
            ]
            candidates.append({
                "object": name,
                "track_key": key,
                "segment_index": i,
                "hand": hand,
                "start_frame": start,
                "end_frame": end,
                "target_start_frame": int(segs[target_i]["start_frame"]),
                "target_segment_index": int(target_i),
                "direct_intervals": ds,
                "first_direct_frame": min(
                    (int(x["start_frame"]) for x in ds), default=None
                ),
                "direct_confidence": max(
                    (float(x["confidence"]) for x in ds), default=0.0
                ),
                "resolved_start_frame": start,
                "resolved_end_frame": end,
                "resolution_reasons": [],
                "unresolved_conflict": False,
            })
    return candidates


def resource_conflict(a, b, max_concurrent):
    if a["object"] == b["object"]:
        return False
    if a["hand"] == b["hand"]:
        return True
    return int(max_concurrent) <= 1


def resolve_candidates(candidates, blockers, schema):
    policy = schema.get("manipulation_policy", {})
    max_concurrent = int(policy.get("max_concurrent_portable_objects", 2))

    # First, direct evidence after a non-portable interaction blocks speculative
    # backward extension of an object hold through that interaction.
    for c in candidates:
        fd = c["first_direct_frame"]
        if fd is None or fd <= c["resolved_start_frame"]:
            continue
        relevant = [
            b for b in blockers
            if b["hand"] == c["hand"]
            and overlap(
                c["resolved_start_frame"], fd,
                b["start_frame"], b["end_frame"],
            )
        ]
        if relevant:
            c["resolved_start_frame"] = int(fd)
            c["resolution_reasons"].append({
                "type": "trim_prefix_after_nonportable_interaction",
                "first_direct_frame": int(fd),
                "blockers": relevant,
            })

    # Then resolve cross-object overlap. Earlier-completing objects keep an
    # ambiguous prefix until the later object's direct ownership begins.
    ordered = sorted(
        range(len(candidates)),
        key=lambda i: (
            candidates[i]["target_start_frame"],
            candidates[i]["start_frame"],
        ),
    )

    for ai_pos, ai in enumerate(ordered):
        a = candidates[ai]
        for bi in ordered[ai_pos + 1:]:
            b = candidates[bi]
            if not resource_conflict(a, b, max_concurrent):
                continue
            if not overlap(
                a["resolved_start_frame"], a["resolved_end_frame"],
                b["resolved_start_frame"], b["resolved_end_frame"],
            ):
                continue

            # By construction a completes no later than b.
            boundary = b.get("first_direct_frame")
            if (
                boundary is not None
                and a["resolved_start_frame"] <= boundary <= a["resolved_end_frame"]
            ):
                old_a_end = a["resolved_end_frame"]
                old_b_start = b["resolved_start_frame"]
                a["resolved_end_frame"] = min(a["resolved_end_frame"], int(boundary) - 1)
                b["resolved_start_frame"] = max(b["resolved_start_frame"], int(boundary))
                a["resolution_reasons"].append({
                    "type": "end_at_next_object_direct_ownership",
                    "next_object": b["object"],
                    "boundary_frame": int(boundary),
                    "previous_end_frame": int(old_a_end),
                })
                b["resolution_reasons"].append({
                    "type": "start_at_direct_ownership_after_competitor",
                    "previous_object": a["object"],
                    "boundary_frame": int(boundary),
                    "previous_start_frame": int(old_b_start),
                })
                continue

            # If the earlier object has direct support and the later one has no
            # direct support, the overlap cannot be safely assigned to the later
            # object. Keep it unresolved for a future targeted re-observation.
            if a["first_direct_frame"] is not None and b["first_direct_frame"] is None:
                b["unresolved_conflict"] = True
                b["resolution_reasons"].append({
                    "type": "unresolved_overlap_without_later_direct_evidence",
                    "competing_object": a["object"],
                })
            elif a["first_direct_frame"] is None and b["first_direct_frame"] is None:
                a["unresolved_conflict"] = True
                b["unresolved_conflict"] = True
                info = {
                    "type": "unresolved_overlap_without_direct_evidence",
                    "objects": [a["object"], b["object"]],
                }
                a["resolution_reasons"].append(info)
                b["resolution_reasons"].append(info)

    for c in candidates:
        if c["resolved_end_frame"] < c["resolved_start_frame"]:
            c["assignment_status"] = "suppressed"
        elif c["unresolved_conflict"]:
            c["assignment_status"] = "ambiguous"
        elif c["first_direct_frame"] is not None:
            c["assignment_status"] = "direct"
        else:
            c["assignment_status"] = "inferred_unique"
    return candidates


def suppressed_segment(seg, start, end, reason):
    return {
        "state": "other",
        "start_frame": int(start),
        "end_frame": int(end),
        "confidence": 0.52,
        "support": "ownership_suppressed",
        "event_ids": [],
        "ownership_suppression_reason": reason,
    }


def apply_resolution(tracked, candidates):
    out = copy.deepcopy(tracked)
    by_track_index = {
        (c["track_key"], int(c["segment_index"])): c
        for c in candidates
    }

    for track_key, track in out.get("tracks", {}).items():
        new_segments = []
        for i, seg0 in enumerate(track.get("segments", [])):
            c = by_track_index.get((track_key, i))
            if c is None:
                new_segments.append(seg0)
                continue

            seg = copy.deepcopy(seg0)
            old_s, old_e = int(seg["start_frame"]), int(seg["end_frame"])
            new_s = int(c["resolved_start_frame"])
            new_e = int(c["resolved_end_frame"])

            if new_s > old_s:
                new_segments.append(
                    suppressed_segment(
                        seg, old_s, min(old_e, new_s - 1),
                        "trimmed_before_owned_interval",
                    )
                )

            if new_e >= new_s:
                seg["start_frame"] = new_s
                seg["end_frame"] = new_e
                seg["ownership_status"] = c["assignment_status"]
                seg["ownership_direct_confidence"] = round(
                    float(c["direct_confidence"]), 3
                )
                seg["ownership_resolution_reasons"] = c["resolution_reasons"]
                if any(
                    x.get("type") == "end_at_next_object_direct_ownership"
                    for x in c["resolution_reasons"]
                ):
                    seg["ownership_end_reason"] = "next_object_direct_ownership"
                    seg["ownership_context_end_frame"] = new_e + 1
                new_segments.append(seg)

            if new_e < old_e:
                new_segments.append(
                    suppressed_segment(
                        seg0, max(old_s, new_e + 1), old_e,
                        "trimmed_after_owned_interval",
                    )
                )

        track["segments"] = sorted(
            [x for x in new_segments if int(x["end_frame"]) >= int(x["start_frame"])],
            key=lambda x: (int(x["start_frame"]), int(x["end_frame"])),
        )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--schema", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    schema = load_schema(args.schema)
    tracked = json.load(open(out_dir / "tracked_entity_states.json", encoding="utf-8"))
    obs = load_jsonl(out_dir / "entity_observations.jsonl")

    direct, blockers = build_direct_evidence(obs, schema)
    candidates = candidate_hand_segments(tracked, schema, direct)
    candidates = resolve_candidates(candidates, blockers, schema)
    resolved = apply_resolution(tracked, candidates)

    resolved["annotation_version"] = "v3.4.4"
    resolved["ownership_resolution_applied"] = True
    resolved["task_schema"] = str(args.schema)
    resolved["task_family"] = schema.get("task_family")

    report = {
        "annotation_version": "v3.4.4",
        "task_family": schema.get("task_family"),
        "policy": schema.get("manipulation_policy", {}),
        "direct_ownership": direct,
        "nonportable_hand_blockers": blockers,
        "candidates": candidates,
        "num_ambiguous_candidates": sum(
            c["assignment_status"] == "ambiguous" for c in candidates
        ),
        "num_suppressed_candidates": sum(
            c["assignment_status"] == "suppressed" for c in candidates
        ),
        "notes": [
            "Manual GT is not read.",
            "Direct ownership evidence cannot be borrowed by a sibling object.",
            "Ambiguous unresolved conflicts are preserved for later targeted re-observation.",
        ],
    }

    (out_dir / "tracked_entity_states_owned.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "hand_object_ownership.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("portable-object ownership:")
    for c in candidates:
        if (
            c["resolved_start_frame"] != c["start_frame"]
            or c["resolved_end_frame"] != c["end_frame"]
            or c["assignment_status"] != "direct"
        ):
            print(
                f"  {c['object']:<12} {c['hand']:<10} "
                f"[{c['start_frame']},{c['end_frame']}] -> "
                f"[{c['resolved_start_frame']},{c['resolved_end_frame']}] "
                f"target={c['target_start_frame']} status={c['assignment_status']}"
            )
    print("saved", out_dir / "tracked_entity_states_owned.json")
    print("saved", out_dir / "hand_object_ownership.json")


if __name__ == "__main__":
    main()
