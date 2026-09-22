#!/usr/bin/env python3
"""V3.3.2 trajectory-level entity state tracking.

Fuses overlapping before/after VLM observations into a physically consistent
trajectory for each entity. The tracker uses:
- confidence-weighted emissions
- persistence
- graph-constrained transitions
- hand/object exclusivity
- weak affordance priors for occluded releases

It does not use the manual regression fixture.
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

UNKNOWN = {"uncertain", "not_visible", None}

SPECS = {
    "door": {
        "states": ["closed", "opening", "open", "closing"],
        "adj": {
            "closed": ["opening", "closing"],
            "opening": ["closed", "open"],
            "open": ["opening", "closing"],
            "closing": ["open", "closed"],
        },
        "stable": {"closed", "open"},
    },
    "dish_rack": {
        "states": ["in", "moving_out", "out", "moving_in"],
        "adj": {
            "in": ["moving_out", "moving_in"],
            "moving_out": ["in", "out"],
            "out": ["moving_out", "moving_in"],
            "moving_in": ["out", "in"],
        },
        "stable": {"in", "out"},
    },
    "cutlery_basket": {
        "states": ["in", "moving_out", "out", "moving_in"],
        "adj": {
            "in": ["moving_out", "moving_in"],
            "moving_out": ["in", "out"],
            "out": ["moving_out", "moving_in"],
            "moving_in": ["out", "in"],
        },
        "stable": {"in", "out"},
    },
    "knife_location": {
        "states": ["tabletop", "right_hand", "left_hand", "cutlery_basket", "other"],
        "adj": {
            "tabletop": ["right_hand", "left_hand"],
            "right_hand": ["tabletop", "cutlery_basket", "other"],
            "left_hand": ["tabletop", "cutlery_basket", "other"],
            "cutlery_basket": ["right_hand", "left_hand"],
            "other": ["right_hand", "left_hand"],
        },
        "target": "cutlery_basket",
    },
    "fork_location": {
        "states": ["tabletop", "right_hand", "left_hand", "cutlery_basket", "other"],
        "adj": {
            "tabletop": ["right_hand", "left_hand"],
            "right_hand": ["tabletop", "cutlery_basket", "other"],
            "left_hand": ["tabletop", "cutlery_basket", "other"],
            "cutlery_basket": ["right_hand", "left_hand"],
            "other": ["right_hand", "left_hand"],
        },
        "target": "cutlery_basket",
    },
    "plate_location": {
        "states": ["tabletop", "right_hand", "left_hand", "dish_rack", "other"],
        "adj": {
            "tabletop": ["right_hand", "left_hand"],
            "right_hand": ["tabletop", "dish_rack", "other"],
            "left_hand": ["tabletop", "dish_rack", "other"],
            "dish_rack": ["right_hand", "left_hand"],
            "other": ["right_hand", "left_hand"],
        },
        "target": "dish_rack",
    },
}

QUALITY_WEIGHT = {"good": 1.0, "partial": 0.8, "poor": 0.45}


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def graph_distance(spec, a, b):
    if a == b:
        return 0
    q = deque([(a, 0)])
    seen = {a}
    while q:
        cur, d = q.popleft()
        for nxt in spec["adj"].get(cur, []):
            if nxt == b:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return 99


def hand_identity_evidence(obs, side, entity):
    """Return extra positive/negative evidence induced by observed hand identity."""
    key = "right_hand" if side == "before" else "right_hand"
    # side refers temporal side, not physical hand.
    state = (obs.get(f"state_{side}") or {})
    conf = (obs.get(f"confidence_{side}") or {})
    out = []
    obj = entity.replace("_location", "")
    for hand in ("right_hand", "left_hand"):
        hv = state.get(hand, "uncertain")
        hc = float(conf.get(hand, 0.0))
        if hv == f"holding_{obj}" and hc >= 0.7:
            out.append(("must", hand, min(0.98, hc)))
        elif hv.startswith("holding_") and hv != f"holding_{obj}" and hc >= 0.8:
            out.append(("not", hand, min(0.98, hc)))
    return out


def build_anchor_evidence(obs):
    """Build evidence plus entity-specific timelines.

    Generic observations define the shared baseline timeline. Dense targeted
    observations add frames only to their target entity, so adding cutlery
    evidence cannot perturb dish-rack/object Viterbi paths.
    """
    by_frame = defaultdict(lambda: defaultdict(list))
    generic_frames = set()
    targeted_frames = defaultdict(set)

    for o in obs:
        q = QUALITY_WEIGHT.get(o.get("window_quality"), 0.7)
        dense_target = o.get("_v331_dense_target_entity")
        if dense_target is None and o.get("_v33_dense_cutlery"):
            dense_target = "cutlery_basket"

        obs_frames = {
            int(o["_window_start_frame"]),
            int(o["_window_end_frame"]),
        }
        if dense_target:
            targeted_frames[dense_target].update(obs_frames)
        else:
            generic_frames.update(obs_frames)

        for side, frame_key in (("before", "_window_start_frame"), ("after", "_window_end_frame")):
            frame = int(o[frame_key])
            st = o.get(f"state_{side}") or {}
            cf = o.get(f"confidence_{side}") or {}
            for entity in SPECS:
                v = st.get(entity, "uncertain")
                c = max(0.0, min(1.0, float(cf.get(entity, 0.0)) * q))
                weight = 1.0
                if dense_target == entity:
                    weight = 1.8
                by_frame[frame][entity].append({
                    "kind": "state",
                    "value": v,
                    "confidence": c,
                    "source_weight": weight,
                    "event_id": int(o["_event_id"]),
                    "side": side,
                    "source": "dense_targeted" if dense_target == entity else "generic",
                })
                if entity.endswith("_location") and not dense_target:
                    for kind, hand, hc in hand_identity_evidence(o, side, entity):
                        by_frame[frame][entity].append({
                            "kind": kind,
                            "value": hand,
                            "confidence": max(0.0, min(1.0, hc * q)),
                            "source_weight": 1.0,
                            "event_id": int(o["_event_id"]),
                            "side": side,
                            "source": "generic",
                        })
    return by_frame, sorted(generic_frames), {k: sorted(v) for k, v in targeted_frames.items()}

def inject_dense_motion_evidence(by_frame, targeted_frames, dense_rows):
    """Add direction-only evidence at dense-window centers.

    Unlike V3.3/V3.3.1 this never invents absolute in/out states.
    """
    for x in dense_rows:
        entity = x.get("entity", "cutlery_basket")
        if entity not in SPECS:
            continue
        motion = x.get("relative_motion")
        if motion not in {"moving_out", "moving_in"}:
            continue
        frame = int(x.get("center_frame", 0))
        conf = max(0.0, min(1.0, float(x.get("confidence", 0.0))))
        targeted_frames.setdefault(entity, [])
        targeted_frames[entity].append(frame)
        by_frame[frame][entity].append({
            "kind": "motion",
            "value": motion,
            "confidence": conf,
            "source_weight": 1.8,
            "event_id": int(x.get("event_id", 10000 + int(x.get("dense_window_id", 0)))),
            "side": "center",
            "source": "dense_relative_motion",
        })


def emission_score(spec, state, evidence):
    score = 0.0
    direct_support = 0.0
    for e in evidence:
        c = max(0.0, min(1.0, float(e.get("confidence", 0.0))))
        effective = c * max(0.0, float(e.get("source_weight", 1.0)))
        kind = e["kind"]
        v = e["value"]
        if kind == "state":
            if v in UNKNOWN:
                continue
            if v == state:
                score += 4.0 * effective
                direct_support = max(direct_support, c)
            else:
                d = graph_distance(spec, state, v) if v in spec["states"] else 99
                if d == 1:
                    score += 0.25 * effective
                else:
                    score -= 3.0 * effective
        elif kind == "motion":
            # Directional evidence constrains the motion state but deliberately
            # does not vote for absolute in/out endpoints.
            if state == v:
                score += 4.0 * effective
                direct_support = max(direct_support, c)
            elif state in {"moving_out", "moving_in"}:
                score -= 4.0 * effective
        elif kind == "must":
            score += 3.5 * effective if state == v else -2.5 * effective
            if state == v:
                direct_support = max(direct_support, c)
        elif kind == "not" and state == v:
            score -= 4.0 * effective

    # Sparse-anchor weak priors. They are deliberately small relative to visual evidence.
    if "target" in spec:
        if state == spec["target"]:
            score += 0.08
        elif state == "other":
            score -= 0.08
    elif state in spec.get("stable", set()):
        score += 0.05
    return score, direct_support


def transition_score(spec, prev, cur, gap_frames):
    d = graph_distance(spec, prev, cur)
    if d == 0:
        return 0.35
    # Larger temporal gaps make state changes more plausible.
    gap_scale = min(1.0, 30.0 / max(30.0, float(gap_frames)))
    base = {1: -0.25, 2: -1.0, 3: -2.0}.get(d, -3.5)
    score = base * gap_scale

    # Affordance-compatible release is a weak prior, only used when visual evidence is missing.
    target = spec.get("target")
    if target and prev in ("right_hand", "left_hand"):
        if cur == target:
            score += 0.28
        elif cur == "tabletop":
            score -= 0.10
        elif cur == "other":
            score -= 0.25
    return score


def viterbi_track(entity, spec, frames, by_frame):
    states = spec["states"]
    if not frames:
        return []
    dp = []
    back = []
    support = []

    first_ev = by_frame[frames[0]].get(entity, [])
    d0, b0, s0 = {}, {}, {}
    for st in states:
        em, sup = emission_score(spec, st, first_ev)
        d0[st] = em
        b0[st] = None
        s0[st] = sup
    dp.append(d0); back.append(b0); support.append(s0)

    for i in range(1, len(frames)):
        gap = max(1, frames[i] - frames[i - 1])
        ev = by_frame[frames[i]].get(entity, [])
        di, bi, si = {}, {}, {}
        for cur in states:
            em, sup = emission_score(spec, cur, ev)
            best_prev = None
            best_score = -1e18
            for prev in states:
                sc = dp[i - 1][prev] + transition_score(spec, prev, cur, gap) + em
                if sc > best_score:
                    best_score = sc
                    best_prev = prev
            di[cur] = best_score
            bi[cur] = best_prev
            si[cur] = sup
        dp.append(di); back.append(bi); support.append(si)

    cur = max(states, key=lambda s: dp[-1][s])
    path = [cur]
    for i in range(len(frames) - 1, 0, -1):
        cur = back[i][cur]
        path.append(cur)
    path.reverse()

    # Remove isolated one-anchor flips when both neighbors agree and the center lacks direct support.
    for i in range(1, len(path) - 1):
        if path[i - 1] == path[i + 1] != path[i] and support[i].get(path[i], 0.0) < 0.85:
            path[i] = path[i - 1]

    anchors = []
    for i, (f, st) in enumerate(zip(frames, path)):
        ev = by_frame[f].get(entity, [])
        match = [e for e in ev if e["kind"] in ("state", "must", "motion") and e["value"] == st]
        direct = max((float(e["confidence"]) for e in match), default=0.0)
        src = sorted({int(e["event_id"]) for e in ev if e["kind"] in ("state", "must", "motion") and e["value"] == st})
        support_type = "observed" if direct >= 0.55 else "interpolated"
        conf = direct if direct >= 0.55 else 0.52
        anchors.append({
            "frame": int(f),
            "state": st,
            "confidence": round(float(conf), 3),
            "support": support_type,
            "event_ids": src,
        })
    return anchors


def anchors_to_segments(anchors):
    if not anchors:
        return []
    segs = []
    start = 0
    for i in range(1, len(anchors) + 1):
        if i == len(anchors) or anchors[i]["state"] != anchors[start]["state"]:
            left = anchors[start]["frame"]
            right = anchors[i - 1]["frame"]
            if start > 0:
                left = int(round((anchors[start - 1]["frame"] + anchors[start]["frame"]) / 2))
            if i < len(anchors):
                right = int(round((anchors[i - 1]["frame"] + anchors[i]["frame"]) / 2))
            vals = anchors[start:i]
            direct = [x["confidence"] for x in vals if x["support"] == "observed"]
            conf = max(direct) if direct else 0.52
            segs.append({
                "state": anchors[start]["state"],
                "start_frame": int(left),
                "end_frame": int(right),
                "confidence": round(float(conf), 3),
                "support": "observed" if direct else "interpolated",
                "event_ids": sorted({eid for x in vals for eid in x["event_ids"]}),
            })
            start = i
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    obs = sorted(load_jsonl(out / "entity_observations.jsonl"), key=lambda x: x["_raw_frame"])
    by_frame, generic_frames, targeted_frames = build_anchor_evidence(obs)
    dense_path = out / "dense_cutlery_transition_evidence.jsonl"
    dense_rows = load_jsonl(dense_path) if dense_path.exists() else []
    inject_dense_motion_evidence(by_frame, targeted_frames, dense_rows)

    tracks = {}
    entity_anchor_frames = {}
    for entity, spec in SPECS.items():
        frames = sorted(set(generic_frames) | set(targeted_frames.get(entity, [])))
        entity_anchor_frames[entity] = frames
        anchors = viterbi_track(entity, spec, frames, by_frame)
        tracks[entity] = {
            "anchors": anchors,
            "segments": anchors_to_segments(anchors),
        }

    result = {
        "annotation_version": "v3.3.2",
        "num_windows": len(obs),
        "anchor_frames": sorted({f for xs in entity_anchor_frames.values() for f in xs}),
        "entity_anchor_frames": entity_anchor_frames,
        "tracks": tracks,
        "notes": [
            "Tracking is confidence-weighted and graph-constrained across the full trajectory.",
            "Unknown/not-visible observations do not erase a persistent state.",
            "Observed hand identity adds cross-entity evidence for object locations.",
            "Occluded releases may use a weak affordance-compatible destination prior; such states are marked interpolated.",
            "Dense targeted frames are added only to the target entity timeline; other entity tracks keep the generic baseline timeline.",
            "Dense evidence contributes only moving_out/moving_in motion-state evidence; it never creates absolute in/out pseudo anchors.",
            "Dense relative-motion source weight is separate from confidence, which is always clamped to [0,1].",
            "The manual regression fixture is not read by this tracker.",
        ],
    }
    path = out / "tracked_entity_states.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for entity in ("door", "dish_rack", "cutlery_basket", "knife_location", "fork_location", "plate_location"):
        seq = " -> ".join(s["state"] for s in tracks[entity]["segments"])
        print(f"{entity}: {seq}")
    print("saved", path)


if __name__ == "__main__":
    main()
