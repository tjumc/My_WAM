#!/usr/bin/env python3
"""V3.4.2 trajectory-level entity state tracking.

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
import sys
from collections import defaultdict, deque
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from schema_runtime import load_schema, build_tracker_specs

UNKNOWN = {"uncertain", "not_visible", None}

SPECS = {}


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

def inject_generic_motion_source_priors(by_frame, generic_frames, obs):
    """Use schema-defined motion-source priors from direct observations."""
    frames = sorted(set(int(x) for x in generic_frames))
    for o in obs:
        if o.get("_v331_dense_target_entity") or o.get("_v33_dense_cutlery"):
            continue
        start = int(o.get("_window_start_frame", 0))
        prev = next((f for f in reversed(frames) if f < start), None)
        if prev is None:
            continue
        for entity, spec in SPECS.items():
            mapping = spec.get("motion_source", {})
            if not mapping:
                continue
            st = o.get("state_before") or {}
            cf = o.get("confidence_before") or {}
            motion = st.get(entity)
            conf = max(0.0, min(1.0, float(cf.get(entity, 0.0))))
            if motion not in mapping or conf < 0.70:
                continue
            by_frame[prev][entity].append({
                "kind": "motion_source_prior",
                "value": mapping[motion],
                "confidence": conf,
                "source_weight": 1.35,
                "event_id": int(o.get("_event_id", -1)),
                "side": "prior_to_window",
                "source": "schema_motion_source_prior",
            })


def inject_dense_motion_evidence(by_frame, targeted_frames, dense_rows):
    """Add directional evidence plus a weak transition-endpoint prior.

    Dense VLM never becomes a direct absolute-state observation. Instead,
    moving_out supports an eventual OUT endpoint and moving_in supports IN.
    Generic direct observations can still override this weak endpoint prior.
    """
    for x in dense_rows:
        entity = x.get("entity", "cutlery_basket")
        if entity not in SPECS:
            continue
        motion = x.get("relative_motion")
        if motion not in {"moving_out", "moving_in"}:
            continue
        center = int(x.get("center_frame", 0))
        end = int(x.get("window_end_frame", center))
        conf = max(0.0, min(1.0, float(x.get("confidence", 0.0))))
        event_id = int(x.get("event_id", 10000 + int(x.get("dense_window_id", 0))))
        endpoint = SPECS.get(entity, {}).get("motion_endpoint", {}).get(motion)
        if endpoint is None:
            continue

        targeted_frames.setdefault(entity, [])
        targeted_frames[entity].extend([center, end])

        by_frame[center][entity].append({
            "kind": "motion",
            "value": motion,
            "confidence": conf,
            "source_weight": 1.8,
            "event_id": event_id,
            "side": "center",
            "source": "dense_relative_motion",
        })
        by_frame[end][entity].append({
            "kind": "motion_endpoint_prior",
            "value": endpoint,
            "confidence": conf,
            "source_weight": 0.85,
            "event_id": event_id,
            "side": "window_end",
            "source": "dense_transition_endpoint_prior",
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
            if state == v:
                score += 4.0 * effective
                direct_support = max(direct_support, c)
            elif state in {"moving_out", "moving_in"}:
                score -= 4.0 * effective
        elif kind == "motion_endpoint_prior":
            # Weak transition completion prior, not a direct visual state.
            if state == v:
                score += 2.4 * effective
            elif state in {"in", "out"}:
                score -= 1.6 * effective
        elif kind == "motion_source_prior":
            # A directly observed motion constrains the physical source state
            # immediately before the motion.
            if state == v:
                score += 3.2 * effective
            elif state in {"closed", "open", "in", "out"}:
                score -= 1.8 * effective
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
        direct_match = [e for e in ev if e["kind"] in ("state", "must", "motion") and e["value"] == st]
        endpoint_match = [e for e in ev if e["kind"] == "motion_endpoint_prior" and e["value"] == st]
        direct = max((float(e["confidence"]) for e in direct_match), default=0.0)
        endpoint = max((float(e["confidence"]) for e in endpoint_match), default=0.0)
        src = sorted({int(e["event_id"]) for e in ev if e["value"] == st})
        if direct >= 0.55:
            support_type = "observed"
            conf = direct
        elif endpoint >= 0.55:
            support_type = "transition_constrained"
            conf = max(0.55, 0.78 * endpoint)
        else:
            support_type = "interpolated"
            conf = 0.52
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
            constrained = [x["confidence"] for x in vals if x["support"] == "transition_constrained"]
            if direct:
                conf = max(direct); support_type = "observed"
            elif constrained:
                conf = max(constrained); support_type = "transition_constrained"
            else:
                conf = 0.52; support_type = "interpolated"
            segs.append({
                "state": anchors[start]["state"],
                "start_frame": int(left),
                "end_frame": int(right),
                "confidence": round(float(conf), 3),
                "support": support_type,
                "event_ids": sorted({eid for x in vals for eid in x["event_ids"]}),
            })
            start = i
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--schema", required=True)
    args = ap.parse_args()

    global SPECS
    schema = load_schema(args.schema)
    SPECS = build_tracker_specs(schema)

    out = Path(args.output_dir)
    obs = sorted(load_jsonl(out / "entity_observations.jsonl"), key=lambda x: x["_raw_frame"])
    by_frame, generic_frames, targeted_frames = build_anchor_evidence(obs)
    inject_generic_motion_source_priors(by_frame, generic_frames, obs)
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
        "annotation_version": "v3.4.2",
        "task_schema": str(args.schema),
        "task_family": schema.get("task_family"),
        "num_windows": len(obs),
        "anchor_frames": sorted({f for xs in entity_anchor_frames.values() for f in xs}),
        "entity_anchor_frames": entity_anchor_frames,
        "tracks": tracks,
        "notes": [
            "Tracking is confidence-weighted and graph-constrained across the full trajectory.",
            "Unknown/not-visible observations do not erase a persistent state.",
            "Observed hand identity adds cross-entity evidence for object locations.",
            "Occluded releases may use a weak affordance-compatible destination prior; such states are marked interpolated.",
            "Early direct motion observations add source-state priors to prevent backward Viterbi state leakage.",
            "Dense targeted frames are added only to the target entity timeline; other entity tracks keep the generic baseline timeline.",
            "Dense evidence contributes moving_out/moving_in motion-state evidence plus a weak transition-endpoint prior; it never becomes a direct absolute-state observation.",
            "Dense relative-motion source weight is separate from confidence, which is always clamped to [0,1].",
            "The manual regression fixture is not read by this tracker.",
        ],
    }
    path = out / "tracked_entity_states.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for entity in SPECS:
        seq = " -> ".join(s["state"] for s in tracks[entity]["segments"])
        print(f"{entity}: {seq}")
    print("saved", path)


if __name__ == "__main__":
    main()
