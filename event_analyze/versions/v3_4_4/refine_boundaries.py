#!/usr/bin/env python3
"""V3.4.4 signal-grounded boundary refinement for deterministic skills."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_episode import load_episode, build_features, robust_z, smooth_1d

MIN_DURATION = {
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


def build_activity(d, fps):
    _, xs, names = build_features(d)
    z = robust_z(xs)
    selected = ["base_speed", "base_yaw_rate", "torso_joint_speed", "left_ee_speed", "right_ee_speed", "left_ee_ang_speed", "right_ee_ang_speed", "left_force_change", "right_force_change"]
    idx = [names.index(n) for n in selected]
    motion = np.mean(np.abs(z[:, idx]), axis=1)
    rg, lg = d["right_grip_cmd"], d["left_grip_cmd"]
    gd = np.abs(np.diff(rg, prepend=rg[0])) + np.abs(np.diff(lg, prepend=lg[0]))
    p95 = np.percentile(gd, 95)
    if p95 > 1e-8:
        gd = np.clip(gd / (p95 + 1e-8), 0, 3)
    return smooth_1d(motion + 0.45 * gd, max(3, int(round(0.25 * fps))))


def valley(x, lo, hi):
    lo, hi = max(0, int(lo)), min(len(x) - 1, int(hi))
    if hi <= lo:
        return lo
    margin = 3
    if hi - lo > 2 * margin:
        lo, hi = lo + margin, hi - margin
    return int(lo + np.argmin(x[lo:hi + 1]))


def minimum_duration_frames(skill, fps):
    return max(1, int(round(MIN_DURATION.get(skill["skill_type"], 0.5) * fps)))


def enforce_duration(s, e, skill, T, fps):
    need = minimum_duration_frames(skill, fps)
    if e - s + 1 >= need:
        return s, e
    rs = max(0, int(skill["provisional_start_frame"]))
    re = min(T - 1, int(skill["provisional_end_frame"]))
    while e - s + 1 < need and (s > rs or e < re):
        if s > rs: s -= 1
        if e - s + 1 < need and e < re: e += 1
    return int(s), int(e)


def lerobot_span(s, e, args):
    if args.lerobot_max_raw_frame is None:
        return {"lerobot_episode_index": args.lerobot_episode, "lerobot_start_frame": s, "lerobot_end_frame": e}
    if s > args.lerobot_max_raw_frame:
        return {"lerobot_episode_index": args.lerobot_episode, "lerobot_start_frame": None, "lerobot_end_frame": None}
    return {"lerobot_episode_index": args.lerobot_episode, "lerobot_start_frame": s, "lerobot_end_frame": min(e, args.lerobot_max_raw_frame)}


def complement(spans, T):
    spans = sorted(spans)
    out, cur = [], 0
    for s, e in spans:
        if s > cur: out.append([cur, s - 1])
        cur = max(cur, e + 1)
    if cur < T: out.append([cur, T - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--lerobot-episode", type=int, default=None)
    ap.add_argument("--lerobot-max-raw-frame", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    candidates = json.load(open(out / "skill_candidates_validated.json", encoding="utf-8"))
    skills = candidates.get("skill_sequence", [])
    d = load_episode(args.hdf5)
    T = len(d["time"])
    activity = build_activity(d, args.fps)

    refined = []
    for i, sk in enumerate(skills):
        ps, pe = int(sk["provisional_start_frame"]), int(sk["provisional_end_frame"])
        left = max(0, ps - int(round(0.75 * args.fps)))
        right = min(T - 1, pe + int(round(0.75 * args.fps)))
        s = valley(activity, left, min(pe, ps + int(round(0.5 * args.fps))))
        e = valley(activity, max(ps, pe - int(round(0.5 * args.fps))), right)
        s, e = min(s, ps), max(e, pe)
        s, e = enforce_duration(s, e, sk, T, args.fps)
        rec = dict(sk)
        rec["raw_start_frame"], rec["raw_end_frame"] = int(s), int(e)
        rec.update(lerobot_span(int(s), int(e), args))
        refined.append(rec)

    # Resolve overlap from right to left. Earlier versions resolved from
    # left to right, so the left neighbor could move a phase start forward and
    # the right neighbor could then move its end backward, collapsing a
    # semantically valid phase to only a few frames.
    #
    # V3.4.4 treats the upstream provisional interval as a semantic ownership
    # prior and the HDF5 activity valley as a numerical refinement cue. Every
    # pairwise cut is constrained so both phases retain their minimum duration
    # whenever that is feasible inside the current raw envelopes.
    refined.sort(key=lambda x: x["raw_start_frame"])

    def resolve_pair(a, b):
        if b["raw_start_frame"] > a["raw_end_frame"]:
            return

        old_a = [int(a["raw_start_frame"]), int(a["raw_end_frame"])]
        old_b = [int(b["raw_start_frame"]), int(b["raw_end_frame"])]

        need_a = minimum_duration_frames(a, args.fps)
        need_b = minimum_duration_frames(b, args.fps)

        # If cut is the final frame owned by a, b starts at cut+1.
        feasible_lo = max(
            old_a[0] + need_a - 1,
            old_b[0] - 1,
        )
        feasible_hi = min(
            old_a[1],
            old_b[1] - need_b,
        )

        a_pe = int(a["provisional_end_frame"])
        b_ps = int(b["provisional_start_frame"])

        if a_pe < b_ps:
            pref_lo, pref_hi = a_pe, b_ps - 1
            mode = "semantic_gap_valley"
        elif a_pe == b_ps:
            pref_lo = pref_hi = a_pe - 1
            mode = "touching_semantic_boundary"
        else:
            # Upstream semantic intervals overlap. The activity valley is used
            # only inside that overlap and is still subject to duration floors.
            pref_lo, pref_hi = b_ps, a_pe
            mode = "overlapping_semantic_intervals"

        if feasible_lo <= feasible_hi:
            search_lo = max(feasible_lo, pref_lo)
            search_hi = min(feasible_hi, pref_hi)
            if search_lo <= search_hi:
                cut = valley(activity, search_lo, search_hi)
            else:
                # Semantic preference is infeasible under the duration floors.
                # Choose the closest feasible cut rather than collapsing either
                # phase below its minimum duration.
                pref = int(round((pref_lo + pref_hi) / 2.0))
                cut = min(feasible_hi, max(feasible_lo, pref))
                mode += "_duration_clamped"
            duration_conflict = False
        else:
            # Extremely crowded candidates can make the two duration floors
            # mutually infeasible. Preserve the semantic ordering and expose
            # the conflict explicitly instead of silently producing a tiny
            # phase.
            raw_lo = max(old_a[0], old_b[0] - 1)
            raw_hi = min(old_a[1], old_b[1] - 1)
            if raw_lo <= raw_hi:
                pref = int(round((pref_lo + pref_hi) / 2.0))
                cut = min(raw_hi, max(raw_lo, pref))
            else:
                cut = min(old_a[1], max(old_a[0], old_b[0] - 1))
            duration_conflict = True
            mode += "_infeasible_duration_conflict"

        a["raw_end_frame"] = max(old_a[0], int(cut))
        b["raw_start_frame"] = min(old_b[1], int(cut) + 1)

        info = {
            "mode": mode,
            "cut_frame": int(cut),
            "duration_conflict": duration_conflict,
            "left_min_duration_frames": int(need_a),
            "right_min_duration_frames": int(need_b),
            "left_original_raw_span": old_a,
            "right_original_raw_span": old_b,
            "left_provisional_end": a_pe,
            "right_provisional_start": b_ps,
        }
        a.setdefault("boundary_arbitration", []).append({
            **info, "role": "left"
        })
        b.setdefault("boundary_arbitration", []).append({
            **info, "role": "right"
        })

        a.update(lerobot_span(a["raw_start_frame"], a["raw_end_frame"], args))
        b.update(lerobot_span(b["raw_start_frame"], b["raw_end_frame"], args))

    for i in range(len(refined) - 2, -1, -1):
        resolve_pair(refined[i], refined[i + 1])

    spans = [[x["raw_start_frame"], x["raw_end_frame"]] for x in refined]
    coarse = []
    for s, e in complement(spans, T):
        z = {"raw_start_frame": s, "raw_end_frame": e, "label_policy": "coarse_task_only"}
        z.update(lerobot_span(s, e, args))
        coarse.append(z)

    task_end = next((x["raw_end_frame"] for x in reversed(refined) if x["skill_type"] == "close_dishwasher_door"), refined[-1]["raw_end_frame"] if refined else T - 1)
    trajectory_instruction = "；".join(x["instruction_zh"] for x in refined)
    result = {
        "annotation_version": "v3.4.4",
        "task_goal": candidates.get("task_goal"),
        "trajectory_instruction": trajectory_instruction,
        "task_end_raw_frame": int(task_end),
        "lerobot_episode_index": args.lerobot_episode,
        "semantic_phases": refined,
        "coarse_only_segments": coarse,
        "notes": [
            "V3.4.4 phases use interaction-validated trajectory-level entity-state-machine skills, not LLM-generated phases.",
            "Entity uncertainty produces coarse-only gaps instead of forced fine labels.",
            "Numerical boundaries are refined with HDF5 activity signals and skill-specific duration priors.",
            "Overlap arbitration runs right-to-left and preserves minimum phase duration whenever feasible.",
            "Upstream provisional intervals are treated as semantic ownership priors rather than being overwritten by an unconstrained activity valley.",
        ],
    }
    (out / "hierarchical_annotations.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    t = d["time"] - d["time"][0]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, activity, label="activity")
    for sk in refined:
        s, e = sk["raw_start_frame"], sk["raw_end_frame"]
        ax.axvspan(t[s], t[min(e, T - 1)], alpha=0.08)
        ax.text(t[s], np.percentile(activity, 95), f"S{sk['skill_id']}", rotation=90, va="top", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "boundaries.png", dpi=160)
    plt.close(fig)
    print("saved", out / "hierarchical_annotations.json")


if __name__ == "__main__":
    main()
