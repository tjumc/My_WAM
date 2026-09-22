#!/usr/bin/env python3
"""V3.4.0 signal-grounded boundary refinement for deterministic skills."""
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


def enforce_duration(s, e, skill, T, fps):
    need = int(round(MIN_DURATION.get(skill["skill_type"], 0.5) * fps))
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

    # Resolve overlaps only between different skills, retaining intentional gaps.
    refined.sort(key=lambda x: x["raw_start_frame"])
    for i in range(len(refined) - 1):
        a, b = refined[i], refined[i + 1]
        if b["raw_start_frame"] <= a["raw_end_frame"]:
            cut = valley(activity, b["raw_start_frame"], a["raw_end_frame"])
            a["raw_end_frame"] = max(a["raw_start_frame"], cut - 1)
            b["raw_start_frame"] = min(b["raw_end_frame"], cut)
            a.update(lerobot_span(a["raw_start_frame"], a["raw_end_frame"], args))
            b.update(lerobot_span(b["raw_start_frame"], b["raw_end_frame"], args))

    spans = [[x["raw_start_frame"], x["raw_end_frame"]] for x in refined]
    coarse = []
    for s, e in complement(spans, T):
        z = {"raw_start_frame": s, "raw_end_frame": e, "label_policy": "coarse_task_only"}
        z.update(lerobot_span(s, e, args))
        coarse.append(z)

    task_end = next((x["raw_end_frame"] for x in reversed(refined) if x["skill_type"] == "close_dishwasher_door"), refined[-1]["raw_end_frame"] if refined else T - 1)
    trajectory_instruction = "；".join(x["instruction_zh"] for x in refined)
    result = {
        "annotation_version": "v3.4.0",
        "task_goal": candidates.get("task_goal"),
        "trajectory_instruction": trajectory_instruction,
        "task_end_raw_frame": int(task_end),
        "lerobot_episode_index": args.lerobot_episode,
        "semantic_phases": refined,
        "coarse_only_segments": coarse,
        "notes": [
            "V3.4.0 phases use interaction-validated trajectory-level entity-state-machine skills, not LLM-generated phases.",
            "Entity uncertainty produces coarse-only gaps instead of forced fine labels.",
            "Numerical boundaries are refined with HDF5 activity signals and skill-specific duration priors.",
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
