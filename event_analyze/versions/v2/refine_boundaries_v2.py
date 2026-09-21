#!/usr/bin/env python3
"""V2 deterministic phase-boundary refinement from HDF5 signals.

Input:
- original HDF5
- candidate_events.json
- event_semantics_v2.jsonl
- semantic_hierarchy_v2.json

Output:
- hierarchical_annotations_v2.json

Qwen never chooses numerical boundaries in this script.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_episode import load_episode, build_features, robust_z, smooth_1d


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_activity(d, fps=30.0):
    X, Xs, names = build_features(d)
    Z = robust_z(Xs)

    selected = [
        "base_speed", "base_yaw_rate", "torso_joint_speed",
        "left_ee_speed", "right_ee_speed",
        "left_ee_ang_speed", "right_ee_ang_speed",
        "left_force_change", "right_force_change",
    ]
    idx = [names.index(n) for n in selected]
    motion = np.mean(np.abs(Z[:, idx]), axis=1)

    rg = d["right_grip_cmd"]
    lg = d["left_grip_cmd"]
    grip_delta = (
        np.abs(np.diff(rg, prepend=rg[0]))
        + np.abs(np.diff(lg, prepend=lg[0]))
    )
    if np.percentile(grip_delta, 95) > 1e-8:
        grip_delta = np.clip(grip_delta / (np.percentile(grip_delta, 95) + 1e-8), 0, 3)
    activity = motion + 0.45 * grip_delta
    return smooth_1d(activity, max(3, int(round(0.25 * fps))))


def valley(activity, lo, hi, edge_margin=3):
    lo = max(0, int(lo))
    hi = min(len(activity) - 1, int(hi))
    if hi <= lo:
        return lo
    if hi - lo > 2 * edge_margin:
        lo2, hi2 = lo + edge_margin, hi - edge_margin
    else:
        lo2, hi2 = lo, hi
    return int(lo2 + np.argmin(activity[lo2:hi2 + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--lerobot-episode", type=int, default=None)
    ap.add_argument("--lerobot-max-raw-frame", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.analysis_dir)
    hierarchy = json.load(open(root / "semantic_hierarchy_v2.json", encoding="utf-8"))
    verified = load_jsonl(root / "event_semantics_v2.jsonl")
    atoms = hierarchy["atomic_events"]

    d = load_episode(args.hdf5)
    T = len(d["time"])
    activity = build_activity(d, fps=args.fps)

    atom_by_id = {a["atomic_id"]: a for a in atoms}
    phases = hierarchy.get("phase_groups", [])
    phase_anchors = []
    for p in phases:
        frames = [atom_by_id[i]["frame"] for i in p["atomic_ids"] if i in atom_by_id]
        if frames:
            phase_anchors.append((min(frames), max(frames)))
        else:
            phase_anchors.append((0, 0))

    # boundaries between adjacent semantic phases are activity valleys
    internal = []
    for i in range(len(phases) - 1):
        left_last = phase_anchors[i][1]
        right_first = phase_anchors[i + 1][0]
        if right_first > left_last:
            b = valley(activity, left_last, right_first)
        else:
            b = int(round((left_last + right_first) / 2))
        internal.append(b)

    # Find earliest unrelated/post-task event after the task completion anchor.
    end_atomic_id = hierarchy.get("task_end_atomic_id")
    end_anchor = None
    if end_atomic_id is not None and int(end_atomic_id) in atom_by_id:
        end_anchor = atom_by_id[int(end_atomic_id)]["frame"]
    elif phase_anchors:
        end_anchor = phase_anchors[-1][1]
    else:
        end_anchor = T - 1

    unrelated_frames = sorted(
        int(e["_raw_frame"]) for e in verified
        if e.get("task_relevance") in ("unrelated", "post_task")
        and int(e["_raw_frame"]) > end_anchor
    )
    max_forward = int(round(3.0 * args.fps))
    search_hi = min(T - 1, end_anchor + max_forward)
    if unrelated_frames:
        search_hi = min(search_hi, unrelated_frames[0])
    task_end = valley(activity, end_anchor, search_hi) if search_hi > end_anchor else end_anchor

    starts = [0] + internal
    ends = [b - 1 for b in internal] + [task_end]

    out_phases = []
    for p, s, e in zip(phases, starts, ends):
        if e < s:
            e = s
        raw_s, raw_e = int(s), int(e)
        lr_s = raw_s
        lr_e = raw_e
        if args.lerobot_max_raw_frame is not None:
            if raw_s > args.lerobot_max_raw_frame:
                lr_s = None
                lr_e = None
            else:
                lr_e = min(raw_e, args.lerobot_max_raw_frame)

        out_phases.append({
            "phase_id": p["phase_id"],
            "instruction": p["instruction"],
            "atomic_ids": p["atomic_ids"],
            "confidence": p.get("confidence"),
            "raw_start_frame": raw_s,
            "raw_end_frame": raw_e,
            "lerobot_episode_index": args.lerobot_episode,
            "lerobot_start_frame": lr_s,
            "lerobot_end_frame": lr_e,
        })

    result = {
        "annotation_version": "v2",
        "task_goal": hierarchy.get("task_goal"),
        "trajectory_instruction": hierarchy.get("trajectory_instruction"),
        "task_end_raw_frame": int(task_end),
        "lerobot_episode_index": args.lerobot_episode,
        "semantic_phases": out_phases,
        "atomic_events": atoms,
        "uncertainties": hierarchy.get("uncertainties", []),
        "notes": [
            "Semantic grouping is produced by Qwen.",
            "All numerical phase boundaries are refined deterministically from HDF5 activity signals.",
            "Frames after task_end_raw_frame should not be supervised with this task unless separately relabeled."
        ],
    }
    (root / "hierarchical_annotations_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # diagnostic plot
    t = d["time"] - d["time"][0]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, activity, label="activity")
    for p in out_phases:
        x = t[min(p["raw_start_frame"], T-1)]
        ax.axvline(x, linestyle="--", alpha=0.5)
        ax.text(x, np.percentile(activity, 95), f"P{p['phase_id']}", rotation=90, va="top")
    ax.axvline(t[min(task_end, T-1)], linestyle="-.", linewidth=2, label="task end")
    ax.set_xlabel("time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "boundaries_v2.png", dpi=160)
    plt.close(fig)

    print("saved", root / "hierarchical_annotations_v2.json")
    print("task_end_raw_frame =", task_end)


if __name__ == "__main__":
    main()
