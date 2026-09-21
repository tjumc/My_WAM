#!/usr/bin/env python3
"""V2.1 signal-grounded refinement for action intervals and semantic phases."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_episode import load_episode, build_features, robust_z, smooth_1d


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_activity(d, fps=30.0):
    _, Xs, names = build_features(d)
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
    grip_delta = np.abs(np.diff(rg, prepend=rg[0])) + np.abs(np.diff(lg, prepend=lg[0]))
    p95 = np.percentile(grip_delta, 95)
    if p95 > 1e-8:
        grip_delta = np.clip(grip_delta / (p95 + 1e-8), 0, 3)
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


def refine_actions(actions, activity, T, fps):
    if not actions:
        return []
    out = []
    for i, a in enumerate(actions):
        first = int(a["first_evidence_frame"])
        last = int(a["last_evidence_frame"])

        if i == 0:
            left_lo = max(0, first - int(round(1.5 * fps)))
        else:
            prev_last = int(actions[i - 1]["last_evidence_frame"])
            left_lo = max(0, int(round((prev_last + first) / 2)))
        start = valley(activity, left_lo, first)

        if i == len(actions) - 1:
            right_hi = min(T - 1, last + int(round(1.5 * fps)))
        else:
            next_first = int(actions[i + 1]["first_evidence_frame"])
            right_hi = min(T - 1, int(round((last + next_first) / 2)))
        end = valley(activity, last, right_hi)
        if end < start:
            end = max(start, last)

        aa = dict(a)
        aa["refined_start_frame"] = int(start)
        aa["refined_end_frame"] = int(end)
        out.append(aa)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--lerobot-episode", type=int, default=None)
    ap.add_argument("--lerobot-max-raw-frame", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.analysis_dir)
    hierarchy = json.load(open(root / "semantic_hierarchy_v2_1.json", encoding="utf-8"))
    windows = load_jsonl(root / "window_semantics_v2_1.jsonl")
    actions = hierarchy.get("action_intervals", [])

    d = load_episode(args.hdf5)
    T = len(d["time"])
    activity = build_activity(d, fps=args.fps)
    actions = refine_actions(actions, activity, T, args.fps)
    action_by_id = {a["action_id"]: a for a in actions}

    phases = hierarchy.get("phase_groups", [])
    valid_phases = []
    for p in phases:
        ids = [i for i in p.get("action_ids", []) if i in action_by_id]
        if not ids:
            continue
        valid_phases.append({**p, "action_ids": ids})

    # Refine task end from the last relevant task action toward the first explicit post-task/unrelated window.
    end_action_id = hierarchy.get("task_end_action_id")
    if end_action_id is not None and int(end_action_id) in action_by_id:
        end_anchor = int(action_by_id[int(end_action_id)]["refined_end_frame"])
    else:
        relevant_actions = [a for a in actions if a.get("task_relevance") == "relevant"]
        end_anchor = int(relevant_actions[-1]["refined_end_frame"]) if relevant_actions else (T - 1)

    unrelated_frames = sorted(
        int(w["_raw_frame"]) for w in windows
        if w.get("task_relevance") in ("post_task", "unrelated") and int(w["_raw_frame"]) > end_anchor
    )
    end_hi = min(T - 1, end_anchor + int(round(3.0 * args.fps)))
    if unrelated_frames:
        end_hi = min(end_hi, unrelated_frames[0])
    task_end = valley(activity, end_anchor, end_hi) if end_hi > end_anchor else end_anchor

    # Build phase boundaries from refined action intervals. Numerical frames never come from Qwen.
    phase_ranges = []
    for p in valid_phases:
        aa = [action_by_id[i] for i in p["action_ids"]]
        phase_ranges.append([
            min(int(x["refined_start_frame"]) for x in aa),
            max(int(x["refined_end_frame"]) for x in aa),
        ])

    internal = []
    for i in range(len(phase_ranges) - 1):
        left_last = phase_ranges[i][1]
        right_first = phase_ranges[i + 1][0]
        if right_first > left_last:
            b = valley(activity, left_last, right_first)
        else:
            b = int(round((left_last + right_first) / 2))
        internal.append(b)

    out_phases = []
    for i, p in enumerate(valid_phases):
        if i == 0:
            s = phase_ranges[i][0]
        else:
            s = internal[i - 1]
        if i < len(valid_phases) - 1:
            e = internal[i] - 1
        else:
            e = min(task_end, max(phase_ranges[i][1], s))
        if e < s:
            e = s

        lr_s, lr_e = int(s), int(e)
        if args.lerobot_max_raw_frame is not None:
            if lr_s > args.lerobot_max_raw_frame:
                lr_s = lr_e = None
            else:
                lr_e = min(lr_e, args.lerobot_max_raw_frame)

        out_phases.append({
            "phase_id": p["phase_id"],
            "instruction": p.get("instruction", ""),
            "action_ids": p["action_ids"],
            "confidence": p.get("confidence"),
            "raw_start_frame": int(s),
            "raw_end_frame": int(e),
            "lerobot_episode_index": args.lerobot_episode,
            "lerobot_start_frame": lr_s,
            "lerobot_end_frame": lr_e,
        })

    prefix = None
    if out_phases and out_phases[0]["raw_start_frame"] > 0:
        prefix = [0, out_phases[0]["raw_start_frame"] - 1]
    suffix = None
    if task_end < T - 1:
        suffix = [int(task_end + 1), int(T - 1)]

    result = {
        "annotation_version": "v2.1",
        "task_goal": hierarchy.get("task_goal"),
        "trajectory_instruction": hierarchy.get("trajectory_instruction"),
        "task_end_raw_frame": int(task_end),
        "lerobot_episode_index": args.lerobot_episode,
        "semantic_phases": out_phases,
        "action_intervals": actions,
        "unassigned_prefix_raw_frames": prefix,
        "post_task_or_unassigned_suffix_raw_frames": suffix,
        "uncertainties": hierarchy.get("uncertainties", []),
        "notes": [
            "Qwen verifies semantic content of local windows, not exact point boundaries.",
            "Multiple semantic windows are clustered into action intervals deterministically.",
            "Qwen only groups action IDs into semantic phases; it never chooses frame numbers.",
            "All numerical action/phase boundaries are refined from HDF5 activity signals.",
            "Unassigned prefix/suffix should keep the coarse task label or be separately relabeled rather than receiving an incorrect fine-grained phase label.",
        ],
    }
    (root / "hierarchical_annotations_v2_1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    t = d["time"] - d["time"][0]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, activity, label="activity")
    for a in actions:
        s = min(int(a["refined_start_frame"]), T - 1)
        e = min(int(a["refined_end_frame"]), T - 1)
        ax.axvspan(t[s], t[e], alpha=0.08)
        ax.text(t[s], np.percentile(activity, 96), f"A{a['action_id']}", rotation=90, va="top", fontsize=8)
    for p in out_phases:
        s = min(p["raw_start_frame"], T - 1)
        ax.axvline(t[s], linestyle="--", alpha=0.6)
        ax.text(t[s], np.percentile(activity, 85), f"P{p['phase_id']}", rotation=90, va="top")
    ax.axvline(t[min(task_end, T - 1)], linestyle="-.", linewidth=2, label="task end")
    ax.set_xlabel("time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "boundaries_v2_1.png", dpi=160)
    plt.close(fig)

    print("saved", root / "hierarchical_annotations_v2_1.json")
    print("task_end_raw_frame =", task_end)


if __name__ == "__main__":
    main()
