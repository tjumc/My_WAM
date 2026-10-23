"""Create boundary metrics and joint/target plots from a robot client trace."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

GROUPS = {"torso": slice(3, 7), "left_arm": slice(7, 14), "right_arm": slice(15, 22)}


def read_events(directory):
    events, warnings = [], []
    for number, line in enumerate((directory / "events.jsonl").read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.append(f"Unreadable event line {number}; recording may be truncated")
    summary = directory / "summary.json"
    if not summary.exists():
        warnings.append("Missing summary.json: recording was not cleanly closed or is still active")
    else:
        status = json.loads(summary.read_text())
        if status.get("dropped") or status.get("writer_error") or not status.get("clean_close"):
            warnings.append(f"Incomplete recording: {status}")
    return events, warnings


def difference(a, b, index):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (25,) or b.shape != (25,):
        return None
    delta = a[index] - b[index]
    return float(np.max(np.abs(delta))) if np.isfinite(delta).all() else None


def boundary_rows(events):
    returns = {e["request_id"]: e for e in events if e["event"] == "sdk_return"}
    rpc = {e["request_id"]: e for e in events if e["event"] == "rpc_return"}
    initial = {e["request_id"] for e in events if e["event"] == "initial_positioning"}
    rows = []
    for e in events:
        if e["event"] != "dispatch":
            continue
        rid = e["request_id"]
        row = {"request_id": rid, "initial_positioning": rid in initial,
               "dispatch_monotonic_ns": e["monotonic_ns"],
               "submitted_steps": e["submitted_steps"],
               "scheduled_duration_s": e["scheduled_duration"],
               "observation_age_ms": e["observation_age_ms"],
               "rpc_ms": rpc.get(rid, {}).get("rpc_ms"),
               "sdk_call_ms": returns.get(rid, {}).get("sdk_call_ms")}
        for group, index in GROUPS.items():
            row[group + "_first_vs_measured_rad"] = difference(
                e["first_target"], e.get("measured_joints"), index)
            # This is the previous SUBMITTED endpoint, not guaranteed executed.
            row[group + "_first_vs_previous_submitted_rad"] = difference(
                e["first_target"], e.get("previous_submitted_last"), index)
        rows.append(row)
    return rows


def plot_joints(directory, output, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = [e for e in events if e["event"] == "joint_sample"
               and e.get("measured_joints") is not None]
    if not samples:
        return ["No joint telemetry: plots cannot show measured arm motion"]
    origin = min(e["monotonic_ns"] for e in events)
    t = np.asarray([(e["monotonic_ns"]-origin)/1e9 for e in samples])
    q = np.asarray([e["measured_joints"] for e in samples], float)
    if q.ndim != 2 or q.shape[1] != 25:
        return ["Unexpected joint telemetry shape; plots skipped"]
    predictions = {e["request_id"]: e for e in events
                   if e["event"] == "prediction" and "artifact" in e}
    dispatches = [e for e in events if e["event"] == "dispatch"]
    warnings = []
    for group in ("left_arm", "right_arm"):
        indices = list(range(25))[GROUPS[group]]
        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        for ax, j in zip(axes, indices):
            ax.plot(t, q[:, j], label="measured joints", linewidth=1)
            for k, e in enumerate(dispatches):
                start = (e["monotonic_ns"]-origin)/1e9
                ax.axvline(start, color="gray", alpha=.25)
                prediction = predictions.get(e["request_id"])
                if prediction is None:
                    continue
                path = directory / prediction["artifact"]
                if not path.is_file():
                    warnings.append(f"Missing {path.name}")
                    continue
                with np.load(path, allow_pickle=False) as data:
                    a = data["selected_actions"]
                target_t = start + np.arange(1, len(a)+1)*e["waypoint_step_dt"]
                ax.plot(target_t, a[:, j], "--", color="tab:orange",
                        label="submitted targets (nominal schedule)" if k == 0 else None)
            ax.set_ylabel(f"q{j} (rad)")
        axes[0].legend()
        axes[0].set_title(group + "; gray lines = chunk submission")
        axes[-1].set_xlabel("Client monotonic time since first event (s)")
        fig.tight_layout()
        fig.savefig(output / (group + ".png"), dpi=160)
        plt.close(fig)
    return sorted(set(warnings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    events, warnings = read_events(args.trace)
    rows = boundary_rows(events)
    if not rows:
        raise ValueError("No dispatch events; expected a client trace")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "boundaries.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plots:
        warnings += plot_joints(args.trace, args.output, events)
    result = {"chunks": len(rows), "warnings": warnings,
              "errors": [e for e in events if e["event"] in
                         ("execution_error", "infer_error")],
              "notes": ["Previous submitted endpoint may not have been reached.",
                        "SDK call duration is measured, not assumed to equal execution time.",
                        "No task success or language understanding is inferred from these metrics."]}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
