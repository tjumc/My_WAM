#!/usr/bin/env python3
"""Create a small text-only review package from local real-robot recordings.

Reads index/metadata only, never copies arrays/images or queries hardware.
Missing SDK return/next observation is unknown, never a task-failure label.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


def read_events(path: Path):
    events, errors = [], []
    if not path.is_file():
        return events, ["events.jsonl missing"]
    last = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                event = json.loads(line)
                event_id = event["event_id"]
                if not isinstance(event_id, int) or event_id <= last:
                    raise ValueError("event_id must be strictly increasing")
                if not isinstance(event.get("metadata"), dict):
                    raise ValueError("metadata missing/not an object")
                if not isinstance(event.get("event"), str):
                    raise ValueError("event type missing")
                if event.get("request_id") is not None and (not isinstance(event["request_id"], int) or event["request_id"] < 1):
                    raise ValueError("invalid request_id")
                last = event_id
                events.append(event)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"line {line_number}: {exc}")
    return events, errors


def _get_metadata(events, name):
    found = [e for e in events if e["event"] == name]
    return found[0]["metadata"] if found else {}


def build_rows(events: list[dict]):
    by_request = defaultdict(list)
    feedback = defaultdict(list)
    for event in events:
        request_id = event.get("request_id")
        if request_id is not None:
            by_request[request_id].append(event)
        if event["event"] == "observation":
            previous = event["metadata"].get("feedback_after_request")
            if previous is not None:
                feedback[previous].append(event)
    rows = []
    for request, items in sorted(by_request.items()):
        names = {e["event"] for e in items}
        obs = _get_metadata(items, "observation")
        prediction = _get_metadata(items, "prediction")
        sdk = _get_metadata(items, "sdk_waypoints_return")
        planned = _get_metadata(items, "sdk_waypoints_attempt")
        following = feedback.get(request, [])
        # Only a naturally recorded observation is usable feedback; do not invent
        # one from predicted action endpoints or a method returning successfully.
        next_event = following[0] if following else None
        next_obs = next_event["metadata"] if next_event else {}
        next_stamp = next_obs.get("state_read_finished", {}).get("monotonic_time_s")
        sdk_stamp = sdk.get("finished", {}).get("monotonic_time_s")
        lag = next_stamp - sdk_stamp if next_stamp is not None and sdk_stamp is not None else None
        ages = [x.get("age_at_selection_s") for x in obs.get("camera_times", [])]
        ages = [x for x in ages if isinstance(x, (int, float))]
        start = sdk.get("started", {}).get("monotonic_time_s")
        received = [x.get("received_monotonic_s") for x in next_obs.get("camera_times", [])]
        newer = (all(x is not None and x >= start for x in received)
                 if received and start is not None else None)
        row = {
            "request_id": request,
            "observation_present": "observation" in names,
            "prediction_present": "prediction" in names,
            "rpc_none": "rpc_none" in names,
            "rpc_wall_seconds": prediction.get("rpc_wall_seconds"),
            "image_wait_seconds": obs.get("image_wait_seconds"),
            "max_camera_receive_age_at_selection_s": max(ages) if ages else None,
            "initial_position_attempt": "sdk_initial_position_attempt" in names,
            "initial_position_return": "sdk_initial_position_return" in names,
            "waypoints_attempt": "sdk_waypoints_attempt" in names,
            "waypoints_return": "sdk_waypoints_return" in names,
            "planned_steps": planned.get("steps"),
            "nominal_action_horizon_s": planned.get("nominal_horizon_s"),
            "sdk_call_wall_seconds": sdk.get("sdk_call_wall_seconds"),
            "sdk_return_type": sdk.get("sdk_return_type"),
            "sdk_return_scalar": sdk.get("sdk_return_scalar"),
            "next_observation_request_id": next_event.get("request_id") if next_event else None,
            "feedback_lag_after_sdk_return_s": lag,
            "feedback_cameras_received_after_command_start": newer,
            "images_requested": obs.get("images_requested"),
            "error_or_interrupt": "loop_error" in names or "loop_interrupted" in names,
            "stage": "", "expected_behavior": "", "observed_outcome": "",
            "first_anomaly_category": "", "notes": "",
        }
        rows.append(row)
    return rows


def summarize(run_dir: Path, output_dir: Path):
    run_dir, output_dir = run_dir.expanduser().resolve(), output_dir.expanduser().resolve()
    if output_dir == run_dir or output_dir in run_dir.parents or run_dir in output_dir.parents:
        raise ValueError("Keep the shareable review package separate from raw recording")
    if output_dir.exists():
        raise FileExistsError("Output already exists; choose a new review directory")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported recording schema")
    closed_path = run_dir / "CLOSED.json"
    closed = json.loads(closed_path.read_text(encoding="utf-8")) if closed_path.is_file() else None
    events, parse_errors = read_events(run_dir / "events.jsonl")
    missing_payloads = []
    for event in events:
        filename = event.get("arrays_file")
        if filename:
            p = (run_dir / filename).resolve()
            if run_dir not in p.parents or not p.is_file():
                missing_payloads.append(event["event_id"])
    rows = build_rows(events)
    gaps = []
    previous = 0
    for event in events:
        if event["event_id"] > previous + 1:
            gaps.append([previous + 1, event["event_id"] - 1])
        previous = event["event_id"]
    runtime = _get_metadata(events, "runtime")
    ending = _get_metadata(events, "loop_end")
    counts = closed.get("counts", {}) if closed else {}
    incomplete = (closed is None or closed.get("status") != "closed" or bool(parse_errors)
                  or bool(gaps) or bool(missing_payloads)
                  or any(counts.get(k, 0) for k in ("observation_pack_error", "command_pack_error")))
    summary = {
        "schema_version": 1, "run_id": manifest["run_id"],
        "recording_status": "incomplete_or_gapped" if incomplete else "closed",
        "recording_counters": counts, "writer_error": closed.get("writer_error") if closed else None,
        "submit_max_ms": closed.get("submit_max_ms") if closed else None,
        "submit_total_ms": closed.get("submit_total_ms") if closed else None,
        "event_count": len(events), "missing_event_id_ranges": gaps,
        "index_parse_errors": parse_errors, "missing_array_event_ids": missing_payloads,
        "request_count": len(rows),
        "waypoints_attempts": sum(r["waypoints_attempt"] for r in rows),
        "waypoints_returns": sum(r["waypoints_return"] for r in rows),
        "commands_without_next_observation": [r["request_id"] for r in rows
            if r["waypoints_attempt"] and r["next_observation_request_id"] is None],
        "model_info": runtime.get("model_info"),
        "effective_execute_steps": runtime.get("effective_execute_steps"),
        "client_arguments": runtime.get("arguments", manifest.get("metadata", {}).get("arguments")),
        "loop_exit": ending, "task_success": None,
        "raw_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "note": "Client request snapshots, no GT and no automatic success/failure/first-anomaly labels.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "requests.csv").open("x", encoding="utf-8-sig", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("request_id\n")
    with (output_dir / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    text = f"""# 真机闭环记录：{manifest['run_id']}

- 记录状态：**{summary['recording_status']}**；索引事件 {len(events)} 条，观测/请求编号 {len(rows)} 个。
- SDK waypoint 尝试 / 返回：{summary['waypoints_attempts']} / {summary['waypoints_returns']}。
- 缺少下一轮观测的指令编号：{summary['commands_without_next_observation']}。
- 写入错误：{summary['writer_error']}；索引解析错误：{len(parse_errors)}。
- 单次入队/复制最大耗时：{summary['submit_max_ms']} ms（不是全系统性能保证）。
- 退出原因：{ending.get('reason', '未记录；可能在初始化或中途退出')}。

## 人工检查

打开 requests.csv，从第一轮向后检查，填 stage、expected_behavior、observed_outcome。
仅在首次实际异常处填写 first_anomaly_category：local_execution / outcome_verification /
phase_selection / runtime_timing / unknown；无证据不强行分类。
此表只列发生过的事件，丢记录会造成空白；缺少返回不自动表示 SDK 未执行。

## 解读边界

这是运行期真实观测和指令记录，不是 GT。不要按时间直接套用示范轨迹的正确动作。
SDK 返回不等于动作完成，下一轮观测也不保证对应计划终点。最后一轮没有反馈时不能补零。
相机时间是客户端回调接收时间，不是传感器采集时间；若反馈图像比下发还旧，不应用来判断效果。
只有逐次推理时的快照，不包含完整 30 Hz 控制过程。必要时另用实验室已有录像核对瞬时抓取/滑落。
正常退出、到达 max_runtime、SDK 返回都不代表任务成功。Ctrl+C 不是本模块新增的硬件急停。

## 存储

本目录仅 report.md、summary.json、requests.csv，可在检查其中的路径/任务信息后提交 GitHub。
完整图片和 NPZ 留原实验存储；不要上传整个原始 rollout 目录。
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    result = summarize(Path(args.run_dir), Path(args.output_dir))
    print(f"Summary: {args.output_dir}; {result['recording_status']}; raw recording unchanged")


if __name__ == "__main__":
    main()
