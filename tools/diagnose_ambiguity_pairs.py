#!/usr/bin/env python3
"""Read-only Astribot-25D diagnostics for saved ambiguity pairs.

Exports actual Parquet state/action rows, per-joint current differences, and
motion statistics. No image decoding, model loading, padding, robot I/O, or
automatic stop/success labels. Requires the existing My_WAM environment.
Supports the current z-score + ConcatLeftAlign configuration ONLY. The NumPy
normalizer follows SingleFieldLinearNormalizer: std+1e-8 and clamp to [-5,5].
Run scripts/diagnose_ambiguity_pairs.sh; existing output paths are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "src", REPO_ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from export_ambiguity_pair_videos import EpisodeReader, parse_source, select_pairs

GROUPS = (("base", 0, 3), ("torso", 3, 7), ("left_arm", 7, 14),
          ("left_gripper", 14, 15), ("right_arm", 15, 22),
          ("right_gripper", 22, 23), ("head", 23, 25))
FLAG_KEYS = ("done", "next.done", "is_last", "terminal", "terminated", "truncated",
             "success", "next.success", "is_success", "task_success", "task_completed",
             "stop", "should_stop")


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to export: {path.name}")
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def joint_names(base_mode: str) -> list[str]:
    names = []
    for group, start, end in GROUPS:
        names.extend(f"{group}_{i}" for i in range(end - start))
    if base_mode == "velocity":
        names[:3] = ["base_vx", "base_vy", "base_omega"]
    elif base_mode == "position":
        names[:3] = ["base_x", "base_y", "base_yaw"]
    return names


def read_settings(config: Path, stats_path: Path) -> tuple[dict, dict]:
    from omegaconf import OmegaConf
    # Resolve only the inspected fields, not unrelated model/environment paths.
    cfg = OmegaConf.load(config)
    train, proc = cfg.data.train, cfg.data.train.processor
    if int(train.get("global_sample_stride", 1)) != 1:
        raise ValueError("This diagnostic requires global_sample_stride=1")
    if str(train.get("norm_stats_source", "")) != "pretrained":
        raise ValueError("This diagnostic requires norm_stats_source=pretrained")
    shape = OmegaConf.to_container(train.shape_meta, resolve=True)
    for kind in ("state", "action"):
        fields = shape[kind]
        if len(fields) != 1 or fields[0]["key"] != "default" or int(fields[0]["raw_shape"]) != 25:
            raise ValueError(f"Expected a single default 25D {kind} field")
    if proc.get("action_state_transforms") is not None:
        raise ValueError("Non-null action_state_transforms are not supported; refusing raw/model mismatch")
    if bool(proc.get("use_stepwise_action_norm", False)):
        raise ValueError("Stepwise action normalization is not supported")
    if str(proc.get("norm_default_mode")) != "z-score" or proc.get("norm_exception_mode") is not None:
        raise ValueError("Only uniform z-score normalization is supported")
    merger = proc.action_state_merger
    if not str(merger.get("_target_", "")).endswith(".ConcatLeftAlign"):
        raise ValueError("Only ConcatLeftAlign padding is supported")
    state_dim, action_dim = int(proc.proprio_output_dim), int(proc.action_output_dim)
    if state_dim < 25 or action_dim < 25:
        raise ValueError("Model dimensions must be at least 25")
    if int(merger.state_target_dim) != state_dim or int(merger.action_target_dim) != action_dim:
        raise ValueError("Merger and processor dimensions disagree")
    horizon = int(train.num_frames) - 1
    if horizon <= 0:
        raise ValueError("Invalid action horizon")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for kind in ("state", "action"):
        field = stats.get(kind, {}).get("default", {})
        for name in ("global_mean", "global_std"):
            a = np.asarray(field.get(name), dtype=np.float32)
            if a.shape != (25,) or not np.isfinite(a).all():
                raise ValueError(f"STATS {kind}.default.{name} must be a finite 25D array")
        if np.any(np.asarray(field["global_std"]) < 0):
            raise ValueError("Negative normalization std")
    return {"horizon": horizon, "state_model_dim": state_dim, "action_model_dim": action_dim,
            "normalization": "float32 x*scale+offset; scale=1/(std+1e-8); clamp [-5,5]",
            "transforms": None, "padding": "ConcatLeftAlign"}, stats


def normalize(x: np.ndarray, stats: dict, kind: str) -> tuple[np.ndarray, np.ndarray]:
    field = stats[kind]["default"]
    mean, std = (np.asarray(field[k], dtype=np.float32) for k in ("global_mean", "global_std"))
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != 25 or mean.shape != (25,) or std.shape != (25,):
        raise ValueError("Normalization requires 25D arrays")
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("Invalid normalization values")
    scale = np.float32(1.0) / (std + np.float32(1e-8))
    y = x * scale - mean / (std + np.float32(1e-8))
    if not np.isfinite(y).all():
        raise ValueError("Normalization overflow")
    clipped = (y < -5.0) | (y > 5.0)
    return np.clip(y, -5.0, 5.0), clipped


def validate_episode(data: dict, length: int, fps: float, tolerance: float) -> dict:
    frames = np.asarray(data["frame"], dtype=np.int64)
    timestamps = np.asarray(data["time"], dtype=np.float64)
    if not math.isfinite(fps) or fps <= 0 or length < 1:
        raise ValueError("Invalid episode length/FPS")
    if frames.shape != (length,) or timestamps.shape != (length,):
        raise ValueError("Parquet row count disagrees with episode metadata")
    order = np.argsort(frames, kind="stable")
    frames, timestamps = frames[order], timestamps[order]
    if not np.array_equal(frames, np.arange(length)):
        raise ValueError("Missing or duplicated frame_index; refusing substitution/padding")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Invalid/nonincreasing episode timestamps")
    if length > 1 and np.max(np.abs(np.diff(timestamps) - 1.0 / fps)) > tolerance:
        raise ValueError("Parquet timestamps disagree with FPS; inspect synchronization")
    out = {"frame": frames, "time": timestamps, "fps": fps, "length": length, "flags": {}}
    for kind in ("state", "action"):
        array = np.asarray(data[kind], dtype=np.float32)
        if array.shape != (length, 25) or not np.isfinite(array).all():
            raise ValueError(f"{kind} must be finite [episode_length,25]")
        out[kind] = array[order]
    for key, values in data.get("flags", {}).items():
        if len(values) != length:
            raise ValueError(f"Flag {key} row count mismatch")
        out["flags"][key] = [values[int(i)] for i in order]
    return out


class NumericEpisodeReader:
    """No video decoding; filter actual episode ID then sort actual frame_index."""
    def __init__(self, root: str, tolerance: float):
        # Reuse the local-only metadata reader; [] requests no cameras.
        self.meta = EpisodeReader(root, []).meta
        self.root, self.fps = Path(root), float(self.meta.fps)
        self.tolerance, self.cache = tolerance, {}
        features = self.meta.info.get("features", {})
        for key in ("observation.state", "action"):
            if list(features.get(key, {}).get("shape", [])) != [25]:
                raise ValueError(f"{key} is not raw 25D")
        import pyarrow.dataset as pads
        self.pads = pads
        self.table_source = pads.dataset(str(self.root / "data"), format="parquet")

    def read(self, episode: int) -> dict:
        if episode not in self.cache:
            if episode not in self.meta.episodes:
                raise ValueError(f"Episode {episode} absent from metadata")
            flag_keys = [k for k in FLAG_KEYS if k in self.table_source.schema.names]
            columns = ["frame_index", "timestamp", "observation.state", "action"] + flag_keys
            table = self.table_source.to_table(columns=columns, filter=self.pads.field("episode_index") == episode)
            data = {"frame": table["frame_index"].to_numpy(), "time": table["timestamp"].to_numpy(),
                    "state": table["observation.state"].to_pylist(), "action": table["action"].to_pylist(),
                    "flags": {k: table[k].to_pylist() for k in flag_keys}}
            self.cache[episode] = validate_episode(data, int(self.meta.episodes[episode]["length"]),
                                                   self.fps, self.tolerance)
        return self.cache[episode]


def current_index(data: dict, source: dict, tolerance: float) -> int:
    idx = int(source["frame"])
    if not 0 <= idx < data["length"] or abs(float(data["time"][idx]) - source["timestamp"]) > tolerance:
        raise ValueError("Saved source frame/timestamp disagrees with Parquet; check dataset mount/version")
    return idx


def padded_rmse(a: np.ndarray, b: np.ndarray, dim: int) -> float:
    # Identical zero padding contributes zero squared error, but dilutes the mean.
    if a.shape != b.shape or a.size == 0 or a.shape[-1] != 25 or dim < 25:
        raise ValueError("Incompatible RMSE arrays/dimensions")
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64) - b), dtype=np.float64) * 25.0 / dim))


def current_comparison(a: dict, b: dict, ia: int, ib: int, stats: dict, names: list[str], model_dim: int):
    raw_a, raw_b = a["state"][ia], b["state"][ib]
    na, ca = normalize(raw_a, stats, "state")
    nb, cb = normalize(raw_b, stats, "state")
    diff = nb.astype(np.float64) - na
    squares = diff ** 2
    total = float(squares.sum())
    groups = {i: group for group, start, end in GROUPS for i in range(start, end)}
    rows = []
    for i, name in enumerate(names):
        rows.append({"dim": i, "joint": name, "group": groups[i],
                     "state_a_raw": float(raw_a[i]), "state_b_raw": float(raw_b[i]),
                     "delta_b_minus_a_raw": float(raw_b[i]) - float(raw_a[i]),
                     "abs_delta_raw": abs(float(raw_b[i]) - float(raw_a[i])),
                     "state_a_normalized": float(na[i]), "state_b_normalized": float(nb[i]),
                     "abs_delta_normalized": abs(float(diff[i])),
                     "squared_error_share": float(squares[i] / total) if total else 0.0,
                     "clipped_a": bool(ca[i]), "clipped_b": bool(cb[i])})
    group_rows = [{"group": group, "dimensions": f"{start}:{end}",
                   "normalized_rmse": float(np.sqrt(np.mean(squares[start:end]))),
                   "normalized_max_abs_difference": float(np.max(np.abs(diff[start:end]))),
                   "squared_error_share": float(squares[start:end].sum() / total) if total else 0.0}
                  for group, start, end in GROUPS]
    return rows, group_rows, {"proprio_rmse_model_padded": padded_rmse(na, nb, model_dim),
                               "proprio_rmse_active25": padded_rmse(na, nb, 25),
                               "max_abs_normalized_difference": float(np.max(np.abs(diff))),
                               "clipped_state_dims_a": np.flatnonzero(ca).tolist(),
                               "clipped_state_dims_b": np.flatnonzero(cb).tolist()}


def motion_rows(data: dict, indices: np.ndarray, side: str, window: str, kind: str,
                names: list[str], base_mode: str) -> list[dict]:
    values, timestamps = data[kind][indices].astype(np.float64), data["time"][indices]
    if not len(indices):
        return []
    delta = np.diff(values, axis=0)
    duration = float(timestamps[-1] - timestamps[0])
    rows = []
    for i, name in enumerate(names):
        velocity = i < 3 and base_mode == "velocity"
        excursion = float(np.max(np.abs(values[:, i] - values[0, i])))
        peak = float(np.max(np.abs(values[:, i])))
        rows.append({"side": side, "window": window, "kind": kind, "dim": i, "joint": name,
                     "samples": len(indices), "duration_seconds": duration,
                     "first_frame": int(data["frame"][indices[0]]), "last_frame": int(data["frame"][indices[-1]]),
                     "first_raw": float(values[0, i]), "last_raw": float(values[-1, i]),
                     "min_raw": float(np.min(values[:, i])), "max_raw": float(np.max(values[:, i])),
                     "range_raw": float(np.ptp(values[:, i])), "max_abs_from_window_start_raw": excursion,
                     "max_abs_value_raw": peak, "total_variation_raw": float(np.abs(delta[:, i]).sum()),
                     "max_abs_step_raw": float(np.max(np.abs(delta[:, i]))) if len(delta) else None,
                     "stationarity_quantity": ("unknown_base_semantics" if i < 3 and base_mode == "unknown"
                                               else "abs_velocity" if velocity else "excursion_from_window_start"),
                     "stationarity_quantity_raw": (None if i < 3 and base_mode == "unknown"
                                                   else peak if velocity else excursion)})
    return rows


def describe_side(data: dict, idx: int, source: dict, horizon: int, seconds: float, tail: float,
                  names: list[str], base_mode: str) -> tuple[dict, list[dict], list[dict]]:
    n, ts = data["length"], data["time"]
    end = min(n, int(np.searchsorted(ts, ts[idx] + seconds + 1e-8, side="right")))
    # Tail never precedes CURRENT and never crosses the episode boundary.
    tail_start = max(idx, int(np.searchsorted(ts, ts[-1] - tail - 1e-8)))
    windows = {"action_chunk": (np.arange(idx, min(n, idx + horizon + 1)),
                                 np.arange(idx, min(n, idx + horizon))),
               "first_seconds": (np.arange(idx, end), np.arange(idx, end)),
               "remaining_recording": (np.arange(idx, n), np.arange(idx, n)),
               "recording_tail": (np.arange(tail_start, n), np.arange(tail_start, n))}
    summaries, motions = {}, []
    for window, (states, actions) in windows.items():
        for kind, indices in (("state", states), ("action", actions)):
            motions.extend(motion_rows(data, indices, source["side"], window, kind, names, base_mode))
        summaries[window] = {"state_samples": len(states), "action_rows": len(actions),
                             "state_duration_seconds": float(ts[states[-1]] - ts[states[0]])}
    summary = {**source, "episode_length": n, "episode_last_frame": int(data["frame"][-1]),
               "episode_last_timestamp": float(ts[-1]), "remaining_recorded_rows_including_current": n - idx,
               "remaining_observed_seconds": float(ts[-1] - ts[idx]),
               "action_horizon": horizon, "full_action_chunk_present": idx + horizon <= n,
               "full_chunk_successor_states_present": idx + horizon < n,
               "requested_first_seconds": seconds, "first_seconds_complete": bool(ts[-1] >= ts[idx] + seconds - 1e-8),
               "requested_tail_seconds": tail, "tail_has_requested_duration": bool(ts[-1] - ts[tail_start] >= tail - 1e-8),
               "windows": summaries, "recorded_flag_fields": list(data["flags"]),
               "last_recorded_flags": {k: values[-1] for k, values in data["flags"].items()},
               "task_completed": None, "stop_demonstrated": None,
               "note": "null means unassessed. Recording end/done is not a stop command. No padding or fabricated labels."}
    sequences = []
    for j in range(idx, n):
        row = {"frame_index": int(data["frame"][j]), "timestamp": float(ts[j]), "relative_seconds": float(ts[j] - ts[idx]),
               "in_gt_action_chunk": bool(j < idx + horizon), "in_first_seconds": bool(j < end),
               "is_last_recorded_row": j == n - 1}
        row.update({f"state__{name}": float(data["state"][j, i]) for i, name in enumerate(names)})
        row.update({f"gt_action__{name}": float(data["action"][j, i]) for i, name in enumerate(names)})
        row.update({f"recorded_flag__{k}": json.dumps(v[j], ensure_ascii=False) for k, v in data["flags"].items()})
        sequences.append(row)
    return summary, motions, sequences


def make_report(pair: dict, summary: dict, current: list[dict], groups: list[dict]) -> str:
    c = summary["comparison"]
    lines = [f"# 第 {pair['rank']} 对：当前关节与后续示范诊断", "",
             "这是原始记录的离线诊断，不是 WAM 推理，也不自动判定任务完成或停止。", "",
             "## 当前状态", f"- 含模型补齐维的 proprio RMSE：{c['proprio_rmse_model_padded']:.6f}",
             f"- 只统计真实 25 维的 proprio RMSE：{c['proprio_rmse_active25']:.6f}",
             "- 原始值单位沿用数据。不同维度不要直接混合比较；差异为直接相减，未做角度回绕。",
             "- 归一化采用训练配置的 z-score，并裁剪到 [-5,5]。裁剪维已在 CSV 标出。", "",
             "归一化差异最大的 5 维："]
    for r in sorted(current, key=lambda r: r["abs_delta_normalized"], reverse=True)[:5]:
        lines.append(f"- {r['joint']}：A={r['state_a_raw']:.6g}，B={r['state_b_raw']:.6g}，"
                     f"原始差值={r['delta_b_minus_a_raw']:+.6g}，归一化绝对差={r['abs_delta_normalized']:.4f}")
    lines.extend(["", "## 剩余记录是否充分"])
    for side in summary["sides"]:
        lines.append(f"- {side['side']}：第 {side['frame']} 帧，轨迹共 {side['episode_length']} 帧；"
                     f"剩余可观测 {side['remaining_observed_seconds']:.3f} 秒；"
                     f"完整 GT chunk={side['full_action_chunk_present']}；"
                     f"完整后继状态={side['full_chunk_successor_states_present']}；"
                     f"后 {side['requested_first_seconds']:g} 秒完整={side['first_seconds_complete']}。")
        lines.append(f"  数据中找到的原始标记字段：{', '.join(side['recorded_flag_fields']) or '无（已检查常见字段）'}。")
    lines.extend(["", "## 怎么判断", "1. 看 current_state.csv 和 current_groups.csv，确认差异是否集中于关键关节。",
                  "2. 看 side_b_sequence.csv：CURRENT 后是继续动作、收手回位，还是保持。这里不含视频的人为暂停。",
                  "3. motion_summary.csv 的 state 是实际状态，action 是记录的 GT 指令；不能互相替代。",
                  "4. 本次 base_mode=" + summary["base_mode"] +
                  "。velocity 模式下底盘要看 max_abs_value_raw；恒定非零速度仍在移动。"
                  "位置型关节看 max_abs_from_window_start_raw；保持非零角度不是运动。",
                  "5. first_seconds 看当前起点附近，recording_tail 看记录末段；末段静止不能反推 CURRENT 已经停止。",
                  "6. 不足的后续记录不会填零或复制。最后一行的 done/terminal 等字段不自动解释为停止。",
                  "", "## 复现检查"])
    for key, val in summary["mining_metric_checks"].items():
        lines.append(f"- {key}：旧值={val['saved']}，重算={val['recomputed']}，一致={val['matches']}。")
    if summary["warnings"]:
        lines.extend(["", "## 注意"] + [f"- {w}" for w in summary["warnings"]])
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pairs-json", required=True)
    p.add_argument("--ranks", nargs="+", type=int, default=[2])
    p.add_argument("--config", default=None, help="Defaults to pairs.json summary.config")
    p.add_argument("--dataset-stats", default=None, help="Defaults to pairs.json summary.dataset_stats")
    p.add_argument("--dataset-dir", default=None, help="Optional single-dataset mount override")
    p.add_argument("--output-dir", default=None, help="New diagnostics only; default beside pairs.json")
    p.add_argument("--after-seconds", type=float, default=3.0)
    p.add_argument("--tail-seconds", type=float, default=2.0)
    p.add_argument("--tolerance-s", type=float, default=0.002)
    p.add_argument("--base-mode", choices=["velocity", "position", "unknown"], default="velocity",
                   help="Explicit assumption for first 3 state/action dimensions; no unit conversions")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for name in ("after_seconds", "tail_seconds", "tolerance_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    path = Path(args.pairs_json).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = select_pairs(payload, args.ranks)
    config_value = args.config or payload.get("summary", {}).get("config")
    stats_value = args.dataset_stats or payload.get("summary", {}).get("dataset_stats")
    if not config_value or not stats_value:
        raise ValueError("Provide config and dataset-stats paths")
    config, stats_path = (Path(x).expanduser().resolve() for x in (config_value, stats_value))
    settings, stats = read_settings(config, stats_path)
    original_roots = {parse_source(p, s)["root"] for p in pairs for s in ("a", "b")}
    if args.dataset_dir and len(original_roots) != 1:
        raise ValueError("One dataset-dir override cannot remap multiple roots")
    sources = [[{**parse_source(p, s, args.dataset_dir), "side": s.upper()} for s in ("a", "b")] for p in pairs]
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else path.parent / "pair_diagnostics" / stamp
    if output == path.parent or any(output == Path(s['root']) or Path(s['root']) in output.parents
                                    for pair_sources in sources for s in pair_sources):
        raise ValueError("Use a NEW diagnostic directory, not the mining directory or raw data")
    # Refuse even an empty existing directory: no accidental overwrite/mixed runs.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"status": "started", "arguments": vars(args), "settings": settings,
               "pairs_json": str(path), "pairs_json_sha256": file_hash(path), "config": str(config),
               "config_sha256": file_hash(config), "dataset_stats": str(stats_path), "stats_sha256": file_hash(stats_path),
               "raw_units": "Retained from dataset; no universal hold threshold or automatic stop/success label"})
    names, readers, completed = joint_names(args.base_mode), {}, []
    print(f"[diagnostic] output={output}\n[diagnostic] ranks={args.ranks}; raw25; base_mode={args.base_mode}", flush=True)
    for pair, pair_sources in zip(pairs, sources):
        if len({(s["root"], s["episode"]) for s in pair_sources}) != 1:
            raise ValueError("Expected two sources from the same dataset/episode")
        root = pair_sources[0]["root"]
        if root not in readers:
            readers[root] = NumericEpisodeReader(root, args.tolerance_s)
        data = readers[root].read(pair_sources[0]["episode"])
        if args.tolerance_s >= 0.5 / data["fps"]:
            raise ValueError("tolerance-s must be less than half a dataset frame")
        indices = [current_index(data, s, args.tolerance_s) for s in pair_sources]
        folder = output / f"pair_{int(pair['rank']):04d}"
        folder.mkdir()
        rows, groups, comparison = current_comparison(data, data, *indices, stats, names, settings["state_model_dim"])
        write_csv(folder / "current_state.csv", rows)
        write_csv(folder / "current_groups.csv", groups)
        horizon = settings["horizon"]
        if all(idx + horizon <= data["length"] for idx in indices):
            aa, ab = [normalize(data["action"][idx:idx + horizon], stats, "action")[0] for idx in indices]
            comparison.update({"action_rmse_model_padded": padded_rmse(aa, ab, settings["action_model_dim"]),
                               "action_rmse_active25": padded_rmse(aa, ab, 25),
                               "action_first_step_rmse": padded_rmse(aa[:1], ab[:1], settings["action_model_dim"]),
                               "action_endpoint_rmse": padded_rmse(aa[-1:], ab[-1:], settings["action_model_dim"])})
        else:
            comparison["action_rmse_model_padded"] = None
        checks, warnings = {}, []
        for old, new in (("proprio_rmse", "proprio_rmse_model_padded"), ("action_rmse", "action_rmse_model_padded"),
                         ("action_first_step_rmse", "action_first_step_rmse"), ("action_endpoint_rmse", "action_endpoint_rmse")):
            if old in pair and comparison.get(new) is not None:
                matches = bool(np.isclose(float(pair[old]), comparison[new], rtol=1e-4, atol=1e-5))
                checks[old] = {"saved": float(pair[old]), "recomputed": comparison[new], "matches": matches}
                if not matches:
                    warnings.append(f"{old} 与筛选结果不一致：先检查 stats/config/数据版本或样本索引，不要直接据此推论。")
        sides, all_motion = [], []
        for source, idx in zip(pair_sources, indices):
            side_info, motions, sequences = describe_side(data, idx, source, horizon, args.after_seconds,
                                                          args.tail_seconds, names, args.base_mode)
            write_csv(folder / f"side_{source['side'].lower()}_sequence.csv", sequences)
            sides.append(side_info)
            all_motion.extend(motions)
        write_csv(folder / "motion_summary.csv", all_motion)
        summary = {"rank": int(pair["rank"]), "base_mode": args.base_mode, "settings": settings,
                   "comparison": comparison, "mining_metric_checks": checks, "sides": sides, "warnings": warnings,
                   "interpretation": "GT discrepancy is not action failure; similarity is not memory necessity; stop remains unassessed."}
        write_json(folder / "summary.json", summary)
        report = make_report(pair, summary, rows, groups)
        (folder / "report.md").write_text(report, encoding="utf-8")
        print(report, flush=True)
        completed.append({"rank": int(pair["rank"]), "directory": folder.name, "warnings": warnings})
    write_json(output / "COMPLETE.json", {"status": "complete", "pairs": completed})
    print(f"[diagnostic] complete: {output}\nNo existing outputs or robot commands were changed.", flush=True)
    return output


if __name__ == "__main__":
    main()
