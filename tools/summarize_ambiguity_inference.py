#!/usr/bin/env python3
"""Aggregate a FIXED ambiguity-inference cohort without running a model.

Reads only the ranks and shared seeds recorded in this run's manifest.json.
Missing/bad pairs are reported, never silently excluded from the planned count.
GT preference is a distance diagnostic, not a correctness or failure label.
Only generated batch_* files in the inference run are written/replaced.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any

TIE_EPS = 1e-6  # Same preference tolerance as infer_ambiguity_pairs.score_predictions.
RATIO_EPS = 1e-12  # Numerical guard only, NOT a calibrated ambiguity threshold.
SPLITS = {
    "train_seen": "训练集已见轨迹（用户声明；未自动核验训练清单）",
    "unknown": "训练重叠未知；不得当作验证集结果",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nonnegative(value: Any, label: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"Nonfinite/negative {label}")
    return x


def same(a: float, b: float, label: str) -> None:
    if not math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError(f"Inconsistent {label}: {a} != {b}")


def unique_positive(values: list, label: str, allow_zero: bool = False) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"Empty/missing {label}")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < (0 if allow_zero else 1) for x in values):
        raise ValueError(f"Invalid {label}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label}")
    return values


def pair_row(folder: Path, rank: int, seeds: list[int]) -> dict:
    summary = read_json(folder / "summary.json")
    if summary.get("rank") != rank or summary.get("num_seeds") != len(seeds) or summary.get("active_dimensions") != 25:
        raise ValueError("Wrong rank, seed count, or active dimensions")
    sources = [r["source"] for r in summary["sources"]]
    if len(sources) != 2 or (sources[0]["root"], sources[0]["episode"]) != (sources[1]["root"], sources[1]["episode"]):
        raise ValueError("Expected two sources from the same episode")
    rows = read_csv(folder / "seed_metrics.csv")
    identities = [(int(r["seed"]), r["side"]) for r in rows]
    expected = {(seed, side) for seed in seeds for side in ("A", "B")}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Missing/duplicate/unexpected per-side seeds")
    paired = read_csv(folder / "paired_seed_metrics.csv")
    if len(paired) != len(seeds) or {int(r["seed"]) for r in paired} != set(seeds):
        raise ValueError("Missing/duplicate/unexpected paired seeds")
    gt = nonnegative(summary["gt_pair_rmse_active25"], "GT separation")
    separations = []
    for r in paired:
        same(nonnegative(r["gt_pair_rmse_active25"], "GT separation"), gt, "GT separation")
        separations.append(nonnegative(r["prediction_pair_rmse_active25"], "prediction separation"))
    separation_mean = statistics.mean(separations)
    same(separation_mean, summary["same_seed_prediction_separation"]["mean"], "prediction separation mean")
    reproduced = summary.get("reference_reproduction", {})
    result = {
        "rank": rank, "dataset_root": sources[0]["root"], "episode": int(sources[0]["episode"]),
        "frame_a": int(sources[0]["frame"]), "frame_b": int(sources[1]["frame"]),
        "num_seeds_per_side": len(seeds),
        "visual_rmse": reproduced.get("visual_rmse"),
        "proprio_rmse_model_padded": reproduced.get("proprio_rmse"),
        "gt_pair_rmse_active25": gt,
        "prediction_pair_rmse_active25_mean": separation_mean,
        "prediction_to_gt_separation_ratio": separation_mean / gt if gt > RATIO_EPS else None,
    }
    for side in ("A", "B"):
        own, other, cross, ties = [], [], 0, 0
        for r in rows:
            if r["side"] != side:
                continue
            a = nonnegative(r["own_gt_rmse_active25"], "own error")
            b = nonnegative(r["other_gt_rmse_active25"], "other error")
            margin = b - a
            preference = "tie" if abs(margin) <= TIE_EPS else ("other_gt" if margin < 0 else "own_gt")
            if r["gt_preference_not_failure_label"] != preference:
                raise ValueError("Saved GT preference disagrees with distances")
            same(margin, float(r["other_minus_own_rmse"]), "GT preference margin")
            cross += int(preference == "other_gt")
            ties += int(preference == "tie")
            own.append(a)
            other.append(b)
        side_summary = summary["sides"][side]
        same(statistics.mean(own), side_summary["own_gt_rmse_active25"]["mean"], "own mean")
        same(statistics.mean(other), side_summary["other_gt_rmse_active25"]["mean"], "other mean")
        if cross != side_summary["other_gt_closer_seed_count"] or ties != side_summary["tie_seed_count"]:
            raise ValueError("CSV and summary GT preference counts disagree")
        prefix = side.lower()
        result.update({
            f"{prefix}_own_rmse_mean": statistics.mean(own),
            f"{prefix}_own_rmse_seed_std": statistics.pstdev(own),
            f"{prefix}_other_rmse_mean": statistics.mean(other),
            f"{prefix}_other_gt_closer_seeds": cross,
            f"{prefix}_tie_seeds": ties,
            f"{prefix}_other_gt_strict_majority": cross > len(seeds) / 2,
            f"{prefix}_own_error_over_gt_separation": statistics.mean(own) / gt if gt > RATIO_EPS else None,
        })
    result["has_any_other_gt_preference"] = bool(result["a_other_gt_closer_seeds"] or result["b_other_gt_closer_seeds"])
    return result


def build_batch(run: Path, data_split: str = "unknown", expected_pairs: int | None = None) -> tuple[list, list, dict]:
    if data_split not in SPLITS:
        raise ValueError("Unknown data split label")
    manifest = read_json(run / "manifest.json")
    ranks = unique_positive(manifest["arguments"].get("ranks"), "manifest ranks")
    seeds = unique_positive(manifest.get("seeds_shared_between_sides"), "manifest seeds", allow_zero=True)
    if expected_pairs is not None and len(ranks) != expected_pairs:
        raise ValueError(f"Planned {len(ranks)} pairs, expected {expected_pairs}; refusing a changed cohort")
    rows, issues = [], []
    for rank in ranks:
        try:
            rows.append(pair_row(run / f"pair_{rank:04d}", rank, seeds))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            issues.append({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    unexpected = sorted(p.name for p in run.glob("pair_[0-9]*") if p.is_dir() and p.name not in {f"pair_{r:04d}" for r in ranks})
    marker_valid = False
    if (run / "COMPLETE.json").is_file():
        marker = read_json(run / "COMPLETE.json")
        marker_valid = (marker.get("status") == "complete" and marker.get("ranks") == ranks
                        and marker.get("num_seeds") == len(seeds)
                        and marker.get("model_calls") == len(ranks) * 2 * len(seeds))
    failed = (run / "FAILED.json").is_file()
    complete = marker_valid and not failed and not issues and not unexpected
    episodes: dict[tuple, list] = {}
    unique_states = set()
    for row in rows:
        episodes.setdefault((row["dataset_root"], row["episode"]), []).append(row)
        unique_states.update((row["dataset_root"], row["episode"], row[f"frame_{side}"]) for side in ("a", "b"))
    episode_rows = []
    for (root, episode), items in sorted(episodes.items()):
        errors = [r[f"{side}_own_rmse_mean"] for r in items for side in ("a", "b")]
        episode_rows.append({"dataset_root": root, "episode": episode, "evaluated_pairs": len(items),
                             "pair_side_mean_own_rmse": statistics.mean(errors),
                             "pairs_with_any_other_gt_preference": sum(r["has_any_other_gt_preference"] for r in items)})
    errors = [r[f"{side}_own_rmse_mean"] for r in rows for side in ("a", "b")]
    counts = {"planned_pairs": len(ranks), "evaluated_pairs": len(rows), "missing_or_invalid_pairs": len(issues),
              "evaluated_pair_sides": 2 * len(rows), "unique_current_states": len(unique_states),
              "covered_episodes": len(episode_rows), "seeds_per_side": len(seeds),
              "planned_model_calls": len(ranks) * 2 * len(seeds),
              "pairs_with_any_other_gt_preference": sum(r["has_any_other_gt_preference"] for r in rows),
              "sides_with_other_gt_strict_majority": sum(r[f"{s}_other_gt_strict_majority"] for r in rows for s in ("a", "b")),
              "numerically_zero_gt_separation_pairs": sum(r["gt_pair_rmse_active25"] <= RATIO_EPS for r in rows)}
    info = {
        "status": "complete" if complete else "partial_or_invalid", "data_split": data_split,
        "data_split_label": SPLITS[data_split], "counts": counts,
        "fixed_ranks": ranks, "shared_seeds": seeds,
        "pairs_json_sha256": manifest.get("pairs_json_sha256"), "checkpoint": manifest.get("checkpoint"),
        "task_prompt": manifest.get("task_prompt"), "issues": issues, "unexpected_pair_directories": unexpected,
        "complete_marker_valid": marker_valid, "failed_marker_present": failed,
        "own_rmse_pair_side_mean": statistics.mean(errors) if errors else None,
        "own_rmse_pair_side_median": statistics.median(errors) if errors else None,
        "own_rmse_pair_side_max": max(errors) if errors else None,
        "own_rmse_episode_balanced_mean": statistics.mean(r["pair_side_mean_own_rmse"] for r in episode_rows) if episode_rows else None,
        "interpretation": "Fixed selected cohort, not all training data. Cross-GT preferences and seeds are not robot failures or independent trials. Low prediction separation alone is not collapse; inspect GT separation and own error. Current observations/proprio can distinguish phases; no causal memory claim.",
    }
    return rows, episode_rows, info


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.5f}"


def report(rows: list[dict], info: dict) -> str:
    c = info["counts"]
    lines = ["# 固定候选集：WAM 批量离线推理", "", f"**数据范围：{info['data_split_label']}。**",
             f"**运行状态：{info['status']}；有效结果 {c['evaluated_pairs']}/{c['planned_pairs']} 对。**",
             "这是固定筛选候选集的诊断，不是整个训练集统计，也不是验证集或真机成功率。", "",
             f"覆盖 {c['covered_episodes']} 条轨迹、{c['unique_current_states']} 个不同当前状态；每侧 {c['seeds_per_side']} 个 seed。",
             f"全部计划为 {c['planned_model_calls']} 次单窗口推理；两侧和所有 pair 使用相同 seed 集合。", "",
             "## 主要结果", "",
             f"- 至少一个 seed 更接近另一侧 GT：**{c['pairs_with_any_other_gt_preference']}/{c['evaluated_pairs']} 对**（分母仅为有效结果）。",
             f"- 严格超过一半 seed 更接近另一侧 GT：**{c['sides_with_other_gt_strict_majority']}/{c['evaluated_pair_sides']} 个 pair-side**。",
             f"- 各 pair-side 对自身 GT 的平均 RMSE：{fmt(info['own_rmse_pair_side_mean'])}；中位数：{fmt(info['own_rmse_pair_side_median'])}；最大：{fmt(info['own_rmse_pair_side_max'])}。",
             f"- 先在各 episode 内平均、再等权平均 episode：{fmt(info['own_rmse_episode_balanced_mean'])}。",
             "", "以上计数只表示距离偏好，不能写成失败概率。同一轨迹、重复端点和多个 seed 不是独立任务试验。",
             "", "## 优先人工复核（最多 10 对；全部结果在 batch_pairs.csv）", "",
             "先按偏向另一侧 GT 的 seed 总数降序，再按两侧中较大的自身 GT 误差排序。不是重新筛选或删除其余样本。",
             "GT 距离很小时，相对比值不稳定；预测距离缩小也可能只是两侧 GT 本来接近。这里不自动设置动作坍缩阈值。", "",
             "| rank | episode | A自身误差 | B自身误差 | A偏向另一侧 | B偏向另一侧 | GT距离 | 预测距离/GT距离 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = sorted(rows, key=lambda r: (-(r["a_other_gt_closer_seeds"] + r["b_other_gt_closer_seeds"]),
                                          -max(r["a_own_rmse_mean"], r["b_own_rmse_mean"]), r["rank"]))
    for r in ordered[:10]:
        n = r["num_seeds_per_side"]
        lines.append(f"| {r['rank']} | {r['episode']} | {fmt(r['a_own_rmse_mean'])} | {fmt(r['b_own_rmse_mean'])} | "
                     f"{r['a_other_gt_closer_seeds']}/{n} | {r['b_other_gt_closer_seeds']}/{n} | {fmt(r['gt_pair_rmse_active25'])} | {fmt(r['prediction_to_gt_separation_ratio'])} |")
    if not c["pairs_with_any_other_gt_preference"] and rows:
        lines += ["", "本批有效结果未出现跨 GT 距离偏好；上表只是误差较大的候选，不代表已发现混淆。"]
    lines += ["", "## 解释边界", "",
              "必须结合当前三相机、proprio、GT 本身是否可互换来判断。预测远离两个 GT 可能是一般拟合误差，不一定是分支混淆。",
              "预测贴合见过的轨迹不能证明泛化；出现错误也不能证明原因是缺少记忆。后续模型比较应保持同一候选集和 checkpoint 评估协议。",
              "本实验仅预测一个 action chunk，不验证长期停止、重复开门或闭环成功。", "",
              "batch_episodes.csv 提供逐轨迹统计；batch_pairs.csv 保留所有有效 pair 的原编号与连续指标。",
              "manifest.json 保存候选文件哈希、checkpoint 路径、task 和 seed；各 pair 的原始数组/曲线用于进一步复核。"]
    if not info["status"] == "complete":
        lines += ["", "## 未完成或不一致", "", "**不能把缺失项记作成功或记作没有混淆。请检查 run.log / FAILED.json。**"]
        lines.extend(f"- rank {r['rank']}: {r['error']}" for r in info["issues"])
        if info["unexpected_pair_directories"]:
            lines.append("- 发现计划外 pair 目录，未纳入统计：" + ", ".join(info["unexpected_pair_directories"]))
        if not info["complete_marker_valid"]:
            lines.append("- 没有与 manifest 一致的 COMPLETE.json。")
        if info["failed_marker_present"]:
            lines.append("- 存在 FAILED.json。")
    return "\n".join(lines) + "\n"


def atomic_text(path: Path, text: str) -> None:
    # Replace only this tool's own generated aggregate files, not pair outputs.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(text)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return "\ufeff" + stream.getvalue()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-split", choices=tuple(SPLITS), default="unknown")
    p.add_argument("--expected-pairs", type=int, default=None, help="Guard against an accidentally changed cohort size")
    args = p.parse_args(argv)
    if args.expected_pairs is not None and args.expected_pairs <= 0:
        p.error("--expected-pairs must be positive")
    run = Path(args.run_dir).expanduser().resolve()
    rows, episodes, info = build_batch(run, args.data_split, args.expected_pairs)
    atomic_text(run / "batch_pairs.csv", csv_text(rows))
    atomic_text(run / "batch_episodes.csv", csv_text(episodes))
    atomic_text(run / "batch_summary.json", json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_text(run / "batch_report.md", report(rows, info))
    c = info["counts"]
    print(f"[batch-summary] {info['status']}; evaluated={c['evaluated_pairs']}/{c['planned_pairs']}; "
          f"any-cross-GT={c['pairs_with_any_other_gt_preference']}; read {run / 'batch_report.md'}", flush=True)
    return 0 if info["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
