#!/usr/bin/env python3
"""Four-condition input-swap sensitivity probe; no training or robot I/O.

AA=(image A, proprio A), BB=(image B, proprio B), BA=(image B, proprio A),
AB=(image A, proprio B). All use identical task context and matched seeds.
Reuse the verified current-only preprocessing/infer_action path. Rerun both
natural baselines in the same process; never mix old prediction artifacts.
Crossed inputs may be off-distribution and have NO ground-truth action label.
Metrics describe finite input sensitivity, not importance or failure rates.
Only small text reports belong in Git; _raw/ arrays are locally ignored.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_ambiguity_pairs as base

CONDITIONS = ("AA", "BB", "BA", "AB")  # first letter=image, second=proprio
EDGES = {
    "image_at_proprio_A": (0, 2), "image_at_proprio_B": (1, 3),
    "proprio_at_image_A": (0, 3), "proprio_at_image_B": (1, 2),
}
METRICS = tuple(EDGES) + ("image_effect", "proprio_effect",
                        "baseline_output_separation", "interaction_rms")
NOTICE = (
    "训练集已见候选（用户确认），不是验证集、闭环或真机试验。"
    "交换图像/状态可能产生物理不一致输入；混合输入没有正确GT，参考距离不是误差或失败率。"
    "proprio影响大不等于忽略视觉：绝对关节动作本来依赖机器人姿态。"
    "两种输入变化的幅度没有匹配，不能据此给模态重要性排名，也不能证明需要记忆。"
)


def mixed_observations(observations: list[base.Observation]) -> list[base.Observation]:
    """Natural [AA,BB] followed by [BA,AB]; no reference/GT payload allowed."""
    import torch
    if len(observations) != 2:
        raise ValueError("Need exactly two current observations")
    a, b = observations
    for name in ("image", "proprio", "context", "context_mask"):
        x, y = getattr(a, name), getattr(b, name)
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"Nonfinite/mismatched {name}")
    if not torch.equal(a.context, b.context) or not torch.equal(a.context_mask, b.context_mask):
        raise ValueError("Task context must be identical; do not swap task/phase instructions")
    # The reused runner clones each tensor before calling the model.
    return [a, b, replace(a, image=b.image), replace(b, image=a.image)]


def run_four(model: Any, observations: list, seeds: list[int], settings: dict, args):
    combinations = mixed_observations(observations)
    predictions, timings = [], []
    for offset in (0, 2):
        print(f"[swap] conditions={CONDITIONS[offset:offset+2]}; runner side=A/B maps in this order", flush=True)
        pred, timing = base.run_inference(
            model, combinations[offset:offset + 2], seeds, settings["horizon"],
            settings["action_model_dim"], args.num_inference_steps, args.sigma_shift, args.tiled,
        )
        predictions.append(pred)
        timings.append(timing)
    return np.concatenate(predictions, axis=1), np.concatenate(timings, axis=1)


def measure(predictions: np.ndarray, gt: np.ndarray, seeds: list[int], rank: int):
    """Matched-seed 2x2 finite differences, always excluding padded dims."""
    p, g = np.asarray(predictions, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if (p.ndim != 4 or p.shape[:2] != (len(seeds), 4) or p.shape[2] < 1
            or p.shape[-1] < 25 or not seeds or len(seeds) != len(set(seeds))):
        raise ValueError("Expected finite [distinct seeds,4,H,D>=25] predictions")
    if g.shape != (2, p.shape[2], 25) or not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("Invalid GT/prediction arrays")
    p = p[..., :25]
    rows, summaries, distances = [], [], []
    for group, lo, hi in (("all25", 0, 25),) + base.GROUPS:
        z = p[..., lo:hi]
        group_rows = []
        for n, seed in enumerate(seeds):
            row = {"rank": rank, "group": group, "seed": seed}
            row.update({key: base.rmse(z[n, i], z[n, j]) for key, (i, j) in EDGES.items()})
            row["image_effect"] = (row["image_at_proprio_A"] + row["image_at_proprio_B"]) / 2
            row["proprio_effect"] = (row["proprio_at_image_A"] + row["proprio_at_image_B"]) / 2
            row["baseline_output_separation"] = base.rmse(z[n, 0], z[n, 1])
            interaction = z[n, 0] + z[n, 1] - z[n, 2] - z[n, 3]
            row["interaction_rms"] = float(np.sqrt(np.mean(interaction ** 2)))
            rows.append(row)
            group_rows.append(row)
        summary = {"rank": rank, "group": group, "num_seeds": len(seeds)}
        for key in METRICS:
            values = [row[key] for row in group_rows]
            summary[key + "_mean"] = float(np.mean(values))
            summary[key + "_seed_std"] = float(np.std(values))
        # Sampling spread, not an independent-task confidence interval.
        for c, name in enumerate(CONDITIONS):
            center = z[:, c].mean(axis=0)
            summary[f"{name}_seed_spread_rms"] = float(np.sqrt(np.mean((z[:, c] - center) ** 2)))
        summary["AA_own_gt_rmse_mean"] = float(np.mean([base.rmse(x, g[0, :, lo:hi]) for x in z[:, 0]]))
        summary["BB_own_gt_rmse_mean"] = float(np.mean([base.rmse(x, g[1, :, lo:hi]) for x in z[:, 1]]))
        summaries.append(summary)
    for n, seed in enumerate(seeds):
        for c, condition in enumerate(CONDITIONS):
            distances.append({
                "rank": rank, "seed": seed, "condition": condition,
                "image_source": condition[0], "proprio_source": condition[1],
                "recorded_gt_for_this_input": condition in ("AA", "BB"),
                "distance_to_gt_A_reference": base.rmse(p[n, c], g[0]),
                "distance_to_gt_B_reference": base.rmse(p[n, c], g[1]),
                "distance_to_AA_prediction_same_seed": base.rmse(p[n, c], p[n, 0]),
                "distance_to_BB_prediction_same_seed": base.rmse(p[n, c], p[n, 1]),
                "outside_training_clip_elements25": int((np.abs(p[n, c]) > 5).sum()),
            })
    return rows, summaries, distances


def new_output(output: Path, pairs_path: Path, roots: set[str]):
    """Protect inputs; ignore all bulky or incidental files by default."""
    base.create_output(output, pairs_path, roots)
    (output / ".gitignore").write_text(
        "# Keep only compact text results in Git; NEVER force-add _raw.\n"
        "*\n!.gitignore\n!report.md\n!pair_summary.csv\n!group_summary.csv\n"
        "!manifest.json\n!STARTED.json\n!COMPLETE.json\n!FAILED.json\n", encoding="utf-8")
    raw = output / "_raw"
    raw.mkdir()
    return raw


def save_pair(raw_dir: Path, item: dict, predictions: np.ndarray, timings: np.ndarray,
              seeds: list[int], stats: dict):
    rank, refs, obs = int(item["pair"]["rank"]), item["references"], item["observations"]
    gt = np.stack([r["normalized_gt"] for r in refs])
    rows, summaries, distances = measure(predictions, gt, seeds, rank)
    # Raw ground truth is preserved; inverse normalization of predictions is unclipped.
    target = raw_dir / f"pair_{rank:04d}.npz"
    with target.open("xb") as file:
        np.savez_compressed(
            file, conditions=np.asarray(CONDITIONS), seeds=np.asarray(seeds),
            prediction_normalized=predictions, prediction_raw25=base.denormalize_predictions(predictions, stats),
            gt_normalized25=gt, gt_raw25=np.stack([r["raw_gt"] for r in refs]),
            image_original=np.stack([o.image.cpu().float().numpy() for o in obs]),
            proprio_original=np.stack([o.proprio.cpu().float().numpy() for o in obs]),
            relative_command_seconds=np.stack([r["relative_times"] for r in refs]),
            model_inference_seconds=timings,
        )
    base.write_csv(raw_dir / f"pair_{rank:04d}_seed_sensitivity.csv", rows)
    base.write_csv(raw_dir / f"pair_{rank:04d}_reference_distances.csv", distances)
    base.write_json(raw_dir / f"pair_{rank:04d}.json", {
        "rank": rank, "conditions": CONDITIONS, "verified_input_metrics": item["verified"],
        "sources": [r["source"] for r in refs],
        "clipped_gt_elements": [r["clipped_gt_elements"] for r in refs],
        "clipped_current_dims": [r["clipped_current_dims"] for r in refs],
        "array_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "note": NOTICE,
    })
    for summary in summaries:
        summary.update(episode=int(refs[0]["source"]["episode"]),
                       frame_a=int(refs[0]["source"]["frame"]), frame_b=int(refs[1]["source"]["frame"]),
                       visual_rmse=item["verified"]["visual_rmse"],
                       proprio_rmse_model_padded=item["verified"]["proprio_rmse"])
    return summaries


def report(output: Path, summaries: list[dict], planned: list[int], status: str):
    all25 = [s for s in summaries if s["group"] == "all25"]
    if summaries:
        base.write_csv(output / "pair_summary.csv", all25)
        base.write_csv(output / "group_summary.csv", summaries)
    finished = {int(s["rank"]) for s in all25}
    missing = [r for r in planned if r not in finished]
    lines = ["# WAM 当前图像 / proprio 输入交换", "",
             f"运行状态：{status}；完成 {len(finished)}/{len(planned)} 对；未完成：{missing}。", "", NOTICE, "",
             "AA/BB 是原始组合，BA/AB 是混合组合；首字母为图像来源，次字母为 proprio 来源。",
             "原始和交换组合均在本轮运行，相同 seed 比较，未复用旧推理数组。", "",
             "## 结果（归一化真实25维，排除补齐维）", "",
             "| pair | AA自身GT误差 | BB自身GT误差 | 换图像的预测变化 | 换proprio的预测变化 | 原始预测间距离 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for s in all25:
        lines.append(f"| {s['rank']} | {s['AA_own_gt_rmse_mean']:.5f} | {s['BB_own_gt_rmse_mean']:.5f} | "
                     f"{s['image_effect_mean']:.5f} | {s['proprio_effect_mean']:.5f} | "
                     f"{s['baseline_output_separation_mean']:.5f} |")
    lines += ["", "## 如何读", "",
              "换图像：平均 RMSE(BA,AA) 与 RMSE(AB,BB)，各比较保持 proprio 不变。",
              "换proprio：平均 RMSE(AB,AA) 与 RMSE(BA,BB)，各比较保持图像不变。",
              "group_summary.csv 分别列出上述四个效应，避免两侧效应被平均掩盖；另列底盘、双臂、夹爪等分组。",
              "interaction_rms 为 RMS(AA+BB-BA-AB)，描述2x2非加性，不是任务级因果归因。",
              "*_seed_std 是本组seed波动；*_seed_spread_rms 是输出围绕seed均值的散布，不是独立任务置信区间。",
              "只有AA/BB有对应的真实示范；混合条件的GT距离只保存在_raw供定位，不能当成正确/错误标签。",
              "图像交换的是三相机当前帧整体，没有输入历史、未来GT、阶段或完成提示。",
              "小幅变化不等于视觉永远无用；较大变化也不证明使用了任务相关语义。没有自动设置模态主导阈值。", "",
              "## 存储", "",
              "上传 report.md、pair_summary.csv、group_summary.csv、manifest.json 和运行状态即可。",
              "_raw/保存逐seed数值和可复查数组，默认被Git忽略；不生成批量图片/视频。不要git add -f。",
              "本轮未修改原始数据、旧结果或任何已有输出路径；运行失败的缺项不记为零敏感性。"]
    with (output / "report.md").open("x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ("pairs-json", "checkpoint", "text-cache-dir", "task"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 9, 16, 43])
    for name in ("config", "dataset-stats", "dataset-dir", "output-dir", "model-base-path"):
        p.add_argument(f"--{name}")
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--vae-device-mode", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--sigma-shift", type=float)
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--video-backend", choices=["pyav", "torchcodec", "video_reader"], default="pyav")
    p.add_argument("--tolerance-s", type=float, default=.002)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.num_seeds < 1 or args.num_inference_steps < 1
            or not 0 <= args.seed < 2**63 - args.num_seeds
            or not args.task.strip() or not math.isfinite(args.tolerance_s) or args.tolerance_s <= 0):
        raise ValueError("Invalid seeds, steps, task or tolerance")
    if args.sigma_shift is not None and (not math.isfinite(args.sigma_shift) or args.sigma_shift <= 0):
        raise ValueError("sigma-shift must be finite/positive")
    pairs_path = Path(args.pairs_json).expanduser().resolve()
    original = pairs_path.read_bytes()
    payload = json.loads(original)
    pairs = base.select_pairs(payload, args.ranks)
    planned = [int(p["rank"]) for p in pairs]
    old_roots = {base.parse_source(p, s)["root"] for p in pairs for s in ("a", "b")}
    if len(old_roots) != 1:
        raise ValueError("One original dataset root required; no ambiguous mount remapping")
    roots = {base.parse_source(p, s, args.dataset_dir)["root"] for p in pairs for s in ("a", "b")}
    root = next(iter(roots))
    config = Path(args.config or payload["summary"]["config"]).expanduser().resolve()
    stats_path = Path(args.dataset_stats or payload["summary"]["dataset_stats"]).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for path in (config, stats_path, checkpoint, Path(root) / "meta/info.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.model_base_path:
        assets = Path(args.model_base_path).expanduser().resolve()
        if not assets.is_dir():
            raise FileNotFoundError(assets)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(assets)
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else pairs_path.parent / "input_swap" / stamp
    raw = new_output(output, pairs_path, roots)
    summaries = []
    try:
        import torch
        from omegaconf import OmegaConf
        if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        cfg = base.config_for_inference(args, config, stats_path, root)
        settings, stats = base.read_settings(config, stats_path)
        if (int(cfg.model.proprio_dim) != settings["state_model_dim"]
                or int(cfg.model.action_dit_config.action_dim) != settings["action_model_dim"]):
            raise ValueError("Model/dataset dimensions disagree")
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        prompt = DEFAULT_PROMPT.format(task=args.task)
        cache = Path(args.text_cache_dir).expanduser().resolve() / (
            hashlib.sha256(prompt.encode()).hexdigest() + f".t5_len{int(cfg.data.train.context_len)}.wan22ti2v5b.pt")
        if not cache.is_file():
            raise FileNotFoundError(f"Missing original task cache: {cache}")
        # Check small input files again after running; checkpoint identity is size/mtime, not a full hash.
        tracked = [pairs_path, config, stats_path, cache, Path(__file__), Path(base.__file__)]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
        ck = checkpoint.stat()
        try:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            head = None
        seeds = list(range(args.seed, args.seed + args.num_seeds))
        base.write_json(output / "manifest.json", {
            "arguments": vars(args), "ranks": planned, "conditions": CONDITIONS,
            "condition_meaning": "first=image, second=proprio; AA/BB natural, BA/AB potentially inconsistent",
            "data_scope": "train_seen_user_confirmed", "git_head": head, "file_sha256": hashes,
            "checkpoint": {"path": str(checkpoint), "size_bytes": ck.st_size, "mtime_ns": ck.st_mtime_ns},
            "seeds_shared_by_all_conditions": seeds, "baseline_policy": "rerun, no cached predictions",
            "task_prompt": prompt, "settings": settings,
            "resolved_model": OmegaConf.to_container(cfg.model, resolve=True),
            "resolved_dataset": OmegaConf.to_container(cfg.data.train, resolve=True), "note": NOTICE,
        })
        print(f"[swap] ranks={planned} seeds={seeds}; calls={len(pairs)*4*len(seeds)}; output={output}", flush=True)
        prepared = base.prepare_pairs(args, pairs, payload, cfg, settings, stats)
        model = base.load_model(args, cfg)
        # Shared text context is saved once, not copied into every pair archive.
        first = prepared[0]["observations"][0]
        with (raw / "task_context.npz").open("xb") as file:
            np.savez_compressed(file, context=first.context.cpu().float().numpy(), mask=first.context_mask.cpu().numpy())
        for n, item in enumerate(prepared, 1):
            print(f"[swap] pair {n}/{len(prepared)} rank={item['pair']['rank']}", flush=True)
            predictions, timings = run_four(model, item["observations"], seeds, settings, args)
            summaries.extend(save_pair(raw, item, predictions, timings, seeds, stats))
        if pairs_path.read_bytes() != original or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError("An input/config/code/cache file changed during the run")
        after = checkpoint.stat()
        if (after.st_size, after.st_mtime_ns) != (ck.st_size, ck.st_mtime_ns):
            raise RuntimeError("Checkpoint size/mtime changed during the run")
        report(output, summaries, planned, "complete")
        base.write_json(output / "COMPLETE.json", {"status": "complete", "ranks": planned,
                        "num_seeds": len(seeds), "model_calls": len(pairs)*4*len(seeds), "no_robot_io": True})
        print(f"[swap] COMPLETE: {output}\nUpload report.md, pair_summary.csv, group_summary.csv and manifest.json only.", flush=True)
    except BaseException as exc:
        # Preserve partial results, but never label an incomplete run as complete.
        try:
            if not (output / "report.md").exists():
                report(output, summaries, planned, "failed_or_interrupted")
            if not (output / "FAILED.json").exists():
                base.write_json(output / "FAILED.json", {"status": "failed_or_interrupted",
                                "error": f"{type(exc).__name__}: {exc}",
                                "saved_pair_ranks": [s["rank"] for s in summaries if s["group"] == "all25"]})
        except Exception as logging_error:
            print(f"[swap] Could not write failure summary: {logging_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
