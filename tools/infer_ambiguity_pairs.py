#!/usr/bin/env python3
"""Offline, current-input-only FastWAM inference on saved ambiguity pairs.

Run scripts/infer_ambiguity_pairs.sh in the existing My_WAM environment.
The existing RobotVideoDataset reads recorded windows for preprocessing/GT;
ONLY video[:,0], proprio[0], and the same cached task context reach infer_action.
No future video, GT action, phase label, reset instruction, or robot connection
is supplied to the policy. No model training or recursive rollout is performed.

Supports the current Astribot raw-25D, z-score, ConcatLeftAlign, main FastWAM
configuration. Metrics exclude padded dimensions. Cross-GT preference and
same-seed prediction separation are descriptive, NOT execution failure rates
or proof of history dependence. The selected episodes are not auto-held-out.
Only a new pair_inference run directory is written; existing paths are intact.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
for directory in (REPO_ROOT / "src", REPO_ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from diagnose_ambiguity_pairs import (
    GROUPS, NumericEpisodeReader, current_index, joint_names,
    normalize, padded_rmse, read_settings, write_csv, write_json,
)
from export_ambiguity_pair_videos import parse_source, select_pairs


@dataclass(frozen=True)
class Observation:
    """This is the ONLY sample payload accessible to model inference."""
    image: Any
    proprio: Any
    context: Any
    context_mask: Any


def extract_observation(sample: dict) -> Observation:
    import torch
    video, state = sample["video"], sample["proprio"]
    context, mask = sample["context"], sample["context_mask"]
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] < 1:
        raise ValueError("Expected video [3,T,H,W]")
    if state.ndim != 2 or state.shape[0] < 1 or context.ndim != 2:
        raise ValueError("Expected proprio [T,D] and context [L,D]")
    if mask.ndim != 1 or mask.shape[0] != context.shape[0]:
        raise ValueError("Invalid context mask")
    # clone() releases the future-window storage; future values are never read here.
    image, proprio = video[:, 0].detach().cpu().clone(), state[0].detach().cpu().clone()
    for value in (image, proprio, context):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite policy input")
    if image.min() < -1.0001 or image.max() > 1.0001:
        raise ValueError("Input image must use the dataset's [-1,1] normalization")
    return Observation(image.unsqueeze(0), proprio.unsqueeze(0),
                       context.detach().cpu().clone().unsqueeze(0),
                       mask.detach().cpu().bool().clone().unsqueeze(0))


def reference_window(data: dict, source: dict, horizon: int, stats: dict, tolerance: float) -> dict:
    i = current_index(data, source, tolerance)
    # H actions + the actual successor state, all strictly in the same episode.
    if horizon <= 0 or i + horizon >= data["length"]:
        raise ValueError(f"ep={source['episode']} frame={i}: incomplete GT/successor; no padding allowed")
    raw_gt = data["action"][i:i + horizon].copy()
    normalized_gt, action_clipped = normalize(raw_gt, stats, "action")
    current_norm, state_clipped = normalize(data["state"][i], stats, "state")
    return {"raw_gt": raw_gt, "normalized_gt": normalized_gt,
            "raw_states": data["state"][i:i + horizon + 1].copy(),
            "current_norm": current_norm,
            "relative_times": data["time"][i:i + horizon] - data["time"][i],
            "state_relative_times": data["time"][i:i + horizon + 1] - data["time"][i],
            "clipped_gt_elements": int(action_clipped.sum()),
            "clipped_current_dims": np.flatnonzero(state_clipped).tolist(),
            "remaining_seconds": float(data["time"][-1] - data["time"][i]),
            "episode_length": int(data["length"]), "fps": float(data["fps"]),
            "source": dict(source)}


def verify_sample(sample: dict, reference: dict, settings: dict) -> None:
    action = sample["action"].detach().cpu().float().numpy()
    state = sample["proprio"][0].detach().cpu().float().numpy()
    if action.shape != (settings["horizon"], settings["action_model_dim"]):
        raise ValueError("Dataset action horizon/dimension mismatch")
    if state.shape != (settings["state_model_dim"],):
        raise ValueError("Dataset proprio dimension mismatch")
    for name in ("image_is_pad", "action_is_pad", "proprio_is_pad"):
        if name in sample and bool(sample[name].any().item()):
            raise ValueError(f"Padded reference: {name}")
    for actual, expected, name in ((action[..., :25], reference["normalized_gt"], "action"),
                                   (state[:25], reference["current_norm"], "state")):
        if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Dataset {name} != independently read Parquet; inspect indices/config/stats")
    if np.any(action[..., 25:] != 0) or np.any(state[25:] != 0):
        raise ValueError("Expected zero ConcatLeftAlign padding")


def locate_source_index(raw_dataset: Any, source: dict, tolerance: float) -> int:
    """Resolve actual episode/frame IDs, never trust old global row offsets."""
    offset = 0
    matches = []
    for child in raw_dataset.multi_dataset._datasets:
        if Path(child.root).resolve() == Path(source["root"]).resolve():
            table = child.hf_dataset.select_columns(
                ["episode_index", "frame_index", "timestamp"]).with_format(None)
            episodes = np.asarray(table["episode_index"], dtype=np.int64)
            frames = np.asarray(table["frame_index"], dtype=np.int64)
            ids = np.flatnonzero((episodes == source["episode"]) & (frames == source["frame"]))
            for local in ids:
                ts = float(table[int(local)]["timestamp"])
                if abs(ts - source["timestamp"]) > tolerance:
                    raise ValueError("Saved timestamp disagrees with HF row")
                matches.append(offset + int(local))
        offset += int(child.num_frames)
    if len(matches) != 1:
        raise ValueError(f"Expected one physical row for saved source; found {len(matches)}")
    return matches[0]


def verify_pair_metrics(pair: dict, observations: list[Observation], references: list[dict],
                        settings: dict, feature_hw: list[int]) -> dict:
    import torch
    import torch.nn.functional as F
    ga, gb = (ref["normalized_gt"] for ref in references)
    values = {
        "proprio_rmse": padded_rmse(references[0]["current_norm"], references[1]["current_norm"],
                                    settings["state_model_dim"]),
        "action_rmse": padded_rmse(ga, gb, settings["action_model_dim"]),
        "action_first_step_rmse": padded_rmse(ga[0], gb[0], settings["action_model_dim"]),
        "action_endpoint_rmse": padded_rmse(ga[-1], gb[-1], settings["action_model_dim"]),
    }
    if len(feature_hw) != 2 or any(int(x) <= 0 for x in feature_hw):
        raise ValueError("Invalid mining visual_feature_hw")
    images = [F.adaptive_avg_pool2d((obs.image.float() + 1) * .5, feature_hw)
              for obs in observations]
    values["visual_rmse"] = float(torch.mean((images[0] - images[1]) ** 2).sqrt())
    for name, value in values.items():
        if name in pair and not math.isclose(value, float(pair[name]), abs_tol=1e-4, rel_tol=1e-3):
            raise ValueError(f"{name}: mined={pair[name]}, reloaded={value}; do not compare different data/preprocessing")
    return values


def run_inference(model: Any, observations: list[Observation], seeds: list[int],
                  horizon: int, action_dim: int, steps: int, sigma_shift: float | None,
                  tiled: bool) -> tuple[np.ndarray, np.ndarray]:
    import torch
    if len(observations) != 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need two observations and a nonempty set of distinct seeds")
    predictions, timings = [], []
    for seed in seeds:
        paired, durations = [], []
        for side, obs in zip(("A", "B"), observations):
            def tensor(x, boolean=False):
                return x.to(device=model.device, dtype=torch.bool if boolean else model.torch_dtype).clone()
            # Explicit allowlist: NO action/GT/future video/source metadata.
            kwargs = dict(prompt=None, input_image=tensor(obs.image), proprio=tensor(obs.proprio),
                          context=tensor(obs.context), context_mask=tensor(obs.context_mask, True),
                          action_horizon=horizon, negative_prompt=None, text_cfg_scale=1.0,
                          num_inference_steps=steps, sigma_shift=sigma_shift, seed=seed,
                          rand_device="cpu", tiled=tiled)
            if torch.device(model.device).type == "cuda":
                torch.cuda.synchronize(model.device)
            start = time.perf_counter()
            with torch.inference_mode():
                out = model.infer_action(**kwargs)
            if torch.device(model.device).type == "cuda":
                torch.cuda.synchronize(model.device)
            elapsed = time.perf_counter() - start
            pred = out["action"].detach().cpu().float().numpy()
            if pred.shape != (horizon, action_dim) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid prediction for side={side}, seed={seed}: {pred.shape}")
            paired.append(pred.copy())
            durations.append(elapsed)
            print(f"[infer-pair] seed={seed} side={side} shape={pred.shape} seconds={elapsed:.3f}", flush=True)
        predictions.append(paired)
        timings.append(durations)
    return np.asarray(predictions, dtype=np.float32), np.asarray(timings)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("RMSE requires finite equal nonempty shapes")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def stats_of(values: list[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "std_across_seeds": float(x.std()),
            "min": float(x.min()), "max": float(x.max())}


def score_predictions(predictions: np.ndarray, gt: np.ndarray, seeds: list[int]) -> tuple[list, list, list, dict]:
    if predictions.ndim != 4 or predictions.shape[0] != len(seeds) or predictions.shape[1] != 2:
        raise ValueError("Expected predictions [seeds,2,H,D]")
    if gt.shape != predictions.shape[1:] or predictions.shape[-1] < 25:
        raise ValueError("GT/prediction dimensions disagree")
    if not np.isfinite(predictions).all() or not np.isfinite(gt).all() or not seeds:
        raise ValueError("Nonfinite/empty predictions or GT")
    p, g = predictions[..., :25], gt[..., :25]
    gt_distance = rmse(g[0], g[1])
    per_seed, per_pair, per_group = [], [], []
    for n, seed in enumerate(seeds):
        for s, side in enumerate(("A", "B")):
            own, other = rmse(p[n, s], g[s]), rmse(p[n, s], g[1 - s])
            # Positive margin => closer to own GT; negative => closer to other GT.
            margin = other - own
            preference = "tie" if abs(margin) <= 1e-6 else ("own_gt" if margin > 0 else "other_gt")
            per_seed.append({"seed": seed, "side": side, "own_gt_rmse_active25": own,
                             "other_gt_rmse_active25": other, "other_minus_own_rmse": margin,
                             "gt_preference_not_failure_label": preference,
                             "own_gt_mae_active25": float(np.abs(p[n, s].astype(np.float64) - g[s]).mean()),
                             "first_step_rmse_active25": rmse(p[n, s, 0], g[s, 0]),
                             "last_step_rmse_active25": rmse(p[n, s, -1], g[s, -1]),
                             "prediction_outside_training_clip_elements": int((np.abs(p[n, s]) > 5).sum())})
            for group, lo, hi in GROUPS:
                per_group.append({"seed": seed, "side": side, "group": group,
                                  "own_gt_rmse_normalized": rmse(p[n, s, :, lo:hi], g[s, :, lo:hi]),
                                  "other_gt_rmse_normalized": rmse(p[n, s, :, lo:hi], g[1-s, :, lo:hi])})
        separation = rmse(p[n, 0], p[n, 1])
        per_pair.append({"seed": seed, "prediction_pair_rmse_active25": separation,
                         "gt_pair_rmse_active25": gt_distance,
                         "prediction_to_gt_separation_ratio": separation / gt_distance if gt_distance > 1e-12 else None})
    summary = {"active_dimensions": 25, "num_seeds": len(seeds), "gt_pair_rmse_active25": gt_distance,
               "same_seed_prediction_separation": stats_of([x["prediction_pair_rmse_active25"] for x in per_pair]),
               "sides": {}}
    for side in ("A", "B"):
        rows = [x for x in per_seed if x["side"] == side]
        summary["sides"][side] = {
            "own_gt_rmse_active25": stats_of([x["own_gt_rmse_active25"] for x in rows]),
            "other_gt_rmse_active25": stats_of([x["other_gt_rmse_active25"] for x in rows]),
            "other_gt_closer_seed_count": sum(x["gt_preference_not_failure_label"] == "other_gt" for x in rows),
            "tie_seed_count": sum(x["gt_preference_not_failure_label"] == "tie" for x in rows),
        }
    summary["interpretation"] = (
        "Distances to recorded demonstrations, not valid-action labels or robot success rates. "
        "Shared seeds control sampling noise, not history; current proprio may distinguish sides. "
        "No held-out split is created, no memory intervention is tested, and no termination label is inferred.")
    return per_seed, per_pair, per_group, summary


def denormalize_predictions(predictions: np.ndarray, stats: dict) -> np.ndarray:
    field = stats["action"]["default"]
    mean = np.asarray(field["global_mean"], dtype=np.float32)
    std = np.asarray(field["global_std"], dtype=np.float32)
    scale = np.float32(1) / (std + np.float32(1e-8))
    offset = -mean / (std + np.float32(1e-8))
    # Do NOT clip predictions or re-normalize recorded GT for the raw plots.
    raw = (predictions[..., :25].astype(np.float32) - offset) / scale
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite denormalized prediction")
    return raw


def plot_results(output: Path, pred_raw: np.ndarray, references: list[dict], names: list[str], rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "joint_curves"
    directory.mkdir()
    means = pred_raw.mean(axis=0)
    for dim, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(8, 4))
        for s, side in enumerate(("A", "B")):
            times = references[s]["relative_times"]
            ax.plot(times, references[s]["raw_gt"][:, dim], label=f"{side}: recorded GT")
            ax.plot(times, means[s, :, dim], linestyle="--", label=f"{side}: prediction mean")
        ax.set(xlabel="Seconds relative to CURRENT (command timestamps)", ylabel="Recorded units",
               title=f"{name} | mean is a summary, not an executed trajectory")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / f"{dim:02d}_{name}.png", dpi=120)
        plt.close(fig)
    for side in ("A", "B"):
        selected = [x for x in rows if x["side"] == side]
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label in (("own_gt_rmse_active25", "To own GT"), ("other_gt_rmse_active25", "To other GT")):
            ax.plot([x["seed"] for x in selected], [x[key] for x in selected], marker="o", label=label)
        ax.set(xlabel="Seed", ylabel="Normalized RMSE (25 active dims)", title=f"Side {side}: GT distances, not failures")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"side_{side.lower()}_seed_errors.png", dpi=120)
        plt.close(fig)


def save_pair(output: Path, pair: dict, observations: list[Observation], references: list[dict],
              predictions: np.ndarray, timings: np.ndarray, seeds: list[int], stats: dict,
              verified: dict, base_mode: str, plots: bool) -> dict:
    import torch
    from torchvision.utils import save_image
    output.mkdir(exist_ok=False)
    gt25 = np.stack([x["normalized_gt"] for x in references])
    gt = np.pad(gt25, ((0, 0), (0, 0), (0, predictions.shape[-1] - 25)))
    seed_rows, pair_rows, group_rows, summary = score_predictions(predictions, gt, seeds)
    for n, row in enumerate(seed_rows):
        row["model_inference_seconds"] = float(timings[n // 2, n % 2])
    raw = denormalize_predictions(predictions, stats)
    np.savez_compressed(output / "actions.npz", seeds=np.asarray(seeds),
                        prediction_normalized=predictions, prediction_raw25=raw,
                        gt_normalized_model=gt, gt_raw25=np.stack([x["raw_gt"] for x in references]),
                        actual_state_raw25=np.stack([x["raw_states"] for x in references]),
                        action_relative_seconds=np.stack([x["relative_times"] for x in references]),
                        state_relative_seconds=np.stack([x["state_relative_times"] for x in references]))
    np.savez_compressed(output / "policy_inputs.npz",
                        current_image=np.stack([x.image[0].numpy() for x in observations]),
                        current_proprio=np.stack([x.proprio[0].numpy() for x in observations]),
                        task_context=observations[0].context.float().numpy(),
                        context_mask=observations[0].context_mask.numpy())
    save_image(torch.cat([((x.image[0] + 1) * .5).clamp(0, 1) for x in observations], dim=-1),
               output / "current_policy_inputs.png")
    write_csv(output / "seed_metrics.csv", seed_rows)
    write_csv(output / "paired_seed_metrics.csv", pair_rows)
    write_csv(output / "group_metrics.csv", group_rows)
    names = joint_names(base_mode)
    records = []
    for n, seed in enumerate(seeds):
        for s, side in enumerate(("A", "B")):
            for step, values in enumerate(raw[n, s]):
                records.append({"seed": seed, "side": side, "step": step,
                                "relative_seconds": float(references[s]["relative_times"][step]),
                                **{name: float(value) for name, value in zip(names, values)}})
    write_csv(output / "predictions_raw.csv", records)
    summary.update(rank=int(pair["rank"]), reference_reproduction=verified,
                   sources=[{k: v for k, v in r.items() if k not in (
                       "raw_gt", "normalized_gt", "raw_states", "current_norm", "relative_times", "state_relative_times")}
                            for r in references], base_mode=base_mode,
                   array_axes="predictions: [seed,A_or_B,action_step,dimension]; actual_state: [A_or_B,H+1,25]")
    write_json(output / "summary.json", summary)
    lines = [f"# 第 {pair['rank']} 对：WAM 离线推理", "",
             f"每侧 {len(seeds)} 个 seed；每次 {predictions.shape[2]} 步。只输入当前图像、当前 proprio、相同任务文本。",
             "评分只统计真实 25 维；GT 是示范，不是所有正确动作的集合。", "",
             "| 侧别 | 对本侧 GT 的 RMSE（均值±seed标准差） | 对另一侧 GT 的 RMSE | 更接近另一侧 GT 的 seed 数 |",
             "|---|---:|---:|---:|"]
    for side in ("A", "B"):
        s = summary["sides"][side]
        e = s["own_gt_rmse_active25"]
        lines.append(f"| {side} | {e['mean']:.5f} ± {e['std_across_seeds']:.5f} | "
                     f"{s['other_gt_rmse_active25']['mean']:.5f} | {s['other_gt_closer_seed_count']}/{len(seeds)} |")
    lines += ["", f"GT 两侧距离：{summary['gt_pair_rmse_active25']:.5f}；"
              f"同 seed 两侧预测距离均值：{summary['same_seed_prediction_separation']['mean']:.5f}。",
              "", "## 解读边界",
              "- 更接近另一侧 GT 只是分支混淆线索，不等于真机失败；当前关节差异也可能区分阶段。",
              "- 预测距离较小不自动证明动作坍缩；GT 本来相似时尤其不能这样解释。",
              "- 完成态应与真实减速/收尾 GT 比较，不能把任何非零动作视为错误。",
              "- 未自动划分测试集，不代表未见轨迹的泛化结果，也没有证明记忆缺失是原因。",
              "- 原始 GT 保留未裁剪值；预测反归一化未裁剪。存在训练归一化裁剪时，两种空间误差可能不同。",
              "- joint_curves 的虚线是多 seed 均值，并非实际执行轨迹；逐 seed 数值见 CSV/NPZ。",
              "- 当前不进行连续 rollout，因此不能断言未来几秒会重新开门或长期保持。", "",
              "文件：seed_metrics.csv、paired_seed_metrics.csv、group_metrics.csv、actions.npz、",
              "policy_inputs.npz、predictions_raw.csv、current_policy_inputs.png、joint_curves/。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plots:
        plot_results(output, raw, references, names, seed_rows)
    print(f"[infer-pair] saved {output}; report.md", flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pairs-json", required=True)
    p.add_argument("--ranks", nargs="+", type=int, default=[2])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None, help="Use the checkpoint's saved training config")
    p.add_argument("--dataset-stats", default=None)
    p.add_argument("--dataset-dir", default=None, help="Optional single dataset mount override")
    p.add_argument("--text-cache-dir", required=True)
    p.add_argument("--task", default="dishwasher", help="Original task, shared by both sides; never a phase label")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    p.add_argument("--vae-device-mode", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--model-base-path", default=None)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--video-backend", choices=["pyav", "torchcodec", "video_reader"], default="pyav")
    p.add_argument("--tolerance-s", type=float, default=.002)
    p.add_argument("--base-mode", choices=["velocity", "position", "unknown"], default="velocity")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def config_for_inference(args, config: Path, stats_path: Path, root: str):
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    from experiments.astribot.fastwam_portal_server import _apply_wan_model_overrides
    register_default_resolvers()
    # These only affect this diagnostic process, not the source config files.
    os.environ["WAM_TASK_INSTRUCTION"] = args.task
    cfg = OmegaConf.load(config)
    if str(cfg.model.get("_target_")) != "fastwam.runtime.create_fastwam":
        raise ValueError("This script targets main FastWAM, not Joint/IDM")
    if str(cfg.model.video_dit_config.video_attention_mask_mode) != "first_frame_causal":
        raise ValueError("Main infer_action requires first_frame_causal")
    if bool(cfg.model.video_dit_config.get("action_conditioned", False)):
        raise ValueError("Expected the non-action-conditioned main FastWAM configuration")
    if cfg.data.train.get("override_instruction") not in (None, args.task):
        raise ValueError("--task differs from the saved training instruction")
    cfg.model.load_text_encoder = False
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    train = cfg.data.train
    train.dataset_dirs = [root]
    train.episodes = None
    train.pretrained_norm_stats = str(stats_path)
    train.is_training_set = False
    train.val_set_proportion = 0.0  # no automatic train/test claims
    train.skip_padding_as_possible = False
    train.override_instruction = args.task
    train.text_embedding_cache_dir = str(Path(args.text_cache_dir).expanduser().resolve())
    train.video_backend = args.video_backend
    train.video_tolerance_s = args.tolerance_s
    _apply_wan_model_overrides(cfg)
    return cfg


def prepare_pairs(args, pairs: list[dict], payload: dict, cfg, settings: dict, stats: dict) -> list[dict]:
    import torch
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    import fastwam.datasets.lerobot.base_lerobot_dataset as base
    dataset = instantiate(cfg.data.train)
    readers, prepared = {}, []
    previous_attempts = base.MAX_GETITEM_ATTEMPT
    base.MAX_GETITEM_ATTEMPT = 1  # no random sample replacement on decode error
    try:
        for pair in pairs:
            observations, references = [], []
            sources = [parse_source(pair, s, args.dataset_dir) for s in ("a", "b")]
            if (sources[0]["root"], sources[0]["episode"]) != (sources[1]["root"], sources[1]["episode"]):
                raise ValueError("Expected a within-episode pair")
            for source in sources:
                root = source["root"]
                if root not in readers:
                    readers[root] = NumericEpisodeReader(root, args.tolerance_s)
                reader = readers[root]
                if args.tolerance_s >= .5 / reader.fps:
                    raise ValueError("Timestamp tolerance must be below half a frame")
                reference = reference_window(reader.read(source["episode"]), source, settings["horizon"], stats, args.tolerance_s)
                idx = locate_source_index(dataset.lerobot_dataset, source, args.tolerance_s)
                sample = dataset._get(idx)
                verify_sample(sample, reference, settings)
                if sample["prompt"] != DEFAULT_PROMPT.format(task=args.task):
                    raise ValueError("Unexpected language condition")
                obs = extract_observation(sample)
                if obs.context.shape[-1] != int(cfg.model.video_dit_config.text_dim):
                    raise ValueError("Cached text embedding dimension != model text_dim")
                observations.append(obs)
                references.append(reference)
                print(f"[input] rank={pair['rank']} ep={source['episode']} frame={source['frame']} verified; current-only model input", flush=True)
            if not torch.equal(observations[0].context, observations[1].context) or not torch.equal(observations[0].context_mask, observations[1].context_mask):
                raise ValueError("Both sides must receive identical task context")
            verified = verify_pair_metrics(pair, observations, references, settings,
                                           payload.get("summary", {}).get("visual_feature_hw", [24, 20]))
            prepared.append(dict(pair=pair, observations=observations, references=references, verified=verified))
    finally:
        base.MAX_GETITEM_ATTEMPT = previous_attempts
    return prepared


def load_model(args, cfg):
    import torch
    from hydra.utils import instantiate
    from experiments.astribot.fastwam_portal_server import (
        _mixed_precision_to_model_dtype, _restore_official_conv3d_forward,
        patch_fastwam_encode_input_image_latents_tensor,
    )
    dtype = _mixed_precision_to_model_dtype(args.mixed_precision or str(cfg.get("mixed_precision", "bf16")))
    print(f"[model] loading checkpoint={args.checkpoint}; device={args.device}; dtype={dtype}", flush=True)
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    model.load_checkpoint(str(Path(args.checkpoint).expanduser().resolve()))
    model = model.to(args.device).eval()
    model.device, model.torch_dtype = torch.device(args.device), dtype
    _restore_official_conv3d_forward(model)
    patch_fastwam_encode_input_image_latents_tensor(model, vae_device_mode=args.vae_device_mode)
    return model


def create_output(path: Path, pairs_path: Path, roots: set[str]) -> None:
    if path == pairs_path.parent or any(path == Path(root) or Path(root) in path.parents for root in roots):
        raise ValueError("Use a new inference subdirectory, not the mining directory or raw dataset")
    path.mkdir(parents=True, exist_ok=True)
    if any(p.name not in {"run.log", "launcher.sh"} for p in path.iterdir()):
        raise FileExistsError("Output contains previous results; use a new run directory")
    write_json(path / "STARTED.json", {"status": "started"})


def main(argv=None):
    args = parse_args(argv)
    if args.num_seeds < 1 or args.num_inference_steps < 1 or not 0 <= args.seed < 2**63 - args.num_seeds:
        raise ValueError("Invalid seed count, start seed, or inference steps")
    if not math.isfinite(args.tolerance_s) or args.tolerance_s <= 0 or not args.task.strip():
        raise ValueError("Invalid tolerance/task")
    if args.sigma_shift is not None and (not math.isfinite(args.sigma_shift) or args.sigma_shift <= 0):
        raise ValueError("sigma-shift must be finite/positive")
    pairs_path = Path(args.pairs_json).expanduser().resolve()
    original = pairs_path.read_bytes()
    payload = json.loads(original)
    pairs = select_pairs(payload, args.ranks)
    old_roots = {parse_source(p, s)["root"] for p in pairs for s in ("a", "b")}
    if len(old_roots) != 1:
        raise ValueError("Run one source dataset at a time; no ambiguous multi-mount remapping")
    roots = {parse_source(p, s, args.dataset_dir)["root"] for p in pairs for s in ("a", "b")}
    root = next(iter(roots))
    config = Path(args.config or payload["summary"]["config"]).expanduser().resolve()
    stats_path = Path(args.dataset_stats or payload["summary"]["dataset_stats"]).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for path in (config, stats_path, checkpoint, Path(root) / "meta/info.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; edit paths in the launcher")
    if args.model_base_path:
        assets = Path(args.model_base_path).expanduser().resolve()
        if not assets.is_dir():
            raise FileNotFoundError(assets)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(assets)
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else pairs_path.parent / "pair_inference" / stamp
    create_output(output, pairs_path, roots)
    try:
        import torch
        from omegaconf import OmegaConf
        if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        cfg = config_for_inference(args, config, stats_path, root)
        settings, stats = read_settings(config, stats_path)
        if int(cfg.model.proprio_dim) != settings["state_model_dim"] or int(cfg.model.action_dit_config.action_dim) != settings["action_model_dim"]:
            raise ValueError("Model and dataset dimensions disagree")
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        prompt = DEFAULT_PROMPT.format(task=args.task)
        cache = Path(args.text_cache_dir).expanduser().resolve() / (
            hashlib.sha256(prompt.encode()).hexdigest() + f".t5_len{int(cfg.data.train.context_len)}.wan22ti2v5b.pt")
        if not cache.is_file():
            raise FileNotFoundError(f"Missing ORIGINAL task embedding cache: {cache}; use the cache for this checkpoint")
        seeds = list(range(args.seed, args.seed + args.num_seeds))
        try:
            git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            git_head = None
        ck = checkpoint.stat()
        write_json(output / "manifest.json", {
            "arguments": vars(args), "seeds_shared_between_sides": seeds, "git_head": git_head,
            "pairs_json": str(pairs_path), "pairs_json_sha256": hashlib.sha256(original).hexdigest(),
            "config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "stats": str(stats_path), "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "checkpoint": {"path": str(checkpoint), "size_bytes": ck.st_size, "mtime_ns": ck.st_mtime_ns},
            "task_prompt": prompt, "text_cache": str(cache), "text_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
            "resolved_model": OmegaConf.to_container(cfg.model, resolve=True),
            "resolved_dataset": OmegaConf.to_container(cfg.data.train, resolve=True), "settings": settings,
            "note": "Not a held-out benchmark. GT/future windows are used only for scoring. No training, rollout, or robot I/O."})
        print(f"[infer-pair] ranks={[p['rank'] for p in pairs]}, seeds={seeds}, horizon={settings['horizon']}; output={output}", flush=True)
        prepared = prepare_pairs(args, pairs, payload, cfg, settings, stats)
        model = load_model(args, cfg)
        for item in prepared:
            predictions, timings = run_inference(model, item["observations"], seeds, settings["horizon"],
                                                  settings["action_model_dim"], args.num_inference_steps,
                                                  args.sigma_shift, args.tiled)
            save_pair(output / f"pair_{int(item['pair']['rank']):04d}", item["pair"], item["observations"],
                      item["references"], predictions, timings, seeds, stats, item["verified"], args.base_mode, not args.no_plots)
        if pairs_path.read_bytes() != original:
            raise RuntimeError("Input pairs.json changed during this run; inspect provenance")
        write_json(output / "COMPLETE.json", {"status": "complete", "ranks": [p["rank"] for p in pairs],
                                             "num_seeds": len(seeds), "model_calls": len(pairs) * 2 * len(seeds)})
        print(f"[infer-pair] COMPLETE: {output}\nRead pair_*/report.md. These are offline GT distances, not robot failure rates.", flush=True)
    except Exception as exc:
        write_json(output / "FAILED.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
