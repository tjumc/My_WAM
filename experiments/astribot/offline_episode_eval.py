#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay a recorded Astribot episode offline and save WAM-predicted actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.astribot.fastwam_portal_server import AstribotFastWAMPolicy, _setup_logging
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.utils.config_resolvers import register_default_resolvers

DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "astribot_washclothes_posttrain32_from_step10k"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "train" / "config.yaml"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "train" / "checkpoints" / "weights" / "step_050000.pt"
DEFAULT_DATASET_STATS = DEFAULT_RUN_DIR / "stats.json"
DEFAULT_TASK = "put the clothes into the washing machine"


def _resolve_path(path: str | os.PathLike[str], *, base: Path = PROJECT_ROOT) -> Path:
    out = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if not out.is_absolute():
        out = base / out
    return out.resolve()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _chw_to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected single-frame image [1,C,H,W], got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
    if arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        if arr_f.size > 0 and float(np.nanmax(arr_f)) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_numpy_2d(value: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 2D [T,D], got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_npz_episode(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    data = np.load(path)
    missing = [key for key in ("img0", "img1", "img2", "state") if key not in data]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}; expected img0/img1/img2/state")
    episode = {
        "img0": np.asarray(data["img0"]),
        "img1": np.asarray(data["img1"]),
        "img2": np.asarray(data["img2"]),
        "state": _to_numpy_2d(data["state"], name="state"),
    }
    if "action" in data:
        episode["action"] = _to_numpy_2d(data["action"], name="action")
    elif "gt_action" in data:
        episode["action"] = _to_numpy_2d(data["gt_action"], name="gt_action")
    lengths = {key: episode[key].shape[0] for key in ("img0", "img1", "img2", "state")}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Episode input lengths mismatch: {lengths}")
    return episode


def _build_raw_lerobot_dataset(cfg, dataset_dirs: list[str] | None) -> BaseLerobotDataset:
    data_cfg = cfg.data.train
    dirs = dataset_dirs or _as_list(data_cfg.get("dataset_dirs"))
    if not dirs:
        raise ValueError("No dataset dirs found. Pass --dataset_dir or set cfg.data.train.dataset_dirs.")

    shape_meta = OmegaConf.to_container(data_cfg.shape_meta, resolve=True)
    dataset = BaseLerobotDataset(
        dataset_dirs=dirs,
        shape_meta=shape_meta,
        obs_size=2,
        action_size=1,
        val_set_proportion=0.0,
        is_training_set=False,
        global_sample_stride=1,
        image_obs_indices=[0],
        video_tolerance_s=float(data_cfg.get("video_tolerance_s", 1e-4)),
        pretrained_norm_stats=data_cfg.get("pretrained_norm_stats"),
        norm_stats_source=str(data_cfg.get("norm_stats_source", "meta")),
        pretrained_norm_stats_dataset_sources=data_cfg.get("pretrained_norm_stats_dataset_sources"),
    )
    dataset._set_return_images(True)
    return dataset


def _load_lerobot_episode(
    cfg,
    *,
    dataset_dirs: list[str] | None,
    episode_index: int,
    start_step: int,
    max_steps: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray | None, str]:
    dataset = _build_raw_lerobot_dataset(cfg, dataset_dirs)
    if episode_index < 0 or episode_index >= len(dataset.episode_data_index["from"]):
        raise IndexError(
            f"episode_index={episode_index} out of range [0, {len(dataset.episode_data_index['from'])})"
        )

    ep_start = int(dataset.episode_data_index["from"][episode_index].item())
    ep_end = int(dataset.episode_data_index["to"][episode_index].item())
    episode_len = ep_end - ep_start
    if episode_len <= 1:
        raise ValueError(f"Episode {episode_index} is too short: length={episode_len}")

    action_meta = dataset.action_meta[0]
    action_key = action_meta["lerobot_key"]
    gt_action = None
    try:
        ep_table = dataset.multi_dataset.get_episode_data(episode_index, columns=[action_key])
        gt_action = _to_numpy_2d(ep_table[action_key], name=action_key)
    except Exception as exc:
        print(f"[offline-episode] warning: failed to load GT action for episode {episode_index}: {exc}")

    max_available = episode_len - 1 - int(start_step)
    if gt_action is not None:
        max_available = min(max_available, gt_action.shape[0] - int(start_step))
    if max_available <= 0:
        raise ValueError(
            f"No evaluable steps: episode_len={episode_len}, gt_len={None if gt_action is None else gt_action.shape[0]}, "
            f"start_step={start_step}"
        )
    num_steps = max_available if max_steps is None else min(max_available, int(max_steps))

    samples: list[dict[str, Any]] = []
    for local_step in range(int(start_step), int(start_step) + num_steps):
        sample = dataset[ep_start + local_step]
        images = sample["images"]
        state = sample["state"][dataset.state_meta[0]["key"]][0].detach().cpu().numpy().astype(np.float32)
        samples.append(
            {
                "step": local_step,
                "img0": _chw_to_hwc_uint8(images[dataset.image_meta[0]["key"]][0]),
                "img1": _chw_to_hwc_uint8(images[dataset.image_meta[1]["key"]][0]),
                "img2": _chw_to_hwc_uint8(images[dataset.image_meta[2]["key"]][0]),
                "state": np.ascontiguousarray(state, dtype=np.float32),
                "task": str(sample.get("task", "")),
            }
        )

    source = f"lerobot episode_index={episode_index} global_range=[{ep_start},{ep_end})"
    if gt_action is not None:
        gt_action = np.ascontiguousarray(gt_action[int(start_step) : int(start_step) + num_steps], dtype=np.float32)
    return samples, gt_action, source


def _load_npz_samples(
    input_npz: str,
    *,
    start_step: int,
    max_steps: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray | None, str]:
    episode = load_npz_episode(input_npz)
    total = episode["state"].shape[0]
    gt_action = episode.get("action")
    max_available = total - int(start_step)
    if gt_action is not None:
        max_available = min(max_available, gt_action.shape[0] - int(start_step))
    if max_available <= 0:
        raise ValueError(f"No evaluable steps in {input_npz}: total={total}, start_step={start_step}")
    num_steps = max_available if max_steps is None else min(max_available, int(max_steps))
    lo = int(start_step)
    hi = lo + num_steps

    samples = []
    for step in range(lo, hi):
        samples.append(
            {
                "step": step,
                "img0": _chw_to_hwc_uint8(episode["img0"][step]),
                "img1": _chw_to_hwc_uint8(episode["img1"][step]),
                "img2": _chw_to_hwc_uint8(episode["img2"][step]),
                "state": np.ascontiguousarray(episode["state"][step], dtype=np.float32),
                "task": "",
            }
        )
    if gt_action is not None:
        gt_action = np.ascontiguousarray(gt_action[lo:hi], dtype=np.float32)
    return samples, gt_action, f"npz episode={input_npz}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline episode eval: replay recorded Astribot observations and save predicted actions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=os.environ.get("CONFIG", str(DEFAULT_CONFIG)))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    parser.add_argument("--dataset_stats", default=os.environ.get("DATASET_STATS", str(DEFAULT_DATASET_STATS)))
    parser.add_argument("--dataset_dir", action="append", default=None, help="Override dataset dir; repeatable.")
    parser.add_argument("--input_npz", default=os.environ.get("INPUT_NPZ"), help="Optional episode npz input.")
    parser.add_argument("--episode_index", type=int, default=int(os.environ.get("EPISODE_INDEX", "0")))
    parser.add_argument("--start_step", type=int, default=int(os.environ.get("START_STEP", "0")))
    parser.add_argument("--max_steps", type=int, default=None if os.environ.get("MAX_STEPS") is None else int(os.environ["MAX_STEPS"]))
    parser.add_argument("--replan_steps", type=int, default=int(os.environ.get("REPLAN_STEPS", "1")))
    parser.add_argument("--output_npz", default=os.environ.get("OUTPUT_NPZ"))
    parser.add_argument("--output_json", default=os.environ.get("OUTPUT_JSON"))
    parser.add_argument("--task", default=os.environ.get("TASK_PROMPT"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=os.environ.get("MIXED_PRECISION"))
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--action_mode", default=os.environ.get("ACTION_MODE", "cmd_absolute_joint_vel"))
    parser.add_argument("--num_inference_steps", type=int, default=int(os.environ.get("NUM_INFERENCE_STEPS", "10")))
    parser.add_argument("--sigma_shift", type=float, default=None)
    parser.add_argument("--text_cfg_scale", type=float, default=float(os.environ.get("TEXT_CFG_SCALE", "1.0")))
    parser.add_argument("--negative_prompt", default=os.environ.get("NEGATIVE_PROMPT", ""))
    parser.add_argument("--rand_device", default=os.environ.get("RAND_DEVICE", "cpu"))
    parser.add_argument("--vae_device_mode", choices=["gpu", "cpu"], default=os.environ.get("VAE_DEVICE_MODE", "cpu"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument(
        "--model_base_path",
        default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH"),
        help="Optional DIFFSYNTH_MODEL_BASE_PATH override.",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    if args.replan_steps <= 0:
        raise ValueError("--replan_steps must be positive.")
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(_resolve_path(args.model_base_path))

    register_default_resolvers()
    cfg = OmegaConf.load(_resolve_path(args.config))
    OmegaConf.resolve(cfg)

    if args.input_npz:
        samples, gt_actions, source = _load_npz_samples(
            args.input_npz,
            start_step=args.start_step,
            max_steps=args.max_steps,
        )
    else:
        samples, gt_actions, source = _load_lerobot_episode(
            cfg,
            dataset_dirs=args.dataset_dir,
            episode_index=args.episode_index,
            start_step=args.start_step,
            max_steps=args.max_steps,
        )

    task_prompt = args.task or next((sample["task"] for sample in samples if sample.get("task")), DEFAULT_TASK)
    if not task_prompt:
        task_prompt = DEFAULT_TASK

    policy = AstribotFastWAMPolicy(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        task_prompt=task_prompt,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        n_action_steps=args.n_action_steps,
        action_mode=args.action_mode,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        rand_device=args.rand_device,
        tiled=args.tiled,
        vae_device_mode=args.vae_device_mode,
        seed=args.seed,
    )

    pred_actions: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    chunk_starts: list[int] = []
    executed_steps: list[int] = []
    latencies_ms: list[float] = []

    cursor = 0
    total = len(samples)
    print("[offline-episode] source:", source)
    print("[offline-episode] task:", task_prompt)
    print("[offline-episode] steps:", total, "replan_steps:", args.replan_steps)
    print("[offline-episode] model_info:", json.dumps(policy.get_model_info(), ensure_ascii=False, indent=2))

    while cursor < total:
        sample = samples[cursor]
        t0 = time.time()
        chunk = policy.infer(
            t0,
            sample["img0"],
            sample["img1"],
            sample["img2"],
            sample["state"],
            his_effort=None,
            delay=0,
        )
        latency_ms = (time.time() - t0) * 1000.0
        if chunk is None:
            raise RuntimeError(f"WAM inference returned None at episode step {sample['step']}")
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        take = min(int(args.replan_steps), chunk.shape[0], total - cursor)
        pred_actions.append(chunk[:take])
        pred_chunks.append(chunk)
        chunk_starts.append(int(sample["step"]))
        executed_steps.extend(int(samples[cursor + i]["step"]) for i in range(take))
        latencies_ms.append(latency_ms)

        print(
            f"[offline-episode] step={sample['step']} take={take} "
            f"chunk_shape={chunk.shape} latency={latency_ms:.1f}ms"
        )
        cursor += take

    pred_actions_np = np.concatenate(pred_actions, axis=0) if pred_actions else np.empty((0, policy.raw_action_dim))
    pred_chunks_np = np.stack(pred_chunks, axis=0) if pred_chunks else np.empty((0, policy.action_horizon, policy.raw_action_dim))
    executed_steps_np = np.asarray(executed_steps, dtype=np.int64)
    chunk_starts_np = np.asarray(chunk_starts, dtype=np.int64)
    latencies_np = np.asarray(latencies_ms, dtype=np.float32)

    metrics: dict[str, Any] = {
        "source": source,
        "task": task_prompt,
        "episode_index": int(args.episode_index),
        "start_step": int(args.start_step),
        "num_pred_actions": int(pred_actions_np.shape[0]),
        "num_chunks": int(pred_chunks_np.shape[0]),
        "replan_steps": int(args.replan_steps),
        "action_dim": int(policy.raw_action_dim),
        "action_horizon": int(policy.action_horizon),
        "latency_ms_mean": float(latencies_np.mean()) if latencies_np.size else None,
        "latency_ms_max": float(latencies_np.max()) if latencies_np.size else None,
    }

    gt_executed = None
    if gt_actions is not None:
        gt_executed = gt_actions[executed_steps_np - int(args.start_step)]
        if gt_executed.shape != pred_actions_np.shape:
            raise RuntimeError(f"GT/pred shape mismatch: gt={gt_executed.shape} pred={pred_actions_np.shape}")
        diff = pred_actions_np - gt_executed
        metrics.update(
            {
                "action_l1": float(np.abs(diff).mean()),
                "action_l2": float(np.square(diff).mean()),
                "action_rmse": float(np.sqrt(np.square(diff).mean())),
                "action_max_abs": float(np.abs(diff).max()),
            }
        )
        print(
            "[offline-episode] action metrics:",
            f"l1={metrics['action_l1']:.6f}",
            f"l2={metrics['action_l2']:.6f}",
            f"rmse={metrics['action_rmse']:.6f}",
            f"max_abs={metrics['action_max_abs']:.6f}",
        )

    print(
        "[offline-episode] pred stats:",
        f"shape={pred_actions_np.shape}",
        f"min={float(pred_actions_np.min()):+.6f}",
        f"max={float(pred_actions_np.max()):+.6f}",
        f"mean={float(pred_actions_np.mean()):+.6f}",
        f"std={float(pred_actions_np.std()):+.6f}",
    )

    output_npz = args.output_npz
    if output_npz is None:
        output_npz = str(
            PROJECT_ROOT
            / "runs"
            / "offline_episode_eval"
            / f"episode_{int(args.episode_index):06d}_start_{int(args.start_step):06d}.npz"
        )
    output_path = _resolve_path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "pred_actions": pred_actions_np,
        "pred_chunks": pred_chunks_np,
        "chunk_starts": chunk_starts_np,
        "executed_steps": executed_steps_np,
        "latencies_ms": latencies_np,
        "metrics_json": np.asarray(json.dumps(metrics, ensure_ascii=False)),
    }
    if gt_executed is not None:
        save_payload["gt_actions"] = gt_executed
        save_payload["action_diff"] = pred_actions_np - gt_executed
    np.savez_compressed(output_path, **save_payload)
    print("[offline-episode] saved npz:", output_path)

    if args.output_json:
        json_path = _resolve_path(args.output_json)
    else:
        json_path = output_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[offline-episode] saved metrics:", json_path)
    print("[offline-episode] OK")


if __name__ == "__main__":
    main()
