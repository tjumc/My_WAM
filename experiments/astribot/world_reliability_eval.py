#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline diagnostic for Fast-WAM world-prediction reliability.

The diagnostic compares the model's imagined future latent with the latent
encoding of the real future observation window from RobotVideoDataset.
It deliberately excludes the first latent time step because Fast-WAM clamps
that step to the observed first frame during sampling.

This first version is intended for the current Astribot Fast-WAM setup where
the video branch is not action-conditioned. Therefore the resulting metric is
an expectation-vs-reality divergence, not an action-consequence score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.astribot.fastwam_portal_server import (
    _apply_wan_model_overrides,
    _mixed_precision_to_model_dtype,
    _resolve_path,
    _restore_official_conv3d_forward,
    _setup_logging,
    patch_fastwam_encode_input_image_latents_tensor,
)
from fastwam.utils.config_resolvers import register_default_resolvers


def _has_padding(sample: dict[str, Any]) -> bool:
    for key in ("image_is_pad", "action_is_pad", "proprio_is_pad"):
        mask = sample.get(key)
        if mask is not None and bool(torch.as_tensor(mask).any().item()):
            return True
    return False


def _source_description(dataset, sample_index: int) -> str:
    raw = getattr(dataset, "lerobot_dataset", None)
    describe = getattr(raw, "_describe_sample_source", None)
    if callable(describe):
        try:
            return str(describe(sample_index))
        except Exception:
            pass
    return f"dataset_index={sample_index}"


@torch.no_grad()
def _encode_real_video_latents(
    model,
    video: torch.Tensor,
    *,
    tiled: bool,
    vae_device_mode: str,
) -> torch.Tensor:
    """Encode a preprocessed [1,C,T,H,W] real video into the WAM VAE space."""
    if video.ndim != 5:
        raise ValueError(f"Expected real video [B,C,T,H,W], got {tuple(video.shape)}")

    if vae_device_mode == "gpu":
        real = video.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
        latents = model._encode_video_latents(real, tiled=tiled)
    elif vae_device_mode == "cpu":
        real = video.to(device="cpu", dtype=torch.float32)
        latents = model.vae.encode(
            real,
            device="cpu",
            tiled=tiled,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
    else:
        raise ValueError(f"Unsupported vae_device_mode={vae_device_mode!r}")

    if isinstance(latents, list):
        if len(latents) != 1:
            raise RuntimeError(f"Expected one real-video latent tensor, got {len(latents)}")
        latents = latents[0]
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise RuntimeError(
            f"Unexpected VAE latent output: type={type(latents)}, "
            f"shape={getattr(latents, 'shape', None)}"
        )
    return latents.detach().to(device="cpu", dtype=torch.float32)


def _future_metrics(pred_latents: torch.Tensor, real_latents: torch.Tensor) -> dict[str, Any]:
    if pred_latents.shape != real_latents.shape:
        raise ValueError(
            f"Pred/real latent shape mismatch: {tuple(pred_latents.shape)} vs "
            f"{tuple(real_latents.shape)}"
        )
    if pred_latents.ndim != 5 or pred_latents.shape[2] <= 1:
        raise ValueError(
            "Reliability metrics require at least two latent time steps so the "
            f"observed first step can be excluded, got {tuple(pred_latents.shape)}"
        )

    pred_future = pred_latents[:, :, 1:].float()
    real_future = real_latents[:, :, 1:].float()
    pred_flat = pred_future.flatten(1)
    real_flat = real_future.flatten(1)

    cosine_similarity = F.cosine_similarity(pred_flat, real_flat, dim=1, eps=1e-8).mean()
    mse = F.mse_loss(pred_future, real_future)
    mae = F.l1_loss(pred_future, real_future)

    pred_final = pred_future[:, :, -1:].flatten(1)
    real_final = real_future[:, :, -1:].flatten(1)
    final_cosine_similarity = F.cosine_similarity(
        pred_final, real_final, dim=1, eps=1e-8
    ).mean()
    final_mse = F.mse_loss(
        pred_future[:, :, -1:],
        real_future[:, :, -1:],
    )

    per_latent_mse = (
        (pred_future - real_future)
        .square()
        .mean(dim=(0, 1, 3, 4))
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return {
        "cosine_divergence": float((1.0 - cosine_similarity).item()),
        "mse": float(mse.item()),
        "rmse": float(math.sqrt(max(float(mse.item()), 0.0))),
        "mae": float(mae.item()),
        "final_cosine_divergence": float((1.0 - final_cosine_similarity).item()),
        "final_mse": float(final_mse.item()),
        "per_latent_mse": per_latent_mse,
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Fast-WAM imagined future latents against real future latents "
            "using the exact RobotVideoDataset preprocessing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Training config used by the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Fast-WAM checkpoint (.pt).")
    parser.add_argument(
        "--dataset_stats",
        required=True,
        help="stats.json used by the checkpoint; overrides cfg.data.train.pretrained_norm_stats.",
    )
    parser.add_argument(
        "--dataset_dir",
        action="append",
        default=None,
        help="Optional dataset directory override; repeat for multiple datasets.",
    )
    parser.add_argument(
        "--text_embedding_cache_dir",
        default=None,
        help="Optional cached text embedding directory override.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--vae_device_mode", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--sigma_shift", type=float, default=None)
    parser.add_argument("--text_cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=32,
        help="Stride over RobotVideoDataset sample indices; 32 reduces heavy window overlap.",
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--output_npz", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--model_base_path",
        default=None,
        help="Optional DIFFSYNTH_MODEL_BASE_PATH override for Wan/VAE assets.",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    if args.sample_stride <= 0:
        raise ValueError("--sample_stride must be positive.")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)
    stats_path = _resolve_path(args.dataset_stats)
    for path in (config_path, checkpoint_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(_resolve_path(args.model_base_path))

    register_default_resolvers()
    cfg = OmegaConf.load(config_path)
    cfg.model.load_text_encoder = False
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    cfg.data.train.pretrained_norm_stats = str(stats_path)
    cfg.data.train.is_training_set = True
    cfg.data.train.val_set_proportion = 0.0
    if args.dataset_dir:
        cfg.data.train.dataset_dirs = list(args.dataset_dir)
    if args.text_embedding_cache_dir:
        cfg.data.train.text_embedding_cache_dir = str(
            _resolve_path(args.text_embedding_cache_dir)
        )
    _apply_wan_model_overrides(cfg)
    OmegaConf.resolve(cfg)

    precision = args.mixed_precision or str(cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    model.load_checkpoint(str(checkpoint_path))
    model = model.to(args.device).eval()
    model.device = torch.device(args.device)
    model.torch_dtype = model_dtype
    _restore_official_conv3d_forward(model)
    patch_fastwam_encode_input_image_latents_tensor(
        model,
        vae_device_mode=args.vae_device_mode,
    )

    action_conditioned = bool(getattr(model.video_expert, "action_conditioned", False))
    if action_conditioned:
        raise RuntimeError(
            "This diagnostic currently targets the main Fast-WAM setup with "
            "video_dit_config.action_conditioned=false. Action-conditioned world "
            "evaluation must explicitly define which action chunk conditions each future."
        )

    dataset = instantiate(cfg.data.train)
    total = len(dataset)
    if args.start_index < 0 or args.start_index >= total:
        raise IndexError(f"--start_index={args.start_index} outside dataset length {total}")

    rows: list[dict[str, Any]] = []
    per_latent_rows: list[np.ndarray] = []
    skipped_padding = 0
    skipped_error = 0

    index = int(args.start_index)
    while index < total and len(rows) < int(args.max_samples):
        try:
            # Use _get instead of __getitem__: __getitem__ silently substitutes a random
            # sample on error, which is undesirable for a deterministic diagnostic.
            sample = dataset._get(index)
            if _has_padding(sample):
                skipped_padding += 1
                index += int(args.sample_stride)
                continue

            video = sample["video"]
            action = sample["action"]
            proprio = sample["proprio"]
            context = sample["context"]
            context_mask = sample["context_mask"]

            if video.ndim != 4:
                raise ValueError(f"Expected sample video [C,T,H,W], got {tuple(video.shape)}")
            if action.ndim != 2:
                raise ValueError(f"Expected sample action [T,D], got {tuple(action.shape)}")
            if proprio.ndim != 2:
                raise ValueError(f"Expected sample proprio [T,D], got {tuple(proprio.shape)}")

            num_video_frames = int(video.shape[1])
            action_horizon = int(action.shape[0])
            expected_raw_obs = int(cfg.data.train.num_frames)
            ratio = int(cfg.data.train.action_video_freq_ratio)
            expected_video_frames = (expected_raw_obs - 1) // ratio + 1
            if num_video_frames != expected_video_frames:
                raise RuntimeError(
                    f"Temporal alignment mismatch: dataset returned {num_video_frames} video "
                    f"frames, expected {expected_video_frames} from raw_obs={expected_raw_obs}, "
                    f"action_video_freq_ratio={ratio}."
                )
            if action_horizon != expected_raw_obs - 1:
                raise RuntimeError(
                    f"Action alignment mismatch: got horizon={action_horizon}, "
                    f"expected {expected_raw_obs - 1}."
                )

            input_image = video[:, 0].unsqueeze(0).to(
                device=model.device,
                dtype=model.torch_dtype,
            )
            context_b = context.unsqueeze(0).to(
                device=model.device,
                dtype=model.torch_dtype,
            )
            context_mask_b = context_mask.unsqueeze(0).to(
                device=model.device,
                dtype=torch.bool,
            )
            proprio_b = proprio[0].unsqueeze(0).to(
                device=model.device,
                dtype=model.torch_dtype,
            )

            sample_seed = int(args.seed) + int(index)
            with torch.inference_mode():
                pred = model.infer_joint(
                    prompt=None,
                    input_image=input_image,
                    num_video_frames=num_video_frames,
                    action_horizon=action_horizon,
                    action=None,
                    proprio=proprio_b,
                    context=context_b,
                    context_mask=context_mask_b,
                    negative_prompt=None,
                    text_cfg_scale=float(args.text_cfg_scale),
                    num_inference_steps=int(args.num_inference_steps),
                    sigma_shift=args.sigma_shift,
                    seed=sample_seed,
                    rand_device="cpu",
                    tiled=bool(args.tiled),
                    test_action_with_infer_action=False,
                    return_video_latents=True,
                    decode_video=False,
                )
                real_latents = _encode_real_video_latents(
                    model,
                    video.unsqueeze(0),
                    tiled=bool(args.tiled),
                    vae_device_mode=args.vae_device_mode,
                )

            pred_latents = pred["video_latents"]
            metrics = _future_metrics(pred_latents, real_latents)
            per_latent_rows.append(metrics.pop("per_latent_mse"))
            rows.append(
                {
                    "sample_index": int(index),
                    "source": _source_description(dataset, index),
                    "seed": sample_seed,
                    "num_video_frames": num_video_frames,
                    "action_horizon": action_horizon,
                    "latent_shape": list(pred_latents.shape),
                    **metrics,
                }
            )
            print(
                f"[world-reliability] idx={index} "
                f"cos_div={metrics['cosine_divergence']:.6f} "
                f"mse={metrics['mse']:.6f} "
                f"final_cos_div={metrics['final_cosine_divergence']:.6f}"
            )
        except Exception as exc:
            skipped_error += 1
            print(f"[world-reliability] warning: idx={index} failed: {exc}")

        index += int(args.sample_stride)

    if not rows:
        raise RuntimeError(
            "No valid reliability samples were evaluated. Check dataset paths, padding, "
            "start index, stride, and text embedding cache."
        )

    keys = (
        "cosine_divergence",
        "mse",
        "rmse",
        "mae",
        "final_cosine_divergence",
        "final_mse",
    )
    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "dataset_stats": str(stats_path),
        "dataset_length": int(total),
        "evaluated_samples": len(rows),
        "skipped_padding": int(skipped_padding),
        "skipped_error": int(skipped_error),
        "start_index": int(args.start_index),
        "sample_stride": int(args.sample_stride),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "num_inference_steps": int(args.num_inference_steps),
        "action_conditioned": action_conditioned,
        "interpretation": (
            "expectation-vs-reality divergence; current main Fast-WAM video branch "
            "is not action-conditioned"
        ),
        "metrics": {key: _stats([float(row[key]) for row in rows]) for key in keys},
    }

    if args.output_npz:
        output_npz = _resolve_path(args.output_npz)
    else:
        output_npz = (
            PROJECT_ROOT
            / "runs"
            / "world_reliability_eval"
            / f"{checkpoint_path.stem}_world_reliability.npz"
        )
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    per_latent_mse = np.stack(per_latent_rows, axis=0)
    np.savez_compressed(
        output_npz,
        sample_index=np.asarray([row["sample_index"] for row in rows], dtype=np.int64),
        source=np.asarray([row["source"] for row in rows], dtype=str),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        cosine_divergence=np.asarray(
            [row["cosine_divergence"] for row in rows], dtype=np.float32
        ),
        mse=np.asarray([row["mse"] for row in rows], dtype=np.float32),
        rmse=np.asarray([row["rmse"] for row in rows], dtype=np.float32),
        mae=np.asarray([row["mae"] for row in rows], dtype=np.float32),
        final_cosine_divergence=np.asarray(
            [row["final_cosine_divergence"] for row in rows], dtype=np.float32
        ),
        final_mse=np.asarray([row["final_mse"] for row in rows], dtype=np.float32),
        per_latent_mse=per_latent_mse,
        summary_json=np.asarray(json.dumps(summary, ensure_ascii=False)),
    )

    output_json = (
        _resolve_path(args.output_json)
        if args.output_json
        else output_npz.with_suffix(".json")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"summary": summary, "samples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[world-reliability] saved:", output_npz)
    print("[world-reliability] summary:", output_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
