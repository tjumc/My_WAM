#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline FastWAM inference smoke test for Astribot real-robot inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.astribot.fastwam_portal_server import AstribotFastWAMPolicy, _setup_logging

DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "astribot_washclothes_posttrain32_from_step10k"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "train" / "config.yaml"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "train" / "checkpoints" / "weights" / "step_050000.pt"
DEFAULT_DATASET_STATS = DEFAULT_RUN_DIR / "stats.json"


def _make_uint8_pattern(shape: tuple[int, int, int], offset: int) -> np.ndarray:
    h, w, c = shape
    if c != 3:
        raise ValueError(f"Expected RGB shape, got {shape}")
    y = np.arange(h, dtype=np.uint16)[:, None]
    x = np.arange(w, dtype=np.uint16)[None, :]
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = (x + offset) % 256
    img[..., 1] = (y + 2 * offset) % 256
    img[..., 2] = ((x // 2 + y // 2 + 3 * offset) % 256).astype(np.uint8)
    return img


def make_fake_real_obs(state_dim: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    img0 = _make_uint8_pattern((720, 1280, 3), offset=7)
    img1 = _make_uint8_pattern((360, 640, 3), offset=31)
    img2 = _make_uint8_pattern((360, 640, 3), offset=61)

    # Vel-mode Astribot state is chassis velocity + torso/arms/grippers, flattened to 25D.
    state = np.zeros((state_dim,), dtype=np.float32)
    state[: min(3, state_dim)] = rng.normal(loc=0.0, scale=0.005, size=min(3, state_dim))
    if state_dim > 3:
        state[3:] = rng.normal(loc=0.0, scale=0.02, size=state_dim - 3)
    return img0, img1, img2, state


def load_npz_obs(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    missing = [key for key in ("img0", "img1", "img2", "state") if key not in data]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}; expected img0/img1/img2/state")
    return data["img0"], data["img1"], data["img2"], data["state"].astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one offline FastWAM inference with real-robot-shaped Astribot inputs."
    )
    parser.add_argument("--config", default=os.environ.get("CONFIG", str(DEFAULT_CONFIG)))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    parser.add_argument("--dataset_stats", default=os.environ.get("DATASET_STATS", str(DEFAULT_DATASET_STATS)))
    parser.add_argument(
        "--task",
        default=os.environ.get("TASK_PROMPT", "put the clothes into the washing machine"),
    )
    parser.add_argument("--input_npz", default=None, help="Optional npz with img0/img1/img2/state.")
    parser.add_argument("--output_npz", default=None, help="Optional path to save inputs and predicted actions.")
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
    parser.add_argument("--vae_device_mode", choices=["gpu", "cpu"], default=os.environ.get("VAE_DEVICE_MODE", "gpu"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fake_obs_seed", type=int, default=0)
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
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.model_base_path

    policy = AstribotFastWAMPolicy(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        task_prompt=args.task,
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

    if args.input_npz:
        img0, img1, img2, state = load_npz_obs(args.input_npz)
        source = args.input_npz
    else:
        img0, img1, img2, state = make_fake_real_obs(policy.raw_state_dim, seed=args.fake_obs_seed)
        source = "synthetic fake real-shaped observation"

    print("[offline] input source:", source)
    print("[offline] model_info:", json.dumps(policy.get_model_info(), ensure_ascii=False, indent=2))
    print("[offline] img shapes:", img0.shape, img1.shape, img2.shape, "state:", state.shape)

    t0 = time.time()
    actions = policy.infer(t0, img0, img1, img2, state, his_effort=None, delay=0)
    if actions is None:
        raise RuntimeError("FastWAM inference returned None; check logs above.")

    finite = bool(np.isfinite(actions).all())
    print("[offline] actions shape:", actions.shape)
    print("[offline] actions finite:", finite)
    print(
        "[offline] actions stats:",
        f"min={float(actions.min()):+.6f}",
        f"max={float(actions.max()):+.6f}",
        f"mean={float(actions.mean()):+.6f}",
        f"std={float(actions.std()):+.6f}",
    )
    print("[offline] first action:", np.array2string(actions[0], precision=4, suppress_small=False))

    expected_shape = (policy.action_horizon, policy.raw_action_dim)
    if actions.shape != expected_shape:
        raise RuntimeError(f"Bad action shape: expected {expected_shape}, got {actions.shape}")
    if not finite:
        raise RuntimeError("Action output contains NaN/Inf.")

    if args.output_npz:
        output_path = Path(args.output_npz)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            img0=img0,
            img1=img1,
            img2=img2,
            state=state,
            actions=actions,
        )
        print("[offline] saved:", output_path)

    print("[offline] OK")


if __name__ == "__main__":
    main()
