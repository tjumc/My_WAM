#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastWAM Portal server for Astribot real-robot inference.

This keeps the same Portal RPC surface used by pi0_astribot/evaluate/infer_client_vel.py:
  - infer(curtime, img0, img1, img2, state, his_effort=None, delay=0)
  - get_model_info()
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

try:
    import portal
except ModuleNotFoundError:
    portal = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.config_resolvers import register_default_resolvers

LOGGER = logging.getLogger("astribot_fastwam_server")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision={mixed_precision!r}; expected no/fp16/bf16.")


def _restore_official_conv3d_forward(module: torch.nn.Module) -> None:
    for child in module.children():
        if isinstance(child, torch.nn.Conv3d):
            child._conv_forward = torch.nn.Conv3d._conv_forward.__get__(child, torch.nn.Conv3d)
        _restore_official_conv3d_forward(child)


def patch_fastwam_encode_input_image_latents_tensor(model: torch.nn.Module, vae_device_mode: str = "gpu") -> None:
    """Keep first-frame VAE encoding on CPU to reduce GPU memory pressure."""
    if vae_device_mode == "gpu":
        return
    if vae_device_mode != "cpu":
        raise ValueError(f"Unsupported vae_device_mode={vae_device_mode!r}")

    model.vae = model.vae.to(device="cpu", dtype=torch.float32).eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    @torch.no_grad()
    def patched_encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ):
        target_device = torch.device(getattr(self, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        target_dtype = getattr(self, "torch_dtype", torch.bfloat16)

        if input_image.dim() != 4:
            raise RuntimeError(f"Unexpected input_image shape: {tuple(input_image.shape)}, expected [B,C,H,W]")

        zs = []
        for i in range(input_image.shape[0]):
            image = input_image[i].detach().to(device="cpu", dtype=torch.float32).contiguous()
            video = image.unsqueeze(1).contiguous()
            z = self.vae.encode(
                [video],
                device="cpu",
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if not isinstance(z, torch.Tensor):
                raise RuntimeError(f"VAE encode returned non-tensor type: {type(z)}")
            zs.append(z.to(device=target_device, dtype=target_dtype))

        return zs[0] if len(zs) == 1 else torch.cat(zs, dim=0)

    model._encode_input_image_latents_tensor = types.MethodType(
        patched_encode_input_image_latents_tensor,
        model,
    )


def _resolve_path(path: str | os.PathLike[str], *, base: Path = PROJECT_ROOT) -> Path:
    out = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if not out.is_absolute():
        out = base / out
    return out.resolve()


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a bool-like value, got {value!r}")


def _apply_wan_model_overrides(cfg) -> None:
    """Cloud train configs may not contain the local tokenizer repo layout used at inference."""
    model_id = os.environ.get("WAN22_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")
    tokenizer_model_id = os.environ.get("WAN22_TOKENIZER_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")
    redirect_common_files = _env_bool("WAN22_REDIRECT_COMMON_FILES")

    LOGGER.info("Use Wan model_id=%s", model_id)
    cfg.model.model_id = model_id
    LOGGER.info("Use Wan tokenizer_model_id=%s", tokenizer_model_id)
    cfg.model.tokenizer_model_id = tokenizer_model_id
    if redirect_common_files is not None:
        LOGGER.info(
            "Override cfg.model.redirect_common_files from WAN22_REDIRECT_COMMON_FILES=%s",
            redirect_common_files,
        )
        cfg.model.redirect_common_files = redirect_common_files


def _load_text_context_cache(cache_dir: str | os.PathLike[str], prompt: str, context_len: int):
    """Load precomputed text context so real-robot inference can skip the text encoder."""
    cache_dir = _resolve_path(cache_dir)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache: {cache_path}. "
            "Set TEXT_CACHE_DIR or precompute this prompt before real-robot inference."
        )

    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.ndim != 2:
        raise ValueError(f"Cached `context` must be 2D [L,D], got {tuple(context.shape)} in {cache_path}")
    if context_mask.ndim != 1:
        raise ValueError(f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}")
    if context.shape[0] != context_len:
        raise ValueError(
            f"Cached context_len mismatch: expected {context_len}, got {context.shape[0]} in {cache_path}"
        )
    if context_mask.shape[0] != context_len:
        raise ValueError(
            f"Cached mask_len mismatch: expected {context_len}, got {context_mask.shape[0]} in {cache_path}"
        )

    context = context.to(dtype=torch.bfloat16).contiguous()
    # Match RobotVideoDataset: zero padded text tokens, then allow attention over the full fixed length.
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask, dtype=torch.bool)
    return context, context_mask, cache_path


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC image with 3/4 channels, got shape={arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32, copy=False)
        if arr_f.size > 0 and float(np.nanmax(arr_f)) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_np(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(_as_rgb_uint8(image)).resize((width, height), Image.BILINEAR), dtype=np.uint8)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(_as_rgb_uint8(image))
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


class AstribotFastWAMPolicy:
    def __init__(
        self,
        *,
        config_path: str,
        checkpoint_path: str,
        dataset_stats_path: str,
        task_prompt: str,
        device: str,
        mixed_precision: str | None,
        action_horizon: int | None,
        n_action_steps: int | None,
        action_mode: str,
        num_inference_steps: int,
        sigma_shift: float | None,
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        vae_device_mode: str,
        seed: int | None,
    ):
        register_default_resolvers()
        self.config_path = _resolve_path(config_path)
        self.checkpoint_path = _resolve_path(checkpoint_path)
        self.dataset_stats_path = _resolve_path(dataset_stats_path)
        self.task_prompt = str(task_prompt)
        self.action_mode = str(action_mode)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.seed = seed

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.dataset_stats_path.exists():
            raise FileNotFoundError(f"Dataset stats not found: {self.dataset_stats_path}")

        cfg = OmegaConf.load(self.config_path)
        cfg.model.load_text_encoder = False
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.action_dit_pretrained_path = None
        _apply_wan_model_overrides(cfg)
        OmegaConf.resolve(cfg)
        self.cfg = cfg

        self.prompt = DEFAULT_PROMPT.format(task=self.task_prompt)
        text_cache_dir = os.environ.get("TEXT_CACHE_DIR") or cfg.data.train.get("text_embedding_cache_dir")
        if text_cache_dir is None:
            raise ValueError("TEXT_CACHE_DIR or cfg.data.train.text_embedding_cache_dir is required.")
        context_len = int(cfg.data.train.get("context_len", cfg.model.get("tokenizer_max_len", 128)))
        self.context, self.context_mask, self.context_cache_path = _load_text_context_cache(
            text_cache_dir,
            self.prompt,
            context_len,
        )

        precision = mixed_precision or str(cfg.get("mixed_precision", "bf16"))
        model_dtype = _mixed_precision_to_model_dtype(precision)

        LOGGER.info("Instantiating FastWAM from %s", self.config_path)
        self.model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
        LOGGER.info("Loading WAM checkpoint: %s", self.checkpoint_path)
        self.model.load_checkpoint(str(self.checkpoint_path))
        self.model = self.model.to(device).eval()
        self.model.device = torch.device(device)
        self.model.torch_dtype = model_dtype
        self.context = self.context.to(device=self.model.device, dtype=self.model.torch_dtype)
        self.context_mask = self.context_mask.to(device=self.model.device)
        _restore_official_conv3d_forward(self.model)
        patch_fastwam_encode_input_image_latents_tensor(self.model, vae_device_mode=vae_device_mode)
        LOGGER.info("Loaded cached text context: %s", self.context_cache_path)

        self.processor = instantiate(cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(self.dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.image_meta = list(self.processor.shape_meta["images"])
        self.state_meta = list(self.processor.shape_meta["state"])
        self.action_meta = list(self.processor.shape_meta["action"])
        if len(self.state_meta) != 1 or len(self.action_meta) != 1:
            raise ValueError("Astribot real inference expects one state key and one action key.")
        self.state_key = self.state_meta[0]["key"]
        self.action_key = self.action_meta[0]["key"]
        self.raw_state_dim = int(self.state_meta[0]["raw_shape"])
        self.raw_action_dim = int(self.action_meta[0]["raw_shape"])
        self.model_action_dim = int(cfg.data.train.processor.action_output_dim)
        self.model_proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
        default_horizon = int(cfg.data.train.num_frames) - 1
        self.action_horizon = int(action_horizon if action_horizon is not None else default_horizon)
        self.n_action_steps = int(n_action_steps if n_action_steps is not None else self.action_horizon)
        self.concat_multi_camera = str(cfg.data.train.get("concat_multi_camera", "robotwin"))
        self.video_size = list(cfg.data.train.get("video_size", [384, 320]))

        LOGGER.info(
            "FastWAM ready: raw_state_dim=%d raw_action_dim=%d model_dims=(proprio=%d, action=%d) "
            "horizon=%d n_action_steps=%d action_mode=%s",
            self.raw_state_dim,
            self.raw_action_dim,
            self.model_proprio_dim,
            self.model_action_dim,
            self.action_horizon,
            self.n_action_steps,
            self.action_mode,
        )

    def _prepare_image_tensor(self, img0: np.ndarray, img1: np.ndarray, img2: np.ndarray) -> torch.Tensor:
        # Keep this layout identical to RobotVideoDataset concat_multi_camera='robotwin'.
        if self.concat_multi_camera == "robotwin":
            cam_top = _resize_np(img0, width=320, height=256)
            cam_left = _resize_np(img1, width=160, height=128)
            cam_right = _resize_np(img2, width=160, height=128)
            bottom = np.concatenate([cam_left, cam_right], axis=1)
            mosaic = np.concatenate([cam_top, bottom], axis=0)
        else:
            images = [img0, img1, img2]
            resized = []
            for image, meta in zip(images, self.image_meta):
                _, h, w = meta["shape"]
                resized.append(_resize_np(image, width=int(w), height=int(h)))
            if self.concat_multi_camera == "horizontal":
                mosaic = np.concatenate(resized, axis=1)
            elif self.concat_multi_camera == "vertical":
                mosaic = np.concatenate(resized, axis=0)
            elif self.concat_multi_camera in {"none", "None", "null"}:
                mosaic = resized[0]
            else:
                raise ValueError(f"Unsupported concat_multi_camera={self.concat_multi_camera!r}")

        # RobotVideoDataset applies a final resize/crop after camera concatenation.
        mosaic = _center_crop_resize(mosaic, width=int(self.video_size[1]), height=int(self.video_size[0]))

        x = torch.from_numpy(mosaic.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()
        x = x.to(device=self.model.device, dtype=self.model.torch_dtype)
        return x * (2.0 / 255.0) - 1.0

    def _prepare_proprio_tensor(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.raw_state_dim:
            raise ValueError(f"Expected state dim {self.raw_state_dim}, got {state.shape[0]}")

        batch = {"state": {self.state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        # WAM was trained with ConcatLeftAlign padding from raw 25D to model 32D.
        batch = self.processor.action_state_merger.forward(batch)
        proprio = batch["state"]
        if proprio.shape[-1] != self.model_proprio_dim:
            raise RuntimeError(f"Bad proprio dim after preprocessing: {tuple(proprio.shape)}")
        return proprio.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action = action.to(dtype=torch.float32, device="cpu")
        dummy_state = torch.zeros(
            (action.shape[0], action.shape[1], self.model_proprio_dim),
            dtype=torch.float32,
        )
        batch = {"action": action, "state": dummy_state}
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            for transform in reversed(self.processor.action_state_transforms):
                batch = transform.backward(batch)
        out = batch["action"][self.action_key][0].numpy()
        return np.ascontiguousarray(out[:, : self.raw_action_dim], dtype=np.float32)

    def infer(
        self,
        curtime: float,
        img0: np.ndarray,
        img1: np.ndarray,
        img2: np.ndarray,
        state: np.ndarray,
        his_effort: np.ndarray | None = None,
        delay: float = 0,
    ) -> np.ndarray | None:
        start = time.time()
        try:
            image = self._prepare_image_tensor(img0, img1, img2)
            proprio = self._prepare_proprio_tensor(state)

            with torch.inference_mode():
                pred = self.model.infer_action(
                    prompt=None,
                    context=self.context,
                    context_mask=self.context_mask,
                    input_image=image,
                    action_horizon=self.action_horizon,
                    proprio=proprio,
                    negative_prompt=None,
                    text_cfg_scale=1.0,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift,
                    seed=self.seed,
                    rand_device=self.rand_device,
                    tiled=self.tiled,
                )
            actions = self._denormalize_action(pred["action"])
            latency_ms = (time.time() - start) * 1000.0
            transport_ms = (start - float(np.asarray(curtime).reshape(-1)[0])) * 1000.0
            LOGGER.info(
                "infer done: actions=%s latency=%.1fms transport_delay=%.1fms client_delay=%s",
                actions.shape,
                latency_ms,
                transport_ms,
                delay,
            )
            return actions
        except Exception as exc:
            LOGGER.error("WAM inference failed: %s", exc)
            LOGGER.error(traceback.format_exc())
            return None

    def get_model_info(self) -> dict[str, Any]:
        return {
            "use_effort": False,
            "model_name": "fastwam",
            "action_mode": self.action_mode,
            "chunk_size": self.action_horizon,
            "n_action_steps": self.n_action_steps,
            "action_dim": self.raw_action_dim,
            "state_dim": self.raw_state_dim,
            "num_inference_steps": self.num_inference_steps,
            "checkpoint": str(self.checkpoint_path),
            "dataset_stats": str(self.dataset_stats_path),
            "text_context_cache": str(self.context_cache_path),
            "task_prompt": self.task_prompt,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a FastWAM Portal server compatible with Astribot infer_client_vel.py."
    )
    parser.add_argument("--config", default="configs/train/astribot_posttrain32.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to WAM .pt weight checkpoint.")
    parser.add_argument("--dataset_stats", required=True, help="Path to stats.json used by this checkpoint.")
    parser.add_argument("--task", required=True, help="Natural-language task prompt used for policy conditioning.")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument(
        "--action_mode",
        default="cmd_absolute_joint",
        help="Reported to the robot client; include 'vel' only if the checkpoint was trained with chassis velocity.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--sigma_shift", type=float, default=None)
    parser.add_argument("--text_cfg_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--rand_device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--vae_device_mode", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model_base_path",
        default=None,
        help="Optional DIFFSYNTH_MODEL_BASE_PATH override for Wan/VAE/text-encoder assets.",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    if portal is None:
        raise ModuleNotFoundError(
            "The `portal` package is required for Astribot RPC. "
            "Please run this server in an environment where `import portal` works."
        )
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(_resolve_path(args.model_base_path))

    LOGGER.info("Starting Astribot FastWAM Portal server on port %d", args.port)
    LOGGER.info("Config        : %s", _resolve_path(args.config))
    LOGGER.info("Checkpoint    : %s", _resolve_path(args.checkpoint))
    LOGGER.info("Dataset stats : %s", _resolve_path(args.dataset_stats))
    LOGGER.info("Task prompt   : %s", args.task)
    LOGGER.info("Model base    : %s", os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "<env not set>"))

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

    server = portal.Server(args.port)
    server.bind("infer", policy.infer)
    server.bind("get_model_info", policy.get_model_info)
    LOGGER.info("Portal server ready. Waiting for robot client...")
    server.start()


if __name__ == "__main__":
    main()
