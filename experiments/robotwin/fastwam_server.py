
'''
python experiments/robotwin/fastwam_server.py \
  --checkpoint ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --dataset_stats ./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --host 0.0.0.0 \
  --port 8765 \
  --device cuda \
  --vae_device_mode cpu
'''
import sys
import os
import types
import argparse
import logging
import traceback
from pathlib import Path

# 必须放最前面
os.environ["MODELSCOPE_NO_DOWNLOAD"] = "1"
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/media/jz08/49630fca-f8b9-4c76-a173-2bcf51fee8a9/wam_ckpt/model_components"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from flask import Flask, request, jsonify
import json_numpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _restore_official_conv3d_forward(module: torch.nn.Module):
    for child in module.children():
        if isinstance(child, torch.nn.Conv3d):
            child._conv_forward = torch.nn.Conv3d._conv_forward.__get__(child, torch.nn.Conv3d)
        _restore_official_conv3d_forward(child)


def patch_fastwam_encode_input_image_latents_tensor(model, vae_device_mode: str = "cpu"):
    """
    根据 vae_device_mode 决定：
    - cpu: VAE 首帧编码走 CPU/FP32，再搬回主设备
    - gpu: 保持原始 GPU 路径，不做 CPU fallback
    """
    if vae_device_mode == "gpu":
        logger.info("VAE device mode = gpu, skip CPU fallback patch.")
        return

    if vae_device_mode != "cpu":
        raise ValueError(f"Unsupported vae_device_mode: {vae_device_mode}")

    @torch.no_grad()
    def patched_encode_input_image_latents_tensor(
        self,
        input_image,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        try:
            target_param = next(self.parameters())
            target_device = target_param.device
            target_dtype = target_param.dtype
        except StopIteration:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            target_dtype = torch.bfloat16

        if input_image.dim() != 4:
            raise RuntimeError(
                f"Unexpected input_image shape: {tuple(input_image.shape)}, expected [B, C, H, W]"
            )

        batch_size = input_image.shape[0]
        zs = []

        logger.info(
            f"[CPU fallback] encode_input_image_latents: "
            f"input_shape={tuple(input_image.shape)}, "
            f"input_device={input_image.device}, input_dtype={input_image.dtype}, "
            f"target_device={target_device}, target_dtype={target_dtype}"
        )

        self.vae = self.vae.to(device="cpu", dtype=torch.float32).eval()

        try:
            for i in range(batch_size):
                image = input_image[i].detach().to(device="cpu", dtype=torch.float32).contiguous()

                # [C, H, W] -> [C, 1, H, W]
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

                z = z.to(device=target_device, dtype=target_dtype)
                zs.append(z)

        finally:
            self.vae = self.vae.to(device=target_device, dtype=target_dtype).eval()

        out = zs[0] if len(zs) == 1 else torch.cat(zs, dim=0)

        logger.info(
            f"[CPU fallback] encoded latents: "
            f"shape={tuple(out.shape)}, device={out.device}, dtype={out.dtype}"
        )
        return out

    model._encode_input_image_latents_tensor = types.MethodType(
        patched_encode_input_image_latents_tensor, model
    )
    logger.info("FastWAM image latent encoding CPU fallback enabled.")


class FastWAMServer:
    def __init__(
        self,
        checkpoint_path: str,
        dataset_stats_path: str,
        device: str = "cuda",
        vae_device_mode: str = "cpu",
    ):
        self.device = device
        self.vae_device_mode = vae_device_mode
        logger.info("Loading FastWAM model...")
        self.model = self._load_model(checkpoint_path, dataset_stats_path)
        logger.info("Model loaded successfully!")

    def _load_model(self, checkpoint_path: str, dataset_stats_path: str):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config_name = "sim_robotwin.yaml"
        configs_root = (PROJECT_ROOT / "configs").resolve()
        overrides = ["task=robotwin_uncond_3cam_384_1e-4"]

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            cfg = compose(config_name=config_name, overrides=overrides)

        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg_copy.load_text_encoder = True

        logger.info("Instantiating model from config...")
        model_dtype = _mixed_precision_to_model_dtype("bf16")
        model = instantiate(model_cfg_copy, model_dtype=model_dtype, device="cpu")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model.load_checkpoint(checkpoint_path)

        logger.info(f"Moving model to device: {self.device}")
        model = model.to(self.device).eval()

        model.device = torch.device(self.device)
        try:
            model.torch_dtype = next(model.parameters()).dtype
        except StopIteration:
            pass

        logger.info("Restoring official Conv3d forward...")
        _restore_official_conv3d_forward(model)
        logger.info("Official Conv3d forward restored.")

        logger.info(f"Applying VAE device mode: {self.vae_device_mode}")
        patch_fastwam_encode_input_image_latents_tensor(
            model,
            vae_device_mode=self.vae_device_mode,
        )

        from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self.processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(cfg.EVALUATION.action_horizon) if cfg.EVALUATION.action_horizon is not None else 24
        self.num_inference_steps = int(cfg.EVALUATION.num_inference_steps) if cfg.EVALUATION.num_inference_steps is not None else 20
        self.sigma_shift = cfg.EVALUATION.sigma_shift
        self.text_cfg_scale = float(cfg.EVALUATION.text_cfg_scale) if cfg.EVALUATION.text_cfg_scale is not None else 1.0
        self.negative_prompt = str(cfg.EVALUATION.negative_prompt) if cfg.EVALUATION.negative_prompt is not None else ""
        self.rand_device = str(self.device)
        self.tiled = bool(cfg.EVALUATION.tiled) if cfg.EVALUATION.tiled is not None else False
        self._num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1

        try:
            first_param = next(model.parameters())
            self.model_param_device = first_param.device
            self.model_param_dtype = first_param.dtype
        except StopIteration:
            self.model_param_device = torch.device(self.device)
            self.model_param_dtype = model_dtype

        logger.info(f"Model param device: {self.model_param_device}")
        logger.info(f"Model param dtype : {self.model_param_dtype}")

        return model

    def _prepare_image_tensor(self, image0: np.ndarray, image1: np.ndarray, image2: np.ndarray) -> torch.Tensor:
        from PIL import Image

        head_pil = Image.fromarray(image0)
        left_pil = Image.fromarray(image1)
        right_pil = Image.fromarray(image2)

        head_resized = head_pil.resize((320, 256), Image.BILINEAR)
        left_resized = left_pil.resize((160, 128), Image.BILINEAR)
        right_resized = right_pil.resize((160, 128), Image.BILINEAR)

        head_np = np.array(head_resized, dtype=np.uint8)
        left_np = np.array(left_resized, dtype=np.uint8)
        right_np = np.array(right_resized, dtype=np.uint8)

        bottom = np.concatenate([left_np, right_np], axis=1)
        image = np.concatenate([head_np, bottom], axis=0)

        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()
        image_tensor = image_tensor.to(device=self.model_param_device, dtype=self.model_param_dtype)
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0

        return image_tensor

    def _prepare_proprio_tensor(self, proprio: np.ndarray) -> torch.Tensor:
        state_batch = {
            "state": {
                "default": torch.tensor(np.array(proprio, copy=True), dtype=torch.float32).unsqueeze(0)
            }
        }
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)

        proprio_tensor = state_batch["state"]["default"]
        proprio_tensor = proprio_tensor.to(device=self.model_param_device, dtype=self.model_param_dtype)
        return proprio_tensor

    def _sanity_check_tensors(self, image_tensor: torch.Tensor, proprio_tensor: torch.Tensor):
        if image_tensor.device != self.model_param_device:
            image_tensor = image_tensor.to(self.model_param_device)
        if proprio_tensor.device != self.model_param_device:
            proprio_tensor = proprio_tensor.to(self.model_param_device)

        if image_tensor.dtype != self.model_param_dtype:
            image_tensor = image_tensor.to(dtype=self.model_param_dtype)
        if proprio_tensor.dtype != self.model_param_dtype:
            proprio_tensor = proprio_tensor.to(dtype=self.model_param_dtype)

        return image_tensor, proprio_tensor

    def infer(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        image2: np.ndarray,
        proprio: np.ndarray,
        language_instruction: str,
    ) -> np.ndarray:
        try:
            image_tensor = self._prepare_image_tensor(image0, image1, image2)
            proprio_tensor = self._prepare_proprio_tensor(proprio)
            image_tensor, proprio_tensor = self._sanity_check_tensors(image_tensor, proprio_tensor)

            logger.info(
                f"infer input_image: shape={tuple(image_tensor.shape)}, "
                f"device={image_tensor.device}, dtype={image_tensor.dtype}"
            )
            logger.info(
                f"infer proprio    : shape={tuple(proprio_tensor.shape)}, "
                f"device={proprio_tensor.device}, dtype={proprio_tensor.dtype}"
            )

            from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
            prompt = DEFAULT_PROMPT.format(task=language_instruction)

            with torch.no_grad():
                pred = self.model.infer_action(
                    prompt=prompt,
                    input_image=image_tensor,
                    action_horizon=self.action_horizon,
                    proprio=proprio_tensor,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift,
                    seed=None,
                    rand_device=self.rand_device,
                    tiled=self.tiled,
                    negative_prompt=self.negative_prompt,
                    text_cfg_scale=self.text_cfg_scale,
                )

            action_tensor = pred["action"]

            if action_tensor.dim() == 2:
                action_tensor = action_tensor.unsqueeze(0)
            elif action_tensor.dim() != 3:
                raise RuntimeError(f"Unexpected action tensor shape: {tuple(action_tensor.shape)}")

            normalizer = self.processor.normalizer.normalizers["action"]["default"]
            actions = normalizer.backward(
                action_tensor.to(dtype=torch.float32, device="cpu")
            )[0].numpy()


            return actions.astype(np.float32)

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            logger.error(traceback.format_exc())
            horizon = int(self.action_horizon) if hasattr(self, "action_horizon") else 25
            return np.zeros((horizon, 14), dtype=np.float32)

    def health_info(self):
        return {
            "status": "ok",
            "model_loaded": self.model is not None,
            "vae_device_mode": self.vae_device_mode,
            "model_param_device": str(self.model_param_device),
            "model_param_dtype": str(self.model_param_dtype),
            "model_device_attr": str(getattr(self.model, "device", "<missing>")),
            "model_torch_dtype_attr": str(getattr(self.model, "torch_dtype", "<missing>")),
        }


@app.route("/act", methods=["POST"])
def act():
    try:
        data = request.json

        proprio_np = json_numpy.loads(data["proprio"])
        instruction = data["language_instruction"]
        head_view = json_numpy.loads(data["image0"])
        left_view = json_numpy.loads(data["image1"])
        right_view = json_numpy.loads(data["image2"])

        actions = model.infer(
            image0=head_view,
            image1=left_view,
            image2=right_view,
            proprio=proprio_np,
            language_instruction=instruction,
        )

        return jsonify({
            "action": json_numpy.dumps(actions),
            "status": "success",
        })

    except Exception as e:
        logger.error(f"Error in /act endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "status": "error",
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(model.health_info())


def main():
    global model

    parser = argparse.ArgumentParser(
        description="FastWAM HTTP Server (Compatible with RoboTwin client)"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset_stats", type=str, required=True, help="Path to dataset stats JSON")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda", help="Main device to use")
    parser.add_argument(
        "--vae_device_mode",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="VAE first-frame encoding device mode",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FastWAM Server Starting")
    logger.info("=" * 60)
    logger.info(f"Checkpoint      : {args.checkpoint}")
    logger.info(f"Dataset Stats   : {args.dataset_stats}")
    logger.info(f"Device          : {args.device}")
    logger.info(f"VAE Device Mode : {args.vae_device_mode}")
    logger.info("=" * 60)

    logger.info(f"torch.__version__                = {torch.__version__}")
    logger.info(f"torch.version.cuda              = {torch.version.cuda}")
    logger.info(f"torch.cuda.is_available()       = {torch.cuda.is_available()}")
    logger.info(f"torch.backends.cudnn.available  = {torch.backends.cudnn.is_available()}")
    logger.info(f"torch.backends.cudnn.enabled    = {torch.backends.cudnn.enabled}")
    try:
        logger.info(f"torch.backends.cudnn.version()  = {torch.backends.cudnn.version()}")
    except Exception:
        logger.info("torch.backends.cudnn.version()  = <unavailable>")

    model = FastWAMServer(
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        device=args.device,
        vae_device_mode=args.vae_device_mode,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Health check   : http://{args.host}:{args.port}/health")
    logger.info(f"Action endpoint: http://{args.host}:{args.port}/act")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()