"""Replay captured observations with paired seeds and different cached prompts.

No robot SDK, RPC client, or motion command is used. Metrics measure changes
in predicted actions, not task success, refusal, or semantic understanding.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from wam_diagnostics import load_observation, sha256_file

GROUPS = {"torso": slice(3, 7), "left_arm": slice(7, 14), "right_arm": slice(15, 22)}


def load_prompts(path):
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or len(prompts) < 2:
        raise ValueError("Expected a JSON list of at least two {id, text} objects; first is baseline")
    ids = set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or any(
            not isinstance(prompt.get(k), str) or not prompt[k].strip() for k in ("id", "text")
        ):
            raise ValueError("Each prompt needs nonempty string id and text")
        if prompt["id"] in ids:
            raise ValueError("Prompt IDs must be unique")
        ids.add(prompt["id"])
    return prompts


def action_metrics(action, baseline):
    if action.shape != baseline.shape or action.ndim != 2 or action.shape[1] != 25 or not len(action):
        raise ValueError("Expected matching nonempty [T,25] action arrays")
    if not np.isfinite(action).all() or not np.isfinite(baseline).all():
        raise ValueError("Cannot score nonfinite actions")
    metrics = {}
    for name, index in GROUPS.items():
        delta = action[:, index].astype(np.float64) - baseline[:, index]
        metrics[name + "_paired_rmse_rad"] = float(np.sqrt(np.mean(delta**2)))
        metrics[name + "_first_step_max_delta_rad"] = float(np.max(np.abs(delta[0])))
    return metrics


def run_matrix(policy, observations, prompts, seeds, set_prompt, output):
    """Injectable policy interface also supports CPU-only behavioral tests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = None
        for oi, path in enumerate(observations):
            obs = load_observation(path)
            baseline = {}
            for pi, prompt in enumerate(prompts):
                set_prompt(prompt["text"])
                for seed in seeds:
                    policy.seed = seed
                    # A processor may transform an input array in place; every
                    # prompt/seed must still receive the same recorded input.
                    action = policy.infer(time.time(), **{k: v.copy() for k, v in obs.items()})
                    if action is None:
                        raise RuntimeError(f"Inference failed: observation={oi}, prompt={prompt['id']}, seed={seed}")
                    action = np.asarray(action)
                    if pi == 0:
                        baseline[seed] = action.copy()
                    metrics = action_metrics(action, baseline[seed])
                    filename = f"obs_{oi:04d}_prompt_{pi:02d}_seed_{seed}.npz"
                    np.savez(output / filename, actions=action)
                    row = {"observation_index": oi, "observation": str(path),
                           "prompt_id": prompt["id"], "seed": seed,
                           "artifact": filename, **metrics}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, nargs="+", required=True,
                        help="Captured request/infer_result NPZ files, not action-only NPZ files")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    parser.add_argument("--action-mode", required=True,
                        help="Copy the deployed server action_mode exactly")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--vae-device-mode", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate inputs/paths only; does not load model or validate text caches")
    args = parser.parse_args()
    prompts = load_prompts(args.prompts)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Seeds must be unique integers in [0, 2**32)")
    if args.output.exists():
        parser.error("Output directory must not already exist")
    for path in (args.config, args.checkpoint, args.dataset_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.text_cache_dir.is_dir():
        raise FileNotFoundError(args.text_cache_dir)
    for path in args.observations:
        load_observation(path)
    print(f"{len(args.observations)} observations x {len(prompts)} prompts x {len(args.seeds)} seeds")
    if args.validate_only:
        print("Input validation passed; model and text-cache validation not performed.")
        return

    # Import model dependencies only after CPU preflight. This module starts no server.
    from fastwam_portal_server import (AstribotFastWAMPolicy, DEFAULT_PROMPT,
                                      _load_text_context_cache, _setup_logging)
    from fastwam.utils.config_resolvers import register_default_resolvers
    from omegaconf import OmegaConf
    _setup_logging()
    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    context_len = int(cfg.data.train.get("context_len", cfg.model.get("tokenizer_max_len", 128)))
    caches, cache_manifest = {}, {}
    for prompt in prompts:
        ctx, mask, path = _load_text_context_cache(
            args.text_cache_dir.resolve(), DEFAULT_PROMPT.format(task=prompt["text"]), context_len)
        caches[prompt["text"]] = (ctx, mask, path)
        cache_manifest[prompt["id"]] = {"path": str(path), "sha256": sha256_file(path)}
    os.environ["TEXT_CACHE_DIR"] = str(args.text_cache_dir.resolve())
    policy = AstribotFastWAMPolicy(
        config_path=str(args.config.resolve()), checkpoint_path=str(args.checkpoint.resolve()),
        dataset_stats_path=str(args.dataset_stats.resolve()), task_prompt=prompts[0]["text"],
        device=args.device, mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon, n_action_steps=None, action_mode=args.action_mode,
        num_inference_steps=args.num_inference_steps, sigma_shift=args.sigma_shift,
        text_cfg_scale=1.0, negative_prompt="", rand_device=args.rand_device,
        tiled=args.tiled, vae_device_mode=args.vae_device_mode, seed=args.seeds[0])

    def set_prompt(text):
        ctx, mask, path = caches[text]
        policy.task_prompt, policy.prompt = text, DEFAULT_PROMPT.format(task=text)
        policy.context = ctx.to(device=policy.model.device, dtype=policy.model.torch_dtype)
        policy.context_mask = mask.to(device=policy.model.device)
        policy.context_cache_path = path

    manifest = {"schema_version": 1, "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items() if k != "observations"},
                "model": policy.get_model_info(), "prompts": prompts,
                "text_caches": cache_manifest,
                "config_sha256": sha256_file(args.config),
                "stats_sha256": sha256_file(args.dataset_stats),
                "server_source_sha256": sha256_file(Path(__file__).with_name("fastwam_portal_server.py")),
                "checkpoint_size": args.checkpoint.stat().st_size,
                "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
                "observations": [{"path": str(p), "sha256": sha256_file(p)} for p in args.observations],
                "notes": ["Paired seeds reduce sampling confounds; GPU determinism is not guaranteed.",
                          "Action change is not task success or semantic correctness.",
                          "Use the same observations for expert/shared models; preserve each model's own stats."]}
    try:
        rows = run_matrix(policy, args.observations, prompts, args.seeds, set_prompt, args.output)
        manifest.update(status="complete", comparisons=len(rows))
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if args.output.is_dir():
            (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        policy.trace.close()
    print(f"Saved {len(rows)} predictions and paired metrics to {args.output}")


if __name__ == "__main__":
    main()
