#!/usr/bin/env python
"""Iterate RobotVideoDataset from a train YAML config for cache/decode checks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instantiate RobotVideoDataset and iterate it without printing samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/train/post_train.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--work-dir", default="./runs/dataset_iter_debug")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=0, help="If > 0, iterate with DataLoader batches.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--simulate-world-size",
        type=int,
        default=None,
        help="Sequentially simulate this many ranks in one process. If unset, uses WORLD_SIZE/RANK env.",
    )
    parser.add_argument("--disable-text-cache", action="store_true")
    parser.add_argument("--disable-video-cache", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def touch_tensors(sample, torch):
    for key in ("video", "action", "proprio", "context", "context_mask"):
        if isinstance(sample, dict):
            value = sample.get(key)
        else:
            value = None
        if isinstance(value, torch.Tensor):
            _ = tuple(value.shape)


def infer_batch_count(batch, torch) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.ndim > 0 else 1
    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
    return 1


def iterate_indices(dataset, indices, *, epoch, epochs, rank_label, torch, tqdm, stop_on_error):
    ok = 0
    errors = 0
    progress = tqdm(indices, desc=f"{rank_label} epoch {epoch}/{epochs}", dynamic_ncols=True)
    for idx in progress:
        try:
            sample = dataset[idx]
            touch_tensors(sample, torch)
            ok += 1
        except Exception as exc:
            errors += 1
            progress.write(f"[dataset-iter][ERROR] {rank_label} epoch={epoch} idx={idx}: {type(exc).__name__}: {exc}")
            if stop_on_error:
                raise
        progress.set_postfix(ok=ok, errors=errors)
    return ok, errors


def iterate_loader(dataset, indices, *, args, epoch, rank, world_size, torch, tqdm):
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, indices)
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        **loader_kwargs,
    )
    rank_label = f"rank {rank}/{world_size}"
    ok = 0
    errors = 0
    progress = tqdm(
        range(len(loader)),
        desc=f"{rank_label} epoch {epoch}/{args.epochs}",
        dynamic_ncols=True,
        disable=(args.simulate_world_size is None and rank != 0),
    )
    iterator = iter(loader)
    for _ in progress:
        try:
            batch = next(iterator)
            touch_tensors(batch, torch)
            ok += infer_batch_count(batch, torch)
        except StopIteration:
            break
        except Exception as exc:
            errors += 1
            progress.write(f"[dataset-iter][ERROR] {rank_label} epoch={epoch}: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise
            break
        progress.set_postfix(ok=ok, errors=errors)
    return ok, errors


def main():
    args = parse_args()

    if args.disable_video_cache:
        os.environ["FASTWAM_VIDEO_DECODER_CACHE_SIZE"] = "0"

    import datasets as _hf_datasets
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from fastwam.utils import misc
    from fastwam.utils.config_resolvers import register_default_resolvers

    _hf_datasets.disable_caching()
    register_default_resolvers()
    misc.register_work_dir(args.work_dir)

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    dataset_cfg = cfg.data[args.split]
    if args.disable_text_cache:
        dataset_cfg.text_context_cache_size = 0

    print(f"[dataset-iter] config={args.config}")
    print(f"[dataset-iter] split={args.split}")
    print(f"[dataset-iter] disable_text_cache={args.disable_text_cache}")
    print(f"[dataset-iter] video_cache_size={os.environ.get('FASTWAM_VIDEO_DECODER_CACHE_SIZE', '<default>')}")
    print(f"[dataset-iter] target={dataset_cfg.get('_target_', '<missing>')}")
    print(f"[dataset-iter] text_embedding_cache_dir={dataset_cfg.get('text_embedding_cache_dir')}")

    dataset = instantiate(dataset_cfg)
    dataset_len = len(dataset)
    if dataset_len <= 0:
        raise RuntimeError("Dataset is empty.")

    start_index = max(int(args.start_index), 0)
    if start_index >= dataset_len:
        raise ValueError(f"--start-index must be < len(dataset), got {start_index} >= {dataset_len}")
    samples_per_epoch = dataset_len - start_index
    if args.max_samples is not None:
        samples_per_epoch = min(samples_per_epoch, max(int(args.max_samples), 0))

    print(f"[dataset-iter] dataset_class={dataset.__class__.__module__}.{dataset.__class__.__name__}")
    print(f"[dataset-iter] len={dataset_len}")
    print(f"[dataset-iter] start_index={start_index}")
    print(f"[dataset-iter] samples_per_epoch={samples_per_epoch}")
    print(f"[dataset-iter] epochs={args.epochs}")
    print(f"[dataset-iter] batch_size={args.batch_size}")
    print(f"[dataset-iter] num_workers={args.num_workers}")

    all_indices = list(range(start_index, start_index + samples_per_epoch))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    if args.simulate_world_size is not None:
        world_size = int(args.simulate_world_size)
        ranks = list(range(world_size))
        print(f"[dataset-iter] simulate_world_size={world_size}")
    else:
        world_size = env_world_size
        ranks = [env_rank]
        print(f"[dataset-iter] world_size={world_size} rank={env_rank}")

    total_ok = 0
    total_errors = 0
    wall_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        ok = 0
        errors = 0
        for rank in ranks:
            rank_indices = all_indices[rank::world_size]
            if args.batch_size > 0:
                cur_ok, cur_errors = iterate_loader(
                    dataset,
                    rank_indices,
                    args=args,
                    epoch=epoch + 1,
                    rank=rank,
                    world_size=world_size,
                    torch=torch,
                    tqdm=tqdm,
                )
            else:
                cur_ok, cur_errors = iterate_indices(
                    dataset,
                    rank_indices,
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    rank_label=f"rank {rank}/{world_size}",
                    torch=torch,
                    tqdm=tqdm,
                    stop_on_error=args.stop_on_error,
                )
            ok += cur_ok
            errors += cur_errors

        total_ok += ok
        total_errors += errors
        elapsed = time.perf_counter() - epoch_start
        speed = ok / elapsed if elapsed > 0 else 0.0
        print(
            f"[dataset-iter] epoch={epoch + 1}/{args.epochs} "
            f"ok={ok} errors={errors} elapsed={elapsed:.2f}s speed={speed:.2f} sample/s"
        )

    total_elapsed = time.perf_counter() - wall_start
    total_speed = total_ok / total_elapsed if total_elapsed > 0 else 0.0
    print(
        f"[dataset-iter] done ok={total_ok} errors={total_errors} "
        f"elapsed={total_elapsed:.2f}s speed={total_speed:.2f} sample/s"
    )
    if total_errors:
        raise RuntimeError(f"Dataset iteration finished with {total_errors} errors.")


if __name__ == "__main__":
    main()
