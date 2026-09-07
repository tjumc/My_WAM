#!/usr/bin/env python
"""Load RobotVideoDataset from a train YAML config and print sample shapes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

torch = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instantiate RobotVideoDataset and inspect one or more samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/train/robotwin.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random", action="store_true", help="Sample random indices instead of --index.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--work-dir", default="./runs/dataset_debug")
    parser.add_argument("--batch-size", type=int, default=0, help="If > 0, also test DataLoader collation.")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def summarize_value(name: str, value: Any, indent: int = 0, *, full_print: bool = False):
    prefix = " " * indent
    if isinstance(value, torch.Tensor):
        if full_print:
            print(f"{prefix}{name}: Tensor shape={tuple(value.shape)} dtype={value.dtype}")
            print(value)
            return

        summary = f"{prefix}{name}: Tensor shape={tuple(value.shape)} dtype={value.dtype}"
        if value.numel() > 0 and (value.is_floating_point() or value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)):
            numeric = value.detach()
            if value.dtype == torch.bool:
                numeric = numeric.to(torch.int32)
            summary += (
                f" min={numeric.min().item():.6g}"
                f" max={numeric.max().item():.6g}"
                f" mean={numeric.float().mean().item():.6g}"
            )
        print(summary)
    elif isinstance(value, dict):
        print(f"{prefix}{name}: dict keys={list(value.keys())}")
        for key, child in value.items():
            summarize_value(str(key), child, indent + 2, full_print=full_print)
    elif isinstance(value, (list, tuple)):
        if full_print:
            print(f"{prefix}{name}: {type(value).__name__} len={len(value)} value={value!r}")
            return
        print(f"{prefix}{name}: {type(value).__name__} len={len(value)}")
        for i, child in enumerate(value[:5]):
            summarize_value(f"[{i}]", child, indent + 2, full_print=full_print)
        if len(value) > 5:
            print(f"{prefix}  ...")
    elif isinstance(value, str):
        short = value if full_print or len(value) <= 180 else value[:177] + "..."
        print(f"{prefix}{name}: str len={len(value)} value={short!r}")
    else:
        print(f"{prefix}{name}: {type(value).__name__} value={value!r}")


def print_sample(sample: dict[str, Any]):
    print(f"sample: dict keys={list(sample.keys())}")
    for key, value in sample.items():
        summarize_value(key, value, indent=2, full_print=(key not in {"video", "context"}))


def main():
    global torch
    args = parse_args()

    import datasets as _hf_datasets
    import torch as torch_mod
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from fastwam.utils import misc
    from fastwam.utils.config_resolvers import register_default_resolvers

    torch = torch_mod
    _hf_datasets.disable_caching()
    register_default_resolvers()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    misc.register_work_dir(args.work_dir)

    dataset_cfg = cfg.data[args.split]
    print(f"[dataset-test] config={args.config}")
    print(f"[dataset-test] split={args.split}")
    print(f"[dataset-test] target={dataset_cfg.get('_target_', '<missing>')}")
    print(f"[dataset-test] dataset_dirs={list(dataset_cfg.get('dataset_dirs', []))}")
    print(f"[dataset-test] pretrained_norm_stats={dataset_cfg.get('pretrained_norm_stats')}")
    print(f"[dataset-test] text_embedding_cache_dir={dataset_cfg.get('text_embedding_cache_dir')}")

    dataset = instantiate(dataset_cfg)
    print(f"[dataset-test] dataset_class={dataset.__class__.__module__}.{dataset.__class__.__name__}")
    print(f"[dataset-test] len={len(dataset)}")

    if len(dataset) <= 0:
        raise RuntimeError("Dataset is empty.")

    rng = random.Random(0)
    for sample_no in range(args.num_samples):
        idx = rng.randrange(len(dataset)) if args.random else args.index + sample_no
        idx %= len(dataset)
        print(f"\n[dataset-test] sample index={idx}")
        sample = dataset[idx]
        print_sample(sample)

    if args.batch_size > 0:
        print(f"\n[dataset-test] DataLoader batch_size={args.batch_size} num_workers={args.num_workers}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        batch = next(iter(loader))
        summarize_value("batch", batch)


if __name__ == "__main__":
    main()
