#!/usr/bin/env python3
"""
Pre-compute dataset normalization stats and save to JSON.
Run this ONCE before training to avoid stats computation during distributed training.

Usage:
    # Use Hydra task config (same way as training)
    python scripts/precompute_stats.py task=pick_place_1e-4 --output runs/pick_place_stats.json

    # Override dataset dir
    python scripts/precompute_stats.py task=pick_place_1e-4 data.train.dataset_dirs=[/data/Midea_DATa] --output runs/pick_place_stats.json
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf


def _parse_output_arg():
    """Extract --output from sys.argv before Hydra sees it."""
    output = "runs/dataset_stats.json"
    argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        else:
            argv.append(sys.argv[i])
            i += 1
    sys.argv = argv
    return output


output_path = _parse_output_arg()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
    from hydra.utils import instantiate

    train_cfg = cfg.data.train

    dataset_dirs = list(train_cfg.dataset_dirs)
    shape_meta   = OmegaConf.to_container(train_cfg.shape_meta, resolve=True)
    action_size  = int(train_cfg.get("action_size", 1))
    obs_size     = int(train_cfg.get("obs_size", 2))

    print(f"[INFO] dataset_dirs : {dataset_dirs}")
    print(f"[INFO] output       : {output_path}")

    # Build dataset (single process, no distributed)
    dataset = BaseLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        action_size=action_size,
        obs_size=obs_size,
        val_set_proportion=0.0,
        is_training_set=True,
    )

    # Build processor
    processor_cfg = train_cfg.get("processor", None)
    if processor_cfg is not None:
        processor = instantiate(processor_cfg)
        processor.train()
    else:
        print("[WARN] No processor config found; stats will be raw (no transform).")
        processor = None

    # Compute stats
    episodes_num = dataset.multi_dataset.num_episodes
    print(f"[INFO] Computing stats over {episodes_num} episodes...")
    stats = dataset.get_dataset_stats(processor)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(out))
    print(f"[DONE] Stats saved to: {out}")
    print(f"       Use in training: data.train.pretrained_norm_stats={out} data.val.pretrained_norm_stats={out}")


if __name__ == "__main__":
    main()

