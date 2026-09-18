#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Force current repository's src/ to have highest priority.
src_str = str(SRC_ROOT)
if src_str in sys.path:
    sys.path.remove(src_str)
sys.path.insert(0, src_str)

print(f"[PATH CHECK] REPO_ROOT = {REPO_ROOT}")
print(f"[PATH CHECK] SRC_ROOT  = {SRC_ROOT}")

import argparse
import inspect

from hydra.utils import instantiate
from omegaconf import OmegaConf

import fastwam
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

print(f"[PATH CHECK] fastwam = {fastwam.__file__}")
print(f"[PATH CHECK] RobotVideoDataset = {inspect.getfile(RobotVideoDataset)}")
print(f"[PATH CHECK] signature = {inspect.signature(RobotVideoDataset.__init__)}")

def main():
    parser = argparse.ArgumentParser(
        description="Export the exact FastWAM normalization stats used by a training config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training yaml, e.g. configs/train/robotwin_proxy_videolambda0.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/robotwin_proxy_norm_stats.json",
        help="Output FastWAM-format normalization stats JSON.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print("=" * 80)
    print(f"Config : {config_path}")
    print(f"Output : {output_path}")
    print("=" * 80)

    # Load exactly the training YAML used by FastWAM.
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    train_cfg = cfg.data.train

    print("\n[Config normalization settings]")
    print(
        "  norm_stats_source:",
        train_cfg.get("norm_stats_source", "meta (default)"),
    )
    print(
        "  pretrained_norm_stats:",
        train_cfg.get("pretrained_norm_stats", None),
    )

    print("\n[1/3] Instantiating training dataset...")
    
    dataset = instantiate(train_cfg)

    if not hasattr(dataset, "lerobot_dataset"):
        raise RuntimeError(
            f"Expected RobotVideoDataset, got {type(dataset).__name__}"
        )

    base_dataset = dataset.lerobot_dataset

    if base_dataset.processors_by_dataset is None:
        raise RuntimeError(
            "processors_by_dataset has not been initialized. "
            "The dataset processor was not configured correctly."
        )

    num_datasets = len(base_dataset.processors_by_dataset)

    print(f"\n[2/3] Found {num_datasets} underlying LeRobot dataset(s).")

    if num_datasets != 1:
        print(
            "\nWARNING: More than one underlying dataset was found. "
            "FastWAM may use separate normalization statistics for each dataset."
        )
        print(
            "This exporter currently expects the RoboTwin proxy setup to contain "
            "one underlying dataset."
        )
        raise RuntimeError(
            f"Expected exactly 1 underlying dataset, got {num_datasets}"
        )

    raw_dataset = base_dataset.multi_dataset._datasets[0]
    processor = base_dataset.processors_by_dataset[0]

    # Report the normalization source that the dataset actually selected.
    actual_source = base_dataset._get_norm_stats_source_for_dataset(raw_dataset)

    print("\n[Actual normalization source]")
    print(f"  dataset root      : {raw_dataset.root}")
    print(f"  norm_stats_source : {actual_source}")

    if actual_source == "meta":
        print(
            "  -> Training used the LeRobot dataset's meta/stats.json "
            "(converted internally to FastWAM format)."
        )
    elif actual_source == "pretrained":
        print(
            "  -> Training used pretrained_norm_stats."
        )
    else:
        print(f"  -> Source: {actual_source}")

    # IMPORTANT:
    # Do NOT call processor.normalizer.get_stats() here.
    #
    # That returns the already-selected global statistics with keys like
    # mean/std, while FastWAMServer.set_normalizer_from_stats() expects the
    # original FastWAM stats dictionary containing:
    #
    # global_mean / global_std / global_min / ...
    #
    # processor.normalizer.stats is exactly the stats dictionary passed
    # to LinearNormalizer during dataset construction.
    stats = processor.normalizer.stats

    if "state" not in stats or "action" not in stats:
        raise RuntimeError(
            f"Invalid FastWAM stats format. Top-level keys: {list(stats.keys())}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[3/3] Saving FastWAM-format stats...")
    save_dataset_stats_to_json(stats, str(output_path))

    print("\nExport completed.")
    print(f"Saved to: {output_path}")

    print("\n[Summary]")
    print(f"  source : {actual_source}")
    print(f"  state keys  : {list(stats['state'].keys())}")
    print(f"  action keys : {list(stats['action'].keys())}")

    for kind in ("state", "action"):
        print(f"\n  {kind}:")
        for key, field_stats in stats[kind].items():
            print(f"    {key}: {list(field_stats.keys())}")

    print("\nExpected FastWAM fields include:")
    print("  global_min")
    print("  global_max")
    print("  global_mean")
    print("  global_std")
    print("  global_q01")
    print("  global_q99")

if __name__ == "__main__":
    main()