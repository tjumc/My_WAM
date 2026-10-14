#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine history-ambiguous state pairs from Fast-WAM / LeRobot trajectories.

Goal:
    Find pairs from the SAME episode where
      1) current multi-camera observations are visually very similar,
      2) current proprio states are similar,
      3) GT future action chunks are different.

This is only a pair-mining diagnostic. It does NOT load or run Fast-WAM.
Candidates are not proof of history dependence or action failure. State/action
RMSE retains the original normalized model dimensions (including padding).
Temporal deduplication reduces near-repeat pairs, not statistical dependence.

Example:
    python tools/find_history_ambiguity_pairs.py \
        --config configs/train/astribot_posttrain32.yaml \
        --dataset_stats /path/to/stats.json \
        --dataset_dir /path/to/dataset \
        --output_dir outputs/history_ambiguity \
        --max_samples 256 \
        --samples_per_episode 64 \
        --min_frame_gap 150 \
        --max_proprio_rmse 0.5 \
        --dedup_frame_gap 30 \
        --top_k 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find visually similar current states with divergent GT action chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--dataset_stats", required=True)
    p.add_argument(
        "--dataset_dir",
        action="append",
        default=None,
        help="Override cfg.data.train.dataset_dirs; repeat for multiple datasets.",
    )
    p.add_argument("--output_dir", default="outputs/history_ambiguity")

    # Sampling
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=5000)
    p.add_argument("--samples_per_episode", type=int, default=64)
    p.add_argument(
        "--sample_stride",
        type=int,
        default=4,
        help="Candidate start-frame stride before random subsampling within each episode.",
    )
    p.add_argument(
        "--min_frame_gap",
        type=int,
        default=150,
        help="Minimum raw-frame distance between two starts in the same episode.",
    )

    # Visual feature = low-resolution pooled robotwin image.
    # This deliberately avoids introducing another pretrained vision model.
    p.add_argument("--feature_h", type=int, default=24)
    p.add_argument("--feature_w", type=int, default=20)

    # Exploratory threshold; not a calibrated definition of identical state.
    p.add_argument(
        "--max_proprio_rmse",
        type=float,
        default=0.5,
        help="Keep pairs whose current normalized proprio RMSE is <= this value.",
    )
    p.add_argument(
        "--min_action_rmse",
        type=float,
        default=0.0,
        help="Keep pairs whose normalized GT action-chunk RMSE is >= this value.",
    )
    p.add_argument(
        "--candidate_k_per_episode",
        type=int,
        default=200,
        help="Keep this many most visually similar candidate pairs per episode before global ranking.",
    )
    p.add_argument(
        "--dedup_frame_gap", type=int, default=30,
        help=("Suppress a worse pair when BOTH sorted frame endpoints are within "
              "this raw-frame radius of a retained pair in the same episode. "
              "Applied BEFORE candidate_k_per_episode; 0 disables deduplication."),
    )
    p.add_argument("--log_every", type=int, default=50,
                   help="Print loading progress every N attempted windows.")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument(
        "--save_pair_images",
        type=int,
        default=50,
        help="Save side-by-side PNGs for the first N output pairs; 0 disables.",
    )
    return p.parse_args()


def has_padding(sample: dict[str, Any]) -> bool:
    for key in ("image_is_pad", "action_is_pad", "proprio_is_pad"):
        value = sample.get(key)
        if value is not None and bool(torch.as_tensor(value).any().item()):
            return True
    return False


def valid_episode_starts(dataset, raw_window: int) -> tuple[np.ndarray, np.ndarray]:
    raw = dataset.lerobot_dataset
    ep_index = raw.episode_data_index
    ep_from = torch.as_tensor(ep_index["from"]).cpu().numpy().astype(np.int64)
    ep_to = torch.as_tensor(ep_index["to"]).cpu().numpy().astype(np.int64)

    if ep_from.shape != ep_to.shape:
        raise RuntimeError(f"episode_data_index mismatch: {ep_from.shape} vs {ep_to.shape}")
    return ep_from, ep_to


def choose_indices(
    ep_from: np.ndarray,
    ep_to: np.ndarray,
    *,
    raw_window: int,
    sample_stride: int,
    samples_per_episode: int,
    max_samples: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Return (global_idx, episode_ordinal, frame_offset) tuples."""
    rng = np.random.default_rng(seed)
    episode_order = np.arange(len(ep_from))
    rng.shuffle(episode_order)

    selected: list[tuple[int, int, int]] = []
    for ep in episode_order:
        start = int(ep_from[ep])
        end = int(ep_to[ep])  # exclusive
        valid_last = end - raw_window
        if valid_last < start:
            continue

        candidates = np.arange(start, valid_last + 1, sample_stride, dtype=np.int64)
        if candidates.size == 0:
            continue

        take = min(samples_per_episode, int(candidates.size))
        picks = rng.choice(candidates, size=take, replace=False)
        picks = np.sort(np.asarray(picks, dtype=np.int64))

        for idx in picks:
            selected.append((int(idx), int(ep), int(idx - start)))
            if len(selected) >= max_samples:
                return selected
    return selected


def source_description(dataset, idx: int) -> str:
    describe = getattr(dataset.lerobot_dataset, "_describe_sample_source", None)
    if callable(describe):
        try:
            return str(describe(idx))
        except Exception:
            pass
    return f"global_idx={idx}"


def visual_feature(current_image: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    current_image: [C,H,W], Fast-WAM normalized to [-1,1].
    Return a flattened [0,1] low-resolution feature.
    """
    x = current_image.float().unsqueeze(0)
    x = (x + 1.0) * 0.5
    x = F.adaptive_avg_pool2d(x, (h, w))
    return x.flatten().cpu()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a[None], b[None], dim=1, eps=1e-8).item())


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


def load_records(dataset, selected, feature_h: int, feature_w: int, log_every: int = 50):
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    records = []
    skipped_padding = 0
    skipped_error = 0

    for n, (idx, episode, frame_offset) in enumerate(selected):
        try:
            # main() also disables BaseLerobotDataset's inner random retry.
            # _get alone does NOT bypass that inner retry.
            sample = dataset._get(idx)
            if has_padding(sample):
                skipped_padding += 1
                continue

            video = sample["video"]       # [C,T,H,W], exact RobotVideoDataset preprocessing
            action = sample["action"]     # [H_action,D]
            proprio = sample["proprio"]   # [H_action,D]

            if video.ndim != 4 or action.ndim != 2 or proprio.ndim != 2:
                raise ValueError(
                    f"Unexpected shapes video={tuple(video.shape)}, "
                    f"action={tuple(action.shape)}, proprio={tuple(proprio.shape)}"
                )

            if not all(bool(torch.isfinite(t).all()) for t in (video, action, proprio)):
                raise ValueError("Non-finite image, action or proprio values")
            # Clone the first-frame view: otherwise its storage retains the
            # ENTIRE future video window for every sampled record.
            current = video[:, 0].detach().cpu().clone()
            records.append(
                {
                    "idx": int(idx),
                    "episode": int(episode),
                    "frame_offset": int(frame_offset),
                    "image": current,
                    "visual_feature": visual_feature(current, feature_h, feature_w),
                    "proprio": proprio[0].detach().float().cpu().clone(),
                    "action": action.float().cpu(),
                    "source": source_description(dataset, idx),
                }
            )
        except Exception as exc:
            skipped_error += 1
            print(f"[mine] warning idx={idx}: {type(exc).__name__}: {exc}")

        finally:
            if (n + 1) % log_every == 0 or n + 1 == len(selected):
                print(
                    f"[mine] loaded {n + 1}/{len(selected)} windows; "
                    f"valid={len(records)} padding={skipped_padding} errors={skipped_error}",
                    flush=True,
                )

    return records, skipped_padding, skipped_error


def _pair_sort_key(pair):
    left, right = sorted((pair["frame_a"], pair["frame_b"]))
    return (pair["visual_rmse"], pair["episode"], left, right)


def deduplicate_temporal_pairs(pairs, frame_radius: int):
    """Greedy, score-ordered suppression in the two-endpoint time plane.

    (264,820) and (264,824) are near repeats with radius=30.
    (264,820) and (264,1220) are NOT: sharing just one endpoint is allowed.
    Different episodes never suppress one another. Radius=0 disables this.
    """
    if frame_radius < 0:
        raise ValueError("dedup_frame_gap must be nonnegative")
    ordered = sorted(pairs, key=_pair_sort_key)
    if frame_radius == 0:
        return ordered, 0
    kept = []
    endpoints_by_episode = defaultdict(list)
    for pair in ordered:
        left, right = sorted((pair["frame_a"], pair["frame_b"]))
        previous = endpoints_by_episode[pair["episode"]]
        if any(abs(left - a) <= frame_radius and abs(right - b) <= frame_radius
               for a, b in previous):
            continue
        kept.append(pair)
        previous.append((left, right))
    return kept, len(ordered) - len(kept)


def mine_pairs(
    records,
    *,
    min_frame_gap: int,
    max_proprio_rmse: float,
    min_action_rmse: float,
    candidate_k_per_episode: int,
    dedup_frame_gap: int = 30,
):
    """Filter -> visual ranking -> temporal dedup -> per-episode cap.

    Returns candidates plus stage counts; never relaxes thresholds to fill K.
    Proprio quantiles cover ALL time-gap-valid pairs, before the proprio gate,
    not only the visually selected Top-K. This helps calibrate later runs.
    """
    by_episode = defaultdict(list)
    for rec in records:
        by_episode[rec["episode"]].append(rec)

    count_keys = (
        "all_pairs", "after_frame_gap", "after_proprio", "after_action",
        "after_temporal_dedup", "after_episode_limit", "removed_temporal_duplicates",
    )
    totals = dict.fromkeys(count_keys, 0)
    episode_stats = []
    proprio_distances = []
    candidates = []
    for episode, unsorted_recs in sorted(by_episode.items()):
        recs = sorted(unsorted_recs, key=lambda r: r["frame_offset"])
        counts = dict.fromkeys(count_keys, 0)
        counts["all_pairs"] = len(recs) * (len(recs) - 1) // 2
        episode_pairs = []
        if len(recs) >= 2:
            feats = torch.stack([r["visual_feature"] for r in recs]).float()
            visual_distances = torch.cdist(feats, feats, p=2) / math.sqrt(feats.shape[1])
            for i, a in enumerate(recs):
                for j in range(i + 1, len(recs)):
                    b = recs[j]
                    if b["frame_offset"] - a["frame_offset"] < min_frame_gap:
                        continue
                    counts["after_frame_gap"] += 1
                    p_rmse = rmse(a["proprio"], b["proprio"])
                    proprio_distances.append(p_rmse)
                    if p_rmse > max_proprio_rmse:
                        continue
                    counts["after_proprio"] += 1
                    a_rmse = rmse(a["action"], b["action"])
                    if a_rmse < min_action_rmse:
                        continue
                    counts["after_action"] += 1
                    episode_pairs.append({
                        "episode": int(episode),
                        "idx_a": a["idx"], "idx_b": b["idx"],
                        "frame_a": a["frame_offset"], "frame_b": b["frame_offset"],
                        "frame_gap": b["frame_offset"] - a["frame_offset"],
                        "visual_rmse": float(visual_distances[i, j].item()),
                        "visual_cosine": cosine_similarity(a["visual_feature"], b["visual_feature"]),
                        "proprio_rmse": p_rmse,
                        "action_rmse": a_rmse,
                        "action_first_step_rmse": rmse(a["action"][0], b["action"][0]),
                        "action_endpoint_rmse": rmse(a["action"][-1], b["action"][-1]),
                        "source_a": a["source"], "source_b": b["source"],
                        "_rec_a": a, "_rec_b": b,
                    })

        unique, removed = deduplicate_temporal_pairs(episode_pairs, dedup_frame_gap)
        counts["removed_temporal_duplicates"] = removed
        counts["after_temporal_dedup"] = len(unique)
        limited = unique[:candidate_k_per_episode]
        counts["after_episode_limit"] = len(limited)
        candidates.extend(limited)
        episode_stats.append({"episode": int(episode), "samples": len(recs), **counts})
        for key in count_keys:
            totals[key] += counts[key]
        print(
            f"[mine] ep={episode} pairs={counts['all_pairs']} "
            f"gap={counts['after_frame_gap']} proprio={counts['after_proprio']} "
            f"action={counts['after_action']} unique={len(unique)} "
            f"near_duplicates_removed={removed} retained={len(limited)}",
            flush=True,
        )

    quantiles = {}
    if proprio_distances:
        values = np.quantile(proprio_distances, [0, .1, .25, .5, .75, .9, 1])
        quantiles = dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"),
                             (float(x) for x in values)))
    candidates.sort(key=_pair_sort_key)
    return candidates, {
        "pair_filter_counts": totals,
        "episode_pair_counts": episode_stats,
        "proprio_rmse_after_frame_gap_quantiles": quantiles,
    }


def save_outputs(output_dir: Path, pairs, top_k: int, save_pair_images: int, summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pairs[:top_k]

    csv_fields = [
        "rank",
        "episode",
        "idx_a",
        "idx_b",
        "frame_a",
        "frame_b",
        "frame_gap",
        "visual_rmse",
        "visual_cosine",
        "proprio_rmse",
        "action_rmse",
        "action_first_step_rmse",
        "action_endpoint_rmse",
        "source_a",
        "source_b",
    ]

    csv_path = output_dir / "pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rank, pair in enumerate(pairs, start=1):
            row = {k: pair[k] for k in csv_fields if k != "rank"}
            row["rank"] = rank
            writer.writerow(row)

    json_rows = []
    for rank, pair in enumerate(pairs, start=1):
        row = {k: v for k, v in pair.items() if not k.startswith("_")}
        row["rank"] = rank
        json_rows.append(row)

    json_path = output_dir / "pairs.json"
    json_path.write_text(
        json.dumps({"summary": summary, "pairs": json_rows}, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    image_dir = output_dir / "pair_images"
    image_dir.mkdir(exist_ok=True)
    if save_pair_images > 0 and pairs:
        from torchvision.utils import save_image
    for rank, pair in enumerate(pairs[:save_pair_images], start=1):
        a = (pair["_rec_a"]["image"].float() + 1.0) * 0.5
        b = (pair["_rec_b"]["image"].float() + 1.0) * 0.5
        canvas = torch.cat([a.clamp(0, 1), b.clamp(0, 1)], dim=-1)
        save_image(canvas, image_dir / f"pair_{rank:04d}.png")

    print(f"[mine] wrote {csv_path}")
    print(f"[mine] wrote {json_path}")
    if save_pair_images > 0:
        print(f"[mine] wrote pair previews to {image_dir}")


def main():
    args = parse_args()
    for name in ("max_samples", "sample_stride", "min_frame_gap",
                 "candidate_k_per_episode", "top_k", "feature_h", "feature_w", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.samples_per_episode < 2 or args.max_samples < 2:
        raise ValueError("max_samples and samples_per_episode must be >= 2")
    if args.dedup_frame_gap < 0:
        raise ValueError("--dedup_frame_gap must be nonnegative")
    if args.save_pair_images < 0:
        raise ValueError("--save_pair_images must be nonnegative")
    if math.isnan(args.max_proprio_rmse) or args.max_proprio_rmse < 0:
        raise ValueError("--max_proprio_rmse must be nonnegative (inf disables)")
    if not math.isfinite(args.min_action_rmse) or args.min_action_rmse < 0:
        raise ValueError("--min_action_rmse must be finite and nonnegative")

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    import fastwam.datasets.lerobot.base_lerobot_dataset as base

    # Enforce strict identity even when this Python script is invoked directly.
    # This changes only the current process, not repository training code.
    base.MAX_GETITEM_ATTEMPT = 1
    register_default_resolvers()
    cfg = OmegaConf.load(Path(args.config).expanduser().resolve())

    stats_path = Path(args.dataset_stats).expanduser().resolve()
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    # Only instantiate the existing training dataset pipeline.
    cfg.data.train.pretrained_norm_stats = str(stats_path)
    cfg.data.train.is_training_set = False
    cfg.data.train.val_set_proportion = 0.0
    cfg.data.train.skip_padding_as_possible = False

    if args.dataset_dir:
        cfg.data.train.dataset_dirs = [
            str(Path(p).expanduser().resolve()) for p in args.dataset_dir
        ]

    # Pair mining does not need language. These values avoid requiring the task
    # instruction/text-cache environment variables just to inspect trajectories.
    cfg.data.train.override_instruction = "history ambiguity mining"
    cfg.data.train.text_embedding_cache_dir = "/tmp/fastwam_unused_text_cache"

    if int(cfg.data.train.get("global_sample_stride", 1)) != 1:
        raise ValueError("This diagnostic requires global_sample_stride=1 for frame alignment")
    dataset = instantiate(cfg.data.train)

    # RobotVideoDataset._get() otherwise reads cached T5 embeddings, even though
    # this script only needs video/action/proprio. Replace that one private helper
    # with a tiny dummy so the rest of _get() stays EXACTLY the same.
    def _dummy_text_context(_prompt: str):
        return (
            torch.zeros((dataset.context_len, 1), dtype=torch.float32),
            torch.ones((dataset.context_len,), dtype=torch.bool),
        )

    dataset._get_cached_text_context = _dummy_text_context

    raw_window = int(cfg.data.train.num_frames)
    ep_from, ep_to = valid_episode_starts(dataset, raw_window)
    selected = choose_indices(
        ep_from,
        ep_to,
        raw_window=raw_window,
        sample_stride=args.sample_stride,
        samples_per_episode=args.samples_per_episode,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print(
        f"[mine] episodes={len(ep_from)} selected_starts={len(selected)} "
        f"raw_window={raw_window} sampled_episodes={len({ep for _, ep, _ in selected})}"
    )
    print(f"[mine] proprio_max={args.max_proprio_rmse} action_min={args.min_action_rmse} "
          f"frame_gap={args.min_frame_gap} dedup_radius={args.dedup_frame_gap}")

    records, skipped_padding, skipped_error = load_records(
        dataset,
        selected,
        feature_h=args.feature_h,
        feature_w=args.feature_w,
        log_every=args.log_every,
    )
    print(
        f"[mine] usable_records={len(records)} "
        f"skipped_padding={skipped_padding} skipped_error={skipped_error}"
    )

    if not records:
        raise RuntimeError("No valid samples; inspect data loading errors before drawing conclusions")

    pairs, mining_stats = mine_pairs(
        records,
        min_frame_gap=args.min_frame_gap,
        max_proprio_rmse=args.max_proprio_rmse,
        min_action_rmse=args.min_action_rmse,
        candidate_k_per_episode=args.candidate_k_per_episode,
        dedup_frame_gap=args.dedup_frame_gap,
    )
    print("[mine] filter_counts:", json.dumps(mining_stats["pair_filter_counts"]))
    print("[mine] proprio RMSE quantiles (all gap-valid pairs):",
          json.dumps(mining_stats["proprio_rmse_after_frame_gap_quantiles"]))

    summary = {
        "diagnostic_version": 2,
        "config": str(Path(args.config).expanduser().resolve()),
        "dataset_stats": str(stats_path),
        "selected_samples": len(selected),
        "usable_samples": len(records),
        "skipped_padding": skipped_padding,
        "skipped_error": skipped_error,
        "candidate_pairs_after_filters": len(pairs),
        "min_frame_gap": args.min_frame_gap,
        "max_proprio_rmse": args.max_proprio_rmse if math.isfinite(args.max_proprio_rmse) else None,
        "proprio_filter_enabled": math.isfinite(args.max_proprio_rmse),
        "dedup_frame_gap": args.dedup_frame_gap,
        "seed": args.seed,
        "sample_stride": args.sample_stride,
        "samples_per_episode": args.samples_per_episode,
        "sampled_episode_count": len({ep for _, ep, _ in selected}),
        "saved_pairs": min(args.top_k, len(pairs)),
        "candidate_k_per_episode": args.candidate_k_per_episode,
        "top_k": args.top_k,
        **mining_stats,
        "min_action_rmse": args.min_action_rmse,
        "visual_feature_hw": [args.feature_h, args.feature_w],
        "note": (
            "visual_rmse is pooled-pixel RMSE on the exact current robotwin image; "
            "proprio/action metrics retain normalized model dimensions including padding. "
            "A null max_proprio_rmse means disabled. Dedup suppresses pairs only when both "
            "frame endpoints are close in the same episode; it does not prove independence. "
            "Thresholds are exploratory; these candidates do not establish history dependence."
        ),
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    save_outputs(
        output_dir,
        pairs,
        top_k=args.top_k,
        save_pair_images=args.save_pair_images,
        summary=summary,
    )

    if pairs:
        print("\nTop pairs:")
        for rank, p in enumerate(pairs[: min(10, args.top_k)], start=1):
            print(
                f"  #{rank:02d} ep={p['episode']} "
                f"frames=({p['frame_a']},{p['frame_b']}) "
                f"visual_rmse={p['visual_rmse']:.5f} "
                f"prop_rmse={p['proprio_rmse']:.5f} "
                f"action_rmse={p['action_rmse']:.5f}"
            )
    else:
        print("[mine] no pairs survived the current filters.")


if __name__ == "__main__":
    main()
