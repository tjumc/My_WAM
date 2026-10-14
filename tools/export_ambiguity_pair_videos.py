#!/usr/bin/env python3
"""Human review videos for My_WAM ambiguity pairs; no inference or robot I/O.

Read the REAL episode/frame/timestamp from source_a/source_b, verify it against
Parquet, and add each camera's LeRobot v3 from_timestamp before video decoding.
Each panel keeps its own absolute time but shares a relative clock (CURRENT=0).
Missing pre/post context is labelled, never borrowed from another episode.
The display uses RGB resizing and the robotwin layout, not bit-exact training
preprocessing. Future frames are for human review, never policy conditioning.

Only new replay files are written. Mining scripts, pairs.json, and their output
path settings are never modified. Run scripts/export_ambiguity_pair_videos.sh.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def parse_source(pair: dict, side: str, dataset_dir: str | None = None) -> dict:
    """Use saved source identity, not the miner's global episode ordinal."""
    text = pair.get(f"source_{side}")
    if not isinstance(text, str):
        raise ValueError(f"Missing source_{side}; use the original mining pairs.json")
    fields = dict(re.findall(r"(?:^|, )([A-Za-z_]\w*)=(.*?)(?=, [A-Za-z_]\w*=|$)", text))
    required = ("dataset_root", "episode_index", "frame_index", "timestamp")
    if any(name not in fields for name in required):
        raise ValueError(f"source_{side} must contain {required}")
    root = Path(dataset_dir or fields["dataset_root"]).expanduser().resolve()
    episode, frame = int(fields["episode_index"]), int(fields["frame_index"])
    timestamp = float(fields["timestamp"])
    if episode < 0 or frame < 0 or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"Invalid source_{side} identity")
    if frame != int(pair[f"frame_{side}"]):
        raise ValueError(f"source_{side} frame_index != saved frame_{side}; inspect mining index alignment")
    return {"root": str(root), "episode": episode, "frame": frame, "timestamp": timestamp}


def select_pairs(payload: dict, ranks: list[int]) -> list[dict]:
    by_rank = {}
    for pair in payload.get("pairs", []):
        rank = int(pair["rank"])
        if rank in by_rank:
            raise ValueError(f"Duplicate rank {rank} in pairs.json")
        by_rank[rank] = pair
    requested = list(dict.fromkeys(ranks))
    if not requested or any(rank <= 0 or rank not in by_rank for rank in requested):
        raise ValueError(f"Requested ranks {requested}; available ranks: {sorted(by_rank)}")
    return [by_rank[rank] for rank in requested]


def relative_grid(before: float, after: float, fps: int) -> np.ndarray:
    if fps <= 0 or any(not math.isfinite(x) or x < 0 for x in (before, after)):
        raise ValueError("Durations must be finite/nonnegative and FPS positive")
    # Integer playback steps guarantee one exact CURRENT sample.
    left, right = int(math.floor(before * fps + 1e-9)), int(math.floor(after * fps + 1e-9))
    return np.arange(-left, right + 1, dtype=np.float64) / fps


def locate_frames(frames: np.ndarray, timestamps: np.ndarray, source: dict,
                  relative: np.ndarray, fps: float, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) == 0 or len(frames) != len(timestamps):
        raise ValueError("Empty or inconsistent episode frame metadata")
    if np.any(np.diff(frames) <= 0) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Episode frame indices/timestamps must be unique and increasing")
    center = int(np.searchsorted(frames, source["frame"]))
    if center == len(frames) or int(frames[center]) != source["frame"]:
        raise ValueError("Saved current frame is absent from the episode Parquet")
    if abs(float(timestamps[center]) - source["timestamp"]) > tolerance:
        raise ValueError("Saved timestamp differs from Parquet; check dataset version/mount")
    targets = source["frame"] + np.rint(relative * fps).astype(np.int64)
    inside = (targets >= frames[0]) & (targets <= frames[-1])
    rows = np.full(len(targets), -1, dtype=np.int64)
    positions = np.searchsorted(frames, targets[inside])
    if np.any(frames[positions] != targets[inside]):
        raise ValueError("Missing frame inside the episode; will not silently substitute")
    rows[inside] = positions
    return targets, rows


def video_location(meta: Any, episode: int, camera: str) -> tuple[Path, float, float | None]:
    record = meta.episodes[episode]
    field = f"videos/{camera}/from_timestamp"
    major = int(str(meta.info["codebase_version"]).lstrip("v").split(".")[0])
    if major >= 3 and field not in record:
        raise ValueError(f"Missing v3 {field}; refusing to assume zero video offset")
    offset = float(record.get(field, 0.0))
    end = record.get(f"videos/{camera}/to_timestamp")
    end = None if end is None else float(end)
    if not math.isfinite(offset) or offset < 0 or (end is not None and (not math.isfinite(end) or end <= offset)):
        raise ValueError("Invalid per-camera episode video interval")
    return Path(meta.root) / meta.get_video_file_path(episode, camera), offset, end


class EpisodeReader:
    def __init__(self, root: str, camera_names: list[str]):
        from fastwam.datasets.lerobot.lerobot.lerobot_dataset import (
            LeRobotDatasetMetadata, canonical_camera_key,
        )
        class LocalMetadata(LeRobotDatasetMetadata):
            def pull_from_repo(self, *args, **kwargs):
                raise FileNotFoundError("Local dataset metadata is incomplete; automatic download is disabled")
        self.root = Path(root)
        if not (self.root / "meta/info.json").is_file():
            raise FileNotFoundError(f"No meta/info.json: {self.root}")
        self.meta = LocalMetadata(repo_id=str(self.root), root=self.root)
        self.fps = float(self.meta.fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Dataset FPS must be positive")
        self.cameras = []
        for name in camera_names:
            matches = [key for key in self.meta.video_keys
                       if canonical_camera_key(key) == canonical_camera_key(name)]
            if len(matches) != 1:
                raise ValueError(f"Camera {name}: expected one match, got {matches}")
            self.cameras.append(matches[0])
        self.cache = {}

    def episode_rows(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        if episode not in self.cache:
            import pyarrow.dataset as pads
            if episode not in self.meta.episodes:
                raise ValueError(f"Episode {episode} is absent from metadata")
            # Filter actual IDs; do not rely on physical row order or global indices.
            table = pads.dataset(str(self.root / "data"), format="parquet").to_table(
                columns=["frame_index", "timestamp"],
                filter=pads.field("episode_index") == episode,
            )
            frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
            timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
            order = np.argsort(frames, kind="stable")
            frames, timestamps = frames[order], timestamps[order]
            length = int(self.meta.episodes[episode]["length"])
            if not np.array_equal(frames, np.arange(length, dtype=np.int64)):
                raise ValueError(f"Episode {episode} incomplete/duplicated Parquet frame indices")
            self.cache[episode] = frames, timestamps
        return self.cache[episode]

    def read(self, source: dict, relative: np.ndarray, tolerance: float, backend: str) -> tuple[list, dict]:
        from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
        frames, timestamps = self.episode_rows(source["episode"])
        targets, rows = locate_frames(frames, timestamps, source, relative, self.fps, tolerance)
        valid_positions = np.flatnonzero(rows >= 0)
        # Three small images per time step; no full-resolution episode held in RAM.
        mosaics = [Image.new("RGB", (320, 384)) if row >= 0 else None for row in rows]
        camera_info = []
        for camera, size, xy in zip(self.cameras, [(320, 256), (160, 128), (160, 128)],
                                    [(0, 0), (0, 256), (160, 256)]):
            path, offset, end = video_location(self.meta, source["episode"], camera)
            if not path.is_file():
                raise FileNotFoundError(path)
            queries = timestamps[rows[valid_positions]] + offset
            if np.any(queries < offset - tolerance) or (end is not None and np.any(queries >= end + tolerance)):
                raise ValueError(f"Query would leave episode video interval: {camera}")
            print(f"[replay] ep={source['episode']} {camera}: {len(queries)} frames, video_offset={offset:.6f}s", flush=True)
            decoded = decode_video_frames(path, queries.tolist(), tolerance, backend)
            if decoded.ndim != 4 or decoded.shape[0] != len(queries) or decoded.shape[1] != 3:
                raise ValueError(f"Unexpected decoded shape: {tuple(decoded.shape)}")
            for k, position in enumerate(valid_positions):
                arr = decoded[k].detach().cpu().permute(1, 2, 0).numpy()
                if not np.isfinite(arr).all():
                    raise ValueError("Decoded image contains nonfinite pixels")
                image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                mosaics[position].paste(image.resize(size, Image.Resampling.BILINEAR), xy)
            del decoded
            camera_info.append({"key": camera, "path": str(path), "video_offset_seconds": offset,
                                "queried_video_timestamps": queries.tolist()})
        actual_times = [None if row < 0 else float(timestamps[row]) for row in rows]
        return mosaics, {**source, "dataset_fps": self.fps, "requested_frames": targets.tolist(),
                         "episode_timestamps": actual_times, "cameras": camera_info,
                         "missing_context_frames": int(np.sum(rows < 0))}


def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_pair(images: list, sources: list[dict], relative: float, index: int, rank: int) -> Image.Image:
    canvas = Image.new("RGB", (640, 480), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    now = abs(relative) < 1e-9
    phase = "CURRENT" if now else ("BEFORE" if relative < 0 else "AFTER")
    highlight = (255, 210, 30) if now else (230, 230, 230)
    for n, (image, source) in enumerate(zip(images, sources)):
        x = n * 320
        draw.text((x + 8, 4), f"Pair {rank:04d} / {'A' if n == 0 else 'B'} / {phase}", font=get_font(16), fill=highlight)
        draw.text((x + 8, 27), f"ep={source['episode']}  frame={source['requested_frames'][index]}", font=get_font(14), fill="white")
        timestamp = source["episode_timestamps"][index]
        text = "episode time: N/A" if timestamp is None else f"episode time: {timestamp:.3f}s"
        draw.text((x + 8, 46), text, font=get_font(12), fill="white")
        if image is None:
            draw.text((x + 16, 210), "NO CONTEXT: episode boundary", font=get_font(14), fill="white")
        else:
            canvas.paste(image, (x, 64))
        draw.text((x + 8, 455), f"relative t={relative:+.2f}s  |  recorded GT", font=get_font(13), fill=highlight)
        if now:
            draw.rectangle((x + 1, 1, x + 318, 478), outline=highlight, width=3)
    draw.line((319, 0, 319, 480), fill=(120, 120, 120), width=2)
    return canvas


def playback(images_a: list, images_b: list, sources: list[dict], relative: np.ndarray,
             rank: int, pause_frames: int):
    for index, rel in enumerate(relative):
        image = render_pair([images_a[index], images_b[index]], sources, float(rel), index, rank)
        yield image
        if abs(rel) < 1e-9:
            for _ in range(pause_frames):
                yield image


def write_json(path: Path, payload: Any):
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pairs-json", required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 9])
    parser.add_argument("--dataset-dir", default=None, help="Optional mount override for one source dataset")
    parser.add_argument("--output-dir", default=None, help="Only new replay files; defaults beside pairs.json")
    parser.add_argument("--before-seconds", type=float, default=3.0)
    parser.add_argument("--after-seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=10, help="Playback FPS, not dataset sampling rate")
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Extra CURRENT-frame hold")
    parser.add_argument("--tolerance-s", type=float, default=0.002)
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec", "video_reader"], default="pyav")
    parser.add_argument("--cameras", nargs=3, default=["head", "left", "right"], help="Top, lower-left, lower-right cameras")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    relative = relative_grid(args.before_seconds, args.after_seconds, args.fps)
    if not math.isfinite(args.pause_seconds) or args.pause_seconds < 0:
        raise ValueError("pause-seconds must be finite/nonnegative")
    if not math.isfinite(args.tolerance_s) or args.tolerance_s <= 0:
        raise ValueError("tolerance-s must be finite/positive")
    path = Path(args.pairs_json).expanduser().resolve()
    original = path.read_bytes()
    payload = json.loads(original)
    pairs = select_pairs(payload, args.ranks)
    all_sources = [[parse_source(pair, side, args.dataset_dir) for side in ("a", "b")] for pair in pairs]
    original_roots = {parse_source(pair, side)["root"] for pair in pairs for side in ("a", "b")}
    if args.dataset_dir and len(original_roots) > 1:
        raise ValueError("A single dataset-dir override cannot remap multiple source roots")
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else path.parent / "pair_videos" / stamp
    # Do not mix a replay run with the mining outputs or raw data.
    if output == path.parent or any(output == Path(s['root']) or Path(s['root']) in output.parents
                                    for sources in all_sources for s in sources):
        raise ValueError("Use a separate replay directory, not the mining directory or raw dataset")
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / "review.csv", output / "manifest.json"]
    for pair in pairs:
        destinations.extend(output / f"pair_{int(pair['rank']):04d}{suffix}"
                            for suffix in (".mp4", "_current.png", ".json", ".partial.mp4"))
    if any(p.exists() for p in destinations):
        raise FileExistsError("Replay outputs already exist; use a new output directory")
    print(f"[replay] input={path}\n[replay] output={output}\n[replay] ranks={args.ranks}; CURRENT aligned at {args.before_seconds}s", flush=True)
    from fastwam.utils.video_io import save_mp4
    readers = {}
    exported = []
    for pair, sources in zip(pairs, all_sources):
        rank = int(pair["rank"])
        images, information = [], []
        for source in sources:
            root = source["root"]
            if root not in readers:
                readers[root] = EpisodeReader(root, args.cameras)
            reader = readers[root]
            if args.fps > reader.fps:
                raise ValueError("Playback FPS must not exceed dataset FPS")
            if args.tolerance_s >= 0.5 / reader.fps:
                raise ValueError("Use tolerance-s < half a dataset frame to prevent ambiguous matching")
            side_images, side_info = reader.read(source, relative, args.tolerance_s, args.video_backend)
            images.append(side_images)
            information.append(side_info)
        if sources[0]["root"] != sources[1]["root"] or sources[0]["episode"] != sources[1]["episode"]:
            raise ValueError("This reviewer expects within-episode pairs; source identities disagree")
        current_index = int(np.flatnonzero(relative == 0)[0])
        pause_frames = int(round(args.pause_seconds * args.fps))
        target = output / f"pair_{rank:04d}.mp4"
        partial = output / f"pair_{rank:04d}.partial.mp4"
        try:
            save_mp4(playback(images[0], images[1], information, relative, rank, pause_frames), str(partial), fps=args.fps)
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("Video encoder produced no output")
            partial.rename(target)
        finally:
            if partial.exists():
                partial.unlink()
        render_pair([images[0][current_index], images[1][current_index]], information, 0.0,
                    current_index, rank).save(output / f"pair_{rank:04d}_current.png")
        metadata = {"rank": rank, "relative_seconds": relative.tolist(), "sides": information,
                    "current_playback_frame": current_index, "extra_current_hold_frames": pause_frames,
                    "fps": args.fps, "decode_tolerance_s": args.tolerance_s,
                    "input_pair_metrics": {k: v for k, v in pair.items() if k.endswith("rmse")},
                    "note": "Recorded GT context, not WAM prediction; RGB display resize is not exact training preprocessing."}
        write_json(output / f"pair_{rank:04d}.json", metadata)
        exported.append({"rank": rank, "episode": sources[0]["episode"], "frame_a": sources[0]["frame"],
                         "frame_b": sources[1]["frame"], "video": target.name, "status": "unreviewed"})
        print(f"[replay] saved {target.name}; missing A/B context = "
              f"{information[0]['missing_context_frames']}/{information[1]['missing_context_frames']}", flush=True)
    fields = ["rank", "episode", "frame_a", "frame_b", "video", "status", "current_input_distinguishable",
              "next_behavior_a", "next_behavior_b", "relevant_history_fact", "swapped_action_still_valid", "notes"]
    with (output / "review.csv").open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)
    write_json(output / "manifest.json", {"pairs_json": str(path), "pairs_json_sha256": hashlib.sha256(original).hexdigest(),
               "arguments": vars(args), "exported": exported,
               "display_only": "Clips include future GT for human review, never feed them to the policy as history."})
    print(f"[replay] complete: {output}\n[replay] no mining files, configuration, or output paths were changed.", flush=True)


if __name__ == "__main__":
    main()
