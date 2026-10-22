#!/usr/bin/env python3
"""Paired offline WAM comparison on a frozen catalog-sampled manifest.

This is a train-seen demonstration fit probe, NOT a held-out/closed-loop success test.
Actions use each checkpoint's own pre/postprocessing; scoring uses one fixed
per-task scale from the corresponding expert's training stats.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

HORIZON = 32
GROUPS = {
    "base": slice(0, 3),
    "torso": slice(3, 7),
    "left_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22),
    "right_gripper": slice(22, 23),
    "head": slice(23, 25),
}
CHUNKS = {"first8": slice(0, 8), "middle16": slice(8, 24), "last8": slice(24, 32)}
SUMMARY_FILES = {"report.md", "tasks.csv", "groups.csv", "chunks.csv",
                 "coverage.csv", "samples.csv", "run_manifest.json",
                 "model_plan.json", ".gitignore"}


def die(message):
    raise RuntimeError(message)


def read_csv(path, delimiter=","):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(seed, name):
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_catalog(rows):
    unique = set()
    result = []
    for row in rows:
        task = row["task"].strip()
        raw = row["dataset_dir"].strip()
        if not task or not raw:
            die("Empty catalog entry")
        root = Path(raw).expanduser().resolve()
        key = (task, str(root))
        if key in unique:
            die(f"Duplicate catalog entry: {key}")
        unique.add(key)
        result.append({"task": task, "dataset_dir": str(root)})
    return result


def read_episodes(root):
    """Read v2.1 JSONL or v3 episode metadata, without video/numeric scans."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        die(f"Missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        die(f"Invalid FPS in {info_path}")
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as handle:
            episodes = [json.loads(line) for line in handle if line.strip()]
    else:
        meta_dir = root / "meta" / "episodes"
        files = sorted(meta_dir.rglob("*.parquet")) if meta_dir.exists() else []
        if not files:
            die(f"Missing episode metadata: {root}")
        import pyarrow.parquet as pq
        episodes = []
        for file in files:
            episodes.extend(pq.read_table(file, columns=["episode_index", "length"]).to_pylist())
    parsed = {}
    for ep in episodes:
        index, length = int(ep["episode_index"]), int(ep["length"])
        if index in parsed:
            die(f"Duplicate episode index {index} in {root}")
        parsed[index] = length
    if len(parsed) != int(info["total_episodes"]):
        die(f"Episode count mismatch in {root}: meta={info['total_episodes']} read={len(parsed)}")
    if sorted(parsed) != list(range(len(parsed))):
        die(f"Non-contiguous episode IDs in {root}; check dataset conversion")
    return [(i, n) for i, n in sorted(parsed.items()) if n > HORIZON], fps


def choose_episodes(by_directory, count, seed, task):
    """Visit distinct directories before drawing additional episodes by volume."""
    import numpy as np
    rng = np.random.default_rng(stable_seed(seed, task))
    available = {root: sorted(items) for root, items in by_directory.items() if items}
    roots = sorted(available)
    if not roots:
        die(f"No eligible episodes for {task}")
    rng.shuffle(roots)
    chosen = []
    for root in roots:
        if len(chosen) >= count:
            break
        idx = int(rng.integers(len(available[root])))
        ep = available[root].pop(idx)
        chosen.append((root, *ep))
    rest = [(root, *ep) for root in sorted(available) for ep in available[root]]
    if len(chosen) < count and rest:
        indexes = rng.choice(len(rest), size=min(count - len(chosen), len(rest)), replace=False)
        chosen.extend(rest[int(i)] for i in indexes)
    return sorted(chosen)


def choose_starts(length, count):
    import numpy as np
    if length <= HORIZON or count < 1:
        return []
    last = length - HORIZON
    return sorted(set(int(round(x)) for x in np.linspace(0, last, min(count, last + 1))))


def ensure_output(out):
    out.mkdir(parents=True, exist_ok=True)
    ignore = "*\n" + "".join(f"!/{name}\n" for name in sorted(SUMMARY_FILES))
    path = out / ".gitignore"
    if path.exists() and path.read_text() != ignore:
        die(f"Different .gitignore already exists: {path}")
    path.write_text(ignore)
    (out / "_raw").mkdir(exist_ok=True)


def prepare(args):
    catalog = normalize_catalog(read_csv(args.catalog, delimiter="\t"))
    task_order = list(dict.fromkeys(row["task"] for row in catalog))
    grouped = defaultdict(dict)
    fps_seen = defaultdict(set)
    for row in catalog:
        episodes, fps = read_episodes(row["dataset_dir"])
        grouped[row["task"]][row["dataset_dir"]] = episodes
        fps_seen[row["task"]].add(fps)
    manifest = []
    for task in task_order:
        selected = choose_episodes(grouped[task], args.episodes_per_task, args.seed, task)
        for root, episode, length in selected:
            for frame in choose_starts(length, args.windows_per_episode):
                key = f"{task}\0{root}\0{episode}\0{frame}"
                manifest.append({
                    "sample_id": hashlib.sha256(key.encode()).hexdigest()[:20],
                    "task": task, "dataset_dir": root,
                    "episode_index": episode, "episode_length": length,
                    "start_frame": frame,
                })
    if not manifest or len(set(r["sample_id"] for r in manifest)) != len(manifest):
        die("Empty or duplicate sample manifest")
    ensure_output(args.output)
    fields = ["sample_id", "task", "dataset_dir", "episode_index", "episode_length", "start_frame"]
    sample_path = args.output / "samples.csv"
    new_text = csv_as_text(manifest, fields)
    if sample_path.exists() and sample_path.read_text(encoding="utf-8") != new_text:
        die(f"Existing sample manifest differs; use a new output directory: {args.output}")
    sample_path.write_text(new_text, encoding="utf-8")
    payload = {
        "purpose": "train-seen expert vs shared model diagnostic, not test-set generalization",
        "catalog_sha256": digest(args.catalog),
        "samples_sha256": digest(sample_path),
        "tasks": task_order,
        "episodes_per_task": args.episodes_per_task,
        "windows_per_episode": args.windows_per_episode,
        "seed": args.seed,
        "horizon": HORIZON,
        "fps_by_task": {task: sorted(fps_seen[task]) for task in task_order},
        "git_head": try_git_head(),
    }
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != payload:
        # A changed git HEAD does not change the sample selection.
        old = json.loads(manifest_path.read_text())
        if {k:v for k,v in old.items() if k != "git_head"} != {k:v for k,v in payload.items() if k != "git_head"}:
            die(f"Existing run manifest differs; use a new output directory: {args.output}")
    else:
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[prepare] {len(task_order)} tasks; {len(manifest)} fixed windows at {sample_path}")
    for task in task_order:
        subset = [r for r in manifest if r["task"] == task]
        print(f"  {task}: {len({(r['dataset_dir'],r['episode_index']) for r in subset})} episodes, "
              f"{len(subset)} windows, {len({r['dataset_dir'] for r in subset})} directories")


def csv_as_text(rows, fields):
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def try_git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def comparison_code():
    # The file hash also identifies a locally applied, uncommitted repair.
    return {"git_head": try_git_head(), "tool_sha256": digest(__file__),
            "gt_reader": "episode-range-mapped-v1"}


def validate_files(rows, samples):
    task_set = set(r["task"] for r in samples)
    experts = {r["task"]: r for r in rows if r["kind"] == "expert"}
    shared = [r for r in rows if r["kind"] == "shared"]
    if len(experts) != len([r for r in rows if r["kind"] == "expert"]) or set(experts) != task_set:
        die(f"Model plan must include exactly one expert for each sampled task: {sorted(task_set)}")
    if len(shared) != 1 or shared[0]["task"] != "__all__":
        die("Model plan must include one shared model with task=__all__")
    plan = []
    for row in rows:
        for key in ("checkpoint", "config", "stats", "text_cache"):
            value = row[key].strip()
            if not value:
                die(f"Missing {key}: {row['kind']} / {row['task']}")
            p = Path(value).expanduser().resolve()
            if key == "text_cache":
                if not p.is_dir():
                    die(f"Missing text cache directory: {p}")
            elif not p.is_file():
                die(f"Missing {key}: {p}")
            row[key] = str(p)
        if row["kind"] == "expert" and not row.get("prompt", "").strip():
            die(f"Missing expert training prompt for task {row['task']}")
        cp = Path(row["checkpoint"])
        stat = cp.stat()
        plan.append({
            **row,
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "config_sha256": digest(row["config"]),
            "stats_sha256": digest(row["stats"]),
        })
    return plan


def fixed_scale(stats_path):
    import numpy as np
    obj = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    try:
        std = np.asarray(obj["action"]["default"]["global_std"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError) as exc:
        die(f"Expected action.default.global_std in {stats_path}: {exc}")
    if std.shape != (25,) or not np.isfinite(std).all() or (std < 0).any():
        die(f"Invalid 25D expert action std: {stats_path}")
    positive = std[std > 1e-5]
    if len(positive) == 0:
        die(f"All expert action scales are zero: {stats_path}")
    floor = max(1e-4, float(np.median(positive)) * .01)
    return np.maximum(std, floor)


def score(pred, gt, scale):
    import numpy as np
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(gt, dtype=np.float64)
    if a.shape != (HORIZON, 25) or b.shape != a.shape:
        die(f"Prediction/GT shape mismatch: {a.shape} / {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        die("Non-finite prediction or GT")
    err = a - b
    nerr = err / scale
    result = {
        "rmse": float(np.sqrt(np.mean(nerr ** 2))),
        "raw_rmse": float(np.sqrt(np.mean(err ** 2))),
    }
    for name, part in GROUPS.items():
        result["group_" + name] = float(np.sqrt(np.mean(nerr[:, part] ** 2)))
        result["raw_group_" + name] = float(np.sqrt(np.mean(err[:, part] ** 2)))
    for name, part in CHUNKS.items():
        result["chunk_" + name] = float(np.sqrt(np.mean(nerr[part] ** 2)))
    return result


def read_scored(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                die(f"Corrupt score line {path}:{line_num}: {exc}")
            key = (row["sample_id"], int(row["seed"]))
            if key in rows:
                die(f"Duplicate score key in {path}: {key}")
            rows[key] = row
    return rows


def stored_prompt(root):
    """Get a bootstrapping prompt for the shared model; subsequent samples use sample.task."""
    import pyarrow.parquet as pq
    meta = Path(root) / "meta"
    tasks_parquet = meta / "tasks.parquet"
    if tasks_parquet.exists():
        table = pq.read_table(tasks_parquet)
        fields = table.column_names
        if "__index_level_0__" in fields:
            texts = table["__index_level_0__"].to_pylist()
        else:
            # Pandas index metadata is not guaranteed; fail rather than guess.
            die(f"Cannot find stored task text column in {tasks_parquet}: {fields}")
        if texts:
            return str(texts[0])
    tasks_jsonl = meta / "tasks.jsonl"
    if tasks_jsonl.exists():
        with tasks_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return str(json.loads(line)["task"])
    die(f"Cannot determine stored task text from {root}")


def configure_model_env(row, task, directories):
    os.environ["TEXT_CACHE_DIR"] = row["text_cache"]
    os.environ["WAM_TEXT_CACHE_DIR"] = row["text_cache"]
    os.environ["WAM_STATS_PATH"] = row["stats"]
    os.environ["WAM_TASK_INSTRUCTION"] = row["prompt"]
    os.environ["WAM_DATASET_DIRS_ENV"] = "\n".join(directories)
    config = Path(row["config"])
    os.environ["WAM_OUTPUT_DIR"] = str(config.parent)


def load_policy(row, init_prompt, args, directories):
    import torch
    from experiments.astribot.fastwam_portal_server import AstribotFastWAMPolicy
    env_row = dict(row, prompt=init_prompt)
    configure_model_env(env_row, row["task"], directories)
    model = AstribotFastWAMPolicy(
        config_path=row["config"], checkpoint_path=row["checkpoint"],
        dataset_stats_path=row["stats"], task_prompt=init_prompt,
        device=args.device, mixed_precision=args.mixed_precision,
        action_horizon=HORIZON, n_action_steps=HORIZON,
        action_mode="cmd_absolute_joint", num_inference_steps=args.num_inference_steps,
        sigma_shift=None, text_cfg_scale=1.0, negative_prompt="",
        rand_device="cpu", tiled=False, vae_device_mode=args.vae_device_mode, seed=None,
    )
    if (model.raw_action_dim, model.raw_state_dim, model.action_horizon) != (25, 25, HORIZON) or int(model.cfg.data.train.num_frames) != HORIZON + 1:
        die(f"Unexpected model action/state/horizon dimensions: {row['checkpoint']}")
    if model.concat_multi_camera != "robotwin" or list(model.video_size) != [384, 320]:
        die(f"Unsupported image preprocessing for comparison: {row['checkpoint']}")
    if int(model.cfg.data.train.get("global_sample_stride", 1)) != 1:
        die("Evaluation supports stride=1 only")
    if int(model.cfg.data.train.get("action_video_freq_ratio", 1)) != 4:
        die("Evaluation expected action_video_freq_ratio=4")
    return model


def set_task_context(policy, prompt, cache_dir):
    """Use the shared model's actual stored training instructions, not expert overrides."""
    import torch
    from experiments.astribot.fastwam_portal_server import _load_text_context_cache
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    full_prompt = DEFAULT_PROMPT.format(task=prompt)
    if policy.prompt == full_prompt:
        return
    length = int(policy.cfg.data.train.get("context_len", 128))
    context, mask, path = _load_text_context_cache(cache_dir, full_prompt, length)
    policy.context = context.to(device=policy.model.device, dtype=policy.model.torch_dtype)
    policy.context_mask = mask.to(device=policy.model.device)
    policy.prompt = full_prompt
    policy.task_prompt = prompt
    policy.context_cache_path = path


def get_raw_dataset(root, model):
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    shape_meta = OmegaConf.to_container(model.cfg.data.train.shape_meta, resolve=True)
    data = BaseLerobotDataset(
        dataset_dirs=[root], shape_meta=shape_meta, obs_size=2, action_size=1,
        val_set_proportion=0., is_training_set=False, global_sample_stride=1,
        image_obs_indices=[0], video_tolerance_s=.002, video_backend="pyav",
        norm_stats_source="meta",
    )
    data._set_return_images(True)
    return data


def episode_source(data, ep_idx):
    """Return the unfiltered, single-directory source and its logical bounds."""
    datasets = data.multi_dataset._datasets
    if len(datasets) != 1 or datasets[0].episodes is not None:
        die("Comparison requires one unfiltered dataset directory")
    dataset = datasets[0]
    bounds = dataset.episode_data_index
    if ep_idx < 0 or ep_idx >= len(bounds["from"]):
        die(f"Episode index out of bounds: ep={ep_idx}")
    start, end = int(bounds["from"][ep_idx]), int(bounds["to"][ep_idx])
    if start < 0 or end <= start:
        die(f"Invalid episode bounds: ep={ep_idx} from={start} to={end}")
    return dataset, start, end


def episode_frame_index(data, ep_idx, frame):
    """BaseLerobotDataset.__getitem__ takes a physical HF row index."""
    dataset, start, end = episode_source(data, ep_idx)
    if frame < 0 or frame >= end - start:
        die(f"Frame index out of bounds: ep={ep_idx} frame={frame}")
    return int(dataset._map_query_indices([start + frame])[0])


def episode_actions(data, ep_idx):
    import numpy as np
    dataset, start, end = episode_source(data, ep_idx)
    key = data.action_meta[0]["lerobot_key"]
    indices = list(range(start, end))
    # This query maps logical indices once and preserves use_obs_as_action.
    # The multi-dataset get_episode_data() reads the whole shared v3 parquet.
    episode = dataset._query_hf_dataset_for_training(
        {name: indices for name in (key, "episode_index", "frame_index")})

    def as_numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    length = end - start
    episode_ids = as_numpy(episode["episode_index"])
    frames = as_numpy(episode["frame_index"])
    if not np.array_equal(episode_ids, np.full(length, ep_idx)):
        die(f"Episode GT identity mismatch: ep={ep_idx} expected {length} rows of that episode")
    if not np.array_equal(frames, np.arange(length)):
        die(f"Episode GT frame order mismatch: ep={ep_idx} expected frames 0..{length - 1}")
    raw = np.asarray(as_numpy(episode[key]), dtype=np.float32)
    if raw.shape != (length, 25):
        die(f"Unexpected episode GT actions: ep={ep_idx} shape={raw.shape} expected=({length}, 25)")
    return raw


def raw_observation(data, sample, frame, ep, gt):
    import numpy as np
    index = int(sample.get("idx", -1))
    expected = episode_frame_index(data, ep, frame)
    if index != expected:
        die(f"Dataset silently replaced requested index {expected} with {index}")
    actual_ep = int(sample["episode_index"])
    actual_fr = int(sample["frame_index"])
    if actual_ep != ep or actual_fr != frame:
        die(f"Dataset source mismatch expected ep/frame={ep}/{frame}, got {actual_ep}/{actual_fr}")
    act = sample["action"][data.action_meta[0]["key"]]
    action0 = act[0].detach().cpu().numpy() if hasattr(act, "detach") else np.asarray(act[0])
    if not np.allclose(action0, gt[frame], atol=1e-5, rtol=1e-5):
        die(f"Current GT mismatch between sample and full episode ep={ep} frame={frame}")
    state = sample["state"][data.state_meta[0]["key"]][0].detach().cpu().numpy().astype(np.float32)
    images = []
    for meta in data.image_meta:
        img = sample["images"][meta["key"]][0]
        arr = img.permute(1, 2, 0).detach().cpu().numpy()
        if arr.dtype != np.uint8 or arr.shape[-1] != 3:
            die(f"Unexpected image: {arr.dtype} {arr.shape}")
        images.append(np.ascontiguousarray(arr[:, :, ::-1]))  # RGB data -> BGR SDK convention
    if len(images) != 3:
        die(f"Expected 3 cameras, found {len(images)}")
    task_text = str(sample.get("task", "")).strip()
    if not task_text:
        die(f"Missing recorded task instruction ep={ep} frame={frame}")
    return images, state, task_text


def run_model_task(row, task, samples, seeds, scales, args, all_directories):
    import numpy as np
    import torch
    import time
    import fastwam.datasets.lerobot.base_lerobot_dataset as base
    is_shared = row["kind"] == "shared"
    key = "shared" if is_shared else "expert"
    dest = args.output / "_raw" / f"{key}_{task}.jsonl"
    previous = read_scored(dest)
    ids = set(r["sample_id"] for r in samples)
    wanted = {(i, seed) for i in ids for seed in seeds}
    invalid = set(previous) - wanted
    if invalid:
        die(f"Stored scores not in frozen sample/seed plan: {dest} ({len(invalid)} keys)")
    if len(previous) == len(wanted):
        print(f"[skip] {key}/{task}: {len(wanted)} scores already stored")
        return
    code = comparison_code()
    initial_prompt = stored_prompt(samples[0]["dataset_dir"]) if is_shared else row["prompt"]
    policy = load_policy(row, initial_prompt, args, all_directories)
    groups = defaultdict(list)
    for item in samples:
        groups[item["dataset_dir"]].append(item)
    original_attempts = base.MAX_GETITEM_ATTEMPT
    base.MAX_GETITEM_ATTEMPT = 1  # Never substitute a different training window.
    try:
        with dest.open("a", encoding="utf-8") as stream:
            for root in sorted(groups):
                directory = get_raw_dataset(root, policy)
                by_episode = defaultdict(list)
                for item in groups[root]:
                    by_episode[int(item["episode_index"])].append(item)
                for ep in sorted(by_episode):
                    gt = episode_actions(directory, ep)
                    if gt.shape[0] != int(by_episode[ep][0]["episode_length"]):
                        die(f"GT length mismatch {root} ep={ep}: got={gt.shape[0]} "
                            f"expected={by_episode[ep][0]['episode_length']}")
                    for item in by_episode[ep]:
                        frame = int(item["start_frame"])
                        missing = [s for s in seeds if (item["sample_id"], s) not in previous]
                        if not missing:
                            continue
                        sample = directory[episode_frame_index(directory, ep, frame)]
                        imgs, state, task_prompt = raw_observation(directory, sample, frame, ep, gt)
                        if is_shared:
                            set_task_context(policy, task_prompt, row["text_cache"])
                        else:
                            expected = row["prompt"]
                            if policy.task_prompt != expected:
                                die(f"Expert prompt changed unexpectedly: {task}")
                        reference = gt[frame:frame + HORIZON]
                        if reference.shape != (HORIZON, 25):
                            die(f"Insufficient future GT: {root} ep={ep} frame={frame}")
                        for seed in missing:
                            policy.seed = int(seed)
                            actions = policy.infer(time.time(), *imgs, state)
                            if actions is None:
                                die(f"Inference failed: {key}/{task} sample={item['sample_id']} seed={seed}")
                            metrics = score(actions, reference, scales[task])
                            line = {
                                "sample_id": item["sample_id"], "seed": int(seed),
                                "task": task, "model": key,
                                "dataset_dir": root, "episode_index": ep,
                                "start_frame": frame, "task_prompt": policy.task_prompt,
                                "comparison_code": code,
                                **metrics,
                            }
                            stream.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
                            stream.flush()
                            previous[(item["sample_id"], seed)] = line
                            print(f"[done] {key}/{task} {len(previous)}/{len(wanted)} "
                                  f"ep={ep} f={frame} seed={seed} rmse={metrics['rmse']:.5f}", flush=True)
                del directory
                gc.collect()
    finally:
        base.MAX_GETITEM_ATTEMPT = original_attempts
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(args):
    import numpy as np
    samples = read_csv(args.output / "samples.csv")
    if not samples:
        die("No samples. Run 'prepare' first.")
    manifest = json.loads((args.output / "run_manifest.json").read_text())
    if digest(args.output / "samples.csv") != manifest["samples_sha256"]:
        die("Sample manifest has changed since preparation")
    rows = read_csv(args.models, delimiter="\t")
    plan = validate_files(rows, samples)
    seeds = [int(args.seed) + i for i in range(args.num_seeds)]
    if args.num_seeds < 1 or args.num_inference_steps < 1:
        die("num_seeds and num_inference_steps must be positive")
    fingerprint = {
        "samples_sha256": manifest["samples_sha256"],
        "models": plan,
        "seeds": seeds,
        "num_inference_steps": args.num_inference_steps,
        "vae_device_mode": args.vae_device_mode,
        "device": args.device, "mixed_precision": args.mixed_precision,
        "preprocessing": "policy._prepare_image_tensor + BaseLerobotDataset raw image",
        "scoring": "per-task expert action.global_std, floor=max(1e-4, 0.01*median(positive_std))",
    }
    plan_path = args.output / "model_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != fingerprint:
            die(f"Existing model plan differs, cannot combine scores: {plan_path}")
    else:
        plan_path.write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n")
    by_task = defaultdict(list)
    for item in samples:
        by_task[item["task"]].append(item)
    expert = {row["task"]: row for row in plan if row["kind"] == "expert"}
    shared = next(row for row in plan if row["kind"] == "shared")
    scales = {task: fixed_scale(expert[task]["stats"]) for task in by_task}
    directories = list(dict.fromkeys(r["dataset_dir"] for r in samples))
    if args.variant in ("all", "expert"):
        for task, windows in by_task.items():
            run_model_task(expert[task], task, windows, seeds, scales, args, directories)
    if args.variant in ("all", "shared"):
        # Reuse the shared model once across every task; prompt switches per raw sample.
        run_shared_tasks(shared, by_task, seeds, scales, args, directories)
    print(f"[run] Completed requested variant={args.variant}; use 'summarize' for paired comparison.")


def run_shared_tasks(row, by_task, seeds, scales, args, directories):
    """Shared policy stays loaded across tasks; save scores per task for safe resume."""
    import gc
    import torch
    import fastwam.datasets.lerobot.base_lerobot_dataset as base
    import numpy as np
    import time
    pending = {}
    for task, items in by_task.items():
        path = args.output / "_raw" / f"shared_{task}.jsonl"
        done = read_scored(path)
        expected = {(r["sample_id"], s) for r in items for s in seeds}
        if set(done) - expected:
            die(f"Unexpected prior shared scores in {path}")
        if len(done) < len(expected):
            pending[task] = (items, path, done, expected)
    if not pending:
        print("[skip] shared model: all task scores already stored")
        return
    code = comparison_code()
    first = next(iter(pending.values()))[0][0]
    bootstrap = stored_prompt(first["dataset_dir"])
    policy = load_policy(row, bootstrap, args, directories)
    original_attempts = base.MAX_GETITEM_ATTEMPT
    base.MAX_GETITEM_ATTEMPT = 1
    try:
        for task, (items, path, done, expected) in pending.items():
            groups = defaultdict(list)
            for item in items:
                groups[item["dataset_dir"]].append(item)
            with path.open("a", encoding="utf-8") as stream:
                for root in sorted(groups):
                    dataset = get_raw_dataset(root, policy)
                    by_episode = defaultdict(list)
                    for item in groups[root]:
                        by_episode[int(item["episode_index"])].append(item)
                    for ep, windows in sorted(by_episode.items()):
                        gt = episode_actions(dataset, ep)
                        if gt.shape[0] != int(windows[0]["episode_length"]):
                            die(f"Shared GT length mismatch {root} ep={ep}: got={gt.shape[0]} "
                                f"expected={windows[0]['episode_length']}")
                        for item in windows:
                            frame = int(item["start_frame"])
                            missing = [s for s in seeds if (item["sample_id"], s) not in done]
                            if not missing:
                                continue
                            sample = dataset[episode_frame_index(dataset, ep, frame)]
                            imgs, state, task_prompt = raw_observation(dataset, sample, frame, ep, gt)
                            set_task_context(policy, task_prompt, row["text_cache"])
                            target = gt[frame:frame + HORIZON]
                            if target.shape != (HORIZON, 25):
                                die(f"Shared future GT is truncated {root} ep={ep} f={frame}")
                            for seed in missing:
                                policy.seed = int(seed)
                                prediction = policy.infer(time.time(), *imgs, state)
                                if prediction is None:
                                    die(f"Shared inference failed task={task} sample={item['sample_id']}")
                                metrics = score(prediction, target, scales[task])
                                record = {
                                    "sample_id": item["sample_id"], "seed": seed, "task": task,
                                    "model": "shared", "dataset_dir": root,
                                    "episode_index": ep, "start_frame": frame,
                                    "task_prompt": policy.task_prompt, **metrics,
                                    "comparison_code": code,
                                }
                                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                                stream.flush()
                                done[(item["sample_id"], seed)] = record
                                print(f"[done] shared/{task} {len(done)}/{len(expected)} "
                                      f"ep={ep} f={frame} seed={seed} rmse={metrics['rmse']:.5f}", flush=True)
                    del dataset
                    gc.collect()
    finally:
        base.MAX_GETITEM_ATTEMPT = original_attempts
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def episode_equal_means(paired, field):
    import numpy as np
    per_episode = defaultdict(list)
    for sample, expert, shared in paired:
        key = (sample["dataset_dir"], int(sample["episode_index"]))
        per_episode[key].append((float(expert[field]), float(shared[field])))
    averages = [np.mean(values, axis=0) for values in per_episode.values()]
    if not averages:
        return None
    a = np.asarray(averages)
    return float(a[:, 0].mean()), float(a[:, 1].mean()), float((a[:, 1] < a[:, 0]).mean())


def summarize(args):
    samples = read_csv(args.output / "samples.csv")
    manifest = json.loads((args.output / "run_manifest.json").read_text())
    if digest(args.output / "samples.csv") != manifest["samples_sha256"]:
        die("Sample manifest changed after prepare")
    plan_path = args.output / "model_plan.json"
    if not plan_path.exists():
        die("No model plan found. Run inference first.")
    plan = json.loads(plan_path.read_text())
    seeds = [int(x) for x in plan["seeds"]]
    by_task = defaultdict(list)
    for sample in samples:
        by_task[sample["task"]].append(sample)
    task_rows, group_rows, chunk_rows, coverage_rows, note_rows = [], [], [], [], []
    code_versions, legacy_scores = set(), 0
    for task, windows in by_task.items():
        expert = read_scored(args.output / "_raw" / f"expert_{task}.jsonl")
        shared = read_scored(args.output / "_raw" / f"shared_{task}.jsonl")
        for records in (expert, shared):
            for record in records.values():
                code = record.get("comparison_code")
                if code is None:
                    legacy_scores += 1
                else:
                    code_versions.add(tuple(str(code.get(k) or "unknown") for k in
                                            ("git_head", "tool_sha256", "gt_reader")))
        sample_lookup = {s["sample_id"]: s for s in windows}
        expected = {(sid, seed) for sid in sample_lookup for seed in seeds}
        if (set(expert) | set(shared)) - expected:
            die(f"Scores outside frozen manifest in task {task}")
        paired = []
        for key in sorted(expected):
            if key not in expert or key not in shared:
                continue
            e, s = expert[key], shared[key]
            reference = sample_lookup[key[0]]
            if any(r["task"] != task or r["model"] != k or
                   r["dataset_dir"] != reference["dataset_dir"] or
                   int(r["episode_index"]) != int(reference["episode_index"]) or
                   int(r["start_frame"]) != int(reference["start_frame"])
                   for r,k in ((e,"expert"),(s,"shared"))):
                die(f"Score source mismatch in task {task} for key {key}")
            paired.append((reference, e, s))
        exp_episodes = len({(x["dataset_dir"], x["episode_index"]) for x in windows})
        covered_episodes = len({(x["dataset_dir"], x["episode_index"]) for x,_,_ in paired})
        coverage_rows.append({
            "task": task, "expected_windows": len(windows), "expected_episodes": exp_episodes,
            "expected_calls_each_model": len(expected), "expert_calls": len(expert),
            "shared_calls": len(shared), "paired_calls": len(paired),
            "paired_episodes": covered_episodes,
            "coverage_percent": round(100. * len(paired) / len(expected), 2),
        })
        mean = episode_equal_means(paired, "rmse")
        if mean is None:
            continue
        a,b,fraction = mean
        expert_prompts = sorted({r["task_prompt"] for _,r,_ in paired})
        shared_prompts = sorted({r["task_prompt"] for _,_,r in paired})
        task_rows.append({
            "task": task, "episodes": covered_episodes, "paired_calls": len(paired),
            "expert_rmse": round(a,6), "shared_rmse": round(b,6),
            "delta_shared_minus_expert": round(b-a,6),
            "relative_delta_percent": round((b/a-1)*100,3) if a else "",
            "fraction_episodes_shared_better": round(fraction,4),
            "expert_prompt_variants": len(expert_prompts), "shared_prompt_variants": len(shared_prompts),
        })
        note_rows.append((task, expert_prompts, shared_prompts))
        for name in GROUPS:
            values = episode_equal_means(paired, "group_" + name)
            raw = episode_equal_means(paired, "raw_group_" + name)
            group_rows.append({
                "task": task, "group": name, "expert_rmse": round(values[0],6),
                "shared_rmse": round(values[1],6), "delta": round(values[1]-values[0],6),
                "expert_raw_rmse": round(raw[0],6), "shared_raw_rmse": round(raw[1],6),
            })
        for name in CHUNKS:
            values = episode_equal_means(paired, "chunk_" + name)
            chunk_rows.append({
                "task": task, "chunk": name, "expert_rmse": round(values[0],6),
                "shared_rmse": round(values[1],6), "delta": round(values[1]-values[0],6),
            })
    write_csv(args.output / "tasks.csv", task_rows, [
        "task","episodes","paired_calls","expert_rmse","shared_rmse",
        "delta_shared_minus_expert","relative_delta_percent",
        "fraction_episodes_shared_better","expert_prompt_variants","shared_prompt_variants"
    ])
    write_csv(args.output / "groups.csv", group_rows, [
        "task","group","expert_rmse","shared_rmse","delta","expert_raw_rmse","shared_raw_rmse"
    ])
    write_csv(args.output / "chunks.csv", chunk_rows, [
        "task","chunk","expert_rmse","shared_rmse","delta"
    ])
    write_csv(args.output / "coverage.csv", coverage_rows, [
        "task","expected_windows","expected_episodes","expected_calls_each_model",
        "expert_calls","shared_calls","paired_calls","paired_episodes","coverage_percent"
    ])
    complete = bool(coverage_rows) and all(
        row["paired_calls"] == row["expected_calls_each_model"] for row in coverage_rows)
    lines = [
        "# Single-task expert vs shared multi-task WAM (offline)",
        "",
        f"Status: {'complete' if complete else 'INCOMPLETE (missing paired inference calls)'}",
        "",
        "This is an on-training-demonstrations fit probe, not a held-out validation/test",
        "or real-robot closed-loop success measurement. A difference is not proof of",
        "causal positive/negative transfer; training budgets and prompt sources can differ.",
        "",
        "Metrics: 32 future action steps, true raw 25D action, inverse-normalized",
        "with each checkpoint's own stats. Scoring scale for both variants is",
        "the task expert's fixed training-action global_std (with documented floor).",
        "Mean is equal-weighted over episodes; windows and diffusion seeds",
        "within an episode are not independent validation cases.",
        "",
        "| Task | Episodes | Expert | Shared | Shared - Expert | Shared better episode % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        lines.append(f"| {row['task']} | {row['episodes']} | {row['expert_rmse']:.5f} | "
                     f"{row['shared_rmse']:.5f} | {row['delta_shared_minus_expert']:+.5f} | "
                     f"{100*row['fraction_episodes_shared_better']:.1f}% |")
    if complete and task_rows and len(task_rows) == len(by_task):
        import numpy as np
        lines += ["", f"Task-equal macro expert RMSE: {np.mean([r['expert_rmse'] for r in task_rows]):.5f}",
                  f"Task-equal macro shared RMSE: {np.mean([r['shared_rmse'] for r in task_rows]):.5f}"]
    lines += ["", "## Pairing and coverage", ""]
    for row in coverage_rows:
        lines.append(f"- {row['task']}: {row['paired_calls']}/{row['expected_calls_each_model']} "
                     f"paired calls ({row['coverage_percent']}%), {row['paired_episodes']} paired episodes.")
    lines += ["", "## Code provenance", "",
              f"Sample preparation git HEAD: `{manifest.get('git_head') or 'unknown'}`."]
    for head, tool_hash, reader in sorted(code_versions):
        lines.append(f"- Scored with git HEAD `{head}`, tool SHA-256 `{tool_hash}`, GT reader `{reader}`.")
    if legacy_scores:
        lines.append(f"- {legacy_scores} stored scores have no code revision (legacy); "
                     "their repair version is unknown.")
    lines += ["", "## Prompt conditions (not forced identical)", ""]
    for task, e, s in note_rows:
        lines.append(f"- {task}: expert={json.dumps(e, ensure_ascii=False)}, "
                     f"shared={json.dumps(s, ensure_ascii=False)}")
    lines += ["", "See groups.csv and chunks.csv for arm/gripper and within-chunk effects.",
              "Source directories are retained in samples.csv but not classified by appearance.",
              "When coverage is incomplete, summary contains only matched calls and must not",
              "be treated as the planned full-sample comparison.", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize] {'complete' if complete else 'INCOMPLETE'}: {args.output / 'report.md'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run", "summarize"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=None, help="TSV with task,dataset_dir")
    p.add_argument("--models", type=Path, default=None, help="TSV with expert/shared checkpoint specs")
    p.add_argument("--episodes-per-task", type=int, default=30)
    p.add_argument("--windows-per-episode", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--variant", choices=["all", "expert", "shared"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vae-device-mode", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    if args.command == "prepare":
        if args.catalog is None or args.episodes_per_task < 1 or args.windows_per_episode < 1:
            die("prepare needs --catalog and positive episode/window counts")
        prepare(args)
    elif args.command == "run":
        if args.models is None:
            die("run needs --models")
        ensure_output(args.output)
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
