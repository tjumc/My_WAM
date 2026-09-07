import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Dict, Optional, Any, DefaultDict
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset, STATE_ACTION_DIM_SLICE

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from fastwam.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor
from .utils.normalizer import load_dataset_stats_from_json

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5

class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1, 
        past_action_size: int = 0, # Excludes the current frame
        obs_size: int = 1, # should be 
        past_obs_size: int = 0,

        # train vs val
        val_set_proportion: float = 0.05, 
        is_training_set: bool = False,
        seed: int = 42,

        # sampling
        global_sample_stride: int = 1,
        image_obs_indices: List[int] | None = None,
        video_tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        pretrained_norm_stats: str | None = None,
        norm_stats_source: Literal["meta", "pretrained", "mixed"] = "meta",
        pretrained_norm_stats_dataset_sources: Optional[List[str] | str] = None,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"
        
        # Expand dataset_dirs: if a dir has no meta/info.json, scan subdirs for subdatasets
        expanded_dirs = []
        for ds_dir in dataset_dirs:
            expanded_dirs.extend(self._expand_dataset_dir(ds_dir))
        assert len(expanded_dirs) > 0, f"No valid datasets found in: {dataset_dirs}"
        self.dataset_dirs = expanded_dirs
        logger.info(f"Loaded {len(self.dataset_dirs)} dataset(s): {self.dataset_dirs}")
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None  # Will be set externally
        self.processors_by_dataset = None
        self.pretrained_norm_stats = pretrained_norm_stats
        self._pretrained_norm_stats_cache = None
        self.norm_stats_source = str(norm_stats_source).strip().lower()
        if self.norm_stats_source not in {"meta", "pretrained", "mixed"}:
            raise ValueError(
                f"Unsupported norm_stats_source: {norm_stats_source}. "
                "Expected one of: ['meta', 'pretrained', 'mixed']."
            )
        self.pretrained_norm_stats_dataset_sources = self._normalize_dataset_sources(
            pretrained_norm_stats_dataset_sources
        )
        if self.norm_stats_source == "mixed" and not self.pretrained_norm_stats_dataset_sources:
            raise ValueError(
                "norm_stats_source='mixed' requires pretrained_norm_stats_dataset_sources, "
                "for example ['midea']."
            )
        metas = []
        for ds_dir in self.dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        self.dataset_fps = {m.repo_id: m.fps for m in metas}
        fps_counts = {fps: fps_list.count(fps) for fps in sorted(set(fps_list))}
        if len(fps_counts) > 1:
            logger.info(
                "Mixed dataset FPS detected; using per-dataset delta_timestamps: %s",
                fps_counts,
            )
        else:
            logger.info("Dataset FPS: %s", fps_list[0])
        
        self.global_sample_stride = global_sample_stride
        if image_obs_indices is None:
            self.image_obs_indices = list(range(obs_size))
        else:
            self.image_obs_indices = [int(idx) for idx in image_obs_indices]
        if not self.image_obs_indices:
            raise ValueError("image_obs_indices must contain at least one frame index.")
        invalid_image_indices = [idx for idx in self.image_obs_indices if idx < 0 or idx >= obs_size]
        if invalid_image_indices:
            raise ValueError(
                f"image_obs_indices must be in [0, {obs_size}), got invalid values: {invalid_image_indices}"
            )
        self.video_tolerance_s = float(video_tolerance_s)
        if video_backend not in {None, "torchcodec", "pyav", "video_reader"}:
            raise ValueError(
                f"Unsupported video_backend={video_backend!r}; "
                "expected one of: torchcodec, pyav, video_reader."
            )
        self.video_backend = video_backend
        logger.info("Video backend: %s", self.video_backend or "auto")

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
        
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
        
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"

        def build_delta_timestamps(fps: int):
            delta_timestamps = {}
            for meta in self.image_meta:
                delta_timestamps[meta["lerobot_key"]] = [
                    ((-past_obs_size + t) * global_sample_stride) / fps for t in self.image_obs_indices
                ]

            for meta in self.state_meta:
                delta_timestamps[meta["lerobot_key"]] = [
                    (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
                ]

            for meta in self.action_meta:
                delta_timestamps[meta["lerobot_key"]] = [
                    (t * global_sample_stride) / fps
                    for t in range(-past_action_size, -past_action_size + action_size)
                ]
            return delta_timestamps

        delta_timestamps = {
            meta.repo_id: build_delta_timestamps(meta.fps)
            for meta in metas
        }

        # Match the VLA pretrain dataloader: always load full datasets and avoid
        # episode-subset filtering. Filtering rewrites the HF row coordinate system,
        # which breaks v3 metadata indices on some XDOF datasets.
        episodes = None

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerances_s=dict.fromkeys(self.dataset_dirs, self.video_tolerance_s),
            video_backend=self.video_backend,
        )
        
        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

    def _expand_dataset_dir(self, ds_dir: str) -> List[str]:
        """If ds_dir has no meta/info.json, scan subdirectories for valid datasets.
        Priority:
          1. ds_dir itself is a valid dataset (has meta/info.json)
          2. ds_dir contains dataloader.json → load paths from it
          3. ds_dir contains subdirectories with meta/info.json → expand them
        """
        ds_path = Path(ds_dir)
        if not ds_path.exists():
            logger.warning(f"Dataset directory does not exist: {ds_dir}")
            return []

        # Priority 1: single valid dataset
        if (ds_path / "meta" / "info.json").exists():
            return [ds_dir]

        # Priority 2: dataloader.json present → load paths from it
        json_config = ds_path / "dataloader.json"
        if json_config.exists():
            import json
            with open(json_config, "r", encoding="utf-8") as f:
                config = json.load(f)
            raw_paths = config.get("dataset_paths", [])
            if raw_paths:
                resolved = []
                for p in raw_paths:
                    path = Path(p)
                    if not path.is_absolute():
                        path = (ds_path / path).resolve()
                    if path.exists() and (path / "meta" / "info.json").exists():
                        resolved.append(str(path))
                    else:
                        logger.warning(f"Path from dataloader.json not valid, skipping: {path}")
                if resolved:
                    logger.info(f"Loaded {len(resolved)} dataset(s) from '{json_config}'")
                    return sorted(resolved)
                logger.warning(f"dataloader.json found but no valid paths in: {json_config}")

        # Priority 3: scan subdirectories
        subdatasets = sorted(
            str(subdir) for subdir in ds_path.iterdir()
            if subdir.is_dir() and (subdir / "meta" / "info.json").exists()
        )
        if subdatasets:
            logger.info(f"Expanded '{ds_dir}' into {len(subdatasets)} subdatasets")
            return subdatasets

        logger.error(f"No valid datasets found in '{ds_dir}' (no meta/info.json, no dataloader.json, no subdatasets). Skipping.")
        return []

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        if raw_shape is not None:
            assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        # state = state[..., :-1, :]  # use state_{t} as observation_t
        if raw_shape is not None:
            assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state
    
    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)        
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image
    
    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample
    
    def _get_episode_data(self, episode_idx):
        columns = [meta["lerobot_key"] for meta in self.state_meta + self.action_meta]
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx, columns=columns)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    @staticmethod
    def _scalar_to_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        if isinstance(value, np.ndarray):
            return int(value.item())
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected scalar value, got {value}")
            return BaseLerobotDataset._scalar_to_int(value[0])
        return int(value)

    @staticmethod
    def _scalar_to_float(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        if isinstance(value, np.ndarray):
            return float(value.item())
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected scalar value, got {value}")
            return BaseLerobotDataset._scalar_to_float(value[0])
        return float(value)

    def _describe_sample_source(self, sample_idx: int) -> str:
        start_idx = 0
        for dataset_idx, dataset in enumerate(self.multi_dataset._datasets):
            end_idx = start_idx + dataset.num_frames
            if sample_idx >= end_idx:
                start_idx = end_idx
                continue

            local_idx = sample_idx - start_idx
            parts = [
                f"dataset_idx={dataset_idx}",
                f"dataset_root={dataset.root}",
                f"global_idx={sample_idx}",
                f"local_idx={local_idx}",
            ]

            try:
                row = dataset.hf_dataset[local_idx]
                ep_idx = self._scalar_to_int(row["episode_index"])
                parts.append(f"episode_index={ep_idx}")
                if "frame_index" in row:
                    parts.append(f"frame_index={self._scalar_to_int(row['frame_index'])}")
                if "index" in row:
                    parts.append(f"absolute_index={self._scalar_to_int(row['index'])}")
                if "timestamp" in row:
                    parts.append(f"timestamp={self._scalar_to_float(row['timestamp']):.6f}")

                try:
                    parts.append(f"data_path={dataset.root / dataset.meta.get_data_file_path(ep_idx)}")
                except Exception as path_err:
                    parts.append(f"data_path=<failed: {path_err}>")

                video_paths = []
                for vid_key in dataset.meta.video_keys:
                    try:
                        video_paths.append(str(dataset.root / dataset.meta.get_video_file_path(ep_idx, vid_key)))
                    except Exception as path_err:
                        video_paths.append(f"{vid_key}=<failed: {path_err}>")
                if video_paths:
                    parts.append(f"video_paths={video_paths}")
            except Exception as row_err:
                parts.append(f"row_lookup=<failed: {type(row_err).__name__}: {row_err}>")

            return ", ".join(parts)

        return f"global_idx={sample_idx}, dataset_lookup=<out of range>, num_frames={len(self)}"

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        # Retry with random indices until we successfully load a frame.
        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        failed_sources: list[str] = []
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                source_info = self._describe_sample_source(sample_idx)
                failed_sources.append(source_info)
                logger.warning(
                    f"Error loading sample {sample_idx} (attempt {attempt}). "
                    f"Source: {source_info}. "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                print(
                    f"[BaseLerobotDataset] Error loading sample {sample_idx} "
                    f"(attempt {attempt}/{MAX_GETITEM_ATTEMPT}). "
                    f"Source: {source_info}. Error: {err}",
                    flush=True,
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc(), flush=True)
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {idx}. Failed sources: {failed_sources}"
            ) from last_exception

        # Get data from lerobot, organized in nested dict
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Preprocess the sample using the processor
        # for quick data loading
        processor = self._get_processor_for_sample(sample)
        if processor is not None:
            sample = processor.preprocess(sample)

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        self.processors_by_dataset = None
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        self.processors_by_dataset = self._build_processors_by_dataset(processor)
        return self

    def _get_processor_for_sample(self, sample):
        if self.processors_by_dataset is None:
            return self.processor
        dataset_index = sample.get("dataset_index", 0)
        if isinstance(dataset_index, torch.Tensor):
            dataset_index = int(dataset_index.item())
        return self.processors_by_dataset[int(dataset_index)]

    def _build_processors_by_dataset(self, processor: BaseProcessor):
        processors = []
        for dataset_idx, dataset in enumerate(self.multi_dataset._datasets):
            cur_processor = deepcopy(processor)
            stats = self._get_norm_stats_for_dataset(dataset_idx, dataset)
            cur_processor.shape_meta = self._build_shape_meta_for_dataset(stats)
            cur_processor.action_state_merger.set_shape_meta(cur_processor.shape_meta)
            if self.is_training_set:
                cur_processor.train()
            else:
                cur_processor.eval()
            cur_processor.set_normalizer_from_stats(stats)
            processors.append(cur_processor)
        if processors:
            self.processor = processors[0]
        return processors

    def _build_shape_meta_for_dataset(self, stats):
        shape_meta = deepcopy(self.shape_meta)
        for meta in shape_meta["state"]:
            meta["raw_shape"] = self._get_stats_field_dim(stats, meta, kind="state")
            meta["shape"] = meta["raw_shape"]
        for meta in shape_meta["action"]:
            meta["raw_shape"] = self._get_stats_field_dim(stats, meta, kind="action")
            meta["shape"] = meta["raw_shape"]
        return shape_meta

    def _get_stats_field_dim(self, stats, meta, kind: Literal["state", "action"]) -> int:
        key = meta["key"]
        field_stats = stats[kind][key]
        value = field_stats.get("global_mean")
        if value is None:
            value = field_stats.get("global_min")
        if value is None:
            raise KeyError(
                f"Cannot infer {kind} dimension for `{key}` from stats. "
                f"Available keys: {list(field_stats.keys())}"
            )
        return int(torch.as_tensor(value).shape[-1])

    def _get_norm_stats_for_dataset(self, dataset_idx: int, dataset):
        norm_stats_source = self._get_norm_stats_source_for_dataset(dataset)
        if norm_stats_source == "meta":
            stats = dataset.meta.stats
            if stats is None:
                raise ValueError(
                    f"Dataset `{dataset.root}` does not have meta/stats.json, "
                    "but norm_stats_source='meta'."
                )
            logger.info("Using dataset meta stats for normalization: %s", dataset.root)
            return self._convert_lerobot_stats_to_fastwam(stats, dataset=dataset)

        stats = self._resolve_pretrained_norm_stats()
        logger.info("Using pretrained_norm_stats for normalization: dataset=%s", dataset.root)
        return stats

    def _normalize_dataset_sources(self, sources: Optional[List[str] | str]) -> set[str]:
        if sources in (None, "", "null"):
            return set()
        if isinstance(sources, str):
            sources = [sources]
        return {str(source).strip().lower() for source in sources if str(source).strip()}

    def _get_norm_stats_source_for_dataset(self, dataset) -> str:
        if self.norm_stats_source in {"meta", "pretrained"}:
            return self.norm_stats_source

        dataset_source = getattr(dataset, "dataset_source", None)
        if dataset_source is not None:
            dataset_source = str(dataset_source).strip().lower()
            if dataset_source in self.pretrained_norm_stats_dataset_sources:
                return "pretrained"

        root = str(getattr(dataset, "root", "")).lower()
        for source in self.pretrained_norm_stats_dataset_sources:
            if source in root:
                return "pretrained"
        return "meta"

    def _resolve_pretrained_norm_stats(self):
        if self._pretrained_norm_stats_cache is not None:
            return self._pretrained_norm_stats_cache
        source = self.pretrained_norm_stats
        if source in (None, "", "null"):
            raise ValueError(
                "pretrained_norm_stats must be set when pretrained normalization is enabled."
            )
        if isinstance(source, (str, Path)):
            self._pretrained_norm_stats_cache = load_dataset_stats_from_json(str(source))
            return self._pretrained_norm_stats_cache
        raise TypeError(
            f"Unsupported pretrained_norm_stats type: {type(source)}. "
            "Expected a single stats file path."
        )

    def _convert_lerobot_stats_to_fastwam(self, lerobot_stats, dataset=None):
        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict)}
        for meta in self.state_meta:
            key = meta["key"]
            field_stats = self._get_lerobot_field_stats(lerobot_stats, meta["lerobot_key"])
            stats["state"][key] = self._field_stats_to_fastwam(field_stats)
        for meta in self.action_meta:
            key = meta["key"]
            lerobot_key = meta["lerobot_key"]
            if dataset is not None and getattr(dataset, "use_obs_as_action", False):
                lerobot_key = "observation.state"
            field_stats = self._get_lerobot_field_stats(lerobot_stats, lerobot_key)
            stats["action"][key] = self._field_stats_to_fastwam(field_stats)
        return self._standardize_stats_dims(stats, dataset)

    def _standardize_stats_dims(self, stats, dataset=None):
        if dataset is None:
            return stats
        dim = STATE_ACTION_DIM_SLICE.get(getattr(dataset, "dataset_source", None))
        if dim is None:
            return stats

        for kind in ("state", "action"):
            for field_stats in stats[kind].values():
                for stat_name, value in list(field_stats.items()):
                    if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[-1] > dim:
                        field_stats[stat_name] = value[..., :dim]
        return stats

    def _get_lerobot_field_stats(self, lerobot_stats, lerobot_key: str):
        if lerobot_key in lerobot_stats:
            return lerobot_stats[lerobot_key]
        fallback_key = lerobot_key.rsplit(".", 1)[0]
        if fallback_key in lerobot_stats:
            return lerobot_stats[fallback_key]
        raise KeyError(
            f"Cannot find stats for `{lerobot_key}` in dataset meta stats. "
            f"Available keys: {list(lerobot_stats.keys())}"
        )

    def _field_stats_to_fastwam(self, field_stats):
        def to_tensor(name: str, fallback: str | None = None):
            source_name = name if name in field_stats else fallback
            if source_name is None or source_name not in field_stats:
                raise KeyError(
                    f"Missing `{name}` in field stats. Available keys: {list(field_stats.keys())}"
                )
            value = field_stats[source_name]
            if isinstance(value, torch.Tensor):
                tensor = value.detach().clone()
            else:
                tensor = torch.as_tensor(value)
            return tensor.float()

        global_min = to_tensor("min")
        global_max = to_tensor("max")
        global_mean = to_tensor("mean")
        global_std = to_tensor("std")
        global_q01 = to_tensor("q01", fallback="min")
        global_q99 = to_tensor("q99", fallback="max")

        return {
            "global_min": global_min,
            "global_max": global_max,
            "global_mean": global_mean,
            "global_std": global_std,
            "global_q01": global_q01,
            "global_q99": global_q99,
            "stepwise_min": global_min,
            "stepwise_max": global_max,
            "stepwise_mean": global_mean,
            "stepwise_std": global_std,
            "stepwise_q01": global_q01,
            "stepwise_q99": global_q99,
        }

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes
        
        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx) 
            batch = preprocessor.action_state_transform(batch)
            return batch
        
        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(range(episodes_num), desc="Iterating dataset to get normalization"):
                batch = process_episode(episode_idx)
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))
        
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in range(episodes_num)]
                
                for future in tqdm(as_completed(futures), total=episodes_num, desc="Iterating dataset to get normalization"):
                    try:
                        batch = future.result()
                        for meta in self.state_meta:
                            key = meta["key"]
                            cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0))
                            state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                            state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))

                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0))
                            action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                            action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        print(traceback.format_exc())
                        raise e

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"], 
                stats["action"][key]["stepwise_std"], 
                stats["action"][key]["global_mean"], 
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].
    
    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)
    
    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window
    
    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0
    
    N, D = x.shape
    
    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)            # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices                     # [N, window_size]

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out
