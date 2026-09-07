#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, List, Literal
import warnings

import datasets
import datasets.features.features as _hf_features
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
import pyarrow.parquet as pq
from datasets import concatenate_datasets
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

if "List" not in _hf_features._FEATURE_TYPES and "Sequence" in _hf_features._FEATURE_TYPES:
    _hf_features._FEATURE_TYPES["List"] = _hf_features._FEATURE_TYPES["Sequence"]

# adapte to own path
from ..constants import HF_LEROBOT_HOME
from .datasets.compute_stats import aggregate_stats, compute_episode_stats
from .datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    TASKS_PATH,
    _validate_feature_names,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    check_timestamps_sync,
    # check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_hf_features_from_features,
    # get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    load_annotations,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from .datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
from .lerobot_utils import load_nested_dataset
import traceback

CODEBASE_VERSION = "v2.1"


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def _is_main_rank() -> bool:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return True


DATASET_SOURCE_ALIASES = (
    ("RoboCOIN", "robocoin"),
    ("RealSourceData", "realsourcedata"),
    ("agibot", "agibot"),
    ("midea", "midea"),
    ("xdof", "xdof"),
    ("robotwin", "robotwin"),
)

CAMERA_KEY_REMAP: dict[str, dict[str, str]] = {
    "RealSourceData": {
        "observation.images.head_camera": "observation.images.cam_high",
        "observation.images.right_hand_camera": "observation.images.cam_right_wrist",
        "observation.images.left_hand_camera": "observation.images.cam_left_wrist",
    },
    "agibot": {
        "observation.images.head": "observation.images.cam_high",
        "observation.images.hand_right": "observation.images.cam_right_wrist",
        "observation.images.hand_left": "observation.images.cam_left_wrist",
    },
}

CAMERA_NAME_ALIASES = {
    "head": "cam_high",
    "head_camera": "cam_high",
    "cam_high": "cam_high",
    "left": "cam_left_wrist",
    "hand_left": "cam_left_wrist",
    "left_hand_camera": "cam_left_wrist",
    "cam_left_wrist": "cam_left_wrist",
    "right": "cam_right_wrist",
    "hand_right": "cam_right_wrist",
    "right_hand_camera": "cam_right_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

OBS_AS_ACTION_SOURCES = {"RealSourceData", "agibot"}

STATE_ACTION_DIM_SLICE: dict[str, int] = {
    "RealSourceData": 16,
    "agibot": 16,
}


def detect_dataset_source(root: str | Path) -> str | None:
    root_lower = str(root).lower()
    for source, alias in DATASET_SOURCE_ALIASES:
        if alias in root_lower:
            return source
    return None


def canonical_camera_key(key: str) -> str:
    prefix = "observation.images."
    if key == "observation.images":
        return key
    suffix = key[len(prefix):] if key.startswith(prefix) else key
    canonical_suffix = CAMERA_NAME_ALIASES.get(suffix, suffix)
    return f"{prefix}{canonical_suffix}"


class _LogicalIndexMapper:
    """Map LeRobot logical `index` values back to physical HF dataset rows.

    Some converted v3 datasets keep `meta/episodes.jsonl` in logical episode
    order while the parquet files are stored in a different physical order.
    The rows still carry the correct logical `index`; this mapper stores only
    contiguous index runs instead of a per-frame dict.
    """

    def __init__(
        self,
        logical_starts: np.ndarray,
        logical_ends: np.ndarray,
        physical_starts: np.ndarray,
    ):
        self.logical_starts = logical_starts.astype(np.int64, copy=False)
        self.logical_ends = logical_ends.astype(np.int64, copy=False)
        self.physical_starts = physical_starts.astype(np.int64, copy=False)

    @property
    def num_runs(self) -> int:
        return int(self.logical_starts.shape[0])

    def to_physical(self, logical_indices: list[int]) -> list[int]:
        if len(logical_indices) == 0:
            return []

        logical = np.asarray([int(idx) for idx in logical_indices], dtype=np.int64)
        run_idx = np.searchsorted(self.logical_starts, logical, side="right") - 1

        ok = run_idx >= 0
        if ok.any():
            valid_run_idx = run_idx[ok]
            ok[ok] = logical[ok] < self.logical_ends[valid_run_idx]

        if not bool(ok.all()):
            missing = logical[~ok][:8].tolist()
            raise IndexError(
                "Logical LeRobot indices are not present in the loaded parquet rows: "
                f"{missing}. This usually means meta episode ranges and parquet rows "
                "use different index coordinates."
            )

        physical = self.physical_starts[run_idx] + (logical - self.logical_starts[run_idx])
        return physical.astype(np.int64, copy=False).tolist()


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        #print(f"[DEBUG LeRobotDatasetMetadata.__init__] repo_id={repo_id}, root={self.root}", flush=True)
        meta_info_path = self.root / 'meta' / 'info.json'
        #print(f"[DEBUG LeRobotDatasetMetadata.__init__] meta/info.json exists: {meta_info_path.exists()}", flush=True)

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            # if is_valid_version(self.revision):
            #     self.revision = get_safe_version(self.repo_id, self.revision)
            print(f"[DEBUG LeRobotDatasetMetadata.__init__] load_metadata FAILED — calling pull_from_repo. WILL TRIGGER HfValidationError if repo_id is a local path!", flush=True)
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        #print(f"[DEBUG LeRobotDatasetMetadata] load_metadata: root={self.root}", flush=True)
        self.info = load_info(self.root)
        #print(f"[DEBUG LeRobotDatasetMetadata] codebase_version={self.info.get('codebase_version')}, fps={self.info.get('fps')}", flush=True)
        # TODO add new check
        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        #print(f"[DEBUG LeRobotDatasetMetadata] tasks loaded: {len(self.tasks)} tasks", flush=True)
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        
        # Always load episodes
        self.episodes = load_episodes(self.root)
        self._ensure_video_features_from_storage()
        
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        elif self._version >= packaging.version.parse("v3.0"):
            # v3.0: Only load global stats, skip per-episode stats
            self.stats = load_stats(self.root)
            self.episodes_stats = {}
        else:
            # v2.1
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values())) if self.episodes_stats else load_stats(self.root)

    def _infer_video_keys_from_episodes(self) -> set[str]:
        video_keys: set[str] = set()
        for ep_dict in self.episodes.values():
            if not isinstance(ep_dict, dict):
                continue
            for field in ep_dict:
                if not isinstance(field, str) or not field.startswith("videos/"):
                    continue
                parts = field.split("/")
                if len(parts) >= 3 and parts[-1] in {
                    "chunk_index",
                    "file_index",
                    "from_timestamp",
                    "to_timestamp",
                }:
                    video_keys.add(parts[1])
        return video_keys

    def _infer_video_keys_from_disk(self) -> set[str]:
        videos_dir = self.root / "videos"
        if not videos_dir.exists():
            return set()

        video_keys: set[str] = set()
        for video_file in videos_dir.rglob("*.mp4"):
            parts = video_file.relative_to(videos_dir).parts
            if len(parts) >= 3 and parts[0].startswith("chunk-"):
                video_keys.add(parts[1])
            elif len(parts) >= 3 and parts[1].startswith("chunk-"):
                video_keys.add(parts[0])
            elif len(parts) >= 2:
                video_keys.add(parts[0])
        return video_keys

    def _ensure_video_features_from_storage(self) -> None:
        inferred_keys = self._infer_video_keys_from_episodes()
        inferred_keys.update(self._infer_video_keys_from_disk())
        if not inferred_keys:
            return

        if not self.info.get("video_path"):
            self.info["video_path"] = DEFAULT_VIDEO_PATH

        added_keys = []
        features = self.info.setdefault("features", {})
        for key in sorted(inferred_keys):
            if key in features:
                continue
            # Some conversion scripts only write videos on disk and omit video
            # features from meta/info.json. Add lightweight feature entries so
            # image queries are decoded from videos instead of read from parquet.
            features[key] = {
                "dtype": "video",
                "shape": (3, 0, 0),
                "names": ["channels", "height", "width"],
            }
            added_keys.append(key)

        if added_keys and _is_main_rank():
            logging.info(
                "[dataset-init] inferred video features from storage for %s: %s",
                self.root,
                added_keys,
            )

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        print(f"[DEBUG LeRobotDatasetMetadata] pull_from_repo called! repo_id={self.repo_id} — THIS IS THE HF VALIDATION ERROR SOURCE", flush=True)
        print(f"[DEBUG LeRobotDatasetMetadata] This means local metadata was NOT found at: {self.root / 'meta'}", flush=True)
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        # v3.0: use file_index from episode metadata
        if self._version >= packaging.version.parse("v3.0") and ep_index in self.episodes:
            ep_dict = self.episodes[ep_index]
            chunk_index = ep_dict.get("data_chunk_index", ep_dict.get("data/chunk_index", 0))
            file_index = ep_dict.get("data_file_index", ep_dict.get("data/file_index", 0))
            fpath = self.data_path.format(chunk_index=chunk_index, file_index=file_index)
            return Path(fpath)
        
        # v2.1: use episode_chunk and episode_index
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        # v3.0: use file_index from episode metadata  
        if self._version >= packaging.version.parse("v3.0") and ep_index in self.episodes:
            ep_dict = self.episodes[ep_index]
            # v3.0 stores video metadata with keys like "videos/cam_high/chunk_index"
            chunk_key = f"videos/{vid_key}/chunk_index"
            file_key = f"videos/{vid_key}/file_index"
            chunk_index = ep_dict.get(chunk_key, 0)
            file_index = ep_dict.get(file_key, 0)
            fpath = self.video_path.format(video_key=vid_key, chunk_index=chunk_index, file_index=file_index)
            return Path(fpath)
        
        # v2.1: use episode_chunk and episode_index
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        raw_file_name: str | None = None, 
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        if raw_file_name is not None:
            episode_dict["raw_file_name"] = raw_file_name

        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        # TODO(aliberts, rcadene): implement sanity check for features
        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, features, use_videos, robot_type)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1", 
        is_compute_episode_stats_image: bool = True,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.dataset_source = detect_dataset_source(self.root)
        self.camera_key_remap = {}
        self.canonical_to_dataset_camera_key = {}
        self.use_obs_as_action = self.dataset_source in OBS_AS_ACTION_SOURCES
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.video_codec = video_codec
        self.is_compute_episode_stats_image = is_compute_episode_stats_image
        self.delta_indices = None
        self.during_training = True

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

                        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        self._configure_camera_key_remap()
        self.delta_timestamps = self._canonicalize_delta_timestamps(delta_timestamps)
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            if self.meta.episodes_stats:  # v2.1 has episode stats, v3.0 doesn't
                episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
                self.stats = aggregate_stats(episodes_stats)

                # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            file_paths = self.get_episodes_file_paths()
            missing = [fpath for fpath in file_paths if not (self.root / fpath).is_file()]
            if missing:
                print(f"[DEBUG LeRobotDataset.__init__] Missing {len(missing)} data files (first 3): {missing[:3]}", flush=True)
            assert len(missing) == 0
            self.hf_dataset = self.load_hf_dataset()
            #print(f"[DEBUG LeRobotDataset.__init__] hf_dataset loaded from local cache, num_rows={len(self.hf_dataset)}", flush=True)
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            #print(f"[DEBUG LeRobotDataset.__init__] Data files not found locally → calling download_episodes. repo_id={self.repo_id} ← THIS WILL TRIGGER HfValidationError!", flush=True)
            # self.revision = get_safe_version(self.repo_id, self.revision)
            self.revision = CODEBASE_VERSION
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # 预建 episode_index -> local(filtered)index 的 O(1) 映射，避免 __getitem__ 里 list.index() O(N) 线性搜索
        if self.episodes is not None:
            self._ep_idx_to_local = {ep_idx: local for local, ep_idx in enumerate(self.episodes)}
        else:
            self._ep_idx_to_local = None

        self._logical_index_mapper = self._build_logical_index_mapper()
        if _is_main_rank():
            logging.info(
                "[dataset-init] LeRobotDataset root=%s episodes=%s len_hf=%d meta_total_frames=%d logical_index_map=%s",
                self.root,
                "None" if self.episodes is None else len(self.episodes),
                len(self.hf_dataset) if self.hf_dataset is not None else -1,
                self.meta.total_frames,
                "identity"
                if self._logical_index_mapper is None
                else f"{self._logical_index_mapper.num_runs} run(s)",
            )

        # Timestamp synchronization checks are disabled for training. Avoid materializing full
        # timestamp/episode_index columns here; on large pretrain mixtures this can cost many GB.
        # check_timestamps_sync(...)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            # check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        #print(f"[DEBUG LeRobotDataset.get_episodes_file_paths] version={self.meta._version}, checking {len(fpaths)} parquet files. First: {fpaths[0] if fpaths else 'N/A'}", flush=True)
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
            #print(f"[DEBUG LeRobotDataset.get_episodes_file_paths] + {len(video_files)} video files. First video: {video_files[0] if video_files else 'N/A'}", flush=True)

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset = self._select_required_hf_columns(hf_dataset)

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _select_required_hf_columns(self, hf_dataset: datasets.Dataset) -> datasets.Dataset:
        required_columns = {"timestamp", "episode_index", "task_index", "frame_index", "index"}
        if self.delta_timestamps is None:
            required_columns.update(key for key in self.features if key not in self.meta.video_keys)
        else:
            for key in self.delta_timestamps:
                if key in self.meta.video_keys:
                    continue
                if self.use_obs_as_action and key == "action":
                    required_columns.add("observation.state")
                else:
                    required_columns.add(key)

        keep_columns = [col for col in hf_dataset.column_names if col in required_columns]
        if len(keep_columns) == len(hf_dataset.column_names):
            return hf_dataset
        dropped_columns = sorted(set(hf_dataset.column_names).difference(keep_columns))
        if dropped_columns and _is_main_rank():
            logging.info(
                "[dataset-init] dataset=%s selected %d/%d parquet columns; dropped unused columns=%s",
                self.root,
                len(keep_columns),
                len(hf_dataset.column_names),
                dropped_columns,
            )
        return hf_dataset.select_columns(keep_columns)

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _configure_camera_key_remap(self) -> None:
        source_remap = CAMERA_KEY_REMAP.get(self.dataset_source, {})
        remap: dict[str, str] = {}
        canonical_to_dataset: dict[str, str] = {}

        for dataset_key in self.meta.camera_keys:
            canonical_key = source_remap.get(dataset_key, canonical_camera_key(dataset_key))
            if canonical_key != dataset_key:
                remap[dataset_key] = canonical_key
            # Prefer an exact canonical key when both canonical and alias keys exist.
            if canonical_key not in canonical_to_dataset or canonical_key == dataset_key:
                canonical_to_dataset[canonical_key] = dataset_key

        self.camera_key_remap = remap
        self.canonical_to_dataset_camera_key = canonical_to_dataset

    def _to_dataset_camera_key(self, key: str) -> str:
        return self.canonical_to_dataset_camera_key.get(key, key)

    def _canonicalize_delta_timestamps(self, delta_timestamps: dict | None) -> dict | None:
        if delta_timestamps is None or not self.canonical_to_dataset_camera_key:
            return delta_timestamps
        return {self._to_dataset_camera_key(key): value for key, value in delta_timestamps.items()}

    def _standardize_camera_keys(self, item: dict) -> dict:
        if not self.camera_key_remap:
            return item

        for dataset_key, canonical_key in self.camera_key_remap.items():
            if dataset_key in item:
                item[canonical_key] = item.pop(dataset_key)

            dataset_pad_key = f"{dataset_key}_is_pad"
            canonical_pad_key = f"{canonical_key}_is_pad"
            if dataset_pad_key in item:
                item[canonical_pad_key] = item.pop(dataset_pad_key)

        return item

    def _standardize_state_action_dims(self, item: dict) -> dict:
        dim = STATE_ACTION_DIM_SLICE.get(self.dataset_source)
        if dim is None:
            return item

        for key in ("observation.state", "action"):
            value = item.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[-1] > dim:
                item[key] = value[..., :dim]
        return item

    @staticmethod
    def _scalar_to_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        if isinstance(value, np.ndarray):
            return int(value.item())
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected scalar value, got {value}")
            return LeRobotDataset._scalar_to_int(value[0])
        return int(value)

    def _item_logical_index(self, item: dict, physical_idx: int) -> int:
        if "index" in item:
            return self._scalar_to_int(item["index"])
        return physical_idx

    def _build_logical_index_mapper(self) -> _LogicalIndexMapper | None:
        if (
            self.hf_dataset is None
            or self.meta._version < packaging.version.parse("v3.0")
            or "index" not in self.hf_dataset.column_names
        ):
            return None

        data_paths = sorted((self.root / "data").glob("*/*.parquet"))
        if not data_paths:
            return None

        selected_episodes = None
        if self.episodes is not None:
            selected_episodes = np.asarray([int(ep) for ep in self.episodes], dtype=np.int64)

        logical_starts: list[int] = []
        logical_ends: list[int] = []
        physical_starts: list[int] = []
        physical_offset = 0

        for path in data_paths:
            columns = ["index"]
            if selected_episodes is not None:
                columns.append("episode_index")
            table = pq.read_table(path, columns=columns)
            indices = np.asarray(table["index"].to_numpy(zero_copy_only=False), dtype=np.int64)
            episode_indices = None
            if "episode_index" in table.column_names:
                episode_indices = np.asarray(
                    table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64
                )

            if selected_episodes is not None:
                if episode_indices is None:
                    return None
                keep = np.isin(episode_indices, selected_episodes, assume_unique=False)
                indices = indices[keep]

            num_rows = int(indices.shape[0])
            if num_rows == 0:
                continue

            logical_indices = indices
            breaks = np.nonzero(np.diff(logical_indices) != 1)[0] + 1
            run_starts = np.concatenate(
                [np.asarray([0], dtype=np.int64), breaks.astype(np.int64, copy=False)]
            )
            run_ends = np.concatenate(
                [breaks.astype(np.int64, copy=False), np.asarray([num_rows], dtype=np.int64)]
            )

            for run_start, run_end in zip(run_starts, run_ends, strict=True):
                run_start = int(run_start)
                run_end = int(run_end)
                logical_starts.append(int(logical_indices[run_start]))
                logical_ends.append(int(logical_indices[run_end - 1]) + 1)
                physical_starts.append(physical_offset + run_start)

            physical_offset += num_rows

        if physical_offset != len(self.hf_dataset):
            if _is_main_rank():
                logging.warning(
                    "[dataset-init] logical index mapper row count mismatch: root=%s mapper_rows=%d hf_rows=%d",
                    self.root,
                    physical_offset,
                    len(self.hf_dataset),
                )

        if not logical_starts:
            return None

        starts = np.asarray(logical_starts, dtype=np.int64)
        ends = np.asarray(logical_ends, dtype=np.int64)
        physical = np.asarray(physical_starts, dtype=np.int64)

        order = np.argsort(starts, kind="mergesort")
        starts = starts[order]
        ends = ends[order]
        physical = physical[order]

        if starts.shape[0] > 1 and bool(np.any(ends[:-1] > starts[1:])):
            if _is_main_rank():
                logging.warning(
                    "[dataset-init] disabling logical index mapper for root=%s because row['index'] "
                    "contains overlapping ranges; falling back to physical row coordinates.",
                    self.root,
                )
            return None

        if _is_main_rank():
            logging.info(
                "[dataset-init] logical index mapper enabled for root=%s rows=%d runs=%d",
                self.root,
                physical_offset,
                starts.shape[0],
            )

        return _LogicalIndexMapper(starts, ends, physical)

    def _map_query_indices(self, q_idx: list[int]) -> list[int]:
        if self._logical_index_mapper is None:
            return q_idx
        return self._logical_index_mapper.to_physical(q_idx)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        if self.meta._version >= packaging.version.parse("v3.0") and 0 <= ep_idx < len(self.meta.episodes):
            ep = self.meta.episodes[ep_idx]
            ep_start = self._scalar_to_int(ep["dataset_from_index"])
            ep_end = self._scalar_to_int(ep["dataset_to_index"])
            base_idx = idx
        else:
            current_ep_idx = self._ep_idx_to_local[ep_idx] if self._ep_idx_to_local is not None else ep_idx
            ep_start = self.episode_data_index["from"][current_ep_idx].item()
            ep_end = self.episode_data_index["to"][current_ep_idx].item()
            base_idx = idx

        query_indices = {
            key: [max(ep_start, min(ep_end - 1, base_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(base_idx + delta < ep_start) | (base_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(self._map_query_indices(query_indices[key]))["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(self._map_query_indices(q_idx))[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
        result = {}
        processed_indices = set()
        index_to_selected = {}
        for key, q_idx in query_indices.items():
            if key not in self.meta.video_keys :
                if 'images' in key and not self.during_training:
                    continue
                select_idx = self._map_query_indices(q_idx)
                q_idx_tuple = tuple(select_idx)
                if q_idx_tuple not in processed_indices:
                    selected_data = self.hf_dataset.select(select_idx)
                    index_to_selected[q_idx_tuple] = selected_data
                    processed_indices.add(q_idx_tuple)
                else:
                    selected_data = index_to_selected[q_idx_tuple]
                result[key] = torch.stack(selected_data[key])
        # 显式释放 Arrow Table 引用，让 GC 更早回收 select() 产生的 Arrow buffer
        index_to_selected.clear()
        return result

    def _query_hf_dataset_for_training(self, query_indices: dict[str, list[int]]) -> dict:
        if not self.use_obs_as_action or "action" not in query_indices:
            return self._query_hf_dataset_fast(query_indices)

        action_indices = query_indices["action"]
        query_without_action = {key: value for key, value in query_indices.items() if key != "action"}
        result = self._query_hf_dataset_fast(query_without_action)

        obs_action = self._query_hf_dataset_fast({"observation.state": action_indices})
        result["action"] = obs_action["observation.state"]
        return result
    
    # no videos
    def get_episode_data(self, episode_id: int) -> dict:
        ep_start = self.episode_data_index["from"][episode_id].item()
        ep_end = self.episode_data_index["to"][episode_id].item()
        q_idx = list(range(ep_start, ep_end))
        # selected_data = self.hf_dataset.select(q_idx)
        selected_data = self.hf_dataset[q_idx]
        res_keys = set(self.hf_dataset.column_names) - set(self.meta.video_keys)
        res = {key : torch.stack(selected_data[key]) for key in res_keys}
        if self.use_obs_as_action and "observation.state" in res:
            res["action"] = res["observation.state"].clone()
        return res

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        ep_dict = self.meta.episodes[ep_idx] if 0 <= ep_idx < len(self.meta.episodes) else None
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)

            from_ts_key = f"videos/{vid_key}/from_timestamp"
            if ep_dict is not None and from_ts_key in ep_dict:
                from_timestamp = ep_dict[from_ts_key]
                query_ts = [from_timestamp + ts for ts in query_ts]

            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            base_idx = idx
            if self._logical_index_mapper is not None:
                base_idx = self._item_logical_index(item, idx)
            query_indices, padding = self._get_query_indices(base_idx, ep_idx)
            query_result = self._query_hf_dataset_for_training(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0 and self.during_training:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        item = self._standardize_camera_keys(item)
        item = self._standardize_state_action_dims(item)

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        if "coarse_task_index" in item:
            coarse_task_index = item["coarse_task_index"].item()
            item["coarse_task"] = self.meta.tasks[coarse_task_index]

        if "operating_hand_index" in item:
            operating_hand_index = item["operating_hand_index"].item()
            item["operating_hand"] = self.meta.tasks[operating_hand_index]
        
        if "subtask_annotation" in item and hasattr(self.meta, "annotations"):
            index = item["subtask_annotation"][0].item()
            item["subtask"] = self.meta.annotations["subtask"][index]
        
        if "atomic_task_index" in item and item["atomic_task_index"] is not None:
            atomic_task_index = item["atomic_task_index"].item()
            item["subtask"] = self.meta.tasks[int(atomic_task_index)]
            
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    # def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image | bytes, fpath: Path) -> None:
    #     if self.image_writer is None:
    #         if isinstance(image, torch.Tensor):
    #             image = image.cpu().numpy()
    #         write_image(image, fpath)
    #     else:
    #         self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict, task: List[str], timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        assert len(task) == 4, "Task frame must be of two elements"
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                # self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None, raw_file_name: str | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set([item for sublist in tasks for item in sublist]))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["coarse_task_index"] = np.array([self.meta.get_task_index(task[0]) for task in tasks])
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task[1]) for task in tasks])
        episode_buffer["coarse_quality_index"] = np.array([self.meta.get_task_index(task[2]) for task in tasks])
        episode_buffer["quality_index"] = np.array([self.meta.get_task_index(task[3]) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "coarse_task_index", "task_index", "coarse_quality_index", "quality_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features, self.is_compute_episode_stats_image)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, raw_file_name)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    # def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
    #     if isinstance(self.image_writer, AsyncImageWriter):
    #         logging.warning(
    #             "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
    #         )

    #     self.image_writer = AsyncImageWriter(
    #         num_processes=num_processes,
    #         num_threads=num_threads,
    #     )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be picklable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True, vcodec=self.video_codec)

        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1", 
        is_compute_episode_stats_image = True,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        # if image_writer_processes or image_writer_threads:
        #     obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.video_codec = video_codec
        obj.is_compute_episode_stats_image = is_compute_episode_stats_image
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        ds_roots = [Path(ds_dir) for ds_dir in dataset_dirs]
        ds_names = [ds_dir for ds_dir in dataset_dirs]
        self.ds_names = ds_names
        self.ds_roots = ds_roots
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(ds_names, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = []
        total_datasets = len(ds_roots)
        init_start = time.perf_counter()
        log_init_timing = _is_main_rank()
        if log_init_timing:
            logging.info(
                "[dataset-init] Loading %d LeRobot dataset(s). "
                "This happens before DataLoader workers are created, so `num_workers` does not speed up this phase.",
                total_datasets,
            )
        for ds_idx, (ds_root, ds_name) in enumerate(zip(ds_roots, ds_names, strict=True)):
            ds_start = time.perf_counter()
            if log_init_timing:
                elapsed = ds_start - init_start
                avg = elapsed / ds_idx if ds_idx > 0 else 0.0
                eta = avg * (total_datasets - ds_idx) if ds_idx > 0 else 0.0
                logging.info(
                    "[dataset-init] start %d/%d dataset=%s elapsed=%s avg=%s eta=%s",
                    ds_idx + 1,
                    total_datasets,
                    ds_root,
                    _format_duration(elapsed),
                    _format_duration(avg),
                    _format_duration(eta),
                )
            try:
                #print(f"[DEBUG MultiLeRobotDataset.__init__] Creating LeRobotDataset: ds_name={ds_name}, ds_root={ds_root}", flush=True)

                # 支持不同 fps 的子数据集：
                # delta_timestamps 有两种格式：
                #   1. per-dataset dict: {ds_name: {key: [offsets]}}  ← base_lerobot_dataset 传入
                #   2. 全局 dict: {key: [offsets]}  ← 单 fps 场景直接传入
                # 对格式1，直接取各子数据集自己的 delta_timestamps；
                # 对格式2（全局），直接传入，各子数据集用自己的 fps 计算 delta_indices。
                if delta_timestamps is not None and ds_name in delta_timestamps:
                    # 格式1：per-dataset，每个子数据集有独立的偏移量（已按各自 fps 计算好）
                    per_ds_dt = delta_timestamps[ds_name]
                else:
                    # 格式2：全局 delta_timestamps，LeRobotDataset 内部用自身 fps 转换
                    per_ds_dt = delta_timestamps

                _dataset = LeRobotDataset(
                    ds_name,
                    root=ds_root,
                    episodes=episodes[ds_name] if episodes else None,
                    image_transforms=image_transforms,
                    delta_timestamps=per_ds_dt,
                    tolerance_s=self.tolerances_s[ds_name],
                    download_videos=download_videos,
                    video_backend=video_backend,
                )
                #print(f"[DEBUG MultiLeRobotDataset.__init__] LeRobotDataset created OK: fps={_dataset.fps} num_frames={_dataset.num_frames}", flush=True)
                self._datasets.append(_dataset)
                if log_init_timing:
                    ds_elapsed = time.perf_counter() - ds_start
                    elapsed = time.perf_counter() - init_start
                    completed = ds_idx + 1
                    avg = elapsed / completed
                    eta = avg * (total_datasets - completed)
                    logging.info(
                        "[dataset-init] done %d/%d dataset=%s rows=%d frames=%d took=%s elapsed=%s avg=%s eta=%s",
                        completed,
                        total_datasets,
                        ds_root,
                        len(_dataset.hf_dataset),
                        _dataset.num_frames,
                        _format_duration(ds_elapsed),
                        _format_duration(elapsed),
                        _format_duration(avg),
                        _format_duration(eta),
                    )
            except Exception as e:
                # logging.error(e)
                if log_init_timing:
                    logging.error(
                        "[dataset-init] failed %d/%d dataset=%s after %s",
                        ds_idx + 1,
                        total_datasets,
                        ds_root,
                        _format_duration(time.perf_counter() - ds_start),
                    )
                logging.error(f"Exception while process ds_root: {ds_root}, ds_name: {ds_name}")
                traceback.print_exc()
                raise e
        if log_init_timing:
            logging.info(
                "[dataset-init] finished %d/%d dataset(s) in %s",
                len(self._datasets),
                total_datasets,
                _format_duration(time.perf_counter() - init_start),
            )
        self._dataset_num_frames = np.asarray(
            [int(dataset.num_frames) for dataset in self._datasets],
            dtype=np.int64,
        )
        self.cumulative_sizes = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(self._dataset_num_frames, dtype=np.int64)]
        )
        self.total_length = int(self.cumulative_sizes[-1])

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for ds_name, ds in zip(self.ds_names, self._datasets, strict=True):
            # Disable warning of empty extra keys do
            extra_keys = set(ds.features).difference(intersection_features)
            disabled_extra_keys = {
                key for key in extra_keys if not key.startswith("observation.images.")
            }
            if disabled_extra_keys:
                logging.warning(
                    f"keys {disabled_extra_keys} of {ds_name} were disabled as they are not contained in all the "
                    "other datasets."
                )
            self.disabled_features.update(disabled_extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # FastWAM builds one processor/normalizer per underlying dataset from
        # dataset.meta.stats in BaseLerobotDataset. A global aggregate is unused
        # here and is not well-defined for mixed post-train datasets whose stats
        # schemas can differ.
        self.stats = {}

    def set_during_training(self, during_training: bool):
        for dataset in self._datasets:
            dataset.during_training = during_training

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.ds_names)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        若所有子数据集 fps 相同则返回该值；若不同则返回最大 fps（用于 tolerance_s 等场景）。
        各子数据集已通过 per-dataset delta_timestamps 独立处理自己的 fps，此属性仅作参考。
        """
        fps_set = {ds.meta.info["fps"] for ds in self._datasets}
        if len(fps_set) == 1:
            return fps_set.pop()
        # 多 fps 场景：返回最大值，调用方不应依赖此值做时间戳计算
        return max(fps_set)

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return self.total_length

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4
    
    def get_episode_data(self, episode_idx: int, columns: list[str] | None = None) -> dict:
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                dataset_episode_idx = dataset.episodes[episode_idx] if dataset.episodes is not None else episode_idx
                file = str(dataset.root / dataset.meta.get_data_file_path(dataset_episode_idx))
                read_columns = columns
                needs_obs_as_action = dataset.use_obs_as_action and (
                    columns is None or "action" in columns
                )
                if needs_obs_as_action and columns is not None:
                    read_columns = [
                        "observation.state" if col == "action" else col
                        for col in columns
                    ]
                    read_columns = list(dict.fromkeys(read_columns))
                    if "observation.state" not in read_columns:
                        read_columns.append("observation.state")
                table = pq.read_table(str(file), columns=read_columns)

                result_dict = {}
                for col_name in table.column_names:
                    col = table[col_name]
                    try:
                        np_arr = col.to_numpy(zero_copy_only=True)
                    except Exception:
                        raw = col.to_numpy()
                        np_arr = np.stack(raw) if raw.dtype == object else raw
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The given NumPy array is not writable",
                            category=UserWarning,
                        )
                        # deal with string in parquet file
                        if np_arr.dtype == 'O':                        
                            result_dict[col_name] = np_arr
                        else:
                            result_dict[col_name] = torch.from_numpy(np_arr)
                if needs_obs_as_action and "observation.state" in result_dict:
                    result_dict["action"] = result_dict["observation.state"].clone()
                return result_dict
            else:
                episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx, side="right") - 1)
        local_idx = int(idx - self.cumulative_sizes[dataset_idx])
        item = self._datasets[dataset_idx][local_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Dataset Names: '{self.ds_names}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
