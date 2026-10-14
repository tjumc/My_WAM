"""Offline tests: no robot data, model checkpoint, or GPU required."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "pair_replay", Path(__file__).resolve().parents[1] / "tools/export_ambiguity_pair_videos.py"
)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def sample_pair(root="/datasets/demo"):
    pair = {"rank": 1, "episode": 999, "frame_a": 10, "frame_b": 20, "action_rmse": 0.6}
    for side, frame in (("a", 10), ("b", 20)):
        pair[f"source_{side}"] = (
            f"dataset_idx=0, dataset_root={root}, global_idx={100+frame}, local_idx={100+frame}, "
            f"episode_index=7, frame_index={frame}, absolute_index={100+frame}, "
            f"timestamp={frame/30:.6f}, data_path={root}/data/chunk-000/file-000.parquet, "
            f"video_paths=['{root}/videos/head.mp4', '{root}/videos/left.mp4']"
        )
    return pair


def test_source_prefers_real_episode_not_ordinal():
    s = replay.parse_source(sample_pair(), "a")
    assert s["episode"] == 7 and s["frame"] == 10
    assert s["root"] == "/datasets/demo"


def test_mount_override_does_not_change_input():
    pair = sample_pair()
    before = copy.deepcopy(pair)
    assert replay.parse_source(pair, "b", "/new/mount")["root"] == "/new/mount"
    assert pair == before


def test_source_missing_or_disagrees_fails():
    with pytest.raises(ValueError):
        replay.parse_source({"source_a": "episode_index=7"}, "a")
    pair = sample_pair()
    pair["frame_a"] = 11
    with pytest.raises(ValueError, match="alignment"):
        replay.parse_source(pair, "a")


def test_rank_selection_and_dedup():
    one = sample_pair()
    nine = {**sample_pair(), "rank": 9}
    assert replay.select_pairs({"pairs": [nine, one]}, [1, 9, 1]) == [one, nine]
    with pytest.raises(ValueError):
        replay.select_pairs({"pairs": [one]}, [9])
    with pytest.raises(ValueError):
        replay.select_pairs({"pairs": [one, one]}, [1])


def test_relative_time_grid_is_centered():
    relative = replay.relative_grid(3, 3, 10)
    assert len(relative) == 61
    assert relative[30] == 0 and relative[0] == -3 and relative[-1] == 3


@pytest.mark.parametrize("before,after,fps", [(-1, 3, 10), (3, float("nan"), 10), (3, 3, 0)])
def test_invalid_grid(before, after, fps):
    with pytest.raises(ValueError):
        replay.relative_grid(before, after, fps)


def test_episode_boundary_is_missing_not_clamped():
    frames = np.arange(40)
    source = {"frame": 10, "timestamp": 10/30}
    relative = replay.relative_grid(1, 1, 10)
    target, rows = replay.locate_frames(frames, frames/30, source, relative, 30, .002)
    assert rows[10] == 10 and target[10] == 10
    assert rows[0] == -1 and rows[-1] == -1
    assert np.all(target[rows < 0] < 0) or target[-1] >= len(frames)


def test_timestamp_mismatch_and_internal_gap_fail():
    frames = np.arange(40)
    relative = replay.relative_grid(.1, .1, 10)
    with pytest.raises(ValueError, match="timestamp differs"):
        replay.locate_frames(frames, frames/30, {"frame": 10, "timestamp": 5}, relative, 30, .002)
    frames = np.delete(frames, 13)
    with pytest.raises(ValueError, match="Missing frame inside"):
        replay.locate_frames(frames, frames/30, {"frame": 10, "timestamp": 10/30}, relative, 30, .002)


def test_v3_nonzero_camera_offset_and_missing_offset():
    meta = types.SimpleNamespace(root=Path("/tmp/data"), info={"codebase_version": "v3.0"},
                                 episodes={7: {"videos/head/from_timestamp": 12.5, "videos/head/to_timestamp": 15}},
                                 get_video_file_path=lambda episode, key: Path(f"videos/{key}/file.mp4"))
    path, offset, end = replay.video_location(meta, 7, "head")
    assert str(path) == "/tmp/data/videos/head/file.mp4" and offset == 12.5 and end == 15
    with pytest.raises(ValueError, match="Missing v3"):
        replay.video_location(meta, 7, "left")


def test_v2_episode_file_zero_offset():
    meta = types.SimpleNamespace(root=Path("/tmp/data"), info={"codebase_version": "v2.1"},
                                 episodes={7: {}}, get_video_file_path=lambda episode, key: Path("episode_7.mp4"))
    assert replay.video_location(meta, 7, "head")[1:] == (0, None)


def fake_information(source, relative):
    return {**source, "requested_frames": (source["frame"] + np.rint(relative*30).astype(int)).tolist(),
            "episode_timestamps": (source["timestamp"] + relative).tolist(), "missing_context_frames": 0}


def test_playback_pause_and_missing_panel():
    relative = replay.relative_grid(.1, .1, 10)
    sources = [fake_information(replay.parse_source(sample_pair(), side), relative) for side in ("a", "b")]
    images = [None, Image.new("RGB", (320,384)), Image.new("RGB", (320,384))]
    result = list(replay.playback(images, images, sources, relative, 1, 10))
    assert len(result) == 13 and result[0].size == (640,480)
    assert np.array_equal(np.asarray(result[1]), np.asarray(result[11]))


def test_decoder_receives_v3_offsets_not_frame_over_fps_only(tmp_path):
    torch = pytest.importorskip("torch")
    camera_names = ["head", "left", "right"]
    meta = types.SimpleNamespace(root=tmp_path, info={"codebase_version": "v3.0"},
        episodes={7: {f"videos/{key}/from_timestamp": offset for key, offset in zip(camera_names, [10, 20, 30])}},
        get_video_file_path=lambda episode, key: Path(f"{key}.mp4"))
    for key in camera_names:
        (tmp_path / f"{key}.mp4").touch()
    reader = replay.EpisodeReader.__new__(replay.EpisodeReader)
    reader.meta, reader.fps, reader.cameras = meta, 30, camera_names
    reader.episode_rows = lambda ep: (np.arange(31), np.arange(31)/30)
    called = []
    module = types.ModuleType("fake_video_utils")
    def decode(path, queries, tolerance, backend):
        called.append(queries)
        return torch.zeros((len(queries), 3, 8, 8))
    module.decode_video_frames = decode
    with mock.patch.dict(sys.modules, {"fastwam.datasets.lerobot.lerobot.datasets.video_utils": module}):
        source = {"root": str(tmp_path), "episode": 7, "frame": 10, "timestamp": 10/30}
        images, info = reader.read(source, replay.relative_grid(1, 1, 10), .002, "pyav")
    assert images[0] is None and info["missing_context_frames"] > 0
    assert len(called) == 3
    assert np.allclose(np.asarray(called[1])-np.asarray(called[0]), 10)
    assert np.allclose(np.asarray(called[2])-np.asarray(called[0]), 20)


def test_json_never_overwrites(tmp_path):
    path = tmp_path / "test.json"
    replay.write_json(path, {"test": 1})
    with pytest.raises(FileExistsError):
        replay.write_json(path, {"test": 2})
    assert json.loads(path.read_text()) == {"test": 1}


def test_end_to_end_with_synthetic_reader_and_real_mp4(tmp_path, monkeypatch):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    pair = sample_pair(str(tmp_path / "raw_data"))
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs": [pair]}))
    original = path.read_bytes()
    class SyntheticReader:
        fps = 30
        def __init__(self, root, cameras):
            pass
        def read(self, source, relative, tolerance, backend):
            info = fake_information(source, relative)
            images = [Image.new("RGB", (320, 384), (int(40+100*(r+0.2)), 50, 80)) for r in relative]
            return images, info
    module = types.ModuleType("fake_writer")
    def save_mp4(frames, dest, fps):
        with imageio.get_writer(dest, fps=fps, codec="libx264", format="FFMPEG", pixelformat="yuv420p") as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
    module.save_mp4 = save_mp4
    monkeypatch.setattr(replay, "EpisodeReader", SyntheticReader)
    with mock.patch.dict(sys.modules, {"fastwam.utils.video_io": module}):
        args = ["--pairs-json", str(path), "--ranks", "1", "--before-seconds", ".2", "--after-seconds", ".2",
                "--fps", "10", "--pause-seconds", ".2", "--output-dir", str(tmp_path / "replay")]
        replay.main(args)
        with pytest.raises(FileExistsError):
            replay.main(args)
    assert original == path.read_bytes()
    with imageio.get_reader(str(tmp_path / "replay/pair_0001.mp4"), format="FFMPEG") as video:
        assert video.count_frames() == 7
        assert video.get_data(0).shape == (480, 640, 3)
    assert (tmp_path / "replay/pair_0001_current.png").is_file()
    assert (tmp_path / "replay/review.csv").is_file()
    assert json.loads((tmp_path / "replay/manifest.json").read_text())["exported"][0]["status"] == "unreviewed"
