"""Synthetic/unit tests only; no WAM checkpoint, robot, or real dataset required.

Run: python -m pytest tests/test_infer_ambiguity_pairs.py -q
"""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import infer_ambiguity_pairs as m


def stats():
    return {k: {"default": {"global_mean": np.zeros(25).tolist(),
                            "global_std": np.ones(25).tolist()}} for k in ("state", "action")}


def data(length=50):
    return {"state": np.arange(length * 25, dtype=np.float32).reshape(length, 25) / 500,
            "action": np.arange(length * 25, dtype=np.float32).reshape(length, 25) / 700,
            "time": np.arange(length) / 30, "frame": np.arange(length), "length": length, "fps": 30.}


def source(frame=4):
    return {"root": "/a-dataset", "episode": 4, "frame": frame, "timestamp": frame / 30}


def sample(horizon=4, current=.1):
    return {"video": torch.zeros(3, 9, 32, 32), "proprio": torch.full((horizon, 32), current),
            "context": torch.ones(4, 6), "context_mask": torch.ones(4, dtype=torch.bool),
            "action": torch.zeros(horizon, 32), "image_is_pad": torch.zeros(9, dtype=torch.bool),
            "action_is_pad": torch.zeros(horizon, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(horizon, dtype=torch.bool)}


def scored_inputs(n=2, horizon=4):
    gt = np.zeros((2, horizon, 32), dtype=np.float32)
    gt[1, :, :25] = 1
    return np.repeat(gt[None], n, axis=0), gt, list(range(42, 42 + n))


class FakeModel:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self, mutate=False, bad_shape=False):
        self.calls = []
        self.mutate, self.bad_shape = mutate, bad_shape

    def infer_action(self, **kwargs):
        self.calls.append(kwargs)
        generator = torch.Generator().manual_seed(kwargs["seed"])
        result = torch.randn(kwargs["action_horizon"], 32, generator=generator) + kwargs["proprio"][0, 0]
        if self.mutate:
            kwargs["input_image"].add_(1)
            kwargs["proprio"].add_(100)
        return {"action": result[:, :24] if self.bad_shape else result}


def test_future_gt_never_changes_model_observation():
    a = sample()
    b = copy.deepcopy(a)
    b["video"][:, 1:] = float("nan")
    b["proprio"][1:] = float("nan")
    b["action"] = object()  # accessing GT in extract_observation would be a bug
    oa, ob = m.extract_observation(a), m.extract_observation(b)
    for name in ("image", "proprio", "context", "context_mask"):
        assert torch.equal(getattr(oa, name), getattr(ob, name))
    b["video"][:, 0] += .2
    assert not torch.equal(oa.image, m.extract_observation(b).image)


def test_observation_owns_current_only_storage():
    a = sample()
    obs = m.extract_observation(a)
    assert obs.image.shape == (1, 3, 32, 32)
    a["video"][:, 0] += .3
    assert torch.count_nonzero(obs.image) == 0
    assert obs.image.untyped_storage().nbytes() == obs.image.numel() * obs.image.element_size()


@pytest.mark.parametrize("field", ["video", "proprio", "context"])
def test_nonfinite_current_input_rejected(field):
    a = sample()
    a[field].fill_(float("nan"))
    with pytest.raises(ValueError):
        m.extract_observation(a)


def test_reference_has_h_actions_and_h_plus_one_real_states():
    ref = m.reference_window(data(), source(4), 32, stats(), .002)
    assert ref["raw_gt"].shape == (32, 25)
    assert ref["raw_states"].shape == (33, 25)
    np.testing.assert_allclose(ref["relative_times"][-1], 31 / 30)
    np.testing.assert_allclose(ref["state_relative_times"][-1], 32 / 30)


def test_episode_end_is_not_padded_with_stop_actions():
    with pytest.raises(ValueError, match="incomplete"):
        m.reference_window(data(), source(18), 32, stats(), .002)
    m.reference_window(data(), source(17), 32, stats(), .002)


def test_source_timestamp_mismatch_rejected():
    bad = source()
    bad["timestamp"] += 1
    with pytest.raises(ValueError):
        m.reference_window(data(), bad, 4, stats(), .002)


def matching_sample():
    settings = dict(horizon=4, action_model_dim=32, state_model_dim=32)
    ref = m.reference_window(data(), source(), 4, stats(), .002)
    s = sample()
    s["action"] = torch.from_numpy(np.pad(ref["normalized_gt"], ((0, 0), (0, 7))))
    s["proprio"] = torch.zeros(4, 32)
    s["proprio"][0, :25] = torch.from_numpy(ref["current_norm"])
    return s, ref, settings


def test_dataset_and_independent_parquet_reference_must_agree():
    s, ref, settings = matching_sample()
    m.verify_sample(s, ref, settings)
    s["action"][0, 2] += .1
    with pytest.raises(ValueError, match="Parquet"):
        m.verify_sample(s, ref, settings)


@pytest.mark.parametrize("field", ["image_is_pad", "action_is_pad", "proprio_is_pad"])
def test_any_padded_reference_rejected(field):
    s, ref, settings = matching_sample()
    s[field][0] = True
    with pytest.raises(ValueError, match="Padded"):
        m.verify_sample(s, ref, settings)


def test_nonzero_dimension_padding_rejected():
    s, ref, settings = matching_sample()
    s["action"][0, 31] = 1
    with pytest.raises(ValueError, match="padding"):
        m.verify_sample(s, ref, settings)


def test_inference_allowlist_and_shared_seeds():
    obs = [m.extract_observation(sample(current=.1)), m.extract_observation(sample(current=.4))]
    model = FakeModel()
    pred, timing = m.run_inference(model, obs, [42, 43], 4, 32, 10, None, False)
    assert pred.shape == (2, 2, 4, 32) and timing.shape == (2, 2)
    assert [c["seed"] for c in model.calls] == [42, 42, 43, 43]
    allowed = {"prompt", "input_image", "proprio", "context", "context_mask", "action_horizon",
               "negative_prompt", "text_cfg_scale", "num_inference_steps", "sigma_shift", "seed",
               "rand_device", "tiled"}
    assert all(set(call) == allowed for call in model.calls)
    np.testing.assert_allclose(pred[:, 1] - pred[:, 0], .3, atol=3e-7)


def test_model_cannot_mutate_saved_observation_between_seeds():
    obs = [m.extract_observation(sample()) for _ in range(2)]
    expected, _ = m.run_inference(FakeModel(), obs, [42, 43], 4, 32, 10, None, False)
    actual, _ = m.run_inference(FakeModel(mutate=True), obs, [42, 43], 4, 32, 10, None, False)
    np.testing.assert_array_equal(expected, actual)
    assert obs[0].proprio[0, 0] == pytest.approx(.1)


def test_model_shape_mismatch_rejected():
    obs = [m.extract_observation(sample()) for _ in range(2)]
    with pytest.raises(ValueError, match="prediction"):
        m.run_inference(FakeModel(bad_shape=True), obs, [42], 4, 32, 10, None, False)


def test_metrics_exclude_all_seven_padding_dimensions():
    pred, gt, seeds = scored_inputs()
    pred[..., 25:] = 1000
    rows, pair_rows, groups, summary = m.score_predictions(pred, gt, seeds)
    assert all(row["own_gt_rmse_active25"] == 0 for row in rows)
    assert all(row["other_gt_rmse_active25"] == 1 for row in rows)
    assert all(row["prediction_to_gt_separation_ratio"] == 1 for row in pair_rows)
    assert summary["active_dimensions"] == 25
    assert len(groups) == 2 * 2 * len(m.GROUPS)


def test_collapse_and_cross_gt_preference_are_descriptive():
    pred, gt, seeds = scored_inputs()
    pred[:, 1, :, :25] = 0
    rows, pairs, _, summary = m.score_predictions(pred, gt, seeds)
    assert summary["sides"]["B"]["other_gt_closer_seed_count"] == 2
    assert summary["sides"]["A"]["other_gt_closer_seed_count"] == 0
    assert all(row["prediction_pair_rmse_active25"] == 0 for row in pairs)
    assert "failure_rate" not in summary
    assert [row["other_minus_own_rmse"] for row in rows] == [1, -1, 1, -1]


def test_equal_gt_no_divide_by_zero_or_fake_branch_failure():
    pred, gt, seeds = scored_inputs()
    gt[:] = 0
    rows, pair_rows, _, summary = m.score_predictions(pred, gt, seeds)
    assert all(x["prediction_to_gt_separation_ratio"] is None for x in pair_rows)
    assert all(x["gt_preference_not_failure_label"] == "tie" for x in rows)
    assert summary["sides"]["B"]["other_gt_closer_seed_count"] == 0


def test_nan_prediction_rejected():
    pred, gt, seeds = scored_inputs()
    pred[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        m.score_predictions(pred, gt, seeds)


def test_prediction_inverse_is_not_clipped():
    s = stats()
    s["action"]["default"]["global_mean"] = [2.] * 25
    s["action"]["default"]["global_std"] = [3.] * 25
    p = np.full((1, 2, 4, 32), 8, dtype=np.float32)
    raw = m.denormalize_predictions(p, s)
    np.testing.assert_allclose(raw, 26., atol=1e-5)
    assert raw.shape[-1] == 25


class FakeTable:
    def __init__(self, rows): self.rows = rows
    def select_columns(self, columns): return self
    def with_format(self, fmt): return self
    def __getitem__(self, key):
        return [row[key] for row in self.rows] if isinstance(key, str) else self.rows[key]


def test_source_lookup_uses_actual_ids_not_old_global_offsets():
    rows = [{"episode_index": 4, "frame_index": f, "timestamp": f / 30} for f in [2, 0, 1]]
    raw = SimpleNamespace(multi_dataset=SimpleNamespace(_datasets=[
        SimpleNamespace(root=Path("/other"), num_frames=10),
        SimpleNamespace(root=Path("/a-dataset"), num_frames=3, hf_dataset=FakeTable(rows))]))
    assert m.locate_source_index(raw, source(1), .002) == 12
    rows.append(dict(rows[-1]))
    with pytest.raises(ValueError, match="found 2"):
        m.locate_source_index(raw, source(1), .002)


def test_mining_metric_reproduction_detects_changed_input():
    s, ref, settings = matching_sample()
    obs = m.extract_observation(s)
    verified = m.verify_pair_metrics({}, [obs, obs], [ref, ref], settings, [24, 20])
    assert all(v == 0 for v in verified.values())
    with pytest.raises(ValueError, match="mined"):
        m.verify_pair_metrics({"proprio_rmse": .5}, [obs, obs], [ref, ref], settings, [24, 20])


def test_never_overwrite_mining_or_raw_dataset(tmp_path):
    mining = tmp_path / "mining"
    mining.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    for path in (mining, raw, raw / "nested"):
        with pytest.raises(ValueError):
            m.create_output(path, mining / "pairs.json", {str(raw)})
    output = mining / "pair_inference" / "test"
    output.mkdir(parents=True)
    (output / "run.log").touch()
    (output / "launcher.sh").touch()
    m.create_output(output, mining / "pairs.json", {str(raw)})
    assert (output / "STARTED.json").is_file()
    with pytest.raises(FileExistsError):
        m.create_output(output, mining / "pairs.json", {str(raw)})


def test_synthetic_inference_to_saved_report(tmp_path):
    refs = [m.reference_window(data(), source(i), 4, stats(), .002) for i in (4, 12)]
    obs = [m.extract_observation(sample(current=x)) for x in (.1, .3)]
    pred, timings = m.run_inference(FakeModel(), obs, [42, 43], 4, 32, 10, None, False)
    out = tmp_path / "pair_0002"
    summary = m.save_pair(out, {"rank": 2}, obs, refs, pred, timings, [42, 43], stats(), {}, "velocity", False)
    assert summary["num_seeds"] == 2
    for name in ("report.md", "summary.json", "seed_metrics.csv", "paired_seed_metrics.csv",
                 "group_metrics.csv", "actions.npz", "predictions_raw.csv", "policy_inputs.npz", "current_policy_inputs.png"):
        assert (out / name).is_file(), name
    with np.load(out / "actions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["prediction_normalized"], pred)
        assert saved["actual_state_raw25"].shape == (2, 5, 25)
        np.testing.assert_array_equal(saved["gt_raw25"][0], refs[0]["raw_gt"])
    import json
    json.loads((out / "summary.json").read_text(), parse_constant=lambda x: pytest.fail(x))
    assert "No held-out split" in summary["interpretation"]
    assert "真机失败" in (out / "report.md").read_text()


def test_plot_smoke(tmp_path):
    refs = [m.reference_window(data(), source(i), 4, stats(), .002) for i in (4, 12)]
    pred, gt, seeds = scored_inputs()
    rows, _, _, _ = m.score_predictions(pred, gt, seeds)
    # Two representative joint dimensions suffice for plotting coverage.
    m.plot_results(tmp_path, m.denormalize_predictions(pred, stats()), refs, ["base_vx", "base_vy"], rows)
    assert (tmp_path / "joint_curves/00_base_vx.png").is_file()
    assert (tmp_path / "side_b_seed_errors.png").is_file()


def test_launcher_syntax_and_safe_defaults():
    import subprocess
    launcher = Path(__file__).resolve().parents[1] / "scripts/infer_ambiguity_pairs.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert "PAIR_RANKS=(2)" in text and "NUM_SEEDS=10" in text
    assert "git reset" not in text and "git pull" not in text
    assert "--checkpoint" in text and 'tee "$OUT/run.log"' in text


@pytest.mark.parametrize("exit_code", [0, 7])
def test_launcher_argument_forwarding_and_error_propagation(tmp_path, exit_code):
    import json
    import os
    import subprocess
    root = tmp_path / "workspace with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools/infer_ambiguity_pairs.py").touch()
    stub = tmp_path / "fake_python"
    stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                    "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv))\n"
                    "sys.exit(int(os.environ['STUB_EXIT']))\n")
    stub.chmod(0o755)
    original = Path(__file__).resolve().parents[1] / "scripts/infer_ambiguity_pairs.sh"
    text = original.read_text().replace('PYTHON_BIN="python"', f'PYTHON_BIN="{stub}"')
    launcher = root / "scripts/infer_ambiguity_pairs.sh"
    launcher.write_text(text)
    capture = tmp_path / "arguments.json"
    result = subprocess.run(["bash", str(launcher)], cwd=tmp_path, text=True, capture_output=True,
                            env={**os.environ, "CAPTURE": str(capture), "STUB_EXIT": str(exit_code)})
    assert result.returncode == exit_code
    args = json.loads(capture.read_text())
    assert args[args.index("--ranks") + 1] == "2"
    assert args[args.index("--num-seeds") + 1] == "10"
    output = Path(args[args.index("--output-dir") + 1])
    assert output.is_relative_to(root)
    assert (output / "launcher.sh").is_file() and (output / "run.log").is_file()
    if exit_code:
        assert "推理未完成" in result.stderr
