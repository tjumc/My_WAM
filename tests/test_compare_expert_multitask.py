"""CPU-only tests for the expert-vs-multitask offline comparison."""
import argparse
import ast
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "compare_expert_multitask.py"
spec = importlib.util.spec_from_file_location("wam_compare", PATH)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


@unittest.skipUnless(pq is not None, "Parquet regression fixtures require pyarrow")
class TestEpisodeActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real numeric query and mapper methods without importing
        # the video/HF/model dependency stack. Only torch.stack is CPU-emulated.
        path = ROOT / "src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py"
        tree = ast.parse(path.read_text())
        mapper = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "_LogicalIndexMapper")
        dataset = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and n.name == "LeRobotDataset")
        names = {"_map_query_indices", "_query_hf_dataset_fast",
                 "_query_hf_dataset_for_training"}
        methods = [n for n in dataset.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"np": np, "torch": SimpleNamespace(stack=np.stack)}
        exec(compile(ast.Module(body=[mapper, *methods], type_ignores=[]), str(path), "exec"), namespace)
        cls.numeric_dataset = type("NumericLeRobotDataset", (),
                                   {n.name: namespace[n.name] for n in methods})
        cls.mapper_type = namespace["_LogicalIndexMapper"]

    def make_data(self, lengths=(35, 37), order=None, obs_as_action=False, edit=None, key="action"):
        rows = []
        for episode, length in enumerate(lengths):
            for frame in range(length):
                index = len(rows)
                rows.append({"index": index, "episode_index": episode, "frame_index": frame,
                             key: (100 * index + np.arange(25)).tolist(),
                             "observation.state": (-100 * index - np.arange(25) - 1).tolist()})
        if edit:
            edit(rows)
        order = list(range(len(rows))) if order is None else list(order)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        file = root / "shared.parquet"
        pq.write_table(pa.Table.from_pylist([rows[i] for i in order]), file)

        class ParquetRows:
            def __init__(self, table):
                self.table = table

            def select(self, indices):
                return ParquetRows(self.table.take(pa.array(indices, type=pa.int64())))

            def __getitem__(self, column):
                return [np.asarray(value) for value in self.table[column].to_pylist()]

        dataset = self.numeric_dataset()
        dataset.root, dataset.episodes = root, None
        dataset.episode_data_index = {"from": np.cumsum([0, *lengths[:-1]]),
                                      "to": np.cumsum(lengths)}
        dataset.meta = SimpleNamespace(video_keys=[])
        dataset.during_training, dataset.use_obs_as_action = False, obs_as_action
        dataset.hf_dataset = ParquetRows(pq.read_table(file))
        dataset._logical_index_mapper = None if order == list(range(len(rows))) else self.mapper_type(
            np.arange(len(rows)), np.arange(len(rows)) + 1, np.argsort(order))

        def legacy_read(episode, columns):
            table = pq.read_table(file, columns=columns)
            return {column: np.stack(table[column].to_pylist()) for column in columns}

        multi = SimpleNamespace(_datasets=[dataset], get_episode_data=Mock(side_effect=legacy_read))
        data = SimpleNamespace(multi_dataset=multi, episode_data_index=dataset.episode_data_index,
                               action_meta=[{"lerobot_key": key, "key": "default"}])
        return data, rows

    def test_shared_parquet_selects_only_target_episode(self):
        data, rows = self.make_data()
        # Both episodes really share a file; the legacy reader returns all 72 rows.
        self.assertEqual(data.multi_dataset.get_episode_data(1, ["action"])["action"].shape, (72, 25))
        data.multi_dataset.get_episode_data.reset_mock()
        for ep in (0, 1):
            expected = [row["action"] for row in rows if row["episode_index"] == ep]
            actual = compare.episode_actions(data, ep)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)
        data.multi_dataset.get_episode_data.assert_not_called()

    def test_logical_order_differs_from_physical_rows(self):
        order = np.random.default_rng(42).permutation(72).tolist()
        data, rows = self.make_data(order=order)
        np.testing.assert_array_equal(compare.episode_actions(data, 1),
                                      [row["action"] for row in rows[35:]])
        # Observations use physical rows, whereas GT queries use logical indices.
        self.assertEqual(compare.episode_frame_index(data, 1, 4), order.index(39))
        self.assertNotEqual(order.index(39), 39)
        with self.assertRaisesRegex(RuntimeError, "silently replaced requested index"):
            compare.raw_observation(data, {"idx": 39}, 4, 1, np.zeros((37, 25)))

    def test_normal_single_episode(self):
        data, rows = self.make_data(lengths=(40,))
        actual = compare.episode_actions(data, 0)
        np.testing.assert_array_equal(actual, [row["action"] for row in rows])
        self.assertEqual(actual.shape, (40, 25))
        self.assertEqual(compare.episode_frame_index(data, 0, 7), 7)

    def test_mapped_observation_identity_and_current_gt(self):
        order = np.random.default_rng(42).permutation(72).tolist()
        data, _ = self.make_data(order=order)
        data.state_meta = [{"key": "default"}]
        data.image_meta = [{"key": str(i)} for i in range(3)]
        gt = compare.episode_actions(data, 1)

        class CpuTensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            cpu = detach

            def numpy(self):
                return self.value

            def permute(self, *axes):
                return CpuTensor(self.value.transpose(axes))

        physical = compare.episode_frame_index(data, 1, 4)
        source = data.multi_dataset._datasets[0].hf_dataset.select([physical])
        state = source["observation.state"][0]
        sample = {"idx": physical, "episode_index": 1, "frame_index": 4,
                  "task": "recorded task", "action": {"default": source["action"]},
                  "state": {"default": [CpuTensor(state)]},
                  "images": {str(i): [CpuTensor(np.zeros((3, 2, 2), dtype=np.uint8))]
                             for i in range(3)}}
        images, actual_state, prompt = compare.raw_observation(data, sample, 4, 1, gt)
        self.assertEqual(len(images), 3)
        np.testing.assert_array_equal(actual_state, state)
        self.assertEqual(prompt, "recorded task")
        for changed in ({"episode_index": 0}, {"frame_index": 5}):
            with self.assertRaisesRegex(RuntimeError, "Dataset source mismatch"):
                compare.raw_observation(data, {**sample, **changed}, 4, 1, gt)
        sample["action"]["default"] = [np.zeros(25)]
        with self.assertRaisesRegex(RuntimeError, "Current GT mismatch"):
            compare.raw_observation(data, sample, 4, 1, gt)

    def test_preserves_observation_as_action_with_mapping(self):
        order = np.random.default_rng(7).permutation(72).tolist()
        data, rows = self.make_data(order=order, obs_as_action=True)
        np.testing.assert_array_equal(compare.episode_actions(data, 1),
                                      [row["observation.state"] for row in rows[35:]])

    def test_uses_configured_action_column(self):
        data, rows = self.make_data(lengths=(40,), key="action.joints")
        np.testing.assert_array_equal(compare.episode_actions(data, 0),
                                      [row["action.joints"] for row in rows])

    def test_wrong_episode_identity_rejected(self):
        data, _ = self.make_data(edit=lambda rows: rows[38].update(episode_index=0))
        with self.assertRaisesRegex(RuntimeError, "GT identity mismatch"):
            compare.episode_actions(data, 1)

    def test_bad_frame_order_rejected(self):
        for frame in (0, 5, 99):
            with self.subTest(frame=frame):
                data, _ = self.make_data(edit=lambda rows: rows[38].update(frame_index=frame))
                with self.assertRaisesRegex(RuntimeError, "GT frame order mismatch"):
                    compare.episode_actions(data, 1)

    def test_action_dimension_and_length_rejected(self):
        data, _ = self.make_data(edit=lambda rows: [row.update(action=[0] * 24) for row in rows])
        with self.assertRaisesRegex(RuntimeError, "Unexpected episode GT actions"):
            compare.episode_actions(data, 1)
        data, _ = self.make_data()
        source = data.multi_dataset._datasets[0]
        query = source._query_hf_dataset_for_training
        source._query_hf_dataset_for_training = lambda indices: {
            key: value[:-1] if key == "action" else value for key, value in query(indices).items()}
        with self.assertRaisesRegex(RuntimeError, "Unexpected episode GT actions"):
            compare.episode_actions(data, 1)

    def test_invalid_source_and_indices_rejected(self):
        data, _ = self.make_data()
        for ep in (-1, 2):
            with self.assertRaisesRegex(RuntimeError, "Episode index out of bounds"):
                compare.episode_actions(data, ep)
        for frame in (-1, 37):
            with self.assertRaisesRegex(RuntimeError, "Frame index out of bounds"):
                compare.episode_frame_index(data, 1, frame)
        source = data.multi_dataset._datasets[0]
        original_end = source.episode_data_index["to"][0]
        source.episode_data_index["to"][0] = 0
        with self.assertRaisesRegex(RuntimeError, "Invalid episode bounds"):
            compare.episode_actions(data, 0)
        source.episode_data_index["to"][0] = original_end
        source.episodes = [0, 1]
        with self.assertRaisesRegex(RuntimeError, "one unfiltered dataset"):
            compare.episode_actions(data, 0)
        source.episodes = None
        data.multi_dataset._datasets.append(source)
        with self.assertRaisesRegex(RuntimeError, "one unfiltered dataset"):
            compare.episode_actions(data, 0)


class TestSampling(unittest.TestCase):
    def test_exact_horizon_safe_windows(self):
        self.assertEqual(compare.choose_starts(33, 8), [0, 1])
        self.assertEqual(compare.choose_starts(64, 3), [0, 16, 32])
        self.assertEqual(compare.choose_starts(32, 3), [])
        self.assertEqual(compare.choose_starts(34, 1), [0])

    def test_per_directory_coverage_and_reproducibility(self):
        folders = {
            "/a": [(0, 100), (1, 110), (2, 120)],
            "/b": [(0, 100)],
            "/c": [(0, 100), (1, 110)],
        }
        chosen = compare.choose_episodes(folders, 5, 42, "dishwasher")
        self.assertEqual(len(chosen), 5)
        self.assertEqual(len(set(chosen)), 5)
        self.assertEqual(set(x[0] for x in chosen), set(folders))
        self.assertEqual(chosen, compare.choose_episodes(folders, 5, 42, "dishwasher"))
        self.assertEqual(folders["/a"][0], (0, 100))  # No input mutation.

    def test_prepare_and_same_manifest_guard(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            roots = []
            for task in ("dishwasher", "fridge"):
                for run in range(2):
                    root = td / f"{task}_{run}"
                    meta = root / "meta"
                    meta.mkdir(parents=True)
                    (meta / "info.json").write_text(json.dumps({"fps": 30, "total_episodes": 2}))
                    (meta / "episodes.jsonl").write_text(
                        "\n".join(json.dumps({"episode_index": i, "length": 65 + i})
                                  for i in range(2)) + "\n"
                    )
                    roots.append((task, str(root)))
            catalog = td / "catalog.tsv"
            compare.write_csv(catalog, [dict(task=t, dataset_dir=r) for t, r in roots],
                              ["task", "dataset_dir"])
            # Restore TSV delimiter: use DictWriter directly.
            with catalog.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["task", "dataset_dir"])
                writer.writerows(roots)
            output = td / "reports"
            args = argparse.Namespace(catalog=catalog, output=output,
                                      episodes_per_task=3, windows_per_episode=4, seed=42)
            compare.prepare(args)
            first = (output / "samples.csv").read_bytes()
            compare.prepare(args)
            self.assertEqual(first, (output / "samples.csv").read_bytes())
            samples = compare.read_csv(output / "samples.csv")
            self.assertEqual(len(samples), 24)
            self.assertEqual(len({r["sample_id"] for r in samples}), len(samples))
            for task in ("dishwasher", "fridge"):
                subset = [r for r in samples if r["task"] == task]
                self.assertEqual(len({r["dataset_dir"] for r in subset}), 2)
                self.assertEqual(len({(r["dataset_dir"], r["episode_index"]) for r in subset}), 3)
                self.assertTrue(all(int(r["start_frame"]) + compare.HORIZON <=
                                    int(r["episode_length"]) for r in subset))
            args.windows_per_episode = 5
            with self.assertRaises(RuntimeError):
                compare.prepare(args)


class TestScoring(unittest.TestCase):
    def test_identical_gt_zero_and_raw_error(self):
        target = np.zeros((32, 25), dtype=np.float32)
        target[:, 14] = 1
        scale = np.full((25,), 2., dtype=float)
        self.assertAlmostEqual(compare.score(target, target, scale)["rmse"], 0.)
        predicted = target.copy()
        predicted[:, 14] += 2
        measures = compare.score(predicted, target, scale)
        self.assertAlmostEqual(measures["group_left_gripper"], 1.)
        self.assertAlmostEqual(measures["raw_group_left_gripper"], 2.)
        self.assertAlmostEqual(measures["chunk_first8"], 1 / 5)
        self.assertAlmostEqual(measures["chunk_last8"], 1 / 5)

    def test_fixed_expert_scale_and_floor(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stats.json"
            std = [0.] + [.5] * 24
            path.write_text(json.dumps({"action": {"default": {"global_std": std}}}))
            scale = compare.fixed_scale(path)
            self.assertEqual(scale.shape, (25,))
            self.assertAlmostEqual(float(scale[0]), .005)
            self.assertAlmostEqual(float(scale[1]), .5)
            std[1] = -3
            path.write_text(json.dumps({"action": {"default": {"global_std": std}}}))
            with self.assertRaises(RuntimeError):
                compare.fixed_scale(path)

    def test_shape_guard(self):
        with self.assertRaises(RuntimeError):
            compare.score(np.zeros((32, 32)), np.zeros((32, 25)), np.ones(25))

    def test_duplicate_scores_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scores.jsonl"
            row = {"sample_id": "abc", "seed": 42}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(RuntimeError):
                compare.read_scored(path)


class TestReport(unittest.TestCase):
    def test_uncommitted_source_identified_by_hash(self):
        with patch.object(compare, "try_git_head", return_value="base-commit"):
            code = compare.comparison_code()
        self.assertEqual(code["git_head"], "base-commit")
        self.assertEqual(code["tool_sha256"], compare.digest(PATH))
        self.assertEqual(code["gt_reader"], "episode-range-mapped-v1")

    def test_complete_paired_only_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            compare.ensure_output(out)
            items = [
                {"sample_id": "a", "task": "dishwasher", "dataset_dir": "/x",
                 "episode_index": 0, "episode_length": 65, "start_frame": 1},
                {"sample_id": "b", "task": "dishwasher", "dataset_dir": "/x",
                 "episode_index": 1, "episode_length": 65, "start_frame": 1},
            ]
            samples = out / "samples.csv"
            samples.write_text(compare.csv_as_text(items, [
                "sample_id","task","dataset_dir","episode_index","episode_length","start_frame"
            ]))
            manifest = out / "run_manifest.json"
            manifest.write_text(json.dumps({"samples_sha256": compare.digest(samples),
                                            "git_head": "prepared-before-repair"}))
            plan = out / "model_plan.json"
            plan.write_text(json.dumps({"seeds": [42]}))
            frozen = {path: path.read_bytes() for path in (samples, manifest, plan)}
            for model, metrics in (("expert", [1., 2.]), ("shared", [.5, 3.])):
                with (out / "_raw" / f"{model}_dishwasher.jsonl").open("w") as f:
                    for item, rmse in zip(items, metrics):
                        row = {**item, "seed": 42, "model": model,
                               "task_prompt": "put the dish into the dishwasher",
                               "rmse": rmse}
                        if model == "shared":
                            row["comparison_code"] = {
                                "git_head": "scored-after-repair", "tool_sha256": "a" * 64,
                                "gt_reader": "episode-range-mapped-v1"}
                        for name in compare.GROUPS:
                            row["group_"+name] = rmse
                            row["raw_group_"+name] = rmse
                        for name in compare.CHUNKS:
                            row["chunk_"+name] = rmse
                        f.write(json.dumps(row) + "\n")
            compare.summarize(argparse.Namespace(output=out))
            scores = compare.read_csv(out / "tasks.csv")
            self.assertEqual(len(scores), 1)
            self.assertAlmostEqual(float(scores[0]["expert_rmse"]), 1.5)
            self.assertAlmostEqual(float(scores[0]["shared_rmse"]), 1.75)
            self.assertAlmostEqual(float(scores[0]["fraction_episodes_shared_better"]), .5)
            self.assertIn("Status: complete", (out / "report.md").read_text())
            report = (out / "report.md").read_text()
            self.assertIn("Sample preparation git HEAD: `prepared-before-repair`", report)
            self.assertIn("Scored with git HEAD `scored-after-repair`", report)
            self.assertIn("a" * 64, report)
            self.assertIn("GT reader `episode-range-mapped-v1`", report)
            self.assertIn("2 stored scores have no code revision (legacy)", report)
            for path, original in frozen.items():
                self.assertEqual(path.read_bytes(), original)
            # A missing side must never be substituted or counted as a paired sample.
            (out / "_raw" / "shared_dishwasher.jsonl").write_text(
                (out / "_raw" / "shared_dishwasher.jsonl").read_text().splitlines()[0] + "\n")
            compare.summarize(argparse.Namespace(output=out))
            self.assertIn("INCOMPLETE", (out / "report.md").read_text())
            coverage = compare.read_csv(out / "coverage.csv")
            self.assertEqual(int(coverage[0]["paired_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
