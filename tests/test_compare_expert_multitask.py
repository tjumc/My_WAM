"""CPU-only tests for the expert-vs-multitask offline comparison."""
import argparse
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "compare_expert_multitask.py"
spec = importlib.util.spec_from_file_location("wam_compare", PATH)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


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
            (out / "run_manifest.json").write_text(json.dumps({"samples_sha256": compare.digest(samples)}))
            (out / "model_plan.json").write_text(json.dumps({"seeds": [42]}))
            for model, metrics in (("expert", [1., 2.]), ("shared", [.5, 3.])):
                with (out / "_raw" / f"{model}_dishwasher.jsonl").open("w") as f:
                    for item, rmse in zip(items, metrics):
                        row = {**item, "seed": 42, "model": model,
                               "task_prompt": "put the dish into the dishwasher",
                               "rmse": rmse}
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
            # A missing side must never be substituted or counted as a paired sample.
            (out / "_raw" / "shared_dishwasher.jsonl").write_text(
                (out / "_raw" / "shared_dishwasher.jsonl").read_text().splitlines()[0] + "\n")
            compare.summarize(argparse.Namespace(output=out))
            self.assertIn("INCOMPLETE", (out / "report.md").read_text())
            coverage = compare.read_csv(out / "coverage.csv")
            self.assertEqual(int(coverage[0]["paired_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
