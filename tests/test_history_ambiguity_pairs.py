"""CPU-only synthetic tests; no robot, dataset, checkpoint, or Hydra required.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_history_ambiguity_pairs.py' -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "tools/find_history_ambiguity_pairs.py"
SPEC = importlib.util.spec_from_file_location("ambiguity_miner", SCRIPT)
miner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(miner)


def pair(a, b, score, episode=69):
    return dict(frame_a=a, frame_b=b, visual_rmse=score, episode=episode)


def record(frame, proprio=0.0, action=0.0, visual=0.0, episode=0):
    return dict(idx=frame, episode=episode, frame_offset=frame,
                proprio=torch.full((32,), float(proprio)),
                action=torch.full((32, 32), float(action)),
                visual_feature=torch.full((12,), float(visual)),
                image=torch.zeros(3, 8, 8), source=f"synthetic frame={frame}")


def mine(records, **kwargs):
    options = dict(min_frame_gap=150, max_proprio_rmse=0.5,
                   min_action_rmse=0.0, candidate_k_per_episode=200,
                   dedup_frame_gap=30)
    options.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return miner.mine_pairs(records, **options)


class TemporalDedupTests(unittest.TestCase):
    def test_uploaded_frame_pattern(self):
        candidates = [pair(264, 820, .11548), pair(264, 824, .12302),
                      pair(264, 816, .12308)]
        kept, removed = miner.deduplicate_temporal_pairs(candidates, 30)
        self.assertEqual(kept, [candidates[0]])
        self.assertEqual(removed, 2)

    def test_both_endpoints_must_be_near(self):
        kept, removed = miner.deduplicate_temporal_pairs(
            [pair(264, 820, .1), pair(264, 1220, .2)], 30)
        self.assertEqual((len(kept), removed), (2, 0))

    def test_different_episodes_never_merge(self):
        kept, removed = miner.deduplicate_temporal_pairs(
            [pair(264, 820, .1, 1), pair(264, 820, .2, 2)], 30)
        self.assertEqual((len(kept), removed), (2, 0))

    def test_best_visual_score_wins_and_endpoint_order_is_irrelevant(self):
        worse, best = pair(264, 824, .2), pair(820, 264, .1)
        kept, removed = miner.deduplicate_temporal_pairs([worse, best], 30)
        self.assertEqual(kept, [best])
        self.assertEqual(removed, 1)

    def test_radius_boundary_is_inclusive(self):
        kept, removed = miner.deduplicate_temporal_pairs(
            [pair(0, 300, .1), pair(30, 330, .2), pair(31, 331, .3)], 30)
        self.assertEqual((len(kept), removed), (2, 1))

    def test_zero_disables_and_negative_rejected(self):
        pairs = [pair(0, 300, .1), pair(0, 301, .2)]
        self.assertEqual(miner.deduplicate_temporal_pairs(pairs, 0), (pairs, 0))
        with self.assertRaises(ValueError):
            miner.deduplicate_temporal_pairs(pairs, -1)


class MiningTests(unittest.TestCase):
    def test_proprio_threshold_and_counts(self):
        rows, stats = mine([record(0), record(200, proprio=.5), record(400, proprio=1.01)])
        self.assertEqual([(p['frame_a'], p['frame_b']) for p in rows], [(0, 200)])
        counts = stats['pair_filter_counts']
        self.assertEqual(counts['all_pairs'], 3)
        self.assertEqual(counts['after_frame_gap'], 3)
        self.assertEqual(counts['after_proprio'], 1)
        # Quantiles are computed BEFORE the proprio gate, not from just survivors.
        self.assertGreater(stats['proprio_rmse_after_frame_gap_quantiles']['max'], 1)

    def test_action_threshold_is_optional(self):
        records = [record(0), record(200, action=0.0)]
        self.assertEqual(len(mine(records)[0]), 1)
        self.assertEqual(len(mine(records, min_action_rmse=.1)[0]), 0)

    def test_min_frame_gap_boundary(self):
        self.assertEqual(len(mine([record(0), record(149)])[0]), 0)
        self.assertEqual(len(mine([record(0), record(150)])[0]), 1)

    def test_dedup_precedes_episode_cap(self):
        records = [record(0, visual=0), record(300, visual=.1),
                   record(304, visual=.101), record(800, visual=.9)]
        rows, stats = mine(records, candidate_k_per_episode=2)
        counts = stats['pair_filter_counts']
        self.assertEqual(len(rows), 2)
        self.assertEqual(counts['after_action'], 5)
        self.assertEqual(counts['removed_temporal_duplicates'], 2)
        self.assertEqual(counts['after_temporal_dedup'], 3)
        self.assertEqual(counts['after_episode_limit'], 2)

    def test_no_automatic_threshold_relaxation(self):
        rows, stats = mine([record(0), record(300, proprio=1)])
        self.assertEqual(rows, [])
        self.assertEqual(stats['pair_filter_counts']['after_frame_gap'], 1)
        self.assertEqual(stats['pair_filter_counts']['after_proprio'], 0)

    def test_empty_records(self):
        rows, stats = mine([])
        self.assertEqual(rows, [])
        self.assertEqual(stats['pair_filter_counts']['all_pairs'], 0)
        self.assertEqual(stats['proprio_rmse_after_frame_gap_quantiles'], {})

    def test_empty_output_is_valid_json_with_no_infinity(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            miner.save_outputs(Path(tmp), [], 50, 0, {'max_proprio_rmse': None})
            text = (Path(tmp) / 'pairs.json').read_text()
            self.assertNotIn('Infinity', text)
            self.assertEqual(json.loads(text)['pairs'], [])


class LoadingTests(unittest.TestCase):
    def test_progress_counts_padding_errors_and_compact_image_storage(self):
        video = torch.zeros(3, 9, 8, 8)
        normal = dict(video=video, action=torch.zeros(32, 32), proprio=torch.zeros(32, 32))
        padded = {**normal, 'action_is_pad': torch.ones(32, dtype=torch.bool)}
        def get(index):
            if index == 1:
                return padded
            if index == 2:
                raise RuntimeError('synthetic failure')
            return normal
        dataset = SimpleNamespace(_get=get, lerobot_dataset=SimpleNamespace())
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            records, padding, errors = miner.load_records(
                dataset, [(i, 0, i) for i in range(3)], 2, 2, log_every=2)
        self.assertEqual((len(records), padding, errors), (1, 1, 1))
        self.assertIn('loaded 2/3 windows', log.getvalue())
        self.assertIn('loaded 3/3 windows', log.getvalue())
        image = records[0]['image']
        self.assertEqual(image.untyped_storage().nbytes(), image.numel() * image.element_size())
        self.assertNotEqual(image.data_ptr(), video.data_ptr())

    def test_nonfinite_sample_is_rejected(self):
        sample = dict(video=torch.zeros(3, 9, 8, 8), action=torch.full((32, 32), float('nan')),
                      proprio=torch.zeros(32, 32))
        dataset = SimpleNamespace(_get=lambda _: sample, lerobot_dataset=SimpleNamespace())
        with contextlib.redirect_stdout(io.StringIO()):
            records, _, errors = miner.load_records(dataset, [(0, 0, 0)], 2, 2)
        self.assertEqual((len(records), errors), (0, 1))

    def test_sampling_is_reproducible_and_in_bounds(self):
        options = dict(raw_window=33, sample_stride=4, samples_per_episode=64,
                       max_samples=256, seed=42)
        starts, ends = np.array([0, 400]), np.array([400, 1000])
        first = miner.choose_indices(starts, ends, **options)
        self.assertEqual(first, miner.choose_indices(starts, ends, **options))
        for index, ep, offset in first:
            self.assertEqual(index, int(starts[ep]) + offset)
            self.assertLess(index + options['raw_window'] - 1, int(ends[ep]))


if __name__ == '__main__':
    unittest.main()
