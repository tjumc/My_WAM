"""Synthetic fixed-cohort and launcher tests; no real checkpoint or robot I/O.
Run: python -m pytest tests/test_summarize_ambiguity_inference.py -q
"""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import infer_ambiguity_pairs as inference
import summarize_ambiguity_inference as m


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def add_pair(run, rank, mode='correct', seeds=None, episode=4):
    seeds = [42, 43] if seeds is None else seeds
    gt = np.zeros((2, 4, 32), dtype=np.float32)
    gt[1, :, :25] = 1
    pred = np.repeat(gt[None], len(seeds), axis=0)
    if mode == 'swapped':
        pred = pred[:, ::-1].copy()
    elif mode == 'collapse':
        pred[:, 1, :, :25] = 0
    elif mode == 'equal_gt':
        gt[:] = 0
    elif mode == 'generic_error':
        pred[..., :25] += 5
    elif mode == 'zero_prediction_separation_only':
        pred[:, :, :, :25] = 0.5
    rows, paired, groups, summary = inference.score_predictions(pred, gt, seeds)
    summary.update(rank=rank, sources=[{'source': {'root': '/dataset', 'episode': episode,
                    'frame': rank * 10 + i * 500, 'timestamp': (rank * 10 + i * 500) / 30}} for i in range(2)],
                   reference_reproduction={'visual_rmse': 0.1, 'proprio_rmse': 0.4})
    folder = run / f'pair_{rank:04d}'
    folder.mkdir()
    write_json(folder / 'summary.json', summary)
    write_csv(folder / 'seed_metrics.csv', rows)
    write_csv(folder / 'paired_seed_metrics.csv', paired)
    write_csv(folder / 'group_metrics.csv', groups)
    return folder


def new_run(tmp_path, ranks=None, seeds=None, complete=True):
    ranks = [1, 2] if ranks is None else ranks
    seeds = [42, 43] if seeds is None else seeds
    run = tmp_path / 'pair_inference' / 'run'
    run.mkdir(parents=True)
    write_json(run / 'manifest.json', {'arguments': {'ranks': ranks}, 'seeds_shared_between_sides': seeds,
              'pairs_json_sha256': 'fixed-cohort-hash', 'checkpoint': {'path': '/model.pt'}, 'task_prompt': 'original task'})
    if complete:
        write_json(run / 'COMPLETE.json', {'status': 'complete', 'ranks': ranks, 'num_seeds': len(seeds),
                                         'model_calls': len(ranks) * 2 * len(seeds)})
    return run


def test_fixed_50_pairs_1000_calls_and_no_reselection(tmp_path):
    seeds = list(range(42, 52))
    run = new_run(tmp_path, list(range(1, 51)), seeds)
    for rank in range(1, 51):
        add_pair(run, rank, seeds=seeds, episode=rank % 4)
    rows, episodes, info = m.build_batch(run, 'train_seen', 50)
    assert info['status'] == 'complete'
    assert info['counts']['planned_model_calls'] == 1000
    assert info['counts']['evaluated_pairs'] == 50
    assert info['counts']['pairs_with_any_other_gt_preference'] == 0
    assert len(episodes) == 4
    assert [r['rank'] for r in rows] == list(range(1, 51))
    assert info['fixed_ranks'] == list(range(1, 51))
    assert info['pairs_json_sha256'] == 'fixed-cohort-hash'
    assert '训练集已见轨迹' in m.report(rows, info)
    assert '不能写成失败概率' in m.report(rows, info)


@pytest.mark.parametrize('mode,cross,majority', [('correct', 0, 0), ('swapped', 1, 2), ('collapse', 1, 1),
                                                ('equal_gt', 0, 0), ('zero_prediction_separation_only', 0, 0)])
def test_descriptive_cross_gt_counts_not_failure_labels(tmp_path, mode, cross, majority):
    run = new_run(tmp_path, [1])
    add_pair(run, 1, mode)
    rows, _, info = m.build_batch(run)
    assert info['counts']['pairs_with_any_other_gt_preference'] == cross
    assert info['counts']['sides_with_other_gt_strict_majority'] == majority
    assert not any('failure_rate' in key for key in info)
    if mode == 'equal_gt':
        assert rows[0]['prediction_to_gt_separation_ratio'] is None
    if mode == 'zero_prediction_separation_only':
        assert rows[0]['prediction_to_gt_separation_ratio'] == 0
        assert rows[0]['a_tie_seeds'] == 2
        assert rows[0]['a_own_rmse_mean'] == .5


def test_high_errors_are_visible_even_without_both_side_confusion(tmp_path):
    run = new_run(tmp_path, [1])
    add_pair(run, 1, 'generic_error')
    rows, _, info = m.build_batch(run)
    assert rows[0]['a_own_rmse_mean'] == 5
    assert rows[0]['b_own_rmse_mean'] == 5
    assert info['own_rmse_pair_side_max'] == 5
    assert '一般拟合误差' in m.report(rows, info)


def test_missing_pairs_not_reported_as_success(tmp_path):
    run = new_run(tmp_path, [1, 2, 3], complete=False)
    add_pair(run, 1)
    _, _, info = m.build_batch(run)
    assert info['counts']['planned_pairs'] == 3
    assert info['counts']['evaluated_pairs'] == 1
    assert info['counts']['missing_or_invalid_pairs'] == 2
    assert info['status'] == 'partial_or_invalid'
    assert m.main(['--run-dir', str(run)]) == 2
    assert (run / 'batch_report.md').is_file()


@pytest.mark.parametrize('damage', ['duplicate_seed', 'missing_seed', 'wrong_side', 'nan', 'negative', 'wrong_label', 'summary_mismatch', 'pair_seed_duplicate'])
def test_corrupt_pair_reported_without_hiding_denominator(tmp_path, damage):
    run = new_run(tmp_path)
    folder = add_pair(run, 1)
    add_pair(run, 2)
    path = folder / 'seed_metrics.csv'
    rows = m.read_csv(path)
    if damage == 'duplicate_seed': rows.append(rows[0])
    if damage == 'missing_seed': rows.pop()
    if damage == 'wrong_side': rows[0]['side'] = 'C'
    if damage == 'nan': rows[0]['own_gt_rmse_active25'] = 'nan'
    if damage == 'negative': rows[0]['own_gt_rmse_active25'] = '-1'
    if damage == 'wrong_label': rows[0]['gt_preference_not_failure_label'] = 'other_gt'
    if damage == 'summary_mismatch':
        s = m.read_json(folder / 'summary.json')
        s['sides']['A']['own_gt_rmse_active25']['mean'] = 1
        write_json(folder / 'summary.json', s)
    if damage == 'pair_seed_duplicate':
        p = m.read_csv(folder / 'paired_seed_metrics.csv')
        p[1]['seed'] = p[0]['seed']
        write_csv(folder / 'paired_seed_metrics.csv', p)
    write_csv(path, rows)
    result, _, info = m.build_batch(run)
    assert len(result) == 1 and result[0]['rank'] == 2
    assert info['counts']['planned_pairs'] == 2
    assert info['issues'][0]['rank'] == 1
    assert info['status'] != 'complete'


@pytest.mark.parametrize('bad', ['failed_marker', 'wrong_complete_count', 'unexpected_pair'])
def test_no_false_complete_status(tmp_path, bad):
    run = new_run(tmp_path, [1])
    add_pair(run, 1)
    if bad == 'failed_marker': write_json(run / 'FAILED.json', {'error': 'plot failure'})
    if bad == 'wrong_complete_count':
        x = m.read_json(run / 'COMPLETE.json'); x['model_calls'] = 999
        write_json(run / 'COMPLETE.json', x)
    if bad == 'unexpected_pair': add_pair(run, 999)
    rows, _, info = m.build_batch(run)
    assert len(rows) == 1
    assert info['status'] == 'partial_or_invalid'


def test_duplicate_manifest_ranks_rejected(tmp_path):
    run = new_run(tmp_path, [1, 1])
    with pytest.raises(ValueError, match='Duplicate'):
        m.build_batch(run)


def test_expected_cohort_guard(tmp_path):
    run = new_run(tmp_path, [2])
    with pytest.raises(ValueError, match='expected 50'):
        m.build_batch(run, expected_pairs=50)


def test_macro_episode_average_not_pair_weighted(tmp_path):
    run = new_run(tmp_path, [1, 2, 3])
    add_pair(run, 1, 'correct', episode=1)
    add_pair(run, 2, 'correct', episode=1)
    add_pair(run, 3, 'generic_error', episode=2)
    _, _, info = m.build_batch(run)
    assert info['own_rmse_pair_side_mean'] == pytest.approx(5 / 3)
    assert info['own_rmse_episode_balanced_mean'] == pytest.approx(2.5)


def test_generated_files_preserve_originals_and_allow_resummarizing(tmp_path):
    run = new_run(tmp_path)
    for rank in [1, 2]: add_pair(run, rank)
    originals = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    for _ in range(2):
        assert m.main(['--run-dir', str(run), '--expected-pairs', '2', '--data-split', 'train_seen']) == 0
    assert all(p.read_bytes() == b for p, b in originals.items())
    assert len(m.read_csv(run / 'batch_pairs.csv')) == 2
    s = m.read_json(run / 'batch_summary.json')
    assert s['data_split'] == 'train_seen'
    assert 'valid-action' not in s
    assert '验证集' in (run / 'batch_report.md').read_text()


def test_launcher_keeps_paths_prompt_and_shared_seeds():
    root = Path(__file__).resolve().parents[1]
    launcher = root / 'scripts/infer_ambiguity_batch.sh'
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    text = launcher.read_text()
    assert 'PAIR_RANKS=({1..50})' in text
    assert 'NUM_SEEDS=10' in text and 'SEED=42' in text
    assert 'TASK_PROMPT="put the dish into the dishwasher"' in text
    assert 'DATA_SPLIT="train_seen"' in text
    assert 'git reset' not in text and 'git pull' not in text
    assert 'INFER_STATUS=$?' in text and 'SUMMARY_STATUS=$?' in text
    assert text.count('-u tools/infer_ambiguity_pairs.py') == 1


def test_mock_model_still_sees_only_current_inputs():
    import torch
    sample = {'video': torch.zeros(3, 9, 16, 16), 'proprio': torch.zeros(32, 32),
              'context': torch.zeros(4, 6), 'context_mask': torch.ones(4, dtype=torch.bool)}
    sample['video'][:, 1:] = float('nan')
    sample['proprio'][1:] = float('nan')
    sample['action'] = object()
    obs = inference.extract_observation(sample)
    class Model:
        device = torch.device('cpu')
        torch_dtype = torch.float32
        def __init__(self): self.seeds = []
        def infer_action(self, **kwargs):
            assert 'action' not in kwargs and 'video' not in kwargs
            assert kwargs['input_image'].shape == (1, 3, 16, 16)
            assert kwargs['proprio'].shape == (1, 32)
            assert torch.isfinite(kwargs['input_image']).all()
            self.seeds.append(kwargs['seed'])
            return {'action': torch.zeros(32, 32)}
    model = Model()
    result, _ = inference.run_inference(model, [obs, obs], [42, 43], 32, 32, 10, None, False)
    assert result.shape == (2, 2, 32, 32)
    assert model.seeds == [42, 42, 43, 43]


@pytest.mark.parametrize('fail', [False, True])
def test_full_launcher_with_mock_inference_and_partial_exit(tmp_path, fail):
    import shutil
    root = Path(__file__).resolve().parents[1]
    fake_repo = tmp_path / 'repo'
    (fake_repo / 'scripts').mkdir(parents=True)
    (fake_repo / 'tools').mkdir()
    launcher = fake_repo / 'scripts/infer_ambiguity_batch.sh'
    text = (root / 'scripts/infer_ambiguity_batch.sh').read_text()
    text = text.replace('PYTHON_BIN="python"', f'PYTHON_BIN="{sys.executable}"')
    launcher.write_text(text)
    shutil.copy(root / 'tools/summarize_ambiguity_inference.py', fake_repo / 'tools')
    # Reuse fixture generation; no network, robot, or real model.
    fake = f'''import argparse, json, sys
from pathlib import Path
sys.path.insert(0, {str(root / 'tests')!r})
from test_summarize_ambiguity_inference import add_pair, write_json
p=argparse.ArgumentParser()
p.add_argument('--output-dir'); p.add_argument('--ranks', nargs='+', type=int)
p.add_argument('--seed', type=int); p.add_argument('--num-seeds', type=int); p.add_argument('--task')
a,_=p.parse_known_args()
assert a.ranks == list(range(1,51))
assert a.task == 'put the dish into the dishwasher'
seeds=list(range(a.seed, a.seed+a.num_seeds)); run=Path(a.output_dir)
write_json(run/'manifest.json', {{'arguments': {{'ranks': a.ranks}}, 'seeds_shared_between_sides': seeds}})
for rank in a.ranks:
    if {fail!r} and rank == 2:
        write_json(run/'FAILED.json', {{'error': 'synthetic failure'}})
        sys.exit(7)
    add_pair(run, rank, seeds=seeds)
write_json(run/'COMPLETE.json', {{'status':'complete', 'ranks': a.ranks, 'num_seeds':len(seeds), 'model_calls':len(a.ranks)*2*len(seeds)}})
'''
    (fake_repo / 'tools/infer_ambiguity_pairs.py').write_text(fake)
    result = subprocess.run(['bash', str(launcher)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == (7 if fail else 0), result.stderr + result.stdout
    outputs = list(fake_repo.rglob('batch_summary.json'))
    assert len(outputs) == 1
    info = m.read_json(outputs[0])
    assert info['counts']['planned_pairs'] == 50
    assert info['counts']['evaluated_pairs'] == (1 if fail else 50)
    assert info['status'] == ('partial_or_invalid' if fail else 'complete')
    assert info['data_split'] == 'train_seen'
    assert (outputs[0].parent / 'launcher.sh').is_file()
    assert (outputs[0].parent / 'run.log').is_file()
