"""Synthetic tests; no checkpoint, real dataset, network or robot required."""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import probe_ambiguity_inputs as m


def observations():
    def one(image, state):
        return m.base.Observation(torch.full((1, 3, 16, 16), image),
                                  torch.full((1, 32), state), torch.ones(1, 4, 6),
                                  torch.ones(1, 4, dtype=torch.bool))
    return [one(.1, .2), one(.7, .8)]


class Model:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self, mutate=False):
        self.calls = []
        self.mutate = mutate

    def infer_action(self, **kw):
        self.calls.append({k: v.clone() if torch.is_tensor(v) else v for k, v in kw.items()})
        noise = torch.randn(kw['action_horizon'], 32, generator=torch.Generator().manual_seed(kw['seed']))
        pred = noise * .01 + kw['input_image'].mean() + 2 * kw['proprio'][0, 0]
        if self.mutate:
            kw['input_image'].fill_(999)
            kw['proprio'].fill_(999)
        return {'action': pred}


def run(model=None):
    args = SimpleNamespace(num_inference_steps=10, sigma_shift=None, tiled=False)
    return m.run_four(model or Model(), observations(), [42, 43],
                      {'horizon': 4, 'action_model_dim': 32}, args)


def gt():
    g = np.zeros((2, 4, 25))
    g[0] = .5
    g[1] = 2.3
    return g


def test_combinations_keep_matching_task_and_requested_proprio():
    combo = m.mixed_observations(observations())
    for o, (i, p) in zip(combo, [(0.1, .2), (.7, .8), (.7, .2), (.1, .8)]):
        assert o.image.mean().item() == pytest.approx(i)
        assert o.proprio[0, 0].item() == pytest.approx(p)
        assert torch.equal(o.context, combo[0].context)


def test_different_instruction_is_rejected():
    obs = observations()
    obs[1].context.add_(1)
    with pytest.raises(ValueError, match="identical"):
        m.mixed_observations(obs)


@pytest.mark.parametrize('field', ['image', 'proprio', 'context'])
def test_bad_inputs_rejected(field):
    obs = observations()
    getattr(obs[0], field).fill_(float('nan'))
    with pytest.raises(ValueError):
        m.mixed_observations(obs)


def test_one_seed_per_condition_no_gt_or_future_or_phase_in_model_call():
    model = Model()
    pred, timings = run(model)
    assert pred.shape == (2, 4, 4, 32) and timings.shape == (2, 4)
    assert [x['seed'] for x in model.calls] == [42, 42, 43, 43, 42, 42, 43, 43]
    allowed = {'prompt','input_image','proprio','context','context_mask','action_horizon',
               'negative_prompt','text_cfg_scale','num_inference_steps','sigma_shift','seed','rand_device','tiled'}
    assert all(set(x) == allowed for x in model.calls)
    np.testing.assert_allclose(pred[:, 2] - pred[:, 0], .6, atol=5e-7)
    np.testing.assert_allclose(pred[:, 3] - pred[:, 0], 1.2, atol=5e-7)


def test_model_cannot_mutate_subsequent_conditions():
    a, _ = run(Model())
    b, _ = run(Model(mutate=True))
    np.testing.assert_array_equal(a, b)


def test_four_conditional_effects_and_padding_exclusion():
    pred, _ = run()
    pred[..., 25:] = np.arange(4)[None, :, None, None] * 1000
    rows, summaries, distances = m.measure(pred, gt(), [42, 43], 2)
    assert len(rows) == 16 and len(summaries) == 8 and len(distances) == 8
    for s in summaries:
        assert s['image_at_proprio_A_mean'] == pytest.approx(.6, abs=5e-7)
        assert s['image_at_proprio_B_mean'] == pytest.approx(.6, abs=5e-7)
        assert s['proprio_at_image_A_mean'] == pytest.approx(1.2, abs=5e-7)
        assert s['proprio_at_image_B_mean'] == pytest.approx(1.2, abs=5e-7)
        assert s['interaction_rms_mean'] < 1e-6
        assert s['AA_seed_spread_rms'] > 0
    assert [d['recorded_gt_for_this_input'] for d in distances[:4]] == [True, True, False, False]
    assert 'failure_rate' not in json.dumps(summaries)


@pytest.mark.parametrize('kind', ['image', 'proprio', 'interaction', 'constant'])
def test_known_dependency_signatures(kind):
    values = {'image':[1,3,3,1], 'proprio':[1,3,1,3], 'interaction':[2,12,6,4], 'constant':[1,1,1,1]}[kind]
    pred = np.broadcast_to(np.array(values)[None, :, None, None], (2,4,4,32)).copy()
    _, s, _ = m.measure(pred, gt(), [42,43], 2)
    s = s[0]
    if kind == 'image':
        assert s['image_effect_mean'] == 2 and s['proprio_effect_mean'] == 0
    if kind == 'proprio':
        assert s['image_effect_mean'] == 0 and s['proprio_effect_mean'] == 2
    if kind == 'interaction':
        assert s['interaction_rms_mean'] == 4
    if kind == 'constant':
        assert all(s[k + '_mean'] == 0 for k in m.METRICS)


@pytest.mark.parametrize('bad', ['nan', 'wrong_combo', 'wrong_gt', 'empty_seed', 'duplicate_seed'])
def test_bad_metrics_fail_explicitly(bad):
    pred, _ = run()
    reference, seeds = gt(), [42, 43]
    if bad == 'nan': pred[0, 0, 0, 0] = np.nan
    if bad == 'wrong_combo': pred = pred[:, :3]
    if bad == 'wrong_gt': reference = reference[..., :24]
    if bad == 'empty_seed': seeds = []
    if bad == 'duplicate_seed': seeds = [42, 42]
    with pytest.raises(ValueError):
        m.measure(pred, reference, seeds, 2)


def test_raw_output_is_ignored_and_reports_are_not(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    mining = tmp_path / 'mining'
    mining.mkdir()
    pairs = mining / 'pairs.json'
    pairs.write_text('{}')
    output = mining / 'input_swap' / 'test'
    raw = m.new_output(output, pairs, {str(tmp_path / 'dataset')})
    (raw / 'predictions.csv').touch()
    (raw / 'arrays.npz').touch()
    (output / 'report.md').touch()
    (output / 'group_summary.csv').touch()
    for p in [raw / 'predictions.csv', raw / 'arrays.npz', output / 'run.log', output / 'launcher.sh']:
        assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=tmp_path).returncode == 0
    for p in [output / 'report.md', output / 'group_summary.csv', output / 'manifest.json']:
        assert subprocess.run(['git','check-ignore','-q',str(p)],cwd=tmp_path).returncode == 1
    with pytest.raises(FileExistsError):
        m.new_output(output, pairs, {str(tmp_path / 'dataset')})
    with pytest.raises(ValueError):
        m.new_output(mining, pairs, {str(tmp_path / 'dataset')})


def test_raw_gt_is_not_scored_as_counterfactual_label_or_clipped(tmp_path):
    output = tmp_path / 'out'
    raw = m.new_output(output, tmp_path / 'mining/pairs.json', {str(tmp_path / 'data')})
    pred, timing = run()
    refs = []
    for side, g in enumerate(gt()):
        refs.append(dict(normalized_gt=g, raw_gt=np.full((4,25),9.), relative_times=np.arange(4)/30,
                         clipped_gt_elements=100, clipped_current_dims=[],
                         source={'episode':4, 'frame':side * 20, 'root':'/fake'}))
    stats = {k:{'default':{'global_mean':[0.]*25,'global_std':[1.]*25}} for k in ('action','state')}
    item = dict(pair={'rank':2}, observations=observations(), references=refs,
                verified={'visual_rmse':.1, 'proprio_rmse':.2})
    summaries = m.save_pair(raw, item, pred, timing, [42,43], stats)
    m.report(output, summaries, [1,2], 'failed_or_interrupted')
    text = (output / 'report.md').read_text()
    assert '完成 1/2 对' in text and '未完成：[1]' in text
    assert '物理不一致' in text and '重要性排名' in text and 'AA/BB' in text
    with np.load(raw / 'pair_0002.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['prediction_normalized'],pred)
        np.testing.assert_array_equal(saved['gt_raw25'],9.)
        assert saved['conditions'].tolist() == ['AA','BB','BA','AB']
        assert saved['image_original'].shape == (2,1,3,16,16)
    for p in raw.glob('*.json'):
        json.loads(p.read_text(), parse_constant=lambda x: pytest.fail(x))
    assert not list(output.rglob('*.png'))


def test_empty_partial_report_does_not_make_up_results(tmp_path):
    m.report(tmp_path, [], [2], 'failed_or_interrupted')
    assert '完成 0/1 对' in (tmp_path / 'report.md').read_text()
    assert not (tmp_path / 'pair_summary.csv').exists()


def test_launcher_and_required_task_and_probe_defaults():
    launcher = ROOT / 'scripts/probe_ambiguity_inputs.sh'
    subprocess.run(['bash','-n',str(launcher)],check=True)
    text = launcher.read_text()
    assert 'PAIR_RANKS=(1 2 9 16 43)' in text and 'NUM_SEEDS=10' in text
    assert 'put the dish into the dishwasher' in text
    assert 'git reset' not in text and 'git pull' not in text and 'real_robot_client' not in text
    with pytest.raises(SystemExit):
        m.parse_args(['--pairs-json','p','--checkpoint','c','--text-cache-dir','t'])
    args = m.parse_args(['--pairs-json','p','--checkpoint','c','--text-cache-dir','t','--task','test'])
    assert args.ranks == [1,2,9,16,43]


def test_identical_inputs_have_zero_effect_even_with_multiple_seeds():
    obs = observations()[0]
    args = SimpleNamespace(num_inference_steps=10, sigma_shift=None, tiled=False)
    pred, _ = m.run_four(Model(), [obs, obs], [42,43], {'horizon':4,'action_model_dim':32}, args)
    _, summaries, _ = m.measure(pred, gt(), [42,43], 2)
    assert all(s[k + '_mean'] == 0 for s in summaries for k in m.METRICS)


@pytest.mark.parametrize('interrupt', [False, True])
def test_mocked_main_preserves_inputs_and_reports_completion(tmp_path, monkeypatch, interrupt):
    """Exercise actual CLI orchestration/storage without model or dataset assets."""
    from omegaconf import OmegaConf
    root = tmp_path / 'dataset'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text('{}')
    mining = tmp_path / 'mining'
    mining.mkdir()
    config, stats_path, ckpt = [tmp_path / name for name in ('config.yaml','stats.json','weights.pt')]
    for p in (config, stats_path, ckpt): p.write_text('test fixture')
    pair = {'rank':2, 'frame_a':4, 'frame_b':12}
    sources = []
    for side, frame in (('a',4),('b',12)):
        pair[f'source_{side}'] = f'dataset_root={root}, episode_index=4, frame_index={frame}, timestamp={frame/30}'
        sources.append({'root':str(root),'episode':4,'frame':frame,'timestamp':frame/30})
    pairs = mining / 'pairs.json'
    pairs.write_text(json.dumps({'summary':{'config':str(config),'dataset_stats':str(stats_path)},'pairs':[pair]}))
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    cache = cache_dir / (m.hashlib.sha256(b'test task').hexdigest() + '.t5_len4.wan22ti2v5b.pt')
    cache.write_text('fake context identity')
    refs = [dict(normalized_gt=g,raw_gt=g,relative_times=np.arange(4)/30,
                 source=s,clipped_gt_elements=0,clipped_current_dims=[]) for g,s in zip(gt(),sources)]
    item = dict(pair=pair,observations=observations(),references=refs,
                verified={'visual_rmse':.1,'proprio_rmse':.2})
    cfg = OmegaConf.create({'model':{'proprio_dim':32,'action_dit_config':{'action_dim':32}},
                            'data':{'train':{'context_len':4}}})
    stats = {k:{'default':{'global_mean':[0.]*25,'global_std':[1.]*25}} for k in ('action','state')}
    monkeypatch.setattr(m.base,'config_for_inference',lambda *a:cfg)
    monkeypatch.setattr(m.base,'read_settings',lambda *a:({'horizon':4,'state_model_dim':32,'action_model_dim':32},stats))
    monkeypatch.setattr(m.base,'prepare_pairs',lambda *a:[item])
    monkeypatch.setitem(sys.modules, 'fastwam.datasets.lerobot.robot_video_dataset', SimpleNamespace(DEFAULT_PROMPT='{task}'))
    model = Model()
    loaded = []
    def load(*a):
        loaded.append(True)
        if interrupt: raise KeyboardInterrupt('mock stop')
        return model
    monkeypatch.setattr(m.base,'load_model',load)
    before = {p:p.read_bytes() for p in (config,stats_path,ckpt,pairs,cache)}
    output = mining / 'input_swap' / 'this_run'
    argv = ['--pairs-json',str(pairs),'--ranks','2','--checkpoint',str(ckpt),
            '--text-cache-dir',str(cache_dir),'--task','test task',
            '--output-dir',str(output),'--num-seeds','2','--device','cpu']
    if interrupt:
        with pytest.raises(KeyboardInterrupt): m.main(argv)
        assert (output/'FAILED.json').exists()
        assert not (output/'COMPLETE.json').exists()
        assert '完成 0/1 对' in (output/'report.md').read_text()
    else:
        m.main(argv)
        assert len(model.calls) == 8 and len(loaded) == 1
        assert json.loads((output/'COMPLETE.json').read_text())['model_calls'] == 8
        assert (output/'pair_summary.csv').exists()
        assert (output/'_raw/task_context.npz').exists()
        assert not (output/'FAILED.json').exists()
        manifest = json.loads((output/'manifest.json').read_text())
        assert manifest['conditions'] == ['AA','BB','BA','AB']
        assert manifest['data_scope'] == 'train_seen_user_confirmed'
    assert before == {p:p.read_bytes() for p in before}
