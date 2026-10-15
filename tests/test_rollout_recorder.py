"""Offline tests: synthetic arrays only, never import an Astribot SDK."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/astribot'))
sys.path.insert(0, str(ROOT / 'tools'))
from rollout_recorder import RecorderConfig, RolloutRecorder
from summarize_robot_rollout import read_events, build_rows, summarize


def config(tmp_path, **kwargs):
    return RecorderConfig(root=str(tmp_path / 'raw'), min_free_bytes=0, **kwargs)


def events(rec):
    return read_events(rec.path / 'events.jsonl')[0]


def test_snapshots_arrays_and_metadata(tmp_path):
    r = RolloutRecorder(config(tmp_path), {'task': 'original instruction'})
    a = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    meta = {'nested': [1]}
    assert r.submit('observation', 1, arrays={'image_0': a}, metadata=meta)
    a[:] = 0
    meta['nested'][0] = 99
    result = r.close()
    assert not result['writer_still_alive']
    e = events(r)[0]
    with np.load(r.path / e['arrays_file'], allow_pickle=False) as data:
        assert data['image_0'].sum() == sum(range(24))
    assert e['metadata']['nested'] == [1]
    assert (r.path / 'CLOSED.json').exists()
    assert not (r.path / 'RUNNING.json').exists()


def test_empty_arrays_and_closed_submission(tmp_path):
    r = RolloutRecorder(config(tmp_path), {})
    assert r.submit('rpc_none', 1)
    r.close()
    assert events(r)[0]['arrays_file'] is None
    assert not r.submit('late', 2)


@pytest.mark.parametrize('kwargs', [dict(queue_size=0), dict(image_every=-1),
    dict(max_bytes=0), dict(min_free_bytes=-1), dict(close_timeout=float('nan'))])
def test_bad_configuration_fails_before_hardware(tmp_path, kwargs):
    values = dict(root=str(tmp_path), min_free_bytes=0)
    values.update(kwargs)
    with pytest.raises(ValueError):
        RolloutRecorder(RecorderConfig(**values), {})


def test_low_space_preflight(tmp_path, monkeypatch):
    import rollout_recorder as m
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _: type('D', (), {'free': 0})())
    with pytest.raises(OSError):
        RolloutRecorder(RecorderConfig(root=str(tmp_path), min_free_bytes=1), {})


def test_full_queue_returns_without_disk_wait(tmp_path):
    started, release = threading.Event(), threading.Event()
    class Slow(RolloutRecorder):
        def _write_record(self, record, arrays, index):
            started.set()
            assert release.wait(3)
            return super()._write_record(record, arrays, index)
    r = Slow(config(tmp_path, queue_size=1), {})
    try:
        assert r.submit('first', 1)
        assert started.wait(2)
        # Inflight writer counts toward the bound. No array conversion is made.
        class MustNotCopy:
            def __array__(self, *args, **kwargs):
                raise AssertionError('full queue copied payload')
        assert not r.submit('second', 2, arrays={'x': MustNotCopy()})
        assert r.status()['counts']['dropped_queue_full'] == 1
    finally:
        release.set()
        r.close()
    assert json.loads((r.path / 'CLOSED.json').read_text())['status'] == 'closed_with_gaps'


def test_close_deadline_marks_incomplete_until_drained(tmp_path):
    gate, entered = threading.Event(), threading.Event()
    class Slow(RolloutRecorder):
        def _write_record(self, *args):
            entered.set()
            gate.wait(3)
            return super()._write_record(*args)
    r = Slow(config(tmp_path, close_timeout=.01), {})
    r.submit('event', 1)
    assert entered.wait(2)
    try:
        assert r.close()['writer_still_alive']
        assert (r.path / 'RUNNING.json').is_file()
        assert not (r.path / 'CLOSED.json').exists()
    finally:
        gate.set()
        r._thread.join(3)
    assert not r._thread.is_alive()


@pytest.mark.parametrize('failure', ['budget', 'io'])
def test_writer_error_never_raises_in_control_thread(tmp_path, failure):
    class Broken(RolloutRecorder):
        def _write_record(self, *args):
            if failure == 'io':
                raise OSError('synthetic disk failure')
            return super()._write_record(*args)
    r = Broken(config(tmp_path, max_bytes=1000 if failure == 'budget' else 1024**2), {})
    with pytest.warns(RuntimeWarning, match='disabled'):
        assert r.submit('prediction', 1, arrays={'actions': np.ones((32, 25))})
        result = r.close()
    assert result['writer_error']
    assert result['counts']['write_failed'] == 1
    assert json.loads((r.path / 'CLOSED.json').read_text())['status'] == 'closed_with_gaps'


def test_oversize_object_and_invalid_metadata_rejected(tmp_path):
    r = RolloutRecorder(config(tmp_path, max_event_bytes=100), {})
    assert not r.submit('bad', arrays={'x': np.ones(1000)})
    assert not r.submit('bad', arrays={'x': np.array([object()])})
    assert not r.submit('bad', metadata={'not_json': float('nan')})
    assert not r.submit('../path')
    r.close()
    assert r.status()['counts']['dropped_oversize'] == 1
    assert not events(r)


def test_image_sampling_does_not_mutate_input(tmp_path):
    r = RolloutRecorder(config(tmp_path, image_every=2), {})
    raw = [np.ones(d) for d in [3,4,7,1,7,1,2]]
    for i in (1,2):
        r.observation(i, [np.zeros((2,3,3), dtype=np.uint8)] * 3, [],
                      np.ones(25), raw, {}, {}, None, .1)
    r.close()
    e = events(r)
    with np.load(r.path / e[0]['arrays_file'], allow_pickle=False) as a:
        assert 'image_0' in a and a['raw_joint_positions'].shape == (25,)
    with np.load(r.path / e[1]['arrays_file'], allow_pickle=False) as a:
        assert 'image_0' not in a and a['state'].shape == (25,)


def test_exact_sdk_waypoints_recorded_not_predicted_head(tmp_path):
    r = RolloutRecorder(config(tmp_path), {})
    waypoint = [list(range(d)) for d in [3,4,7,1,7,1]]
    r.command('sdk_waypoints_attempt', 1, ['a']*6, [waypoint], [.033],
              np.ones((1,25)), np.ones(3), 'cmd_absolute_joint_vel')
    r.close()
    e = events(r)[0]
    with np.load(r.path / e['arrays_file'], allow_pickle=False) as a:
        np.testing.assert_equal(a['sdk_waypoints_position23'][0], [x for part in waypoint for x in part])
        assert a['selected_actions_after_blend25'].shape == (1,25)
    assert e['metadata']['predicted_head_commanded'] is False


def test_summary_is_small_and_does_not_claim_completion(tmp_path):
    r = RolloutRecorder(config(tmp_path), {})
    r.submit('observation', 1, metadata={'feedback_after_request': None})
    r.submit('prediction', 1, metadata={'rpc_wall_seconds': .5})
    r.submit('sdk_waypoints_attempt', 1, metadata={'steps': 2, 'nominal_horizon_s': 2/30})
    r.submit('sdk_waypoints_return', 1, metadata={'started': {'monotonic_time_s': 1},
                   'finished': {'monotonic_time_s': 2}, 'sdk_call_wall_seconds': 1})
    r.close()
    out = tmp_path / 'report'
    summary = summarize(r.path, out)
    assert summary['commands_without_next_observation'] == [1]
    assert summary['task_success'] is None
    assert {p.name for p in out.iterdir()} == {'report.md', 'requests.csv', 'summary.json'}
    assert sum(p.stat().st_size for p in out.iterdir()) < 20000
    with pytest.raises(FileExistsError):
        summarize(r.path, out)
    with pytest.raises(ValueError):
        summarize(r.path, r.path / 'inside')


def test_feedback_links_actual_observation_and_stale_images():
    ev = [dict(event_id=1,event='sdk_waypoints_attempt',request_id=1,metadata={}),
          dict(event_id=2,event='sdk_waypoints_return',request_id=1,metadata={
              'started': {'monotonic_time_s': 4}, 'finished': {'monotonic_time_s':5}}),
          dict(event_id=3,event='observation',request_id=2,metadata={
              'feedback_after_request':1, 'state_read_finished': {'monotonic_time_s':5.2},
              'camera_times':[{'received_monotonic_s':3.9}]})]
    rows = build_rows(ev)
    assert rows[0]['next_observation_request_id'] == 2
    assert rows[0]['feedback_lag_after_sdk_return_s'] == pytest.approx(.2)
    assert rows[0]['feedback_cameras_received_after_command_start'] is False


def test_partial_index_remains_visible(tmp_path):
    r = RolloutRecorder(config(tmp_path), {})
    r.submit('event', 1)
    r.close()
    with (r.path / 'events.jsonl').open('a') as f:
        f.write('{"truncated"')
    summary = summarize(r.path, tmp_path / 'report')
    assert summary['recording_status'] == 'incomplete_or_gapped'
    assert summary['index_parse_errors']


def test_copied_run_is_git_ignored(tmp_path):
    import subprocess
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    r = RolloutRecorder(config(tmp_path), {})
    r.submit('event')
    r.close()
    p = r.path / 'manifest.json'
    result = subprocess.run(['git','check-ignore',str(p)],cwd=tmp_path,capture_output=True)
    assert result.returncode == 0


def test_launchers_are_shell_valid_and_leave_original_paths_alone():
    import subprocess
    for f in ['scripts/real_robot/client_record.sh','scripts/summarize_robot_rollout.sh']:
        subprocess.run(['bash','-n',str(ROOT/f)],check=True)
        text = (ROOT/f).read_text()
        assert 'git reset' not in text and 'git pull' not in text
    text = (ROOT/'scripts/real_robot/client_record.sh').read_text()
    assert 'source "$SCRIPT_DIR/config.sh"' in text
    assert 'TASK_PROMPT=' not in text and 'EXECUTE_STEPS=' not in text
    assert 'RECORD_ROOT="${HOME}/wam_rollouts"' in text
