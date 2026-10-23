"""CPU tests: recording, boundary interpretation, replay pairing, server behavior."""
import ast
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "astribot"
sys.path.insert(0, str(SCRIPTS))
from wam_diagnostics import TraceSession, load_observation
from analyze_trace import boundary_rows, read_events
from compare_prompts import action_metrics, load_prompts, run_matrix
from instrument_robot_client import instrument


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def observation(self):
        path = self.root / "obs.npz"
        np.savez(path, img0=np.zeros((4, 5, 3), np.uint8),
                 img1=np.zeros((4, 5, 3), np.uint8),
                 img2=np.zeros((4, 5, 3), np.uint8), state=np.zeros(25))
        return path

    def test_disabled_recording_does_not_create_files(self):
        trace = TraceSession(None, {"not_serializable": object()})
        self.assertFalse(trace.emit("anything", arrays={"unsafe": object()}))
        trace.close()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_snapshot_is_immutable_and_close_drains(self):
        trace = TraceSession(self.root, {"value": np.int64(3)})
        arr = np.arange(25, dtype=np.float32)
        self.assertTrue(trace.emit("prediction", state=arr, arrays={"actions": arr}))
        arr[:] = -999
        trace.close()
        events, warnings = read_events(trace.path)
        self.assertEqual(warnings, [])
        self.assertEqual(events[0]["state"][1], 1)
        with np.load(trace.path / events[0]["artifact"], allow_pickle=False) as data:
            self.assertEqual(data["actions"][1], 1)

    def test_bad_record_drops_without_raising_into_control(self):
        trace = TraceSession(self.root, {})
        self.assertFalse(trace.emit("bad", arrays={"bad": np.array([object()])}))
        self.assertTrue(trace.emit("good", value=float("nan")))
        trace.close()
        events, warnings = read_events(trace.path)
        self.assertIsNone(events[0]["value"])
        self.assertTrue(warnings)
        self.assertEqual(trace.dropped, 1)

    def test_full_queue_is_nonblocking_and_counted(self):
        with patch.object(TraceSession, "_worker", lambda self: None):
            trace = TraceSession(self.root, {}, queue_size=1)
            self.assertTrue(trace.emit("first"))
            self.assertFalse(trace.emit("second"))
            self.assertEqual(trace.dropped, 1)
            trace.close()

    def test_npz_rejects_invalid_state_and_object_images(self):
        path = self.observation()
        self.assertEqual(load_observation(path)["state"].shape, (25,))
        np.savez(path, img0=np.array([object()]), img1=np.zeros((2,2,3), np.uint8),
                 img2=np.zeros((2,2,3), np.uint8), state=np.zeros(25))
        with self.assertRaises(ValueError):
            load_observation(path)

    def test_boundary_metrics_use_joint_units_and_distinguish_actual(self):
        measured, previous, first = np.zeros(25), np.ones(25), np.full(25, 2.0)
        first[:3] = 1000  # base velocity is not a joint angle
        rows = boundary_rows([
            {"event": "initial_positioning", "request_id": 1},
            {"event": "dispatch", "request_id": 1, "monotonic_ns": 10,
             "submitted_steps": 32, "scheduled_duration": 2.56, "observation_age_ms": 2,
             "first_target": first, "measured_joints": measured,
             "previous_submitted_last": previous},
            {"event": "sdk_return", "request_id": 1, "sdk_call_ms": 0.1}])
        self.assertTrue(rows[0]["initial_positioning"])
        self.assertEqual(rows[0]["left_arm_first_vs_measured_rad"], 2)
        self.assertEqual(rows[0]["left_arm_first_vs_previous_submitted_rad"], 1)
        self.assertEqual(rows[0]["sdk_call_ms"], .1)

    def test_prompt_replay_pairs_same_observation_and_seed(self):
        calls = []
        policy = SimpleNamespace(seed=None, prompt=None)
        def infer(now, **obs):
            calls.append((policy.prompt, policy.seed, obs["state"].copy()))
            obs["state"][:] = 999  # Simulate a processor mutating its input.
            return np.full((3,25), policy.seed + (0 if policy.prompt == "correct" else 2.0))
        policy.infer = infer
        prompts = [{"id": "baseline", "text": "correct"}, {"id": "wrong", "text": "other"}]
        rows = run_matrix(policy, [self.observation()], prompts, [7, 11],
                          lambda p: setattr(policy, "prompt", p), self.root / "results")
        self.assertEqual([(x[0], x[1]) for x in calls],
                         [("correct", 7), ("correct", 11), ("other", 7), ("other", 11)])
        for call in calls:
            np.testing.assert_array_equal(call[2], np.zeros(25))
        self.assertEqual([r["left_arm_paired_rmse_rad"] for r in rows], [0,0,2,2])
        self.assertEqual(len(list((self.root / "results").glob("*.npz"))), 4)

    def test_replay_failure_is_not_scored_as_success(self):
        policy = SimpleNamespace(infer=lambda *a, **kw: None)
        with self.assertRaises(RuntimeError):
            run_matrix(policy, [self.observation()], [{"id":"x","text":"x"}], [0],
                       lambda p: None, self.root / "failed")
        with self.assertRaises(ValueError):
            action_metrics(np.full((2,25), np.nan), np.zeros((2,25)))

    def test_duplicate_prompt_ids_and_unknown_client_are_rejected(self):
        path = self.root / "prompts.json"
        path.write_text(json.dumps([{"id":"x","text":"a"}, {"id":"x","text":"b"}]))
        with self.assertRaises(ValueError):
            load_prompts(path)
        with self.assertRaises(ValueError):
            instrument("# an unreviewed client")

    def test_server_recording_preserves_predictions_and_failure_contract(self):
        # Exercise the actual inference method without importing CUDA/model packages.
        tree = ast.parse((SCRIPTS / "fastwam_portal_server.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "AstribotFastWAMPolicy")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "infer")
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        namespace = dict(np=np, time=time, traceback=traceback,
                         torch=SimpleNamespace(inference_mode=nullcontext),
                         LOGGER=logging.getLogger("server-test"))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                     "inference-under-test", "exec"), namespace)
        expected = np.arange(100).reshape(4,25).astype(float)
        def make_policy(trace):
            return SimpleNamespace(
                trace=trace, capture_every=1, _capture_index=0, seed=2, task_prompt="test",
                context=None, context_mask=None, action_horizon=4, num_inference_steps=10,
                sigma_shift=None, rand_device="cpu", tiled=False,
                _prepare_image_tensor=lambda *a: None, _prepare_proprio_tensor=lambda s: s,
                _denormalize_action=lambda a: a,
                model=SimpleNamespace(infer_action=lambda **kw: {"action": expected.copy()}))
        obs = load_observation(self.observation())
        enabled = TraceSession(self.root, {})
        for trace in (TraceSession(None, {}), enabled):
            p = make_policy(trace)
            result = namespace["infer"](p, 123.5, **obs)
            np.testing.assert_array_equal(result, expected)
        p.model.infer_action = lambda **kw: (_ for _ in ()).throw(RuntimeError("test failure"))
        self.assertIsNone(namespace["infer"](p, 124.5, **obs))
        enabled.close()
        events, warnings = read_events(enabled.path)
        self.assertEqual(warnings, [])
        result = next(e for e in events if e["event"] == "infer_result")
        with np.load(enabled.path / result["artifact"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["actions"], expected)
            np.testing.assert_array_equal(data["state"], obs["state"])
        self.assertTrue(any(e["event"] == "infer_error" for e in events))


if __name__ == "__main__":
    unittest.main()
