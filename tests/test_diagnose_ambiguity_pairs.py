"""Synthetic tests: no actual robot data, decoder, GPU, or WAM weights required."""
from __future__ import annotations
import ast
import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import diagnose_ambiguity_pairs as d


def stats():
    return {k: {"default": {"global_mean": [0.] * 25, "global_std": [1.] * 25}}
            for k in ("state", "action")}


def episode(n=100):
    return d.validate_episode({"frame": np.arange(n), "time": np.arange(n) / 30.,
                                "state": np.zeros((n, 25)), "action": np.ones((n, 25)),
                                "flags": {}}, n, 30., .002)


def source(frame, root="/tmp/example", side="A"):
    return {"root": root, "episode": 4, "frame": frame, "timestamp": frame / 30., "side": side}


def config_text(extra=""):
    return '''data:
  train:
    global_sample_stride: 1
    norm_stats_source: pretrained
    num_frames: 33
    shape_meta:
      state: [{key: default, raw_shape: 25}]
      action: [{key: default, raw_shape: 25}]
    processor:
      proprio_output_dim: 32
      action_output_dim: 32
      action_state_transforms: null
      use_stepwise_action_norm: false
      norm_default_mode: z-score
      norm_exception_mode: null
      action_state_merger:
        _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
        state_target_dim: 32
        action_target_dim: 32
model:
  irrelevant: ${oc.env:DIAGNOSTIC_UNUSED_ENV_DOES_NOT_EXIST}
''' + extra


class NumericTests(unittest.TestCase):
    def test_sorted_rows_not_physical_order(self):
        data = {"frame": [2, 0, 1], "time": [2/30., 0., 1/30.],
                "state": np.repeat([[2.], [0.], [1.]], 25, axis=1),
                "action": np.zeros((3, 25)), "flags": {"is_last": [True, False, False]}}
        result = d.validate_episode(data, 3, 30., .002)
        self.assertEqual(result["state"][:, 0].tolist(), [0., 1., 2.])
        self.assertEqual(result["flags"]["is_last"], [False, False, True])

    def test_missing_and_duplicate_frame_rejected(self):
        for frames in ([0, 0, 2], [0, 1, 3]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                d.validate_episode({"frame": frames, "time": np.arange(3)/30.,
                                    "state": np.zeros((3,25)), "action": np.zeros((3,25))}, 3,30.,.002)

    def test_bad_shape_nonfinite_and_time_rejected(self):
        for field in ("state", "action", "time"):
            data = episode(5)
            data[field] = data[field].copy()
            data[field].flat[0] = np.nan
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.validate_episode(data,5,30.,.002)
        data = episode(5)
        data["state"] = np.zeros((5,32))
        with self.assertRaises(ValueError): d.validate_episode(data,5,30.,.002)
        data = episode(5)
        data["time"][2] += .02
        with self.assertRaises(ValueError): d.validate_episode(data,5,30.,.002)

    def test_identity_timestamp_checked(self):
        data = episode()
        self.assertEqual(d.current_index(data,source(4),.002),4)
        bad = source(4); bad["timestamp"] += 1
        with self.assertRaises(ValueError): d.current_index(data,bad,.002)
        with self.assertRaises(ValueError): d.current_index(data,source(100),.002)

    def test_source_mismatch_and_rank_rejected(self):
        p = {"frame_a": 4, "source_a": "dataset_root=/tmp/example, episode_index=4, frame_index=5, timestamp=0.16"}
        with self.assertRaises(ValueError): d.parse_source(p,"a")
        with self.assertRaises(ValueError): d.select_pairs({"pairs":[{"rank":1}]},[2])

    def test_normalizer_matches_repository_torch_class(self):
        import torch
        # Execute ONLY this production class, avoiding unrelated GitPython imports.
        path = ROOT / "src/fastwam/datasets/lerobot/utils/normalizer.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SingleFieldLinearNormalizer")
        namespace = {"torch":torch,"NormMode":str}
        exec(compile(ast.Module(body=[node],type_ignores=[]), str(path), "exec"),namespace)
        rng = np.random.default_rng(42)
        x = (rng.normal(size=(80,25))*8).astype(np.float32)
        s = stats(); field=s["state"]["default"]
        field["global_mean"] = rng.normal(size=25).tolist()
        field["global_std"] = rng.uniform(.01,2.,25).tolist()
        ref = namespace["SingleFieldLinearNormalizer"]({"mean":torch.tensor(field["global_mean"],dtype=torch.float32),
                   "std":torch.tensor(field["global_std"],dtype=torch.float32)},mode="z-score")
        np.testing.assert_allclose(d.normalize(x,s,"state")[0], ref.forward(torch.from_numpy(x)).numpy(),rtol=1e-6,atol=1e-6)

    def test_clipping_explicitly_marked(self):
        x=np.full(25,6.,dtype=np.float32)
        y,mask=d.normalize(x,stats(),"state")
        self.assertTrue(mask.all()); self.assertTrue((y==5.).all())
        s=stats(); s["state"]["default"]["global_std"][0]=0.
        self.assertTrue(np.isfinite(d.normalize(x,s,"state")[0]).all())

    def test_bad_stats_rejected(self):
        s=stats(); s["state"]["default"]["global_std"][0]=-1.
        with self.assertRaises(ValueError): d.normalize(np.zeros(25),s,"state")

    def test_padding_dilution_and_joint_contribution(self):
        data=episode(); data["state"][50,7]=2.
        rows,groups,summary=d.current_comparison(data,data,0,50,stats(),d.joint_names("velocity"),32)
        self.assertAlmostEqual(summary["proprio_rmse_model_padded"],2/np.sqrt(32))
        self.assertAlmostEqual(summary["proprio_rmse_active25"],2/5)
        self.assertEqual(rows[7]["squared_error_share"],1.)
        self.assertEqual(next(r for r in groups if r["group"]=="left_arm")["squared_error_share"],1.)
        rows,_,_=d.current_comparison(data,data,0,0,stats(),d.joint_names("velocity"),32)
        self.assertEqual(sum(r["squared_error_share"] for r in rows),0.)

    def test_horizon_includes_successor_observation(self):
        data=episode(100)
        info,motion,seq=d.describe_side(data,4,source(4),32,3.,2.,d.joint_names("velocity"),"velocity")
        self.assertEqual(info["windows"]["action_chunk"]["action_rows"],32)
        self.assertEqual(info["windows"]["action_chunk"]["state_samples"],33)
        self.assertTrue(info["full_chunk_successor_states_present"])
        self.assertEqual(sum(r["in_gt_action_chunk"] for r in seq),32)

    def test_short_terminal_context_not_padded(self):
        data=episode(100)
        info,_,seq=d.describe_side(data,95,source(95),32,3.,2.,d.joint_names("velocity"),"velocity")
        self.assertEqual(len(seq),5)
        self.assertFalse(info["full_action_chunk_present"])
        self.assertFalse(info["first_seconds_complete"])
        self.assertIsNone(info["stop_demonstrated"])
        self.assertIsNone(info["task_completed"])
        self.assertAlmostEqual(info["remaining_observed_seconds"],4/30.)

    def test_exact_end_action_without_successor(self):
        data=episode(100)
        info,_,_=d.describe_side(data,68,source(68),32,3.,2.,d.joint_names("velocity"),"velocity")
        self.assertTrue(info["full_action_chunk_present"])
        self.assertFalse(info["full_chunk_successor_states_present"])

    def test_stationary_nonzero_position_and_moving_constant_velocity(self):
        data=episode(10)
        data["state"][:,0]=.3
        data["state"][:,7]=1.2
        rows=d.motion_rows(data,np.arange(10),"B","tail","state",d.joint_names("velocity"),"velocity")
        self.assertAlmostEqual(rows[0]["stationarity_quantity_raw"],.3,places=6)
        self.assertEqual(rows[0]["range_raw"],0.)
        self.assertEqual(rows[7]["stationarity_quantity_raw"],0.)
        rows=d.motion_rows(data,np.arange(10),"B","tail","state",d.joint_names("unknown"),"unknown")
        self.assertIsNone(rows[0]["stationarity_quantity_raw"])

    def test_roundtrip_motion_not_hidden_by_endpoint(self):
        data=episode(3); data["state"][:,7]=[0.,1.,0.]
        rows=d.motion_rows(data,np.arange(3),"B","tail","state",d.joint_names("velocity"),"velocity")
        self.assertEqual(rows[7]["last_raw"]-rows[7]["first_raw"],0.)
        self.assertEqual(rows[7]["total_variation_raw"],2.)
        self.assertEqual(rows[7]["max_abs_from_window_start_raw"],1.)

    def test_done_is_exported_not_inferred_as_stop(self):
        data=episode(5); data["flags"]={"done":[False]*4+[True]}
        info,_,seq=d.describe_side(data,4,source(4),32,3.,2.,d.joint_names("velocity"),"velocity")
        self.assertTrue(info["last_recorded_flags"]["done"])
        self.assertEqual(seq[-1]["recorded_flag__done"],"true")
        self.assertIsNone(info["stop_demonstrated"])

    def test_parquet_reader_filters_real_episode(self):
        data=episode(5)
        recorded={}
        class Column:
            def __init__(self,value): self.value=value
            def to_numpy(self): return np.asarray(self.value)
            def to_pylist(self): return list(self.value)
        table={k:Column(v) for k,v in {"frame_index":data["frame"],"timestamp":data["time"],
                "observation.state":data["state"],"action":data["action"],"done":[False]*4+[True]}.items()}
        class DS:
            schema=types.SimpleNamespace(names=list(table)+["episode_index"])
            def to_table(self,**kwargs): recorded.update(kwargs); return table
        class Field:
            def __eq__(self,other): return ("episode_index",other)
        pads=types.ModuleType("pyarrow.dataset")
        pads.dataset=lambda *a,**kw:DS()
        pads.field=lambda key:Field()
        pyarrow=types.ModuleType("pyarrow"); pyarrow.dataset=pads
        meta=types.SimpleNamespace(info={"features":{"observation.state":{"shape":[25]},"action":{"shape":[25]}}},
                                   fps=30, episodes={4:{"length":5}})
        with patch.dict(sys.modules,{"pyarrow":pyarrow,"pyarrow.dataset":pads}),patch.object(d,"EpisodeReader",return_value=types.SimpleNamespace(meta=meta)):
            reader=d.NumericEpisodeReader("/synthetic",.002)
            out=reader.read(4)
        self.assertEqual(recorded["filter"],("episode_index",4))
        self.assertIn("action",recorded["columns"])
        self.assertEqual(out["length"],5)

    def test_config_scope_and_reject_unsupported_transforms(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cfg=root/"config.yaml"; st=root/"stats.json"
            cfg.write_text(config_text()); st.write_text(json.dumps(stats()))
            settings,_=d.read_settings(cfg,st)
            self.assertEqual(settings["horizon"],32)
            cfg.write_text(config_text().replace("action_state_transforms: null","action_state_transforms: []"))
            with self.assertRaises(ValueError): d.read_settings(cfg,st)

    def test_end_to_end_files_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); dataset=root/"raw"; dataset.mkdir()
            cfg=root/"cfg.yaml"; st=root/"stats.json"; pairs=root/"pairs.json"; out=root/"diagnostic"
            cfg.write_text(config_text()); st.write_text(json.dumps(stats()))
            data=episode(100); data["state"][50,7]=.8; data["flags"]={"done":[False]*99+[True]}
            pair={"rank":2,"frame_a":4,"frame_b":50,"proprio_rmse":.8/np.sqrt(32)}
            for s,fr in (("a",4),("b",50)):
                pair[f"source_{s}"]=f"dataset_root={dataset}, episode_index=4, frame_index={fr}, timestamp={fr/30.:.6f}"
            pairs.write_text(json.dumps({"summary":{"config":str(cfg),"dataset_stats":str(st)},"pairs":[pair]}))
            originals={p:p.read_bytes() for p in (cfg,st,pairs)}
            fake=types.SimpleNamespace(read=lambda ep:data)
            argv=["--pairs-json",str(pairs),"--output-dir",str(out)]
            with patch.object(d,"NumericEpisodeReader",return_value=fake),contextlib.redirect_stdout(io.StringIO()):
                d.main(argv)
                with self.assertRaises(FileExistsError): d.main(argv)
            summary=json.loads((out/"pair_0002/summary.json").read_text())
            self.assertTrue(summary["mining_metric_checks"]["proprio_rmse"]["matches"])
            self.assertIsNone(summary["sides"][1]["stop_demonstrated"])
            self.assertTrue((out/"COMPLETE.json").is_file())
            with (out/"pair_0002/current_state.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual(len(list(csv.DictReader(f))),25)
            self.assertEqual(list(dataset.iterdir()),[])
            self.assertTrue(all(p.read_bytes()==content for p,content in originals.items()))


if __name__=="__main__":
    unittest.main()
