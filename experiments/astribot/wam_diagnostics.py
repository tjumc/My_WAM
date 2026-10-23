"""Optional, bounded background recording; numpy + standard library only.

Copy this file beside the instrumented robot client. No SDK/model imports.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def joint_snapshot(robot):
    """Measured joint positions, including base POSITION (not velocity)."""
    try:
        return np.concatenate([np.asarray(x).reshape(-1)
                               for x in robot.get_current_joints_position()[:7]]).copy()
    except Exception as exc:
        print(f"[wam-trace] joint observation unavailable: {exc}", file=sys.stderr)
        return None


class TraceSession:
    """Recording must never block motion on disk I/O or an unbounded queue.

    All event times are on the recording host. Compare monotonic times only
    within a host; use the RPC request timestamp as a correlation identifier.
    Dropped records and writer errors are reported, never silently hidden.
    """

    def __init__(self, root, metadata, *, kind="trace", queue_size=16):
        self.path = None
        self.dropped = 0
        self.written = 0
        self.error = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if not root:
            return
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(root) / f"{kind}_{name}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self._write_json("manifest.json", {"schema_version": 1, "kind": kind,
                         "created_wall_ns": time.time_ns(), "metadata": metadata})
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="wam-trace-writer")
        self._thread.start()

    @property
    def enabled(self):
        return self.path is not None

    def _write_json(self, name, data):
        text = json.dumps(jsonable(data), ensure_ascii=False, allow_nan=False, indent=2)
        tmp = self.path / (name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(self.path / name)

    def emit(self, event, *, arrays=None, **fields):
        if not self.enabled:
            return False
        with self._lock:
            seq = self._seq
            self._seq += 1
        try:
            if self.error or self._stop.is_set() or self._queue.full():
                raise queue.Full
            record = jsonable({**fields, "event": event, "seq": seq,
                               "wall_ns": time.time_ns(),
                               "monotonic_ns": time.monotonic_ns()})
            snapshots = None
            if arrays is not None:
                snapshots = {}
                for key, value in arrays.items():
                    a = np.asarray(value)
                    if a.dtype.hasobject:
                        raise ValueError("object arrays are not supported")
                    snapshots[key] = a.copy()
            self._queue.put_nowait((record, snapshots))
            return True
        except Exception as exc:
            with self._lock:
                self.dropped += 1
            if self.dropped == 1:
                print(f"[wam-trace] record dropped ({type(exc).__name__}); "
                      "check summary.json before analysis", file=sys.stderr)
            return False

    def _worker(self):
        try:
            with (self.path / "events.jsonl").open("x", encoding="utf-8") as stream:
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        record, arrays = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        if arrays is not None:
                            name = f"sample_{record['seq']:06d}.npz"
                            tmp = self.path / (name + ".tmp")
                            with tmp.open("wb") as f:
                                np.savez(f, **arrays)
                            tmp.replace(self.path / name)
                            record["artifact"] = name
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        self.written += 1
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[wam-trace] recording failed: {self.error}", file=sys.stderr)
        finally:
            try:
                self._write_json("summary.json", {
                    "written": self.written, "dropped": self.dropped,
                    "pending": self._queue.qsize(), "writer_error": self.error,
                    "clean_close": self._stop.is_set() and not self.error
                                   and self._queue.empty()})
            except Exception as exc:
                print(f"[wam-trace] cannot save summary: {exc}", file=sys.stderr)

    def close(self, timeout=3.0):
        if not self.enabled:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("[wam-trace] close timed out; recording may be incomplete",
                  file=sys.stderr)

    def start_joint_sampler(self, robot, hz=20.0):
        """Optional telemetry reads. Their host timestamps are not controller time."""
        if not self.enabled or hz == 0:
            return
        if not math.isfinite(hz) or hz < 0:
            raise ValueError("joint sampling frequency must be finite and >= 0")

        def sample():
            while not self._stop.is_set() and not self.error:
                started = time.monotonic_ns()
                q = joint_snapshot(robot)
                self.emit("joint_sample", measured_joints=q,
                          read_start_ns=started, read_end_ns=time.monotonic_ns())
                self._stop.wait(1.0 / hz)

        threading.Thread(target=sample, daemon=True,
                         name="wam-joint-observer").start()


def load_observation(path):
    """Lossless recorded client/server input, never unpickle captured data."""
    with np.load(path, allow_pickle=False) as data:
        required = ("img0", "img1", "img2", "state")
        if any(k not in data for k in required):
            raise ValueError(f"{path}: expected {required}")
        obs = {k: data[k].copy() for k in required}
    if obs["state"].shape != (25,) or not np.isfinite(obs["state"]).all():
        raise ValueError("Expected finite 25-D state")
    for key in required[:3]:
        a = obs[key]
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError(f"{key}: expected uint8 HWC image")
    return obs
