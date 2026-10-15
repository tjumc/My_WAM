"""Bounded, asynchronous, best-effort client-side rollout recording.

No SDK/RPC imports or calls. All arrays are snapshots; disk I/O and compression
run only in the writer thread. submit() never waits for queue/disk availability,
but array copying has a finite, measured cost (this is not a real-time logger).
Queue loss, invalid payloads, disk errors and incomplete shutdown are explicit.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import uuid
import warnings

import numpy as np


@dataclass(frozen=True)
class RecorderConfig:
    root: str
    queue_size: int = 8
    max_bytes: int = 5 * 1024**3
    min_free_bytes: int = 2 * 1024**3
    max_event_bytes: int = 32 * 1024**2
    image_every: int = 1  # 0 disables images, 1 records each policy observation.
    close_timeout: float = 3.0

    def validate(self):
        if not self.root or self.queue_size < 1 or self.max_bytes < 1 or self.min_free_bytes < 0:
            raise ValueError("Invalid recorder directory, queue size or disk budget")
        if self.max_event_bytes < 1 or self.image_every < 0:
            raise ValueError("Invalid recorder event budget/image interval")
        if not math.isfinite(self.close_timeout) or self.close_timeout < 0:
            raise ValueError("Recorder close timeout must be finite/nonnegative")


def clock_stamp() -> dict:
    return {"wall_time_s": time.time(), "monotonic_time_s": time.perf_counter()}


def _json_copy(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Not JSON metadata: {type(item).__name__}")
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


class RolloutRecorder:
    """One control-thread producer, one disk writer; no robot-side background reads.

Manifest setup is synchronous, so construct BEFORE hardware initialization.
Runtime failures disable recording or drop a record; they never issue robot
commands. Closing drains for at most close_timeout; a surviving RUNNING.json
without CLOSED.json means this recording may be incomplete.
"""
    def __init__(self, config: RecorderConfig, metadata: dict):
        config.validate()
        self.config = config
        root = Path(config.root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < config.min_free_bytes:
            raise OSError("Recorder free disk below configured reserve; no robot should be started")
        name = "rollout_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.path = root / name
        self.path.mkdir(exist_ok=False)
        # Protect all generated data even if this run is copied into a Git repo.
        (self.path / ".gitignore").write_text("*\n", encoding="utf-8")
        (self.path / "records").mkdir()
        _atomic_json(self.path / "manifest.json", {
            "schema_version": 1, "created": clock_stamp(), "run_id": name,
            "recorder": {k: v for k, v in vars(config).items() if k != "root"},
            "metadata": _json_copy(metadata),
            "semantics": {
                "clock": "All event clocks belong to CLIENT host; wall time is not a synchronized sensor clock.",
                "camera_time": "Callback receive time, NOT capture time; cameras/state are not claimed synchronized.",
                "observation": "One snapshot at each policy request, not a 30 Hz trajectory/video.",
                "feedback": "Next natural loop observation; not a guaranteed completed action endpoint.",
                "prediction": "Full denormalized RPC response, before client truncation/blend; no GT.",
                "sdk_return": "Method returned only; NOT a success, motion-completion or task-completion label.",
                "head": "Predicted dimensions 23:25 logged but not explicitly commanded; existing head-follow remains.",
                "loss": "Recording best effort; inspect counters, event-id gaps and missing stages before analysis.",
                "safety": "Logger is not an emergency stop; existing hardware safety procedure remains required.",
            },
        })
        _atomic_json(self.path / "RUNNING.json", {"status": "recording", "created": clock_stamp()})
        self._queue = queue.Queue(maxsize=config.queue_size)
        self._slots = threading.BoundedSemaphore(config.queue_size)
        self._stop = threading.Event()
        self._disabled = threading.Event()
        self._lock = threading.Lock()
        self._counts = Counter(attempted=0, accepted=0, written=0)
        self._next_id = 0
        self._bytes = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        self._submit_max_ms = 0.0
        self._submit_total_ms = 0.0
        self._error = None
        self._exit_reason = "not_closed"
        self._thread = threading.Thread(target=self._worker, name="wam-rollout-writer", daemon=True)
        self._thread.start()

    def _count(self, name, n=1):
        with self._lock:
            self._counts[name] += n

    def status(self) -> dict:
        with self._lock:
            return {"counts": dict(self._counts), "bytes_written": self._bytes,
                    "submit_max_ms": self._submit_max_ms, "submit_total_ms": self._submit_total_ms,
                    "writer_error": self._error, "disabled": self._disabled.is_set(),
                    "exit_reason": self._exit_reason}

    def wants_images(self, request_id: int) -> bool:
        return (not self._disabled.is_set() and self.config.image_every > 0
                and (request_id - 1) % self.config.image_every == 0)

    def submit(self, event: str, request_id: int | None = None, *, metadata=None, arrays=None) -> bool:
        """Snapshot data without waiting for disk. Safe to continue on False.

On full queue skip BEFORE copying. Arrays must be numeric (no object/pickle).
No synchronous logging from this function; status is reported at run shutdown.
"""
        started = time.perf_counter()
        acquired = False
        try:
            with self._lock:
                self._next_id += 1
                event_id = self._next_id
                self._counts["attempted"] += 1
            if self._disabled.is_set() or self._stop.is_set():
                self._count("dropped_disabled_or_closed")
                return False
            acquired = self._slots.acquire(blocking=False)
            if not acquired:
                self._count("dropped_queue_full")
                return False
            if not re.fullmatch(r"[a-z][a-z0-9_]*", event):
                raise ValueError("Invalid event name")
            info = _json_copy(metadata or {})
            encoded_len = len(json.dumps(info).encode("utf-8"))
            source = {key: np.asarray(value) for key, value in (arrays or {}).items()}
            if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) for key in source):
                raise ValueError("Invalid array name")
            if any(a.dtype.kind not in "biufc" for a in source.values()):
                raise ValueError("Only numeric arrays may be recorded")
            if encoded_len > 60000 or sum(a.nbytes for a in source.values()) + encoded_len > self.config.max_event_bytes:
                self._count("dropped_oversize")
                return False
            snapshot = {key: np.array(a, copy=True, order="C") for key, a in source.items()}
            record = {"event_id": event_id, "event": event, "request_id": request_id,
                      **clock_stamp(), "metadata": info}
            self._queue.put_nowait((record, snapshot))
            acquired = False  # slot ownership transferred to the writer
            self._count("accepted")
            return True
        except Exception:
            self._count("dropped_invalid_payload")
            return False
        finally:
            if acquired:
                self._slots.release()
            elapsed = (time.perf_counter() - started) * 1000
            with self._lock:
                self._submit_total_ms += elapsed
                self._submit_max_ms = max(self._submit_max_ms, elapsed)

    def _write_record(self, record: dict, arrays: dict, index) -> None:
        # Reserve one full input-size payload + headers. Compression is lossless;
        # disk/network-filesystem writes may still stall, never on control thread.
        estimate = sum(a.nbytes for a in arrays.values()) + 65536
        if self._bytes + estimate > self.config.max_bytes:
            raise OSError("run_size_limit: configured recording budget reached")
        if shutil.disk_usage(self.path).free < self.config.min_free_bytes + estimate:
            raise OSError("low_disk: configured free-space reserve reached")
        stem = f"{record['event_id']:08d}_{record['event']}"
        if arrays:
            final = self.path / "records" / (stem + ".npz")
            temporary = final.with_suffix(".npz.partial")
            try:
                with temporary.open("xb") as file:
                    np.savez_compressed(file, **arrays)
                size = temporary.stat().st_size
                if self._bytes + size > self.config.max_bytes:
                    raise OSError("run_size_limit after compression")
                os.replace(temporary, final)
                record["arrays_file"] = "records/" + final.name
                with self._lock:
                    self._bytes += size
            finally:
                if temporary.exists():
                    temporary.unlink()
        else:
            record["arrays_file"] = None
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        index.write(line)
        index.flush()  # no fsync per event; power loss can leave a partial tail
        with self._lock:
            self._bytes += len(line.encode("utf-8"))
            self._counts["written"] += 1

    def _fail(self, exc):
        with self._lock:
            self._error = f"{type(exc).__name__}: {exc}"
        self._disabled.set()
        # Writer-thread warning only; disk failure never raises in robot loop.
        try:
            warnings.warn(f"Rollout recording disabled: {self._error}", RuntimeWarning)
        except Exception:
            pass  # A process-wide warnings-as-errors setting must not kill the writer.

    def _worker(self):
        index = None
        try:
            index = (self.path / "events.jsonl").open("x", encoding="utf-8")
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    record, arrays = self._queue.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    if self._disabled.is_set():
                        self._count("discarded_after_writer_error")
                    else:
                        try:
                            self._write_record(record, arrays, index)
                        except Exception as exc:
                            self._count("write_failed")
                            self._fail(exc)
                finally:
                    del arrays
                    self._slots.release()
                    self._queue.task_done()
        except Exception as exc:
            self._fail(exc)
            while True:
                try:
                    self._queue.get_nowait()
                    self._count("discarded_after_writer_error")
                    self._slots.release()
                    self._queue.task_done()
                except queue.Empty:
                    break
        finally:
            if index is not None:
                try:
                    index.close()
                except OSError as exc:
                    self._fail(exc)
            try:
                state = self.status()
                counts = state["counts"]
                state["status"] = "closed_with_gaps" if (state["writer_error"] or
                    counts.get("attempted", 0) != counts.get("written", 0) or
                    counts.get("observation_pack_error", 0) or counts.get("command_pack_error", 0)) else "closed"
                state["closed"] = clock_stamp()
                state["task_success"] = None
                _atomic_json(self.path / "CLOSED.json", state)
                (self.path / "RUNNING.json").unlink(missing_ok=True)
            except Exception:
                # Missing CLOSED marker is intentionally visible after I/O failure.
                pass

    def close(self, reason="client_exit") -> dict:
        with self._lock:
            self._exit_reason = reason
        self._stop.set()
        self._thread.join(timeout=self.config.close_timeout)
        result = self.status()
        result["writer_still_alive"] = self._thread.is_alive()
        return result

    def observation(self, request_id, images, camera_times, state, raw_joints,
                    state_started, state_finished, feedback_after_request, wait_seconds):
        """Log existing reads, never query hardware to obtain a feedback sample."""
        try:
            arrays = {"state": state,
                      "raw_joint_positions": np.concatenate([np.asarray(a).reshape(-1) for a in raw_joints[:7]])}
            save_images = self.wants_images(request_id)
            if save_images:
                arrays.update({f"image_{i}": image for i, image in enumerate(images)})
            self.submit("observation", request_id, arrays=arrays, metadata={
                "camera_times": camera_times, "images_requested": save_images,
                "state_read_started": state_started, "state_read_finished": state_finished,
                "feedback_after_request": feedback_after_request,
                "raw_joint_group_lengths": [int(np.asarray(a).size) for a in raw_joints[:7]],
                "image_wait_seconds": wait_seconds,
                "note": "Feedback is the next natural observation, not an SDK endpoint/success measurement.",
            })
        except Exception:
            self._count("observation_pack_error")

    def command(self, event, request_id, command_names, waypoints, time_list,
                selected_actions, state_used_for_base, action_mode):
        try:
            flat = [[x for group in point for x in group] for point in waypoints]
            self.submit(event, request_id, arrays={
                "sdk_waypoints_position23": np.asarray(flat, dtype=np.float64),
                "sdk_time_list_s": np.asarray(time_list, dtype=np.float64),
                "selected_actions_after_blend25": selected_actions,
                "state_used_for_base": state_used_for_base,
            }, metadata={
                "command_names": list(command_names), "group_sizes": [3, 4, 7, 1, 7, 1],
                "action_mode": action_mode, "steps": len(waypoints),
                "nominal_horizon_s": float(time_list[-1]) if time_list else None,
                "predicted_head_commanded": False,
                "call_kwargs": {"use_wbc": False, "joy_controller": False}
                    if event == "sdk_waypoints_attempt" else {},
                "status": "attempt_only; inspect matching return/error event",
            })
        except Exception:
            self._count("command_pack_error")
