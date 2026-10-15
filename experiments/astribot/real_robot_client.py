#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Astribot real-robot client for Fast-WAM.

The client intentionally keeps robot hardware I/O separate from the Fast-WAM
server. It is based on the real Astribot client used in pi0_astribot, but the
state/action semantics are explicit for the 25-D WAM joint representation:

    [base_vx, base_vy, base_omega,
     torso(4), left_arm(7), left_gripper(1),
     right_arm(7), right_gripper(1), head(2)]

For action modes containing `vel`, the first three predicted dimensions are
base velocities. They are integrated into chassis position waypoints for the
Astribot SDK while the remaining joint commands are sent as position waypoints.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque

import numpy as np
import portal

CAMERAS = ("head_rgbd", "left_wrist_rgbd", "right_wrist_rgbd")
MAX_IMAGE_AGE = 0.5
WAM_DIM = 25

images_lock = threading.Lock()
images_dict = {name: {"data": None, "timestamp": 0.0} for name in CAMERAS}

USE_HISTORY_EFFORT_OBS = False
history_effort_queue: deque[np.ndarray] = deque([], maxlen=50)
effort_lock = threading.Lock()
effort_thread_running = True


def image_callback(topic_name, msg, width, height, array: np.ndarray) -> None:
    del width, height
    if msg.format.lower() != "jpeg":
        return
    parts = topic_name.split("/")
    if len(parts) < 3:
        return
    camera_name = parts[2]
    if camera_name not in images_dict:
        return
    with images_lock:
        images_dict[camera_name] = {
            "data": np.asarray(array).copy(),
            "timestamp": time.time(),
            "received_monotonic_s": time.perf_counter(),
        }


def record_effort_continuously(astribot_instance, hz: float = 30.0) -> None:
    global effort_thread_running
    interval = 1.0 / float(hz)
    current = np.concatenate(astribot_instance.get_current_joints_torque()[:7])
    history_effort_queue.extend([current] * history_effort_queue.maxlen)
    start = time.perf_counter()
    frame = 0
    while effort_thread_running:
        frame += 1
        target = start + frame * interval
        current = np.concatenate(astribot_instance.get_current_joints_torque()[:7])
        with effort_lock:
            history_effort_queue.append(current)
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)


def _wait_for_fresh_images(*, return_metadata: bool = False):
    while True:
        now = time.time()
        with images_lock:
            entries = [images_dict[name].copy() for name in CAMERAS]
        if all(entry["data"] is not None for entry in entries):
            ages = [now - float(entry["timestamp"]) for entry in entries]
            if all(age <= MAX_IMAGE_AGE for age in ages):
                images = tuple(entry["data"] for entry in entries)
                if return_metadata:
                    # Same locked snapshot as the images; no later lookup/race.
                    times = [{"camera": camera,
                              "received_wall_time_s": float(entry["timestamp"]),
                              "received_monotonic_s": entry.get("received_monotonic_s"),
                              "age_at_selection_s": float(age)}
                             for camera, entry, age in zip(CAMERAS, entries, ages)]
                    return images, times
                return images
        time.sleep(0.005)


def build_wam_state(
    astribot,
    action_mode: str,
    chassis_origin: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the exact 25-D WAM state and raw current joint arrays."""
    cur_joint_all = astribot.get_current_joints_position()
    if len(cur_joint_all) < 7:
        raise RuntimeError(f"Expected at least 7 Astribot joint groups, got {len(cur_joint_all)}")

    if "vel" in action_mode.lower():
        chassis_velocity = np.asarray(
            astribot.get_current_joints_velocity([astribot.chassis_name])[0],
            dtype=np.float32,
        ).reshape(-1)
        state = np.concatenate(
            [chassis_velocity]
            + [np.asarray(cur_joint_all[i]).reshape(-1) for i in range(1, 7)]
        )
    else:
        chassis_relative = (
            np.asarray(cur_joint_all[0], dtype=np.float32).reshape(-1) - chassis_origin
        )
        state = np.concatenate(
            [chassis_relative]
            + [np.asarray(cur_joint_all[i]).reshape(-1) for i in range(1, 7)]
        )

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (WAM_DIM,):
        raise RuntimeError(
            f"WAM state must be {WAM_DIM}D, got {state.shape}. "
            "Expected base(3)+torso(4)+left(8)+right(8)+head(2)."
        )
    return state, cur_joint_all


def split_wam_action(action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (WAM_DIM,):
        raise ValueError(f"WAM action must be {WAM_DIM}D, got {action.shape}")
    return {
        "base": action[0:3],
        "torso": action[3:7],
        "left_arm": action[7:14],
        "left_gripper": action[14:15],
        "right_arm": action[15:22],
        "right_gripper": action[22:23],
        "head": action[23:25],
    }


def blend_chunk_start(
    actions: np.ndarray,
    current_state: np.ndarray,
    blend_steps: int,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != current_state.shape[0]:
        raise ValueError(f"Cannot blend action/state shapes {actions.shape} and {current_state.shape}")
    count = min(max(int(blend_steps), 0), actions.shape[0])
    if count == 0:
        return actions
    progress = np.arange(1, count + 1, dtype=np.float32) / float(count)
    weights = progress * progress * (3.0 - 2.0 * progress)
    actions[:count] = (
        current_state[None, :] * (1.0 - weights[:, None])
        + actions[:count] * weights[:, None]
    )
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astribot real-robot client for Fast-WAM")
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=2222)
    parser.add_argument(
        "--execute_steps",
        type=int,
        default=None,
        help="Execute the first N actions before replanning; defaults to server n_action_steps.",
    )
    parser.add_argument("--boundary_blend_steps", type=int, default=0)
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument(
        "--action_dt",
        type=float,
        default=1.0 / 30.0,
        help="Temporal spacing of predicted actions; dishwasher data is 30 Hz.",
    )
    parser.add_argument("--max_runtime", type=float, default=1800.0)
    parser.add_argument("--record_dir", default=None,
                        help="Opt-in recorder root. Omit to preserve unrecorded execution.")
    parser.add_argument("--record_queue_size", type=int, default=8)
    parser.add_argument("--record_max_gb", type=float, default=5.0)
    parser.add_argument("--record_min_free_gb", type=float, default=2.0)
    parser.add_argument("--record_image_every", type=int, default=1)
    parser.add_argument("--record_close_timeout", type=float, default=3.0)
    return parser.parse_args()


def _stamp() -> dict:
    return {"wall_time_s": time.time(), "monotonic_time_s": time.perf_counter()}


def _return_description(value) -> dict:
    # Do not stringify arbitrary SDK objects or call their methods.
    scalar = value if type(value) in (type(None), bool, int, str) else None
    return {"sdk_return_type": type(value).__name__, "sdk_return_scalar": scalar}


def _run_client(args, recorder=None) -> None:
    global USE_HISTORY_EFFORT_OBS, history_effort_queue, effort_thread_running

    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    astribot = Astribot(freq=float(args.control_hz))
    astribot.move_to_home()

    cameras_info = astribot.get_cameras_info()
    ready = all(cameras_info[camera]["activate"] for camera in CAMERAS)
    while not ready:
        astribot.deactivate_camera()
        time.sleep(5.0)
        astribot.activate_camera()
        print("Waiting for cameras to be ready...")
        for _ in range(4):
            time.sleep(5.0)
            cameras_info = astribot.get_cameras_info()
            ready = all(cameras_info[camera]["activate"] for camera in CAMERAS)
            if ready:
                break
    print("All cameras are ready.")

    endpoint = f"{args.server_host}:{args.server_port}"
    client = portal.Client(endpoint)
    while not client.connected:
        print(f"...Waiting for inference server at {endpoint}...")
        time.sleep(1.0)

    model_config = client.get_model_info().result()
    action_mode = str(model_config["action_mode"])
    chunk_size = int(model_config["chunk_size"])
    action_dim = int(model_config.get("action_dim", WAM_DIM))
    state_dim = int(model_config.get("state_dim", WAM_DIM))
    if action_dim != WAM_DIM or state_dim != WAM_DIM:
        raise RuntimeError(
            f"This WAM client requires state_dim=action_dim={WAM_DIM}, "
            f"server reported state_dim={state_dim}, action_dim={action_dim}."
        )

    recommended_steps = int(model_config.get("n_action_steps", chunk_size))
    execute_steps = recommended_steps if args.execute_steps is None else int(args.execute_steps)
    if not 1 <= execute_steps <= chunk_size:
        raise ValueError(f"execute_steps must be in [1,{chunk_size}], got {execute_steps}")

    USE_HISTORY_EFFORT_OBS = bool(model_config.get("use_effort", False))
    print("==== Model config:\n", model_config)
    print(
        f"==== Runtime: execute_steps={execute_steps}, action_dt={args.action_dt:.6f}s, "
        f"control_hz={args.control_hz:.1f}"
    )

    subscribers = [
        astribot.register_image_callback(camera, "color", image_callback, need_decode=True)
        for camera in CAMERAS
    ]
    del subscribers
    time.sleep(0.2)

    # Preserve the existing Astribot runtime behavior: the head follows the
    # effector, so predicted head[23:25] is currently not sent explicitly.
    astribot.set_head_follow_effector(enable=True)

    effort_thread = None
    if USE_HISTORY_EFFORT_OBS:
        effort_buffer_len = chunk_size
        history_effort_queue = deque([], maxlen=effort_buffer_len)
        effort_thread_running = True
        effort_thread = threading.Thread(
            target=record_effort_continuously,
            args=(astribot,),
            kwargs={"hz": 30.0},
            daemon=True,
        )
        effort_thread.start()
        while len(history_effort_queue) < effort_buffer_len:
            time.sleep(0.1)

    desired = astribot.get_desired_joints_position(names=[astribot.chassis_name])
    chassis_origin = np.asarray(desired[0], dtype=np.float32).reshape(-1)
    if chassis_origin.shape != (3,):
        raise RuntimeError(f"Expected 3D chassis origin, got {chassis_origin.shape}")

    command_names = [
        astribot.chassis_name,
        astribot.torso_name,
        astribot.arm_left_name,
        astribot.effector_left_name,
        astribot.arm_right_name,
        astribot.effector_right_name,
    ]

    if recorder is not None:
        recorder.submit("runtime", metadata={
            "model_info": model_config, "arguments": vars(args),
            "cameras": list(CAMERAS), "command_names": command_names,
            "effective_execute_steps": execute_steps,
            "head_follow_effector": True,
            "recording_begins_after_startup": True,
        }, arrays={"chassis_origin": chassis_origin})

    all_start = False
    start_time = time.time()
    iteration = 0
    request_id = 0
    previous_command_request = None
    exit_reason = "max_runtime"
    stage = "before_loop"

    try:
        while time.time() - start_time < float(args.max_runtime):
            request_id += 1
            stage = "observation"
            if recorder is None:
                img0, img1, img2 = _wait_for_fresh_images()
            else:
                waiting_since = time.perf_counter()
                (img0, img1, img2), camera_times = _wait_for_fresh_images(return_metadata=True)
                waited = time.perf_counter() - waiting_since
                state_started = _stamp()
            state, cur_joint_all = build_wam_state(astribot, action_mode, chassis_origin)
            if recorder is not None:
                state_finished = _stamp()
                recorder.observation(request_id, (img0, img1, img2), camera_times,
                                     state, cur_joint_all, state_started, state_finished,
                                     previous_command_request, waited)
            previous_command_request = None
            his_effort = (
                np.asarray(history_effort_queue, dtype=np.float32).reshape(-1)
                if USE_HISTORY_EFFORT_OBS
                else None
            )

            if not client.connected:
                print("ERROR: inference server disconnected, exiting...")
                exit_reason = "server_disconnected"
                break

            stage = "rpc"
            if recorder is not None:
                recorder.submit("rpc_attempt", request_id,
                                arrays={"history_effort": his_effort} if his_effort is not None else {})
                rpc_started = _stamp()
            request_time = time.time()
            future = client.infer(request_time, img0, img1, img2, state, his_effort)
            actions_raw = future.result(timeout=2000)
            if recorder is not None:
                rpc_finished = _stamp()
            if actions_raw is None:
                if recorder is not None:
                    recorder.submit("rpc_none", request_id, metadata={
                        "started": rpc_started, "finished": rpc_finished,
                        "rpc_wall_seconds": rpc_finished["monotonic_time_s"] - rpc_started["monotonic_time_s"],
                    })
                print("ERROR: inference server returned None; skipping this cycle")
                continue

            stage = "prepare_command"
            actions_raw = np.asarray(actions_raw, dtype=np.float32)
            if recorder is not None:
                recorder.submit("prediction", request_id, arrays={"actions_raw25": actions_raw}, metadata={
                    "started": rpc_started, "finished": rpc_finished,
                    "rpc_wall_seconds": rpc_finished["monotonic_time_s"] - rpc_started["monotonic_time_s"],
                    "shape": list(actions_raw.shape),
                })
            if actions_raw.ndim != 2 or actions_raw.shape[1] != WAM_DIM:
                raise RuntimeError(f"Expected actions [T,{WAM_DIM}], got {actions_raw.shape}")
            actions = actions_raw[:execute_steps].copy()
            if args.boundary_blend_steps > 0:
                actions = blend_chunk_start(actions, state, args.boundary_blend_steps)

            if "vel" in action_mode.lower():
                current_chassis = np.asarray(cur_joint_all[0], dtype=np.float32).reshape(3)
                chassis_positions = (
                    np.cumsum(actions[:, 0:3], axis=0) * float(args.action_dt)
                    + current_chassis
                )
            else:
                chassis_positions = actions[:, 0:3] + chassis_origin

            waypoints = []
            time_list = []
            for idx, action in enumerate(actions):
                parts = split_wam_action(action)
                command_list = [
                    chassis_positions[idx].tolist(),
                    parts["torso"].tolist(),
                    parts["left_arm"].tolist(),
                    parts["left_gripper"].tolist(),
                    parts["right_arm"].tolist(),
                    parts["right_gripper"].tolist(),
                ]
                waypoints.append(command_list)
                time_list.append(float(args.action_dt) * (idx + 1))

            if not all_start:
                stage = "initial_position_command"
                if recorder is not None:
                    recorder.command("sdk_initial_position_attempt", request_id, command_names,
                                     [waypoints[0]], [], actions[:1], cur_joint_all[0], action_mode)
                    position_started = _stamp()
                previous_command_request = request_id
                position_result = astribot.move_joints_position(command_names, waypoints[0])
                if recorder is not None:
                    recorder.submit("sdk_initial_position_return", request_id,
                                    metadata={"started": position_started, "finished": _stamp(),
                                              **_return_description(position_result)})
                time.sleep(1.0)
                all_start = True

            stage = "waypoints_command"
            if recorder is not None:
                recorder.command("sdk_waypoints_attempt", request_id, command_names, waypoints,
                                 time_list, actions, cur_joint_all[0], action_mode)
                sdk_started = _stamp()
            previous_command_request = request_id
            sdk_result = astribot.move_joints_waypoints(
                command_names,
                waypoints,
                time_list,
                use_wbc=False,
                joy_controller=False,
            )
            if recorder is not None:
                sdk_finished = _stamp()
                recorder.submit("sdk_waypoints_return", request_id, metadata={
                    "started": sdk_started, "finished": sdk_finished,
                    "sdk_call_wall_seconds": sdk_finished["monotonic_time_s"] - sdk_started["monotonic_time_s"],
                    "physical_completion_verified": False,
                    **_return_description(sdk_result),
                })
            iteration += 1
            print(
                f"[wam-client] iter={iteration} state={state.shape} actions={actions.shape} "
                f"server_latency={time.time() - request_time:.3f}s"
            )
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        if recorder is not None:
            recorder.submit("loop_interrupted", request_id, metadata={"stage": stage})
        print("Ctrl+C received; stopping robot client.")
    except Exception as exc:
        exit_reason = "exception"
        if recorder is not None:
            recorder.submit("loop_error", request_id, metadata={"stage": stage,
                            "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if recorder is not None:
            recorder.submit("loop_end", request_id, metadata={
                "reason": exit_reason, "requests": request_id, "sdk_returned_iterations": iteration,
                "command_without_next_observation": previous_command_request,
                "task_success": None,
            })
        effort_thread_running = False
        if effort_thread is not None:
            effort_thread.join(timeout=1.0)
        client.close()


def main() -> None:
    args = parse_args()
    if args.control_hz <= 0 or args.action_dt <= 0:
        raise ValueError("--control_hz and --action_dt must be positive")
    recorder = None
    if args.record_dir is not None:
        # Create/check local recording BEFORE importing or initializing hardware.
        if __package__:
            from .rollout_recorder import RecorderConfig, RolloutRecorder
        else:
            from rollout_recorder import RecorderConfig, RolloutRecorder
        import hashlib
        from pathlib import Path
        config = RecorderConfig(root=args.record_dir, queue_size=args.record_queue_size,
            max_bytes=int(args.record_max_gb * 1024**3),
            min_free_bytes=int(args.record_min_free_gb * 1024**3),
            image_every=args.record_image_every, close_timeout=args.record_close_timeout)
        recorder = RolloutRecorder(config, {"arguments": vars(args),
            "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print(f"[record] local directory: {recorder.path}", flush=True)
    reason = "client_returned"
    try:
        _run_client(args, recorder)
    except BaseException as exc:
        reason = type(exc).__name__
        if recorder is not None:
            recorder.submit("client_exit_error", metadata={"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if recorder is not None:
            # No new SDK reads/commands on exit; never delay existing cleanup indefinitely.
            result = recorder.close(reason)
            print(f"[record] {result}; local files: {recorder.path}", flush=True)
            if result["writer_still_alive"]:
                print("[record] flush deadline reached; missing CLOSED.json means incomplete recording.", flush=True)


if __name__ == "__main__":
    main()
