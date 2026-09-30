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


def _wait_for_fresh_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    while True:
        now = time.time()
        with images_lock:
            entries = [images_dict[name].copy() for name in CAMERAS]
        if all(entry["data"] is not None for entry in entries):
            ages = [now - float(entry["timestamp"]) for entry in entries]
            if all(age <= MAX_IMAGE_AGE for age in ages):
                return tuple(entry["data"] for entry in entries)  # type: ignore[return-value]
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
    return parser.parse_args()


def main() -> None:
    global USE_HISTORY_EFFORT_OBS, history_effort_queue, effort_thread_running

    args = parse_args()
    if args.control_hz <= 0 or args.action_dt <= 0:
        raise ValueError("--control_hz and --action_dt must be positive")

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

    all_start = False
    start_time = time.time()
    iteration = 0

    try:
        while time.time() - start_time < float(args.max_runtime):
            img0, img1, img2 = _wait_for_fresh_images()
            state, cur_joint_all = build_wam_state(astribot, action_mode, chassis_origin)
            his_effort = (
                np.asarray(history_effort_queue, dtype=np.float32).reshape(-1)
                if USE_HISTORY_EFFORT_OBS
                else None
            )

            if not client.connected:
                print("ERROR: inference server disconnected, exiting...")
                break

            request_time = time.time()
            future = client.infer(request_time, img0, img1, img2, state, his_effort)
            actions_raw = future.result(timeout=2000)
            if actions_raw is None:
                print("ERROR: inference server returned None; skipping this cycle")
                continue

            actions_raw = np.asarray(actions_raw, dtype=np.float32)
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
                astribot.move_joints_position(command_names, waypoints[0])
                time.sleep(1.0)
                all_start = True

            astribot.move_joints_waypoints(
                command_names,
                waypoints,
                time_list,
                use_wbc=False,
                joy_controller=False,
            )
            iteration += 1
            print(
                f"[wam-client] iter={iteration} state={state.shape} actions={actions.shape} "
                f"server_latency={time.time() - request_time:.3f}s"
            )
    except KeyboardInterrupt:
        print("Ctrl+C received; stopping robot client.")
    finally:
        effort_thread_running = False
        if effort_thread is not None:
            effort_thread.join(timeout=1.0)
        client.close()


if __name__ == "__main__":
    main()
