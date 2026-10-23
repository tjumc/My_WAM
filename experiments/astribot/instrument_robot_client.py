"""Generate a diagnostic copy of the verified ymn client, without importing it.

The robot SDK is never loaded by this installer. The original file is preserved.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil

EXPECTED_SHA256 = "a71c7853ede3e1ef64e9a2ced71bfabe469ba0c3d970df3173c39d97b9309d4a"


def instrument(source):
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Client differs from the verified ymn version; review before adapting anchors")

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Expected one anchor: {old[:70]!r}")
        source = source.replace(old, new)

    replace("import portal\n", "import portal\nimport atexit\nfrom wam_diagnostics import TraceSession, joint_snapshot, sha256_file\n")
    replace("    args = parser.parse_args()", '''    parser.add_argument("--trace-dir", default=None,
                        help="Enable diagnostic JSONL/NPZ recording in a new session directory")
    parser.add_argument("--capture-every", type=int, default=10,
                        help="Capture raw images/state every N requests; 0 disables image capture")
    parser.add_argument("--trace-joint-hz", type=float, default=20.0,
                        help="Joint telemetry sampling frequency; 0 disables extra reads")
    args = parser.parse_args()
    if args.capture_every < 0:
        parser.error("--capture-every must be >= 0")
    if not np.isfinite(args.trace_joint_hz) or args.trace_joint_hz < 0:
        parser.error("--trace-joint-hz must be finite and >= 0")''')
    replace("    effort_record_hz = control_hz", '''    trace = TraceSession(args.trace_dir, {
        "args": vars(args), "model": model_config,
        "client_source_sha256": sha256_file(__file__),
        "original_source_sha256": "''' + EXPECTED_SHA256 + '''",
        "execute_steps": execute_steps, "waypoint_step_dt": waypoint_step_dt,
        "vel_integrate_dt": vel_integrate_dt, "control_hz": control_hz,
        "control_way": control_way,
    }, kind="client")
    atexit.register(trace.close)
    previous_submitted_last = None
    request_index = 0
    effort_record_hz = control_hz''')
    replace("    pose_thread.start()", "    pose_thread.start()\n    trace.start_joint_sampler(astribot, args.trace_joint_hz)")
    replace("            cur_joint_all = astribot.get_current_joints_position()", "            observation_start_ns = time.monotonic_ns()\n            cur_joint_all = astribot.get_current_joints_position()")
    replace("            future = client.infer(time.time(), img0, img1, img2, cur_state, his_effort)\n            actions_raw = future.result(timeout=2000)", '''            request_index += 1
            request_id = time.time()
            capture = (trace.enabled and args.capture_every > 0
                       and (request_index - 1) % args.capture_every == 0)
            trace.emit("request", request_id=request_id, request_index=request_index,
                       state=cur_state, observation_start_ns=observation_start_ns,
                       image_timestamps={cam: images_dict[cam]["timestamp"] for cam in target_cameras},
                       arrays=({"img0": img0, "img1": img1,
                       "img2": img2, "state": cur_state} if capture else None))
            rpc_start_ns = time.monotonic_ns()
            future = client.infer(request_id, img0, img1, img2, cur_state, his_effort)
            actions_raw = future.result(timeout=2000)
            rpc_end_ns = time.monotonic_ns()
            trace.emit("rpc_return", request_id=request_id,
                       rpc_ms=(rpc_end_ns-rpc_start_ns)/1e6,
                       valid=actions_raw is not None)''')
    replace("            actions = actions_raw[:execute_steps].copy()", '''            actions = actions_raw[:execute_steps].copy()
            trace.emit("prediction", request_id=request_id,
                       finite=bool(np.isfinite(actions_raw).all()),
                       action_shape=list(actions_raw.shape),
                       arrays={"actions": actions_raw, "selected_actions": actions})''')
    replace("                if not all_start:\n", '''                if not all_start:
                    trace.emit("initial_positioning", request_id=request_id)
''')
    replace("                astribot.move_joints_waypoints(command_names, waypoints, time_list, use_wbc=False, joy_controller=False)", '''                q_dispatch = joint_snapshot(astribot) if trace.enabled else None
                trace.emit("dispatch", request_id=request_id, iteration=i,
                           measured_joints=q_dispatch,
                           first_target=actions[0], last_target=actions[-1],
                           previous_submitted_last=previous_submitted_last,
                           submitted_steps=len(actions), waypoint_step_dt=waypoint_step_dt,
                           scheduled_duration=time_list[-1],
                           observation_age_ms=(time.monotonic_ns()-observation_start_ns)/1e6)
                sdk_start_ns = time.monotonic_ns()
                astribot.move_joints_waypoints(command_names, waypoints, time_list, use_wbc=False, joy_controller=False)
                sdk_end_ns = time.monotonic_ns()
                trace.emit("sdk_return", request_id=request_id,
                           sdk_call_ms=(sdk_end_ns-sdk_start_ns)/1e6,
                           measured_joints=joint_snapshot(astribot) if trace.enabled else None)
                if trace.enabled:
                    previous_submitted_last = actions[-1].copy()''')
    replace("    except KeyboardInterrupt:\n", '''    except KeyboardInterrupt:
        trace.emit("stop", reason="keyboard_interrupt")
''')
    replace("    # 停止姿态记录线程\n", '''    except Exception as exc:
        trace.emit("execution_error", error_type=type(exc).__name__, error=str(exc))
        trace.close()
        raise
    trace.emit("loop_exit", elapsed_s=time.time()-t0, iterations=i)
    trace.close()
    # 停止姿态记录线程
''')
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.client.read_text(encoding="utf-8")
    output = instrument(source)
    helper = Path(__file__).with_name("wam_diagnostics.py")
    destination = args.output.parent / helper.name
    if args.output.exists():
        raise FileExistsError(args.output)
    if destination.exists() and destination.read_bytes() != helper.read_bytes():
        raise FileExistsError(f"Different helper exists: {destination}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if helper.resolve() != destination.resolve():
        shutil.copyfile(helper, destination)
    with args.output.open("x", encoding="utf-8") as f:
        f.write(output)
    print(f"Created diagnostic client: {args.output}")
    print(f"Helper: {destination}")


if __name__ == "__main__":
    main()
