#!/usr/bin/env python3
"""V2.1 semantic-window verification for candidate robot events.

A candidate frame is only an attention anchor. Qwen decides whether the surrounding
multi-view window contains a meaningful semantic action; it is NOT asked whether
the center frame is an exact action boundary.
"""
import argparse
import base64
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
from openai import OpenAI

DEFAULT_BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"
DEFAULT_MODEL = "/model/qwen3.5-122b-a10b"


def data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError(f"No JSON object found: {text[:600]}")
        return json.loads(m.group(0))


def build_client(args):
    headers = {}
    user = args.aigc_user or os.getenv("AIGC_USER")
    if user:
        headers["AIGC-USER"] = user
    return OpenAI(
        api_key=args.api_key or os.getenv("AIGC_API_KEY"),
        base_url=args.base_url or os.getenv("AIGC_BASE_URL", DEFAULT_BASE_URL),
        default_headers=headers or None,
    )


def _trend(before, after, eps):
    d = float(after - before)
    if abs(d) <= eps:
        return "stable", d
    return ("increase" if d > 0 else "decrease"), d


def load_hdf5_context(hdf5_path, events, fps=30.0, radius_sec=0.35):
    """Convert raw control/contact signals into neutral local trends.

    Important: command direction is intentionally not mapped to semantic open/close.
    """
    radius = max(2, int(round(radius_sec * fps)))
    with h5py.File(hdf5_path, "r") as f:
        rg = f["command_poses_dict/astribot_gripper_right"][:, 0].astype(float)
        lg = f["command_poses_dict/astribot_gripper_left"][:, 0].astype(float)
        rw = f["endpoint_wrench_dict/astribot_arm_right"][:].astype(float)
        lw = f["endpoint_wrench_dict/astribot_arm_left"][:].astype(float)

    rforce = np.linalg.norm(rw[:, :3], axis=1)
    lforce = np.linalg.norm(lw[:, :3], axis=1)
    out = {}

    def med(x, s, t):
        if t <= s:
            return float(x[min(max(s, 0), len(x) - 1)])
        return float(np.median(x[s:t]))

    for e in events:
        i = int(e["raw_frame"])
        a0, a1 = max(0, i - radius), max(1, i)
        b0, b1 = min(len(rg) - 1, i + 1), min(len(rg), i + radius + 1)

        rgb, rga = med(rg, a0, a1), med(rg, b0, b1)
        lgb, lga = med(lg, a0, a1), med(lg, b0, b1)
        rfb, rfa = med(rforce, a0, a1), med(rforce, b0, b1)
        lfb, lfa = med(lforce, a0, a1), med(lforce, b0, b1)

        rt, rd = _trend(rgb, rga, eps=5.0)
        lt, ld = _trend(lgb, lga, eps=5.0)
        rft, rfd = _trend(rfb, rfa, eps=max(2.0, 0.10 * max(rfb, rfa, 1.0)))
        lft, lfd = _trend(lfb, lfa, eps=max(2.0, 0.10 * max(lfb, lfa, 1.0)))

        out[e["event_id"]] = {
            "right_gripper_command_trend": rt,
            "right_gripper_command_delta": round(rd, 3),
            "left_gripper_command_trend": lt,
            "left_gripper_command_delta": round(ld, 3),
            "right_force_trend": rft,
            "right_force_delta": round(rfd, 3),
            "left_force_trend": lft,
            "left_force_delta": round(lfd, 3),
        }
    return out


def verify_window(client, model, task, event, sensor_context, sheet_path, fps, window_radius_sec, max_tokens=1000):
    radius_frames = int(round(window_radius_sec * fps))
    w0 = max(0, int(event["raw_frame"]) - radius_frames)
    w1 = int(event["raw_frame"]) + radius_frames
    prompt = f"""你正在分析一条人形机器人家庭操作轨迹中的局部时间窗口。

关键定义：候选 frame 只是“注意力锚点”，不是精确语义边界。
你的任务不是判断中心帧是不是动作起点/终点，而是判断整个 ±{window_radius_sec:.1f}s 窗口里是否清楚存在一个有意义的机器人动作。

规则：
1. 如果窗口清楚展示 open / grasp / transport / place / retract 等持续动作，即使中心帧位于动作中间，也应 contains_semantic_action=true。
2. 只有当窗口主要是相机抖动、无法解释的微调、静止、视觉严重不足或不存在稳定可命名动作时，才设 false。
3. action_progress 描述中心帧相对该动作的位置：start / ongoing / end / transition / uncertain。
4. 不要求这个窗口包含精确开始和结束边界；边界稍后由机器人信号算法确定。
5. 高层任务只提供上下文，不得据此脑补看不到的物体或动作。
6. 不得根据夹爪 command 数值猜“张开/闭合”；下面 increase/decrease/stable 没有预设物理含义。
7. 不要跨窗口默认具体物体身份一致。只能可靠说 plate/dish/fork 时就使用类别名称。
8. confidence 要校准；视觉证据明确的持续动作可以 >0.8，证据冲突应降低。

高层任务：{task}
候选中心：frame={event['raw_frame']}, time={event['time_sec']:.3f}s
检测器 score：{event.get('score')}
检测器变化信号：{', '.join(event.get('signals', []))}
分析窗口约为 raw frame {w0} ~ {w1}

中性传感器摘要：
{json.dumps(sensor_context, ensure_ascii=False, indent=2)}

图片是三视角同步 contact sheet：
- 行：head / left / right
- 列：从窗口前部到后部
- 中间列附近对应候选中心

严格输出一个 JSON：
{{
  "contains_semantic_action": true,
  "dominant_action": "navigation|reach|grasp|release|open|close|pull|push|place|transport|retract|reposition|contact|idle|other|uncertain",
  "action_tags": ["最多2个动作标签"],
  "action_progress": "start|ongoing|end|transition|uncertain",
  "atomic_action": "用一句中文描述这个窗口中可观察到的动作，不要强调中心帧必须是边界",
  "active_arm": "left|right|both|none|uncertain",
  "visible_object_category": "dish|plate|fork|rack|door|cabinet|dishwasher|other|none|uncertain",
  "object_identity_confident": false,
  "task_relevance": "relevant|post_task|unrelated|uncertain",
  "confidence": 0.0,
  "evidence": ["最多3条可观察证据"],
  "reason_if_false": "若false，说明为什么整个窗口都不能稳定命名为语义动作"
}}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url(sheet_path)}},
            ],
        }],
        max_tokens=max_tokens,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content
    obj = parse_json(raw)
    obj["_event_id"] = event["event_id"]
    obj["_raw_frame"] = event["raw_frame"]
    obj["_time_sec"] = event["time_sec"]
    obj["_window_start_frame"] = w0
    obj["_window_end_frame"] = w1
    obj["_detector_score"] = event.get("score")
    obj["_detector_signals"] = event.get("signals", [])
    obj["_sensor_context"] = sensor_context
    return obj, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--api-key", default="msk-57d214329a4b63bebb0617ce7dadbddb4d00c382f61be32a015d73c4e6c8d121")
    ap.add_argument("--base-url", default="https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1")
    ap.add_argument("--aigc-user", default="suty11")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--window-radius-sec", type=float, default=1.5)
    args = ap.parse_args()

    root = Path(args.analysis_dir)
    manifest = json.load(open(root / "candidate_events.json", encoding="utf-8"))
    events = manifest["candidate_events"]
    hctx = load_hdf5_context(args.hdf5, events, fps=args.fps)
    client = build_client(args)

    raw_dir = root / "vlm_raw_v2_1"
    raw_dir.mkdir(exist_ok=True)
    out_jsonl = root / "window_semantics_v2_1.jsonl"

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        for e in events:
            matches = sorted((root / "contact_sheets").glob(f"event_{e['event_id']:02d}_f*.jpg"))
            if not matches:
                raise FileNotFoundError(f"No contact sheet for event {e['event_id']}")
            obj, raw = verify_window(
                client, args.model, args.task, e, hctx[e["event_id"]], matches[0],
                fps=args.fps, window_radius_sec=args.window_radius_sec,
            )
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fw.flush()
            (raw_dir / f"window_{e['event_id']:02d}.txt").write_text(raw, encoding="utf-8")
            print(
                f"[{e['event_id']:02d}] keep={obj.get('contains_semantic_action')} "
                f"action={obj.get('dominant_action')} progress={obj.get('action_progress')} "
                f"rel={obj.get('task_relevance')} conf={obj.get('confidence')}"
            )

    print("saved", out_jsonl)


if __name__ == "__main__":
    main()
