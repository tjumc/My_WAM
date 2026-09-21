#!/usr/bin/env python3
"""V2 conservative atomic-event verification.

Key changes vs V1:
- Candidate events are assumed to be over-segmented; FALSE is normal.
- Qwen never interprets raw gripper values as open/closed.
- Local HDF5 statistics are converted into neutral trends (increase/decrease/stable).
- Qwen does NOT choose temporal phase boundaries.
- Object identity is deliberately conservative.
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


def load_hdf5_context(hdf5_path, events, fps=30.0, radius_sec=0.25):
    """Turn raw signals into neutral local trends; never assign semantic open/closed labels."""
    radius = max(2, int(round(radius_sec * fps)))
    with h5py.File(hdf5_path, "r") as f:
        rg = f["command_poses_dict/astribot_gripper_right"][:, 0].astype(float)
        lg = f["command_poses_dict/astribot_gripper_left"][:, 0].astype(float)
        rw = f["endpoint_wrench_dict/astribot_arm_right"][:].astype(float)
        lw = f["endpoint_wrench_dict/astribot_arm_left"][:].astype(float)

    rforce = np.linalg.norm(rw[:, :3], axis=1)
    lforce = np.linalg.norm(lw[:, :3], axis=1)
    out = {}

    for e in events:
        i = int(e["raw_frame"])
        a0, a1 = max(0, i - radius), max(1, i)
        b0, b1 = min(len(rg) - 1, i + 1), min(len(rg), i + radius + 1)

        def med(x, s, t):
            if t <= s:
                return float(x[min(max(s, 0), len(x)-1)])
            return float(np.median(x[s:t]))

        rgb, rga = med(rg, a0, a1), med(rg, b0, b1)
        lgb, lga = med(lg, a0, a1), med(lg, b0, b1)
        rfb, rfa = med(rforce, a0, a1), med(rforce, b0, b1)
        lfb, lfa = med(lforce, a0, a1), med(lforce, b0, b1)

        # thresholds describe numerical change only, not physical semantics
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


def verify_event(client, model, task, event, sensor_context, sheet_path, max_tokens=900):
    prompt = f"""你正在保守地验证一条人形机器人家庭操作轨迹中的候选事件。

重要规则：
1. 候选点由高召回率检测器生成，故意过分割；很多候选点应当判定为 false。
2. 只有当三视角时序图明确显示“离散的动作/接触/物体状态转换”时，才设 is_semantic_event=true。
3. 单纯持续运动、同一动作内部的中间时刻、相机运动、姿态微调，通常应为 false。
4. 不得根据高层任务脑补动作。
5. 不得根据夹爪 command 数值猜“张开/闭合”。下面只给出 increase/decrease/stable，它们没有预先定义的物理语义。
6. 如果只能看出“盘子/餐具”，不要擅自声明它与前后事件是同一个具体物体。
7. confidence 必须校准：证据冲突或身份不确定时应明显低于 0.8。

高层任务：{task}
候选：frame={event['raw_frame']}, time={event['time_sec']:.3f}s
candidate score：{event.get('score')}
检测器变化信号：{', '.join(event.get('signals', []))}

中性传感器摘要：
{json.dumps(sensor_context, ensure_ascii=False, indent=2)}

图片说明：
- 三行分别为 head / left / right
- 列从事件前到事件后
- 中间列附近是候选时刻

请严格输出一个 JSON：
{{
  "is_semantic_event": false,
  "event_type": "navigation_start|navigation_stop|reach|grasp|release|open|close|pull|push|place|retract|reposition|contact|other|uncertain",
  "atomic_action": "只描述能够直接观察到的局部动作；false 时写持续动作/无明确边界",
  "active_arm": "left|right|both|none|uncertain",
  "visible_object_category": "dish|plate|fork|rack|door|cabinet|dishwasher|other|none|uncertain",
  "object_identity_confident": false,
  "physical_transition": "例如 free->contact / contact->free / door_moving / none / uncertain",
  "task_relevance": "relevant|post_task|unrelated|uncertain",
  "boundary_frame_relation": "before|near|after|uncertain",
  "confidence": 0.0,
  "evidence": ["最多3条简短且可观察的证据"],
  "reason_if_false": "若为false，说明为什么它只是同一动作内部或证据不足"
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
    obj["_detector_score"] = event.get("score")
    obj["_detector_signals"] = event.get("signals", [])
    obj["_sensor_context"] = sensor_context
    return obj, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--aigc-user", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    root = Path(args.analysis_dir)
    manifest = json.load(open(root / "candidate_events.json", encoding="utf-8"))
    events = manifest["candidate_events"]
    hctx = load_hdf5_context(args.hdf5, events, fps=args.fps)
    client = build_client(args)

    raw_dir = root / "vlm_raw_v2"
    raw_dir.mkdir(exist_ok=True)
    out_jsonl = root / "event_semantics_v2.jsonl"

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        for e in events:
            matches = sorted((root / "contact_sheets").glob(f"event_{e['event_id']:02d}_f*.jpg"))
            if not matches:
                raise FileNotFoundError(f"No contact sheet for event {e['event_id']}")
            obj, raw = verify_event(
                client, args.model, args.task, e, hctx[e["event_id"]], matches[0]
            )
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fw.flush()
            (raw_dir / f"event_{e['event_id']:02d}.txt").write_text(raw, encoding="utf-8")
            print(
                f"[{e['event_id']:02d}] keep={obj.get('is_semantic_event')} "
                f"type={obj.get('event_type')} rel={obj.get('task_relevance')} "
                f"conf={obj.get('confidence')}  {obj.get('atomic_action')}"
            )

    print("saved", out_jsonl)


if __name__ == "__main__":
    main()
