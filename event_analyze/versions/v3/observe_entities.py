#!/usr/bin/env python3
"""V3 constrained entity-state observation for long-horizon dishwasher trajectories.

The VLM is deliberately NOT asked to name actions or phases. It only reports
observable entity states before/after each proposal window. Skills are inferred
later by deterministic state machines.
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
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"No JSON object found: {text[:600]}")
        return json.loads(m.group(0))


def build_client(args):
    key = args.api_key or os.getenv("AIGC_API_KEY")
    if not key:
        raise RuntimeError("AIGC_API_KEY is required")
    headers = {}
    user = args.aigc_user or os.getenv("AIGC_USER")
    if user:
        headers["AIGC-USER"] = user
    return OpenAI(
        api_key=key,
        base_url=args.base_url or os.getenv("AIGC_BASE_URL", DEFAULT_BASE_URL),
        default_headers=headers or None,
    )


def _trend(a, b, eps):
    d = float(b - a)
    return ("stable" if abs(d) <= eps else ("increase" if d > 0 else "decrease")), d


def load_hdf5_context(path, events, fps=30.0, radius_sec=0.35):
    """Neutral local control/contact summaries; no semantic gripper mapping."""
    r = max(2, int(round(radius_sec * fps)))
    with h5py.File(path, "r") as f:
        rg = f["command_poses_dict/astribot_gripper_right"][:, 0].astype(float)
        lg = f["command_poses_dict/astribot_gripper_left"][:, 0].astype(float)
        rw = f["endpoint_wrench_dict/astribot_arm_right"][:].astype(float)
        lw = f["endpoint_wrench_dict/astribot_arm_left"][:].astype(float)
    rf = np.linalg.norm(rw[:, :3], axis=1)
    lf = np.linalg.norm(lw[:, :3], axis=1)

    def med(x, s, t):
        if t <= s:
            return float(x[min(max(s, 0), len(x) - 1)])
        return float(np.median(x[s:t]))

    out = {}
    for e in events:
        i = int(e["raw_frame"])
        a0, a1 = max(0, i - r), max(1, i)
        b0, b1 = min(len(rg) - 1, i + 1), min(len(rg), i + r + 1)
        rgb, rga = med(rg, a0, a1), med(rg, b0, b1)
        lgb, lga = med(lg, a0, a1), med(lg, b0, b1)
        rfb, rfa = med(rf, a0, a1), med(rf, b0, b1)
        lfb, lfa = med(lf, a0, a1), med(lf, b0, b1)
        rt, rd = _trend(rgb, rga, 5.0)
        lt, ld = _trend(lgb, lga, 5.0)
        rft, rfd = _trend(rfb, rfa, max(2.0, 0.1 * max(rfb, rfa, 1)))
        lft, lfd = _trend(lfb, lfa, max(2.0, 0.1 * max(lfb, lfa, 1)))
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


STATE_TEMPLATE = {
    "robot_location": "uncertain",
    "door": "uncertain",
    "dish_rack": "uncertain",
    "cutlery_basket": "uncertain",
    "right_hand": "uncertain",
    "left_hand": "uncertain",
    "knife_location": "uncertain",
    "fork_location": "uncertain",
    "plate_location": "uncertain",
}

VALID = {
    "robot_location": {"away", "approaching_station", "at_station", "uncertain", "not_visible"},
    "door": {"closed", "opening", "open", "closing", "uncertain", "not_visible"},
    "dish_rack": {"in", "moving_out", "out", "moving_in", "uncertain", "not_visible"},
    "cutlery_basket": {"in", "moving_out", "out", "moving_in", "uncertain", "not_visible"},
    "right_hand": {
        "free", "approaching", "contact_door", "contact_dish_rack", "contact_cutlery_basket",
        "holding_knife", "holding_fork", "holding_plate", "uncertain", "not_visible"
    },
    "left_hand": {
        "free", "approaching", "contact_door", "contact_dish_rack", "contact_cutlery_basket",
        "holding_knife", "holding_fork", "holding_plate", "uncertain", "not_visible"
    },
    "knife_location": {"tabletop", "right_hand", "left_hand", "cutlery_basket", "other", "uncertain", "not_visible"},
    "fork_location": {"tabletop", "right_hand", "left_hand", "cutlery_basket", "other", "uncertain", "not_visible"},
    "plate_location": {"tabletop", "right_hand", "left_hand", "dish_rack", "other", "uncertain", "not_visible"},
}


def sanitize_state(state):
    out = dict(STATE_TEMPLATE)
    for k in out:
        v = state.get(k, "uncertain") if isinstance(state, dict) else "uncertain"
        out[k] = v if v in VALID[k] else "uncertain"
    return out


def sanitize_conf(conf):
    out = {}
    for k in STATE_TEMPLATE:
        try:
            out[k] = max(0.0, min(1.0, float((conf or {}).get(k, 0.0))))
        except Exception:
            out[k] = 0.0
    return out


def observe(client, model, task, event, ctx, sheet, fps, radius_sec, max_tokens=1700):
    r = int(round(radius_sec * fps))
    w0 = max(0, int(event["raw_frame"]) - r)
    w1 = int(event["raw_frame"]) + r
    prompt = f"""你正在做机器人轨迹的实体状态观测。不要命名动作，不要生成 phase，不要根据任务猜测。

高层任务只作为场景背景：{task}
窗口约 raw frame {w0}~{w1}，中心 frame={event['raw_frame']}，time={event['time_sec']:.3f}s。
三行图片严格依次是 head / left / right 相机的同步时序，每行从左到右为早→晚。

必须区分两个不同容器：
- dish_rack（碗篮）：较大的洗碗机拉篮，用于盘子/碗；
- cutlery_basket（餐具篮）：较小的刀叉餐具篮/托盘，用于 knife/fork。
看不清时填 uncertain，严禁把两者都泛化为 rack。

必须区分三个餐具实体：knife / fork / plate。看不清具体类别时保持 uncertain，不要用 plate/dish 代替 knife/fork。

状态枚举：
robot_location: away|approaching_station|at_station|uncertain|not_visible
door: closed|opening|open|closing|uncertain|not_visible
dish_rack: in|moving_out|out|moving_in|uncertain|not_visible
cutlery_basket: in|moving_out|out|moving_in|uncertain|not_visible
right_hand/left_hand: free|approaching|contact_door|contact_dish_rack|contact_cutlery_basket|holding_knife|holding_fork|holding_plate|uncertain|not_visible
knife_location: tabletop|right_hand|left_hand|cutlery_basket|other|uncertain|not_visible
fork_location: tabletop|right_hand|left_hand|cutlery_basket|other|uncertain|not_visible
plate_location: tabletop|right_hand|left_hand|dish_rack|other|uncertain|not_visible

规则：
1. 先独立观察窗口早期 state_before 和晚期 state_after；不要从“你认为发生了什么动作”倒推状态。
2. 如果实体被遮挡，不要延续上一窗口身份，填 not_visible/uncertain。
3. 不得依据夹爪 command 数值判断抓取/释放；以下信号仅作运动/接触辅助，没有预设开合语义。
4. 若右手/左手看起来握着东西但无法区分 knife/fork/plate，hand 填 uncertain，而不是猜类别。
5. confidence_by_entity 必须逐实体校准；真正清晰可见才给 >=0.85。

中性传感器摘要：
{json.dumps(ctx, ensure_ascii=False, indent=2)}

严格输出 JSON：
{{
  "state_before": {json.dumps(STATE_TEMPLATE, ensure_ascii=False)},
  "state_after": {json.dumps(STATE_TEMPLATE, ensure_ascii=False)},
  "confidence_before": {{"robot_location":0.0,"door":0.0,"dish_rack":0.0,"cutlery_basket":0.0,"right_hand":0.0,"left_hand":0.0,"knife_location":0.0,"fork_location":0.0,"plate_location":0.0}},
  "confidence_after": {{"robot_location":0.0,"door":0.0,"dish_rack":0.0,"cutlery_basket":0.0,"right_hand":0.0,"left_hand":0.0,"knife_location":0.0,"fork_location":0.0,"plate_location":0.0}},
  "visible_entities": ["只列清楚可见实体"],
  "evidence": ["最多4条纯视觉证据"],
  "window_quality": "good|partial|poor"
}}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(sheet)}},
        ]}],
        max_tokens=max_tokens,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content
    obj = parse_json(raw)
    obj["state_before"] = sanitize_state(obj.get("state_before", {}))
    obj["state_after"] = sanitize_state(obj.get("state_after", {}))
    obj["confidence_before"] = sanitize_conf(obj.get("confidence_before", {}))
    obj["confidence_after"] = sanitize_conf(obj.get("confidence_after", {}))
    obj.update({
        "_event_id": int(event["event_id"]),
        "_raw_frame": int(event["raw_frame"]),
        "_time_sec": float(event["time_sec"]),
        "_window_start_frame": w0,
        "_window_end_frame": w1,
        "_detector_score": event.get("score"),
        "_detector_signals": event.get("signals", []),
        "_sensor_context": ctx,
    })
    return obj, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--aigc-user", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--window-radius-sec", type=float, default=1.5)
    args = ap.parse_args()

    root = Path(args.episode_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = json.load(open(root / "candidate_events.json", encoding="utf-8"))["candidate_events"]
    contexts = load_hdf5_context(args.hdf5, events, args.fps)
    client = build_client(args)
    raw_dir = out / "vlm_raw"
    raw_dir.mkdir(exist_ok=True)
    outfile = out / "entity_observations.jsonl"

    with open(outfile, "w", encoding="utf-8") as fw:
        for e in events:
            sheets = sorted((root / "contact_sheets").glob(f"event_{e['event_id']:02d}_f*.jpg"))
            if not sheets:
                raise FileNotFoundError(f"missing contact sheet for event {e['event_id']}")
            obj, raw = observe(client, args.model, args.task, e, contexts[e["event_id"]], sheets[0], args.fps, args.window_radius_sec)
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fw.flush()
            (raw_dir / f"window_{e['event_id']:02d}.txt").write_text(raw, encoding="utf-8")
            sb, sa = obj["state_before"], obj["state_after"]
            print(f"[{e['event_id']:02d}] door {sb['door']}->{sa['door']} dish_rack {sb['dish_rack']}->{sa['dish_rack']} cutlery {sb['cutlery_basket']}->{sa['cutlery_basket']}")
    print("saved", outfile)


if __name__ == "__main__":
    main()
