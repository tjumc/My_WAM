#!/usr/bin/env python3
"""V2: deterministic temporal deduplication + Qwen semantic phase grouping.

Qwen is NOT allowed to output numerical phase boundaries.
"""
import argparse
import json
import os
import re
from pathlib import Path
from openai import OpenAI

DEFAULT_BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"
DEFAULT_MODEL = "/model/qwen3.5-122b-a10b"

DUP_CLASS = {
    "grasp": "grasp",
    "release": "place_release",
    "place": "place_release",
    "retract": "retract",
    "reach": "reach",
    "open": "open",
    "close": "close",
    "navigation_start": "navigation_start",
    "navigation_stop": "navigation_stop",
    "pull": "pull",
    "push": "push",
    "reposition": "reposition",
    "contact": "contact",
}


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
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


def load_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def deduplicate(events, min_conf=0.60, duplicate_sec=2.0, fps=30.0):
    """Collapse repeated proposals for the same atomic transition."""
    valid = [
        e for e in events
        if e.get("is_semantic_event") is True
        and float(e.get("confidence", 0.0)) >= min_conf
        and e.get("event_type") not in ("uncertain", None)
    ]
    valid.sort(key=lambda x: x["_raw_frame"])

    groups = []
    for e in valid:
        cls = DUP_CLASS.get(e.get("event_type"), e.get("event_type"))
        arm = e.get("active_arm", "uncertain")
        obj = e.get("visible_object_category", "uncertain")
        placed = False

        if groups:
            g = groups[-1]
            dt = (e["_raw_frame"] - g[-1]["_raw_frame"]) / fps
            same_cls = DUP_CLASS.get(g[-1].get("event_type"), g[-1].get("event_type")) == cls
            same_arm = arm == g[-1].get("active_arm", "uncertain") or "uncertain" in (arm, g[-1].get("active_arm"))
            obj_compatible = (
                obj == g[-1].get("visible_object_category")
                or "uncertain" in (obj, g[-1].get("visible_object_category"))
                or "none" in (obj, g[-1].get("visible_object_category"))
            )
            if dt <= duplicate_sec and same_cls and same_arm and obj_compatible:
                g.append(e)
                placed = True
        if not placed:
            groups.append([e])

    atoms = []
    for aid, g in enumerate(groups):
        # representative balances VLM confidence and detector strength
        def quality(e):
            det = e.get("_detector_score")
            det = 0.0 if det is None else float(det)
            return float(e.get("confidence", 0.0)) + 0.08 * max(-2.0, min(4.0, det))
        rep = max(g, key=quality)
        atoms.append({
            "atomic_id": aid,
            "representative_event_id": rep["_event_id"],
            "member_event_ids": [x["_event_id"] for x in g],
            "frame": rep["_raw_frame"],
            "time_sec": rep["_time_sec"],
            "event_type": rep.get("event_type"),
            "atomic_action": rep.get("atomic_action"),
            "active_arm": rep.get("active_arm"),
            "visible_object_category": rep.get("visible_object_category"),
            "object_identity_confident": rep.get("object_identity_confident", False),
            "physical_transition": rep.get("physical_transition"),
            "task_relevance": rep.get("task_relevance"),
            "confidence": rep.get("confidence"),
        })
    return atoms


def compose_phases(client, model, task, atoms, max_tokens=2200):
    prompt = f"""你正在把已经去重的机器人 atomic events 组合成 semantic phases。

高层任务：{task}

注意：
1. 下面 atomic events 已经过代码去重，不要重新发明不存在的事件。
2. 只做“语义分组和命名”，禁止输出 start_frame/end_frame。
3. 不要跨事件追踪具体物体身份，除非输入明确说明 object_identity_confident=true。
4. task_relevance=unrelated/post_task 的事件不应被强行并入任务阶段。
5. 可以把 grasp -> transport/reposition -> release 组合为一个完整的操作阶段。
6. 同一任务中真实重复发生的 grasp/place 循环应保留。
7. 如果某个事件与高层任务冲突，保留 uncertainty，而不是脑补解释。
8. task_end_atomic_id 表示任务语义结束的最后一个 atomic event；若无法判断则为 null。

Atomic events:
{json.dumps(atoms, ensure_ascii=False, indent=2)}

严格输出：
{{
  "trajectory_instruction": "只基于这些atomic events生成按真实顺序的轨迹描述",
  "phase_groups": [
    {{
      "phase_id": 0,
      "instruction": "...",
      "atomic_ids": [0, 1],
      "confidence": 0.0
    }}
  ],
  "task_end_atomic_id": null,
  "uncertainties": ["..."]
}}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content
    return parse_json(raw), raw


def sanitize_hierarchy(h, atoms):
    valid_ids = {a["atomic_id"] for a in atoms}
    clean = []
    used = set()
    for p in h.get("phase_groups", []):
        ids = [int(i) for i in p.get("atomic_ids", []) if int(i) in valid_ids]
        ids = sorted(set(ids))
        if not ids:
            continue
        clean.append({
            "phase_id": len(clean),
            "instruction": p.get("instruction", ""),
            "atomic_ids": ids,
            "confidence": float(p.get("confidence", 0.0)),
        })
        used.update(ids)
    h["phase_groups"] = clean
    end_id = h.get("task_end_atomic_id")
    if end_id is not None and int(end_id) not in valid_ids:
        h["task_end_atomic_id"] = None
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--task", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.60)
    ap.add_argument("--duplicate-sec", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--aigc-user", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    root = Path(args.analysis_dir)
    events = load_jsonl(root / "event_semantics_v2.jsonl")
    atoms = deduplicate(
        events,
        min_conf=args.min_confidence,
        duplicate_sec=args.duplicate_sec,
        fps=args.fps,
    )
    (root / "atomic_events_v2.json").write_text(
        json.dumps({"atomic_events": atoms}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"verified events: {len(events)} -> deduplicated atomic events: {len(atoms)}")

    client = build_client(args)
    hierarchy, raw = compose_phases(client, args.model, args.task, atoms)
    hierarchy = sanitize_hierarchy(hierarchy, atoms)
    hierarchy["task_goal"] = args.task
    hierarchy["atomic_events"] = atoms

    (root / "semantic_hierarchy_v2.json").write_text(
        json.dumps(hierarchy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raw_dir = root / "vlm_raw_v2"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "phase_compose.txt").write_text(raw, encoding="utf-8")
    print("saved", root / "semantic_hierarchy_v2.json")


if __name__ == "__main__":
    main()
