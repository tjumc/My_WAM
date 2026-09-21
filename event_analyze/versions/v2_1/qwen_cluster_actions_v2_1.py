#!/usr/bin/env python3
"""V2.1 temporal action clustering + semantic phase composition.

Multiple candidate windows that describe the same ongoing action are merged into an
action interval. Qwen then groups action intervals into semantic phases, but it never
chooses numerical frame boundaries.
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"
DEFAULT_MODEL = "/model/qwen3.5-122b-a10b"

FAMILY = {
    "navigation": "navigation",
    "reach": "reach",
    "grasp": "grasp",
    "release": "place",
    "place": "place",
    "transport": "transport",
    "reposition": "transport",
    "open": "open",
    "close": "close",
    "pull": "pull",
    "push": "push",
    "retract": "retract",
    "contact": "contact",
    "idle": "idle",
    "other": "other",
    "uncertain": "uncertain",
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
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def family_set(e):
    tags = e.get("action_tags") or []
    vals = [e.get("dominant_action")] + list(tags)
    out = {FAMILY.get(v, v) for v in vals if v}
    out.discard("uncertain")
    out.discard("idle")
    return out


def arm_compatible(a, b):
    return a == b or "uncertain" in (a, b) or "both" in (a, b)


def object_compatible(a, b):
    return a == b or a in (None, "uncertain", "none", "other") or b in (None, "uncertain", "none", "other")


def weighted_mode(values, weights, default="uncertain"):
    score = {}
    for v, w in zip(values, weights):
        if v is None:
            continue
        score[v] = score.get(v, 0.0) + float(w)
    if not score:
        return default
    return max(score.items(), key=lambda kv: kv[1])[0]


def should_merge(cluster, e, max_gap_frames):
    last = cluster[-1]
    # Semantic family overlap is the primary condition.
    fam_overlap = bool(family_set(last) & family_set(e))
    if not fam_overlap:
        return False

    # Candidate windows may overlap even when centers are not close.
    temporal_gap = int(e["_window_start_frame"]) - int(last["_window_end_frame"])
    if temporal_gap > max_gap_frames:
        return False

    if not arm_compatible(last.get("active_arm", "uncertain"), e.get("active_arm", "uncertain")):
        return False
    if not object_compatible(last.get("visible_object_category"), e.get("visible_object_category")):
        return False

    # start/ongoing/end are all compatible inside one action instance.
    return True


def cluster_actions(events, min_conf=0.55, max_gap_sec=1.0, fps=30.0):
    eligible = [
        e for e in events
        if e.get("contains_semantic_action") is True
        and float(e.get("confidence", 0.0)) >= min_conf
        and e.get("dominant_action") not in (None, "idle", "uncertain")
    ]
    eligible.sort(key=lambda x: x["_raw_frame"])
    max_gap_frames = int(round(max_gap_sec * fps))

    groups = []
    for e in eligible:
        if groups and should_merge(groups[-1], e, max_gap_frames):
            groups[-1].append(e)
        else:
            groups.append([e])

    actions = []
    for aid, g in enumerate(groups):
        weights = [max(0.05, float(x.get("confidence", 0.0))) for x in g]
        dominant = weighted_mode([FAMILY.get(x.get("dominant_action"), x.get("dominant_action")) for x in g], weights)
        relevance = weighted_mode([x.get("task_relevance") for x in g], weights)
        arm = weighted_mode([x.get("active_arm") for x in g], weights)
        obj = weighted_mode([x.get("visible_object_category") for x in g], weights)

        first_center = min(int(x["_raw_frame"]) for x in g)
        last_center = max(int(x["_raw_frame"]) for x in g)
        rough_start = min(int(x["_window_start_frame"]) for x in g)
        rough_end = max(int(x["_window_end_frame"]) for x in g)
        rep = max(g, key=lambda x: float(x.get("confidence", 0.0)))

        tag_counter = Counter()
        for x, w in zip(g, weights):
            for t in family_set(x):
                tag_counter[t] += w
        tags = [k for k, _ in tag_counter.most_common(3)]

        actions.append({
            "action_id": aid,
            "action_type": dominant,
            "action_tags": tags,
            "instruction": rep.get("atomic_action", ""),
            "active_arm": arm,
            "visible_object_category": obj,
            "task_relevance": relevance,
            "confidence": round(sum(weights) / len(weights), 3),
            "member_event_ids": [int(x["_event_id"]) for x in g],
            "member_center_frames": [int(x["_raw_frame"]) for x in g],
            "progress_sequence": [x.get("action_progress") for x in g],
            "rough_start_frame": rough_start,
            "rough_end_frame": rough_end,
            "first_evidence_frame": first_center,
            "last_evidence_frame": last_center,
        })
    return eligible, actions


def compose_phases(client, model, task, actions, max_tokens=2400):
    prompt = f"""你正在把一条长时程机器人轨迹中的 action intervals 组合成更高层 semantic phases。

高层任务：{task}

规则：
1. action intervals 已由代码根据多视角窗口语义和时间重叠聚类得到，不要重新发明不存在的动作。
2. 你只负责“语义分组和命名”，禁止输出任何 start_frame/end_frame。
3. phase_groups 里只填写 action_id。
4. task_relevance=post_task/unrelated 的 action 不应被并入当前任务 phase。
5. 可以把 reach/grasp/transport/place 等连续 action 组合成一个语义完整的阶段。
6. 多次真实重复的取放循环要保留，不要强行合并成一次。
7. 不要跨 action 推断具体物体身份，除非输入证据非常明确。
8. task_end_action_id 表示完成当前高层任务的最后一个 relevant action；若无法判断则为 null。
9. 如果某些 action 互相矛盾，放进 uncertainties，不要自行修补轨迹。

Action intervals:
{json.dumps(actions, ensure_ascii=False, indent=2)}

严格输出：
{{
  "trajectory_instruction": "按真实动作顺序生成一句较详细但简洁的轨迹描述",
  "phase_groups": [
    {{
      "phase_id": 0,
      "instruction": "...",
      "action_ids": [0, 1],
      "confidence": 0.0
    }}
  ],
  "task_end_action_id": null,
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


def sanitize_hierarchy(h, actions):
    valid_ids = {a["action_id"] for a in actions}
    clean = []
    for p in h.get("phase_groups", []):
        ids = sorted(set(int(i) for i in p.get("action_ids", []) if int(i) in valid_ids))
        if not ids:
            continue
        clean.append({
            "phase_id": len(clean),
            "instruction": p.get("instruction", ""),
            "action_ids": ids,
            "confidence": float(p.get("confidence", 0.0)),
        })
    h["phase_groups"] = clean
    end_id = h.get("task_end_action_id")
    if end_id is not None and int(end_id) not in valid_ids:
        h["task_end_action_id"] = None
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--task", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--max-gap-sec", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--api-key", default="msk-57d214329a4b63bebb0617ce7dadbddb4d00c382f61be32a015d73c4e6c8d121")
    ap.add_argument("--base-url", default="https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1")
    ap.add_argument("--aigc-user", default="suty11")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    
    root = Path(args.analysis_dir)
    events = load_jsonl(root / "window_semantics_v2_1.jsonl")
    eligible, actions = cluster_actions(
        events,
        min_conf=args.min_confidence,
        max_gap_sec=args.max_gap_sec,
        fps=args.fps,
    )

    (root / "action_intervals_v2_1.json").write_text(
        json.dumps({
            "num_candidate_windows": len(events),
            "num_semantic_windows": len(eligible),
            "num_action_intervals": len(actions),
            "action_intervals": actions,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"candidate windows: {len(events)} -> semantic windows: {len(eligible)} -> action intervals: {len(actions)}")

    client = build_client(args)
    hierarchy, raw = compose_phases(client, args.model, args.task, actions)
    hierarchy = sanitize_hierarchy(hierarchy, actions)
    hierarchy["task_goal"] = args.task
    hierarchy["action_intervals"] = actions

    (root / "semantic_hierarchy_v2_1.json").write_text(
        json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_dir = root / "vlm_raw_v2_1"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "phase_compose.txt").write_text(raw, encoding="utf-8")
    print("saved", root / "semantic_hierarchy_v2_1.json")


if __name__ == "__main__":
    main()
