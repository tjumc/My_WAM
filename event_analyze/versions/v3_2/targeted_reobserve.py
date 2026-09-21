#!/usr/bin/env python3
"""V3.2 targeted re-observation for poorly observed entities.

The generic V3 observer is kept unchanged. This pass only re-queries entities
whose trajectory is under-observed but whose related objects/hands indicate an
interaction. It never reads the manual regression fixture.
"""
import argparse
import json
from pathlib import Path

from observe_entities import (
    DEFAULT_MODEL,
    build_client,
    data_url,
    parse_json,
)


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def select_cutlery_events(obs, radius=4):
    triggers = []
    for i, o in enumerate(obs):
        vals = []
        for side in ("before", "after"):
            st = o.get(f"state_{side}") or {}
            vals.extend([
                st.get("right_hand"),
                st.get("left_hand"),
                st.get("knife_location"),
                st.get("fork_location"),
            ])
        if any(v in {"holding_knife", "holding_fork", "right_hand", "left_hand", "cutlery_basket"} for v in vals):
            triggers.append(i)

    if not triggers:
        # Fallback: focus on windows where the cutlery basket was not confidently observed.
        triggers = [
            i for i, o in enumerate(obs)
            if (
                (o.get("state_before") or {}).get("cutlery_basket") in {"uncertain", "not_visible"}
                or (o.get("state_after") or {}).get("cutlery_basket") in {"uncertain", "not_visible"}
            )
        ]

    selected = set()
    for i in triggers:
        for j in range(max(0, i - radius), min(len(obs), i + radius + 1)):
            selected.add(j)
    return [obs[i] for i in sorted(selected)]


def normalize_state(v):
    return v if v in {"in", "moving_out", "out", "moving_in", "uncertain", "not_visible"} else "uncertain"


def normalize_relation(v):
    valid = {
        "free", "approaching", "contact_cutlery_basket",
        "pulling_cutlery_basket", "pushing_cutlery_basket",
        "holding_utensil", "uncertain", "not_visible",
    }
    return v if v in valid else "uncertain"


def observe_cutlery(client, model, task, base_obs, sheet):
    prompt = f"""你只需要观察洗碗机里的 cutlery_basket（餐具篮/刀叉篮），不要判断碗篮 dish_rack，也不要生成动作标签。

场景背景：{task}
候选中心 frame={base_obs['_raw_frame']}。
三行图片严格依次是 head / left / right，相同时间序列，每行从左到右为早→晚。

关键定义：
- cutlery_basket：较小的刀叉餐具篮/托盘，用于 knife/fork。
- dish_rack：较大的盘碗拉篮。严禁把 dish_rack 当作 cutlery_basket。
- 如果餐具篮被碗篮、手臂或门遮挡，填 uncertain/not_visible，不要猜。

请独立观察窗口早期和晚期：
cutlery_basket_state: in|moving_out|out|moving_in|uncertain|not_visible
right_hand_relation: free|approaching|contact_cutlery_basket|pulling_cutlery_basket|pushing_cutlery_basket|holding_utensil|uncertain|not_visible

注意：
1. moving_out 表示餐具篮正在从洗碗机内部向外拉出；moving_in 表示正在推回。
2. 只有明确看见餐具篮本体相对洗碗机移动，才能给 moving_out/moving_in。
3. 不要因为右手在移动就假定餐具篮在移动。
4. confidence 只有在餐具篮身份和状态都清楚时才 >=0.85。

严格输出 JSON：
{{
  "cutlery_before": "uncertain",
  "cutlery_after": "uncertain",
  "confidence_before": 0.0,
  "confidence_after": 0.0,
  "right_hand_relation_before": "uncertain",
  "right_hand_relation_after": "uncertain",
  "evidence": ["最多4条"],
  "window_quality": "good|partial|poor"
}}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(sheet)}},
        ]}],
        max_tokens=900,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content
    obj = parse_json(raw)
    obj["cutlery_before"] = normalize_state(obj.get("cutlery_before"))
    obj["cutlery_after"] = normalize_state(obj.get("cutlery_after"))
    obj["right_hand_relation_before"] = normalize_relation(obj.get("right_hand_relation_before"))
    obj["right_hand_relation_after"] = normalize_relation(obj.get("right_hand_relation_after"))
    for k in ("confidence_before", "confidence_after"):
        try:
            obj[k] = max(0.0, min(1.0, float(obj.get(k, 0.0))))
        except Exception:
            obj[k] = 0.0
    obj["_event_id"] = int(base_obs["_event_id"])
    obj["_raw_frame"] = int(base_obs["_raw_frame"])
    return obj, raw


def merge(base, targeted, min_conf=0.65):
    by_id = {int(x["_event_id"]): x for x in targeted}
    out = []
    for src in base:
        o = json.loads(json.dumps(src))
        t = by_id.get(int(o["_event_id"]))
        if t:
            for side in ("before", "after"):
                state = t[f"cutlery_{side}"]
                conf = float(t[f"confidence_{side}"])
                if state not in {"uncertain", "not_visible"} and conf >= min_conf:
                    old_conf = float((o.get(f"confidence_{side}") or {}).get("cutlery_basket", 0.0))
                    # Targeted observation is entity-specific; use it when it is
                    # reasonably confident, even if the generic observer was overconfident.
                    if conf >= min_conf or conf >= old_conf:
                        o[f"state_{side}"]["cutlery_basket"] = state
                        o[f"confidence_{side}"]["cutlery_basket"] = conf

                relation = t.get(f"right_hand_relation_{side}", "uncertain")
                if relation in {
                    "contact_cutlery_basket", "pulling_cutlery_basket", "pushing_cutlery_basket"
                } and conf >= min_conf:
                    current = o[f"state_{side}"].get("right_hand", "uncertain")
                    if current not in {"holding_knife", "holding_fork", "holding_plate"}:
                        o[f"state_{side}"]["right_hand"] = "contact_cutlery_basket"
                        o[f"confidence_{side}"]["right_hand"] = max(
                            float(o[f"confidence_{side}"].get("right_hand", 0.0)), conf
                        )

            o["_v32_targeted_cutlery"] = t
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--aigc-user", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--neighbor-radius", type=int, default=4)
    ap.add_argument("--min-merge-confidence", type=float, default=0.65)
    args = ap.parse_args()

    root = Path(args.episode_dir)
    out = Path(args.output_dir)
    base_path = out / "entity_observations_base.jsonl"
    if not base_path.exists():
        base_path = out / "entity_observations.jsonl"
    base = load_jsonl(base_path)
    selected = select_cutlery_events(base, args.neighbor_radius)
    print("targeted cutlery windows:", [x["_event_id"] for x in selected])

    client = build_client(args)
    raw_dir = out / "vlm_raw_targeted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    targeted = []
    for o in selected:
        sheets = sorted((root / "contact_sheets").glob(f"event_{int(o['_event_id']):02d}_f*.jpg"))
        if not sheets:
            raise FileNotFoundError(f"missing contact sheet for event {o['_event_id']}")
        t, raw = observe_cutlery(client, args.model, args.task, o, sheets[0])
        targeted.append(t)
        (raw_dir / f"cutlery_{int(o['_event_id']):02d}.txt").write_text(raw, encoding="utf-8")
        print(
            f"[{int(o['_event_id']):02d}] "
            f"{t['cutlery_before']}->{t['cutlery_after']} "
            f"conf=({t['confidence_before']:.2f},{t['confidence_after']:.2f}) "
            f"hand={t['right_hand_relation_before']}->{t['right_hand_relation_after']}"
        )

    (out / "targeted_cutlery_observations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in targeted) + ("\n" if targeted else ""),
        encoding="utf-8",
    )
    augmented = merge(base, targeted, args.min_merge_confidence)
    (out / "entity_observations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in augmented) + "\n",
        encoding="utf-8",
    )
    print("saved", out / "entity_observations.jsonl")


if __name__ == "__main__":
    main()
