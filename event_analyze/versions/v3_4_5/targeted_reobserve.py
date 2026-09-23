#!/usr/bin/env python3
"""Schema-driven targeted visual re-observation for V3.4.5.

The reasoning stage decides WHERE and WHAT is ambiguous. This script only asks
the VLM to observe the requested entity states/hand relations. It does not name
actions or phases and never sees manual GT.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

V333_DIR = Path(__file__).resolve().parents[1] / "v3_3_3"
if str(V333_DIR) not in sys.path:
    sys.path.insert(0, str(V333_DIR))

from observe_entities import DEFAULT_MODEL, build_client, parse_json


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def data_url(path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def decode_frame(f, camera, index, cache):
    if camera not in cache:
        sizes = f[f"images_dict/{camera}/rgb_size"][:].astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
        cache[camera] = offsets
    offsets = cache[camera]
    i = int(max(0, min(index, len(offsets) - 2)))
    packed = f[f"images_dict/{camera}/rgb"]
    raw = np.asarray(packed[int(offsets[i]):int(offsets[i + 1])], dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to decode camera={camera} frame={i}")
    return img


def fit(img, width=300, height=168):
    h, w = img.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = x
    return canvas


def available_cameras(f):
    out = []
    for cam in ("head", "left", "right"):
        if f"images_dict/{cam}/rgb_size" in f and f"images_dict/{cam}/rgb" in f:
            out.append(cam)
    if not out:
        raise RuntimeError("no supported cameras found in HDF5")
    return out


def sample_frames(start, end, count):
    if end <= start:
        return [int(start)]
    return np.linspace(int(start), int(end), max(2, int(count))).round().astype(int).tolist()


def build_strip(hdf5_path, frames, out_path):
    rows = []
    with h5py.File(hdf5_path, "r") as f:
        cameras = available_cameras(f)
        cache = {}
        for camera in cameras:
            cells = []
            for j, frame in enumerate(frames):
                img = fit(decode_frame(f, camera, frame, cache))
                cv2.putText(
                    img,
                    f"{camera} t{j} f{int(frame)}",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cells.append(img)
            rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise RuntimeError(f"failed to write {out_path}")
    return cameras


def observation_keys(schema):
    out = {}
    for name, cfg in schema.get("entities", {}).items():
        out[name] = cfg.get("observation_key", name)
    return out


def holding_tokens(name, cfg):
    return list(cfg.get("holding_tokens", [f"holding_{name}"]))


def target_spec_text(schema, target_entities):
    lines = []
    hand_tokens = {"free", "approaching", "uncertain", "not_visible"}
    for name in target_entities:
        cfg = schema.get("entities", {}).get(name)
        if not cfg:
            continue
        key = cfg.get("observation_key", name)
        states = list(cfg.get("states", [])) + ["uncertain", "not_visible"]
        lines.append(
            f"- {name} ({cfg.get('display_name', name)}), key={key}, "
            f"type={cfg.get('type')}, states={'|'.join(states)}"
        )
        hand_tokens.update(cfg.get("contact_tokens", []))
        if cfg.get("type") == "portable_object":
            hand_tokens.update(holding_tokens(name, cfg))
    return "\n".join(lines), sorted(hand_tokens)


def sanitize(schema, target_entities, obj):
    keys = observation_keys(schema)
    state_before = {k: "uncertain" for k in keys.values()}
    state_after = {k: "uncertain" for k in keys.values()}
    state_before.update({"right_hand": "uncertain", "left_hand": "uncertain"})
    state_after.update({"right_hand": "uncertain", "left_hand": "uncertain"})

    conf_before = {k: 0.0 for k in state_before}
    conf_after = {k: 0.0 for k in state_after}

    _, hand_valid = target_spec_text(schema, target_entities)
    hand_valid = set(hand_valid)

    raw_b = obj.get("state_before") or {}
    raw_a = obj.get("state_after") or {}
    raw_cb = obj.get("confidence_before") or {}
    raw_ca = obj.get("confidence_after") or {}

    for name in target_entities:
        cfg = schema.get("entities", {}).get(name)
        if not cfg:
            continue
        key = cfg.get("observation_key", name)
        valid = set(cfg.get("states", [])) | {"uncertain", "not_visible"}
        vb = raw_b.get(key, "uncertain")
        va = raw_a.get(key, "uncertain")
        state_before[key] = vb if vb in valid else "uncertain"
        state_after[key] = va if va in valid else "uncertain"
        try:
            conf_before[key] = max(0.0, min(1.0, float(raw_cb.get(key, 0.0))))
            conf_after[key] = max(0.0, min(1.0, float(raw_ca.get(key, 0.0))))
        except Exception:
            pass

    for hand in ("right_hand", "left_hand"):
        vb = raw_b.get(hand, "uncertain")
        va = raw_a.get(hand, "uncertain")
        if vb == "none":
            vb = "free"
        if va == "none":
            va = "free"
        state_before[hand] = vb if vb in hand_valid else "uncertain"
        state_after[hand] = va if va in hand_valid else "uncertain"
        try:
            conf_before[hand] = max(0.0, min(1.0, float(raw_cb.get(hand, 0.0))))
            conf_after[hand] = max(0.0, min(1.0, float(raw_ca.get(hand, 0.0))))
        except Exception:
            pass

    obj["state_before"] = state_before
    obj["state_after"] = state_after
    obj["confidence_before"] = conf_before
    obj["confidence_after"] = conf_after
    return obj


def query(client, model, task, schema, req, frames, sheet, cameras):
    target_text, hand_tokens = target_spec_text(schema, req["target_entities"])
    keys = observation_keys(schema)
    target_keys = [
        keys[x] for x in req["target_entities"]
        if x in keys
    ]

    state_template = {k: "uncertain" for k in target_keys}
    state_template.update({"right_hand": "uncertain", "left_hand": "uncertain"})
    conf_template = {k: 0.0 for k in state_template}

    prompt = f"""你正在为机器人长时序轨迹做“定点复查”。这里只做可观察实体状态判断，不要命名动作，不要生成 phase，也不要根据任务逻辑猜测缺失状态。

高层任务仅作为场景背景：{task}
raw frame 范围：{req['start_frame']}~{req['end_frame']}
采样帧：{frames}
图片行依次对应相机：{cameras}；每一行从左到右严格为早→晚。

本次只复查以下实体：
{target_text}

手部关系只允许：
{'|'.join(hand_tokens)}

要求：
1. state_before 描述窗口早期，state_after 描述窗口晚期。
2. 必须依据图像中的相对位置、连续运动、抓持或接触关系，不要根据“任务应该如何结束”倒推。
3. 若是可动容器，moving_out/moving_in 只有在至少两个时刻支持连续方向变化时才能使用。
4. 若是 portable object，只有明确看到物体在手中时才输出 right_hand/left_hand。
5. 若手正在接触目标实体但方向不清楚，可以输出对应 contact token；不要猜 pulling/pushing。
6. 遮挡或身份不清时使用 uncertain/not_visible。
7. confidence>=0.85 只用于非常清晰的直接视觉证据。
8. 非本次 target 的实体不要额外判断。

严格输出 JSON：
{{
  "state_before": {json.dumps(state_template, ensure_ascii=False)},
  "state_after": {json.dumps(state_template, ensure_ascii=False)},
  "confidence_before": {json.dumps(conf_template, ensure_ascii=False)},
  "confidence_after": {json.dumps(conf_template, ensure_ascii=False)},
  "evidence_time_indices": [0, 1],
  "evidence": ["最多4条纯视觉证据"],
  "window_quality": "good|partial|poor"
}}"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(sheet)}},
        ]}],
        max_tokens=1400,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content
    obj = sanitize(schema, req["target_entities"], parse_json(raw))
    obj.update({
        "_event_id": 20000 + int(req["request_id"]),
        "_raw_frame": int(round((req["start_frame"] + req["end_frame"]) / 2)),
        "_time_sec": None,
        "_window_start_frame": int(req["start_frame"]),
        "_window_end_frame": int(req["end_frame"]),
        "_v344_targeted": True,
        "_target_entities": list(req["target_entities"]),
        "_ambiguity_kind": req["kind"],
        "_ambiguity_priority": int(req["priority"]),
        "_sample_frames": [int(x) for x in frames],
    })
    return obj, raw


def mergeable(obj, schema, min_conf):
    for name in obj.get("_target_entities", []):
        cfg = schema.get("entities", {}).get(name, {})
        key = cfg.get("observation_key", name)
        cb = float((obj.get("confidence_before") or {}).get(key, 0.0))
        ca = float((obj.get("confidence_after") or {}).get(key, 0.0))
        vb = (obj.get("state_before") or {}).get(key)
        va = (obj.get("state_after") or {}).get(key)
        if max(cb, ca) >= min_conf and (
            vb not in {"uncertain", "not_visible", None}
            or va not in {"uncertain", "not_visible", None}
        ):
            return True
    # Entity-specific contact may itself be useful even if state is uncertain.
    for hand in ("right_hand", "left_hand"):
        c = max(
            float((obj.get("confidence_before") or {}).get(hand, 0.0)),
            float((obj.get("confidence_after") or {}).get(hand, 0.0)),
        )
        values = {
            (obj.get("state_before") or {}).get(hand),
            (obj.get("state_after") or {}).get(hand),
        }
        if c >= min_conf and any(
            isinstance(v, str) and v.startswith(
                ("contact_", "holding_", "pulling_", "pushing_", "opening_", "closing_")
            )
            for v in values
        ):
            return True
    return False


def write_jsonl(path, rows):
    Path(path).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--aigc-user", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    schema = load_json(args.schema)
    policy = schema.get("targeted_perception", {})
    min_conf = float(policy.get("min_confidence_to_merge", 0.65))
    num_samples = int(policy.get("num_samples", 8))
    req_doc = load_json(out / "ambiguity_requests.json")
    requests = req_doc.get("requests", [])

    base_path = out / "entity_observations_base.jsonl"
    base = load_jsonl(base_path)
    target_path = out / "targeted_observations.jsonl"

    if args.reuse and target_path.exists():
        targeted = load_jsonl(target_path)
        merged = base + [x for x in targeted if x.get("_merge_accepted")]
        write_jsonl(out / "entity_observations.jsonl", merged)
        print(f"reused targeted observations: {len(targeted)}, merged={len(merged)-len(base)}")
        return

    if not requests:
        write_jsonl(target_path, [])
        write_jsonl(out / "entity_observations.jsonl", base)
        print("no ambiguity requests; no VLM calls")
        return

    client = build_client(args)
    strip_dir = out / "targeted_temporal_strips"
    raw_dir = out / "vlm_raw_targeted"
    strip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    targeted = []
    accepted = []
    for req in requests:
        frames = sample_frames(req["start_frame"], req["end_frame"], num_samples)
        sheet = strip_dir / (
            f"targeted_{req['request_id']:02d}_{req['start_frame']}_{req['end_frame']}.jpg"
        )
        cameras = build_strip(args.hdf5, frames, sheet)
        obj, raw = query(
            client, args.model, args.task, schema, req, frames, sheet, cameras
        )
        obj["_merge_accepted"] = bool(mergeable(obj, schema, min_conf))
        targeted.append(obj)
        if obj["_merge_accepted"]:
            accepted.append(obj)
        (raw_dir / f"targeted_{req['request_id']:02d}.txt").write_text(
            raw, encoding="utf-8"
        )

        states = {
            name: [
                obj["state_before"].get(schema["entities"][name].get("observation_key", name)),
                obj["state_after"].get(schema["entities"][name].get("observation_key", name)),
            ]
            for name in req["target_entities"]
            if name in schema.get("entities", {})
        }
        print(
            f"[targeted {req['request_id']:02d}] {req['kind']} "
            f"{req['target_entities']} states={states} merge={obj['_merge_accepted']}"
        )

    write_jsonl(target_path, targeted)
    write_jsonl(out / "entity_observations.jsonl", base + accepted)
    print(f"targeted observations: {len(targeted)}, merged: {len(accepted)}")
    print("saved", target_path)


if __name__ == "__main__":
    main()
