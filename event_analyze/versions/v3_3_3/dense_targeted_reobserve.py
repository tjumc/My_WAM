#!/usr/bin/env python3
"""V3.3.3 dense targeted temporal observation for small/occluded articulated entities.

For the current dishwasher ontology this pass targets cutlery_basket. It uses
dense head+right temporal strips decoded directly from the raw HDF5, and asks
for RELATIVE MOTION with respect to the large dish rack rather than absolute
single-frame state classification.

The manual regression fixture is never read.
"""
import argparse
import base64
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np

from observe_entities import DEFAULT_MODEL, build_client, parse_json


STATE_KEYS = [
    "robot_location", "door", "dish_rack", "cutlery_basket",
    "right_hand", "left_hand",
    "knife_location", "fork_location", "plate_location",
]


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _data_url(path):
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _decode_frame(f, camera, index, cache):
    key = camera
    if key not in cache:
        sizes = f[f"images_dict/{camera}/rgb_size"][:].astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
        cache[key] = (sizes, offsets)
    _, offsets = cache[key]
    i = int(max(0, min(index, len(offsets) - 2)))
    packed = f[f"images_dict/{camera}/rgb"]
    raw = np.asarray(packed[int(offsets[i]):int(offsets[i + 1])], dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to decode {camera} frame {i}")
    return img


def _fit(img, width=320, height=180):
    h, w = img.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = x
    return canvas


def build_dense_strip(hdf5_path, frames, out_path):
    rows = []
    with h5py.File(hdf5_path, "r") as f:
        cache = {}
        for camera in ("head", "right"):
            cells = []
            for j, frame in enumerate(frames):
                img = _fit(_decode_frame(f, camera, int(frame), cache))
                cv2.putText(
                    img, f"t{j} f{int(frame)}", (7, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
                cells.append(img)
            rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise RuntimeError(f"failed to write {out_path}")
    return out_path


def hand_events(obs, labels):
    out = []
    for o in obs:
        values = []
        for side in ("before", "after"):
            st = o.get(f"state_{side}") or {}
            values.extend([st.get("right_hand"), st.get("left_hand")])
        if any(v in labels for v in values):
            out.append(int(o["_raw_frame"]))
    return sorted(out)


def sustained_plate_onset(obs, min_hits=2, horizon_frames=360):
    frames = hand_events(obs, {"holding_plate"})
    for i, f in enumerate(frames):
        hits = sum(1 for g in frames[i:] if f <= g <= f + horizon_frames)
        if hits >= min_hits:
            return f
    return frames[0] if frames else None


def search_intervals(obs, total_frames, fps):
    utensil = hand_events(obs, {"holding_knife", "holding_fork"})
    if not utensil:
        return [{"name": "under_observed_cutlery", "start": 0, "end": total_frames - 1}]

    first_u, last_u = utensil[0], utensil[-1]
    plate_onset = sustained_plate_onset(obs)

    pre_start = max(0, first_u - int(round(14.0 * fps)))
    pre_end = max(pre_start + 1, first_u - int(round(0.5 * fps)))

    post_start = min(total_frames - 1, last_u + int(round(0.25 * fps)))
    if plate_onset is not None and plate_onset > last_u + int(round(1.0 * fps)):
        post_end = min(total_frames - 1, plate_onset + int(round(1.5 * fps)))
    else:
        post_end = min(total_frames - 1, last_u + int(round(10.0 * fps)))

    intervals = []
    if pre_end > pre_start:
        intervals.append({"name": "before_utensil_manipulation", "start": pre_start, "end": pre_end})
    if post_end > post_start:
        intervals.append({"name": "after_utensil_manipulation", "start": post_start, "end": post_end})
    return intervals


def sliding_windows(intervals, fps, window_sec=4.5, stride_sec=2.0):
    width = max(2, int(round(window_sec * fps)))
    stride = max(1, int(round(stride_sec * fps)))
    windows = []
    wid = 0
    for interval in intervals:
        s, e = int(interval["start"]), int(interval["end"])
        if e - s + 1 <= width:
            windows.append({"window_id": wid, "region": interval["name"], "start": s, "end": e})
            wid += 1
            continue
        cur = s
        while cur <= e:
            end = min(e, cur + width - 1)
            if end - cur >= int(round(1.5 * fps)):
                windows.append({"window_id": wid, "region": interval["name"], "start": cur, "end": end})
                wid += 1
            if end >= e:
                break
            cur += stride
    return windows


def sample_frames(start, end, count):
    if end <= start:
        return [int(start)]
    return np.linspace(int(start), int(end), int(count)).round().astype(int).tolist()


def normalize_state(v):
    return v if v in {"in", "moving_out", "out", "moving_in", "uncertain", "not_visible"} else "uncertain"


def normalize_motion(v):
    return v if v in {"moving_out", "moving_in", "stationary", "uncertain"} else "uncertain"


def normalize_hand(v):
    valid = {
        "none", "approaching", "contact_cutlery_basket", "pulling_cutlery_basket",
        "pushing_cutlery_basket", "holding_utensil", "uncertain", "not_visible",
    }
    return v if v in valid else "uncertain"


def query_dense(client, model, task, window, frames, sheet):
    prompt = f"""你正在观察一个小型可动实体：洗碗机内的 cutlery_basket（餐具篮/刀叉篮）。

高层任务仅作场景背景：{task}
搜索区域：{window['region']}
raw frame 范围：{window['start']}~{window['end']}

图片：
- 第一行是 head camera；
- 第二行是 right camera；
- 两行严格时间同步；
- 每行 t0→t{len(frames)-1} 从早到晚；
- 帧序列：{frames}

你必须做的是“相对运动判断”，不是单帧物体分类。

参照物：
- dish_rack：较大的盘碗拉篮；
- cutlery_basket：放刀叉的较小餐具篮/托盘。
判断 cutlery_basket 相对于 dish_rack / 洗碗机主体是否发生连续位移。

重点：
1. 若小餐具篮从内部位置连续向机器人/洗碗机外方向移动：motion=moving_out。
2. 若连续向洗碗机内部方向移动：motion=moving_in。
3. 若能清晰辨认且相对位置基本不变：stationary。
4. 若实体身份、参照关系或运动方向看不清：uncertain；不要猜。
5. dish_rack 自己移动不能被误判成 cutlery_basket 相对 dish_rack 移动。
6. 运动方向必须至少由两个不同时间点的相对位置支持。
7. right_hand_relation 只描述右手与 cutlery_basket 的关系，不要因为手在附近就自动判定接触。
8. confidence>=0.85 只用于实体身份和方向都非常清楚的情况。

严格输出 JSON：
{{
  "start_state": "in|out|uncertain|not_visible",
  "end_state": "in|out|uncertain|not_visible",
  "relative_motion": "moving_out|moving_in|stationary|uncertain",
  "right_hand_relation": "none|approaching|contact_cutlery_basket|pulling_cutlery_basket|pushing_cutlery_basket|holding_utensil|uncertain|not_visible",
  "confidence": 0.0,
  "evidence_time_indices": [0,1],
  "evidence": ["最多4条，说明相对位置如何变化"],
  "visibility": "good|partial|poor"
}}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_url(sheet)}},
        ]}],
        max_tokens=1100,
        stream=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content
    obj = parse_json(raw)
    obj["start_state"] = normalize_state(obj.get("start_state"))
    obj["end_state"] = normalize_state(obj.get("end_state"))
    obj["relative_motion"] = normalize_motion(obj.get("relative_motion"))
    obj["right_hand_relation"] = normalize_hand(obj.get("right_hand_relation"))
    try:
        obj["confidence"] = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except Exception:
        obj["confidence"] = 0.0
    obj.update({
        "_dense_window_id": int(window["window_id"]),
        "_region": window["region"],
        "_window_start_frame": int(window["start"]),
        "_window_end_frame": int(window["end"]),
        "_sample_frames": [int(x) for x in frames],
        "_raw_frame": int(round((window["start"] + window["end"]) / 2)),
    })
    return obj, raw


def transition_evidence(x, synthetic_id):
    """Keep dense VLM output as directional motion evidence, not absolute state."""
    return {
        "entity": "cutlery_basket",
        "relative_motion": x.get("relative_motion", "uncertain"),
        "confidence": max(0.0, min(1.0, float(x.get("confidence", 0.0)))),
        "right_hand_relation": x.get("right_hand_relation", "uncertain"),
        "center_frame": int(x["_raw_frame"]),
        "window_start_frame": int(x["_window_start_frame"]),
        "window_end_frame": int(x["_window_end_frame"]),
        "dense_window_id": int(x["_dense_window_id"]),
        "event_id": int(synthetic_id),
        "region": x.get("_region"),
        "evidence": x.get("evidence", []),
        "source": "dense_relative_motion_v3_3_2",
    }

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
    ap.add_argument("--window-sec", type=float, default=4.5)
    ap.add_argument("--stride-sec", type=float, default=2.0)
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--min-merge-confidence", type=float, default=0.62)
    args = ap.parse_args()

    root = Path(args.episode_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_path = out / "entity_observations_base.jsonl"
    if not base_path.exists():
        base_path = out / "entity_observations.jsonl"
    base = load_jsonl(base_path)

    with h5py.File(args.hdf5, "r") as f:
        if "time" in f:
            total_frames = len(f["time"])
        else:
            total_frames = len(f["images_dict/head/rgb_size"])

    intervals = search_intervals(base, total_frames, args.fps)
    windows = sliding_windows(intervals, args.fps, args.window_sec, args.stride_sec)
    print("dense search intervals:", intervals)
    print("dense windows:", [(w["window_id"], w["region"], w["start"], w["end"]) for w in windows])

    client = build_client(args)
    sheet_dir = out / "dense_temporal_strips"
    raw_dir = out / "vlm_raw_dense"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dense = []
    transitions = []
    for w in windows:
        frames = sample_frames(w["start"], w["end"], args.num_samples)
        sheet = sheet_dir / f"dense_{w['window_id']:02d}_{w['start']}_{w['end']}.jpg"
        build_dense_strip(args.hdf5, frames, sheet)
        obj, raw = query_dense(client, args.model, args.task, w, frames, sheet)
        dense.append(obj)
        (raw_dir / f"dense_{w['window_id']:02d}.txt").write_text(raw, encoding="utf-8")

        print(
            f"[dense {w['window_id']:02d}] {w['region']} "
            f"{obj['start_state']}->{obj['end_state']} "
            f"motion={obj['relative_motion']} hand={obj['right_hand_relation']} "
            f"conf={obj['confidence']:.2f}"
        )

        directional = obj["relative_motion"] in {"moving_out", "moving_in"}
        # Keep only direction-level evidence; do not fabricate absolute in/out anchors.
        if obj["confidence"] >= args.min_merge_confidence and directional:
            transitions.append(transition_evidence(obj, 10000 + int(w["window_id"])))

    (out / "dense_cutlery_observations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in dense) + ("\n" if dense else ""),
        encoding="utf-8",
    )
    (out / "dense_cutlery_transition_evidence.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in transitions) + ("\n" if transitions else ""),
        encoding="utf-8",
    )

    # Generic observations remain unchanged; dense evidence is consumed
    # separately by the tracker/validator.
    (out / "entity_observations.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in base) + "\n",
        encoding="utf-8",
    )
    print(f"dense directional evidence: {len(transitions)}/{len(dense)}")
    print("saved", out / "dense_cutlery_transition_evidence.jsonl")


if __name__ == "__main__":
    main()
