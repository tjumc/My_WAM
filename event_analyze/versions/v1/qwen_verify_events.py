#!/usr/bin/env python3
"""Semantic verification for candidate robot events using an OpenAI-compatible VLM endpoint.

Input: analysis directory created by analyze_episode.py
Output:
  event_semantics.jsonl   one structured semantic decision per candidate event
  trajectory_semantics.json  optional second-pass merged hierarchy

Environment variables (can be overridden by CLI):
  AIGC_API_KEY
  AIGC_BASE_URL
  AIGC_USER
"""
import argparse, base64, json, os, re
from pathlib import Path
from openai import OpenAI

DEFAULT_BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"
DEFAULT_MODEL = "/model/qwen3.5-122b-a10b"


def data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def parse_json(text: str):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.S)
        if not m:
            raise ValueError(f"No JSON object found in response: {text[:500]}")
        return json.loads(m.group(0))


def build_client(args):
    headers = {}
    user = args.aigc_user or os.getenv('AIGC_USER')
    if user:
        headers['AIGC-USER'] = user
    return OpenAI(
        api_key=args.api_key or os.getenv('AIGC_API_KEY'),
        base_url=args.base_url or os.getenv('AIGC_BASE_URL', DEFAULT_BASE_URL),
        default_headers=headers or None,
    )


def verify_event(client, model, task, event, sheet_path, max_tokens=900):
    prompt = f"""你正在分析一条人形机器人家庭操作轨迹中的候选事件边界。

高层任务：{task}
候选事件：raw_frame={event['raw_frame']}, time={event['time_sec']:.3f}s
机器人时序检测器认为该点附近的主要变化信号：{', '.join(event.get('signals', []))}
右夹爪 command：{event.get('right_gripper_command')}
左夹爪 command：{event.get('left_gripper_command')}

图片是同一时间窗口的三视角同步 contact sheet：
- 行：head / left / right 三个相机
- 列：从事件前到事件后的连续时间点
- 中间列对应候选事件附近

任务：判断这个候选点是否对应一个有语义意义的机器人动作/状态变化。必须以图片中实际可观察内容为依据；高层任务只能作为上下文，不能据此脑补未观察到的动作。

请严格只输出一个 JSON 对象：
{{
  "is_semantic_event": true,
  "event_type": "navigation_start|navigation_stop|reach|grasp|release|open|close|pull|push|place|retract|reposition|other|uncertain",
  "atomic_action": "用一句中文描述事件附近机器人实际做了什么",
  "active_arm": "left|right|both|none|uncertain",
  "interacted_objects": ["..."],
  "state_before": "事件前的可观察状态",
  "state_after": "事件后的可观察状态",
  "object_state_change": "若有则描述，例如 dishwasher door: closed -> open；没有则写 none",
  "boundary_frame_relation": "before|near|after|uncertain",
  "confidence": 0.0,
  "evidence": ["最多3条简短视觉证据"],
  "notes": "若图片不足以判断，明确说明"
}}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            'role':'user',
            'content':[
                {'type':'text','text':prompt},
                {'type':'image_url','image_url':{'url':data_url(sheet_path)}}
            ]
        }],
        max_tokens=max_tokens,
        stream=False,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    raw = resp.choices[0].message.content
    out = parse_json(raw)
    out['_event_id'] = event['event_id']
    out['_raw_frame'] = event['raw_frame']
    out['_time_sec'] = event['time_sec']
    out['_detector_signals'] = event.get('signals', [])
    return out, raw


def merge_trajectory(client, model, task, events, max_tokens=2400):
    compact=[]
    for e in events:
        compact.append({k:e.get(k) for k in [
            '_event_id','_raw_frame','_time_sec','is_semantic_event','event_type','atomic_action',
            'active_arm','interacted_objects','state_before','state_after','object_state_change','confidence'
        ]})
    prompt = f"""你正在把一条长时程人形机器人轨迹的局部事件标注合并为层次化任务描述。
高层任务：{task}

下面是按时间顺序排列的候选事件标注：
{json.dumps(compact, ensure_ascii=False, indent=2)}

请完成：
1. 删除明显的伪事件或同一个动作内部重复的边界；
2. 将相邻 atomic events 合并为语义完整的 semantic phases；
3. 不要根据高层任务补出事件列表中没有观察证据支持的阶段；
4. start/end frame 应覆盖对应事件附近的实际阶段，可使用相邻事件中点作近似边界；
5. 保留可能重复发生的 grasp/place 循环，不要为了简化而删除实际重复操作。

严格输出 JSON：
{{
  "task_goal": "...",
  "trajectory_instruction": "按真实执行顺序生成一句较详细但简洁的轨迹级描述",
  "semantic_phases": [
    {{
      "phase_id": 0,
      "start_frame": 0,
      "end_frame": 0,
      "instruction": "...",
      "supporting_event_ids": [0],
      "confidence": 0.0
    }}
  ],
  "atomic_events": [
    {{"event_id": 0, "frame": 0, "instruction": "..."}}
  ],
  "uncertainties": ["..."]
}}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{'role':'user','content':prompt}],
        max_tokens=max_tokens,
        stream=False,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    raw=resp.choices[0].message.content
    return parse_json(raw), raw


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('analysis_dir')
    ap.add_argument('--task',required=True,help='coarse task instruction')
    ap.add_argument('--api-key',default=None)
    ap.add_argument('--base-url',default=None)
    ap.add_argument('--aigc-user',default=None)
    ap.add_argument('--model',default=DEFAULT_MODEL)
    ap.add_argument('--skip-merge',action='store_true')
    args=ap.parse_args()
    root=Path(args.analysis_dir)
    manifest=json.load(open(root/'candidate_events.json',encoding='utf-8'))
    client=build_client(args)
    results=[]
    raw_dir=root/'vlm_raw'; raw_dir.mkdir(exist_ok=True)
    out_jsonl=root/'event_semantics.jsonl'
    with open(out_jsonl,'w',encoding='utf-8') as fw:
        for e in manifest['candidate_events']:
            matches=sorted((root/'contact_sheets').glob(f"event_{e['event_id']:02d}_f*.jpg"))
            if not matches:
                raise FileNotFoundError(f"No contact sheet for event {e['event_id']}")
            obj,raw=verify_event(client,args.model,args.task,e,matches[0])
            fw.write(json.dumps(obj,ensure_ascii=False)+'\n'); fw.flush()
            (raw_dir/f"event_{e['event_id']:02d}.txt").write_text(raw,encoding='utf-8')
            results.append(obj)
            print(f"[{e['event_id']:02d}] {obj.get('is_semantic_event')} {obj.get('atomic_action')} conf={obj.get('confidence')}")
    if not args.skip_merge:
        merged,raw=merge_trajectory(client,args.model,args.task,results)
        (root/'trajectory_semantics.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2),encoding='utf-8')
        (raw_dir/'trajectory_merge.txt').write_text(raw,encoding='utf-8')
        print('saved',root/'trajectory_semantics.json')

if __name__=='__main__':
    main()
