#!/usr/bin/env python3
"""V2.2 trajectory-consistent action clustering and semantic phase composition."""
import argparse, json, os, re
from collections import Counter
from pathlib import Path
from openai import OpenAI

DEFAULT_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"
DEFAULT_MODEL="/model/qwen3.5-122b-a10b"
FAMILY={"navigation":"navigation","reach":"reach","grasp":"grasp","release":"place","place":"place","transport":"transport","reposition":"transport","open":"open","close":"close","pull":"pull","push":"push","retract":"retract","contact":"contact","idle":"idle","other":"other","uncertain":"uncertain"}

def parse_json(text):
    text=text.strip()
    if text.startswith("```"):
        text=re.sub(r"^```(?:json)?\s*","",text,flags=re.I); text=re.sub(r"\s*```$","",text)
    try:return json.loads(text)
    except Exception:
        m=re.search(r"\{.*\}",text,re.S)
        if not m: raise
        return json.loads(m.group(0))

def build_client(args):
    key=args.api_key or os.getenv("AIGC_API_KEY")
    if not key: raise RuntimeError("AIGC_API_KEY is required")
    headers={}; user=args.aigc_user or os.getenv("AIGC_USER")
    if user: headers["AIGC-USER"]=user
    return OpenAI(api_key=key,base_url=args.base_url or os.getenv("AIGC_BASE_URL",DEFAULT_BASE_URL),default_headers=headers or None)

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def action_of(e): return e.get("normalized_action") or e.get("proposed_action") or "uncertain"
def family(e): return FAMILY.get(action_of(e),action_of(e))
def obj_of(e):
    for state in (e.get("state_after") or {},e.get("state_before") or {}):
        v=state.get("visible_object_category")
        if v not in (None,"uncertain","not_visible","none"): return "dish" if v=="plate" else v
    return "uncertain"

def arm_compatible(a,b): return a==b or "uncertain" in (a,b) or "both" in (a,b)
def obj_compatible(a,b):
    a="dish" if a=="plate" else a; b="dish" if b=="plate" else b
    return a==b or a in (None,"uncertain","none","other","not_visible") or b in (None,"uncertain","none","other","not_visible")

def overlap_or_near(a,b,max_gap):
    return int(b["_window_start_frame"])-int(a["_window_end_frame"])<=max_gap

def apply_sequence_consistency(events,max_gap_frames):
    xs=sorted((dict(e) for e in events),key=lambda x:x["_raw_frame"])
    for e in xs: e.setdefault("consistency_flags",[])
    for i in range(1,len(xs)-1):
        p,c,n=xs[i-1],xs[i],xs[i+1]
        pa,ca,na=p.get("active_arm","uncertain"),c.get("active_arm","uncertain"),n.get("active_arm","uncertain")
        same_context=family(p)==family(c)==family(n) and obj_compatible(obj_of(p),obj_of(c)) and obj_compatible(obj_of(c),obj_of(n))
        close=overlap_or_near(p,c,max_gap_frames) and overlap_or_near(c,n,max_gap_frames)
        if same_context and close and pa==na and pa in ("left","right") and ca in ("left","right") and ca!=pa:
            c["active_arm"]="uncertain"; c["confidence"]=round(float(c.get("confidence",0))*0.75,3); c["consistency_flags"].append("isolated_arm_flip")
    return xs

def should_merge(g,e,max_gap_frames):
    last=g[-1]
    if family(last)!=family(e): return False
    if not overlap_or_near(last,e,max_gap_frames): return False
    if not arm_compatible(last.get("active_arm","uncertain"),e.get("active_arm","uncertain")): return False
    if not obj_compatible(obj_of(last),obj_of(e)): return False
    return True

def weighted_mode(vals,weights,default="uncertain"):
    s={}
    for v,w in zip(vals,weights):
        if v is not None:s[v]=s.get(v,0)+float(w)
    return max(s.items(),key=lambda kv:kv[1])[0] if s else default

def cluster_actions(events,min_conf=.55,max_gap_sec=1.0,fps=30.0):
    gap=int(round(max_gap_sec*fps)); events=apply_sequence_consistency(events,gap)
    eligible=[e for e in events if e.get("contains_semantic_action") is True and float(e.get("confidence",0))>=min_conf and family(e) not in ("idle","uncertain")]
    groups=[]
    for e in eligible:
        if groups and should_merge(groups[-1],e,gap): groups[-1].append(e)
        else: groups.append([e])
    actions=[]
    for aid,g in enumerate(groups):
        w=[max(.05,float(x.get("confidence",0))) for x in g]
        typ=weighted_mode([family(x) for x in g],w); rel=weighted_mode([x.get("task_relevance") for x in g],w); arm=weighted_mode([x.get("active_arm") for x in g],w); obj=weighted_mode([obj_of(x) for x in g],w)
        rep=max(g,key=lambda x:float(x.get("confidence",0)))
        transitions=[{"event_id":x["_event_id"],"transition_action":x.get("transition_action"),"source":x.get("transition_source"),"state_before":x.get("state_before"),"state_after":x.get("state_after")} for x in g if x.get("transition_action")]
        flags=sorted({f for x in g for f in x.get("consistency_flags",[])})
        actions.append({"action_id":aid,"action_type":typ,"instruction":rep.get("atomic_action",""),"active_arm":arm,"visible_object_category":obj,"task_relevance":rel,"confidence":round(sum(w)/len(w),3),
          "member_event_ids":[int(x["_event_id"]) for x in g],"member_center_frames":[int(x["_raw_frame"]) for x in g],"progress_sequence":[x.get("action_progress") for x in g],
          "semantic_support":[x.get("semantic_support") for x in g],"state_transition_evidence":transitions,"consistency_flags":flags,
          "rough_start_frame":min(int(x["_window_start_frame"]) for x in g),"rough_end_frame":max(int(x["_window_end_frame"]) for x in g),
          "first_evidence_frame":min(int(x["_raw_frame"]) for x in g),"last_evidence_frame":max(int(x["_raw_frame"]) for x in g)})
    # post_task is terminal by definition. Non-terminal occurrences are downgraded to off_task.
    for i,a in enumerate(actions):
        if a.get("task_relevance")=="post_task" and any(x.get("task_relevance")=="relevant" for x in actions[i+1:]):
            a["task_relevance"]="off_task"; a["consistency_flags"].append("nonterminal_post_task_downgraded")
    return events,eligible,actions

def compose(client,model,task,actions,max_tokens=2600):
    prompt=f"""把机器人 action intervals 组合成 semantic phases。高层任务：{task}

规则：
1. 只做语义分组和命名，禁止输出 frame 数值。
2. phase_groups 只能包含 task_relevance=relevant 的 action_id；reverse/off_task/post_task/uncertain 只能作为上下文，不得强行并入。
3. state_transition_evidence 优先于自由文本 instruction。若 instruction 与门/拉篮/抓放状态变化矛盾，以状态变化为准。
4. reach/grasp/transport/place 可以组成阶段；真实重复循环保留。
5. task_end_action_id 必须是 relevant 且 progress_sequence 中含 end 或 transition；ongoing-only action 不能作为任务结束。
6. 不要求 phases 覆盖整条轨迹，允许中间存在 coarse-only gap。
7. 发现矛盾写入 uncertainties，不要自行修补。

Action intervals:
{json.dumps(actions,ensure_ascii=False,indent=2)}

严格输出 JSON：
{{
 "trajectory_instruction":"一句按真实顺序的轨迹描述",
 "phase_groups":[{{"phase_id":0,"instruction":"...","action_ids":[0,1],"confidence":0.0}}],
 "task_end_action_id":null,
 "uncertainties":["..."]
}}"""
    resp=client.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],max_tokens=max_tokens,stream=False,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    raw=resp.choices[0].message.content
    return parse_json(raw),raw

def sanitize(h,actions):
    by={a["action_id"]:a for a in actions}; clean=[]
    for p in h.get("phase_groups",[]):
        ids=sorted({int(i) for i in p.get("action_ids",[]) if int(i) in by and by[int(i)].get("task_relevance")=="relevant"})
        if ids: clean.append({"phase_id":len(clean),"instruction":p.get("instruction",""),"action_ids":ids,"confidence":float(p.get("confidence",0))})
    h["phase_groups"]=clean
    end=h.get("task_end_action_id")
    def end_ok(i):
        if i not in by or by[i].get("task_relevance")!="relevant": return False
        return any(x in ("end","transition") for x in by[i].get("progress_sequence",[]))
    if end is None or not end_ok(int(end)):
        candidates=[i for p in clean for i in p["action_ids"] if end_ok(i)]
        h["task_end_action_id"]=max(candidates) if candidates else None
        if end is not None: h.setdefault("uncertainties",[]).append("LLM task_end_action_id failed consistency checks and was replaced/dropped.")
    return h

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--output-dir",required=True); ap.add_argument("--task",required=True); ap.add_argument("--min-confidence",type=float,default=.55); ap.add_argument("--max-gap-sec",type=float,default=1.0); ap.add_argument("--fps",type=float,default=30.0)
    ap.add_argument("--api-key",default=None); ap.add_argument("--base-url",default=None); ap.add_argument("--aigc-user",default=None); ap.add_argument("--model",default=DEFAULT_MODEL)
    args=ap.parse_args(); out=Path(args.output_dir); events=load_jsonl(out/"window_semantics.jsonl")
    normalized,eligible,actions=cluster_actions(events,args.min_confidence,args.max_gap_sec,args.fps)
    (out/"window_semantics_consistent.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in normalized)+"\n",encoding="utf-8")
    (out/"action_intervals.json").write_text(json.dumps({"num_candidate_windows":len(events),"num_semantic_windows":len(eligible),"num_action_intervals":len(actions),"action_intervals":actions},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"candidate windows: {len(events)} -> semantic windows: {len(eligible)} -> action intervals: {len(actions)}")
    client=build_client(args); h,raw=compose(client,args.model,args.task,actions); h=sanitize(h,actions); h["task_goal"]=args.task; h["action_intervals"]=actions
    (out/"semantic_hierarchy.json").write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"vlm_raw"/"phase_compose.txt").write_text(raw,encoding="utf-8")
    print("saved",out/"semantic_hierarchy.json")
if __name__=="__main__": main()
