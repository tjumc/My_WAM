#!/usr/bin/env python3
"""V2.2 state-transition-grounded semantic window verification."""
import argparse, base64, json, os, re
from pathlib import Path
import h5py
import numpy as np
from openai import OpenAI

DEFAULT_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"
DEFAULT_MODEL="/model/qwen3.5-122b-a10b"

def data_url(path):
    return "data:image/jpeg;base64,"+base64.b64encode(Path(path).read_bytes()).decode()

def parse_json(text):
    text=text.strip()
    if text.startswith("```"):
        text=re.sub(r"^```(?:json)?\s*","",text,flags=re.I)
        text=re.sub(r"\s*```$","",text)
    try: return json.loads(text)
    except Exception:
        m=re.search(r"\{.*\}",text,re.S)
        if not m: raise
        return json.loads(m.group(0))

def build_client(args):
    key=args.api_key or os.getenv("AIGC_API_KEY")
    if not key:
        raise RuntimeError("AIGC_API_KEY is required")
    headers={}
    user=args.aigc_user or os.getenv("AIGC_USER")
    if user: headers["AIGC-USER"]=user
    return OpenAI(api_key=key,base_url=args.base_url or os.getenv("AIGC_BASE_URL",DEFAULT_BASE_URL),default_headers=headers or None)

def trend(a,b,eps):
    d=float(b-a)
    return ("stable" if abs(d)<=eps else ("increase" if d>0 else "decrease")),d

def load_hdf5_context(path,events,fps=30.0,radius_sec=.35):
    r=max(2,int(round(radius_sec*fps)))
    with h5py.File(path,"r") as f:
        rg=f["command_poses_dict/astribot_gripper_right"][:,0].astype(float)
        lg=f["command_poses_dict/astribot_gripper_left"][:,0].astype(float)
        rw=f["endpoint_wrench_dict/astribot_arm_right"][:].astype(float)
        lw=f["endpoint_wrench_dict/astribot_arm_left"][:].astype(float)
    rf=np.linalg.norm(rw[:,:3],axis=1); lf=np.linalg.norm(lw[:,:3],axis=1)
    def med(x,s,t):
        if t<=s: return float(x[min(max(s,0),len(x)-1)])
        return float(np.median(x[s:t]))
    out={}
    for e in events:
        i=int(e["raw_frame"]); a0,a1=max(0,i-r),max(1,i); b0,b1=min(len(rg)-1,i+1),min(len(rg),i+r+1)
        rgb,rga=med(rg,a0,a1),med(rg,b0,b1); lgb,lga=med(lg,a0,a1),med(lg,b0,b1)
        rfb,rfa=med(rf,a0,a1),med(rf,b0,b1); lfb,lfa=med(lf,a0,a1),med(lf,b0,b1)
        rt,rd=trend(rgb,rga,5.0); lt,ld=trend(lgb,lga,5.0)
        rft,rfd=trend(rfb,rfa,max(2.0,.1*max(rfb,rfa,1))); lft,lfd=trend(lfb,lfa,max(2.0,.1*max(lfb,lfa,1)))
        out[e["event_id"]]={"right_gripper_command_trend":rt,"right_gripper_command_delta":round(rd,3),
          "left_gripper_command_trend":lt,"left_gripper_command_delta":round(ld,3),
          "right_force_trend":rft,"right_force_delta":round(rfd,3),
          "left_force_trend":lft,"left_force_delta":round(lfd,3)}
    return out

def transition_action(o):
    b=o.get("state_before") or {}; a=o.get("state_after") or {}
    db,da=b.get("door_state"),a.get("door_state")
    rb,ra=b.get("rack_state"),a.get("rack_state")
    if db in ("closed","closing") and da in ("opening","open"): return "open","door_state"
    if db in ("open","opening") and da in ("closing","closed"): return "close","door_state"
    if rb in ("in","moving_in") and ra in ("moving_out","out"): return "pull","rack_state"
    if rb in ("out","moving_out") and ra in ("moving_in","in"): return "push","rack_state"
    for side in ("right","left"):
        x=b.get(f"{side}_hand_relation"); y=a.get(f"{side}_hand_relation")
        if x in ("free","approaching","contact") and y=="holding": return "grasp",f"{side}_hand_relation"
        if x=="holding" and y in ("released","free"): return "release",f"{side}_hand_relation"
    lb,la=b.get("object_location"),a.get("object_location")
    held_b=any(b.get(f"{s}_hand_relation")=="holding" for s in ("right","left"))
    held_a=any(a.get(f"{s}_hand_relation")=="holding" for s in ("right","left"))
    if held_b and held_a and lb and la and lb!=la and "uncertain" not in (lb,la): return "transport","object_location"
    return None,None

def normalize_result(o):
    inferred,source=transition_action(o)
    proposed=o.get("proposed_action","uncertain")
    flags=[]
    normalized=proposed
    if inferred:
        normalized=inferred
        if proposed not in (inferred,"uncertain","other"):
            flags.append(f"transition_override:{proposed}->{inferred}")
    support=o.get("semantic_support","weak")
    keep=bool(o.get("contains_semantic_action"))
    if support=="weak" and not inferred and float(o.get("confidence",0))<.8:
        keep=False; flags.append("weak_support_rejected")
    o["transition_action"]=inferred
    o["transition_source"]=source
    o["normalized_action"]=normalized
    o["contains_semantic_action"]=keep
    o["consistency_flags"]=flags
    return o

def verify(client,model,task,event,ctx,sheet,fps,radius_sec,max_tokens=1400):
    r=int(round(radius_sec*fps)); w0=max(0,int(event["raw_frame"])-r); w1=int(event["raw_frame"])+r
    prompt=f"""你正在分析人形机器人家庭操作轨迹的一个多视角局部时间窗口。候选中心 frame 只是注意力锚点，不是动作边界。

高层任务：{task}
窗口约 raw frame {w0}~{w1}，中心 frame={event['raw_frame']}，time={event['time_sec']:.3f}s。
检测信号：{', '.join(event.get('signals',[]))}
中性传感器摘要（数值 increase/decrease 没有预定义的张开/闭合含义）：
{json.dumps(ctx,ensure_ascii=False,indent=2)}

先做“状态观察”，再给动作。不要反过来根据动作猜状态。
三行图片依次是 head/left/right，列从早到晚。

状态枚举：
door_state: closed|opening|open|closing|uncertain|not_visible
rack_state: in|moving_out|out|moving_in|uncertain|not_visible
right/left_hand_relation: free|approaching|contact|holding|released|uncertain|not_visible
object_location: tabletop|dishwasher_entrance|dishwasher_inside|rack|hand|uncertain|not_visible

规则：
1. 门 closed→open/opening 是 open；open→closed/closing 是 close。拉篮 in→out 是 pull，out→in 是 push。
2. free/approaching/contact→holding 支持 grasp；holding→released/free 支持 release/place。
3. 持物且位置持续改变支持 transport。持续 goal-directed motion 也可以是真动作，即使没有离散状态跳变。
4. 不得把 door/rack 运动描述成搬盘子；看不清物体就填 uncertain/not_visible。
5. 不得根据高层任务脑补物体，不得根据夹爪 command 数值猜开合。
6. task_relevance: relevant|reverse|off_task|post_task|uncertain。post_task 只在任务已经完成后才用；中途反向/异常动作优先 reverse/off_task。
7. semantic_support: state_transition|goal_directed_motion|both|weak。
8. confidence 必须校准。

严格输出 JSON：
{{
 "state_before":{{"door_state":"uncertain","rack_state":"uncertain","right_hand_relation":"uncertain","left_hand_relation":"uncertain","object_location":"uncertain","visible_object_category":"uncertain"}},
 "state_after":{{"door_state":"uncertain","rack_state":"uncertain","right_hand_relation":"uncertain","left_hand_relation":"uncertain","object_location":"uncertain","visible_object_category":"uncertain"}},
 "contains_semantic_action":true,
 "proposed_action":"navigation|reach|grasp|release|open|close|pull|push|place|transport|retract|reposition|contact|idle|other|uncertain",
 "action_progress":"start|ongoing|end|transition|uncertain",
 "semantic_support":"state_transition|goal_directed_motion|both|weak",
 "atomic_action":"一句中文，只描述可观察动作",
 "active_arm":"left|right|both|none|uncertain",
 "task_relevance":"relevant|reverse|off_task|post_task|uncertain",
 "confidence":0.0,
 "evidence":["最多3条"],
 "reason_if_false":""
}}"""
    resp=client.chat.completions.create(model=model,messages=[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":data_url(sheet)}}]}],max_tokens=max_tokens,stream=False,extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    raw=resp.choices[0].message.content
    o=normalize_result(parse_json(raw))
    o.update({"_event_id":event["event_id"],"_raw_frame":event["raw_frame"],"_time_sec":event["time_sec"],"_window_start_frame":w0,"_window_end_frame":w1,"_detector_score":event.get("score"),"_detector_signals":event.get("signals",[]),"_sensor_context":ctx})
    return o,raw

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--output-dir",required=True); ap.add_argument("--hdf5",required=True); ap.add_argument("--task",required=True)
    ap.add_argument("--api-key",default=None); ap.add_argument("--base-url",default=None); ap.add_argument("--aigc-user",default=None); ap.add_argument("--model",default=DEFAULT_MODEL); ap.add_argument("--fps",type=float,default=30.0); ap.add_argument("--window-radius-sec",type=float,default=1.5)
    args=ap.parse_args(); root=Path(args.episode_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    events=json.load(open(root/"candidate_events.json",encoding="utf-8"))["candidate_events"]; ctx=load_hdf5_context(args.hdf5,events,args.fps); client=build_client(args)
    raw_dir=out/"vlm_raw"; raw_dir.mkdir(exist_ok=True); outfile=out/"window_semantics.jsonl"
    with open(outfile,"w",encoding="utf-8") as fw:
        for e in events:
            sheets=sorted((root/"contact_sheets").glob(f"event_{e['event_id']:02d}_f*.jpg"))
            if not sheets: raise FileNotFoundError(f"missing contact sheet {e['event_id']}")
            o,raw=verify(client,args.model,args.task,e,ctx[e["event_id"]],sheets[0],args.fps,args.window_radius_sec)
            fw.write(json.dumps(o,ensure_ascii=False)+"\n"); fw.flush(); (raw_dir/f"window_{e['event_id']:02d}.txt").write_text(raw,encoding="utf-8")
            print(f"[{e['event_id']:02d}] keep={o['contains_semantic_action']} action={o['normalized_action']} rel={o.get('task_relevance')} conf={o.get('confidence')} flags={o.get('consistency_flags')}")
    print("saved",outfile)
if __name__=="__main__": main()
