#!/usr/bin/env python3
"""V2.2 duration-aware signal refinement with non-forced phase coverage."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from analyze_episode import load_episode, build_features, robust_z, smooth_1d

MIN_DURATION_SEC={"grasp":.25,"place":.6,"release":.25,"contact":.25,"reach":.5,"transport":1.0,"retract":.5,"open":.7,"close":.7,"pull":.6,"push":.6,"navigation":1.0,"reposition":.5}

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def build_activity(d,fps=30.0):
    _,Xs,names=build_features(d); Z=robust_z(Xs)
    selected=["base_speed","base_yaw_rate","torso_joint_speed","left_ee_speed","right_ee_speed","left_ee_ang_speed","right_ee_ang_speed","left_force_change","right_force_change"]
    idx=[names.index(n) for n in selected]; motion=np.mean(np.abs(Z[:,idx]),axis=1)
    rg=d["right_grip_cmd"]; lg=d["left_grip_cmd"]; gd=np.abs(np.diff(rg,prepend=rg[0]))+np.abs(np.diff(lg,prepend=lg[0])); p95=np.percentile(gd,95)
    if p95>1e-8: gd=np.clip(gd/(p95+1e-8),0,3)
    return smooth_1d(motion+.45*gd,max(3,int(round(.25*fps))))

def valley(activity,lo,hi,edge_margin=3):
    lo=max(0,int(lo)); hi=min(len(activity)-1,int(hi))
    if hi<=lo:return lo
    if hi-lo>2*edge_margin: lo2,hi2=lo+edge_margin,hi-edge_margin
    else: lo2,hi2=lo,hi
    return int(lo2+np.argmin(activity[lo2:hi2+1]))

def enforce_min_duration(s,e,a,T,fps):
    need=max(1,int(round(MIN_DURATION_SEC.get(a.get("action_type"),.4)*fps)))
    if e-s+1>=need:return s,e
    rough_s=max(0,int(a.get("rough_start_frame",s))); rough_e=min(T-1,int(a.get("rough_end_frame",e)))
    deficit=need-(e-s+1); left=min((deficit+1)//2,s-rough_s); s-=left; deficit-=left
    right=min(deficit,rough_e-e); e+=right; deficit-=right
    if deficit>0:
        left=min((deficit+1)//2,s); s-=left; deficit-=left
        e=min(T-1,e+deficit)
    return int(s),int(e)

def refine_actions(actions,activity,T,fps):
    out=[]
    for a in actions:
        first=int(a["first_evidence_frame"]); last=int(a["last_evidence_frame"])
        rough_s=max(0,int(a.get("rough_start_frame",first-int(1.5*fps)))); rough_e=min(T-1,int(a.get("rough_end_frame",last+int(1.5*fps))))
        s=valley(activity,rough_s,first); e=valley(activity,last,rough_e)
        s=min(s,first); e=max(e,last); s,e=enforce_min_duration(s,e,a,T,fps)
        aa=dict(a); aa["refined_start_frame"]=s; aa["refined_end_frame"]=e; aa["min_duration_prior_sec"]=MIN_DURATION_SEC.get(a.get("action_type"),.4); out.append(aa)
    return out

def merge_spans(spans,max_gap):
    spans=sorted([list(x) for x in spans])
    if not spans:return []
    out=[spans[0]]
    for s,e in spans[1:]:
        if s-out[-1][1]-1<=max_gap: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out

def resolve_cross_phase_overlaps(phase_spans,activity):
    flat=[]
    for pidx,spans in enumerate(phase_spans):
        for j,(s,e) in enumerate(spans): flat.append([s,e,pidx,j])
    flat.sort()
    for k in range(len(flat)-1):
        a,b=flat[k],flat[k+1]
        if b[0]<=a[1] and a[2]!=b[2]:
            cut=valley(activity,b[0],a[1])
            a[1]=max(a[0],cut-1); b[0]=min(b[1],cut)
    rebuilt=[[] for _ in phase_spans]
    for s,e,pidx,j in flat:
        if e>=s: rebuilt[pidx].append([int(s),int(e)])
    return rebuilt

def complement_segments(spans,start,end):
    spans=sorted(spans); out=[]; cur=start
    for s,e in spans:
        if e<start or s>end: continue
        s=max(s,start); e=min(e,end)
        if s>cur: out.append([cur,s-1])
        cur=max(cur,e+1)
    if cur<=end: out.append([cur,end])
    return out

def lr_span(s,e,args):
    if args.lerobot_max_raw_frame is None:return {"lerobot_episode_index":args.lerobot_episode,"lerobot_start_frame":s,"lerobot_end_frame":e}
    if s>args.lerobot_max_raw_frame:return {"lerobot_episode_index":args.lerobot_episode,"lerobot_start_frame":None,"lerobot_end_frame":None}
    return {"lerobot_episode_index":args.lerobot_episode,"lerobot_start_frame":s,"lerobot_end_frame":min(e,args.lerobot_max_raw_frame)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("episode_dir"); ap.add_argument("--output-dir",required=True); ap.add_argument("--hdf5",required=True); ap.add_argument("--fps",type=float,default=30.0); ap.add_argument("--phase-bridge-gap-sec",type=float,default=.5)
    ap.add_argument("--lerobot-episode",type=int,default=None); ap.add_argument("--lerobot-max-raw-frame",type=int,default=None); args=ap.parse_args()
    out=Path(args.output_dir); h=json.load(open(out/"semantic_hierarchy.json",encoding="utf-8")); windows=load_jsonl(out/"window_semantics_consistent.jsonl")
    d=load_episode(args.hdf5); T=len(d["time"]); activity=build_activity(d,args.fps); actions=refine_actions(h.get("action_intervals",[]),activity,T,args.fps); by={a["action_id"]:a for a in actions}
    valid_phases=[]
    for p in h.get("phase_groups",[]):
        ids=[i for i in p.get("action_ids",[]) if i in by and by[i].get("task_relevance")=="relevant"]
        if ids: valid_phases.append({**p,"action_ids":ids})
    end_id=h.get("task_end_action_id")
    if end_id is not None and int(end_id) in by: end_anchor=int(by[int(end_id)]["refined_end_frame"])
    else:
        c=[a for a in actions if a.get("task_relevance")=="relevant" and any(x in ("end","transition") for x in a.get("progress_sequence",[]))]
        if not c:c=[a for a in actions if a.get("task_relevance")=="relevant"]
        end_anchor=int(c[-1]["refined_end_frame"]) if c else T-1
    post=[a for a in actions if a.get("task_relevance")=="post_task" and int(a["refined_start_frame"])>end_anchor]
    end_hi=min(T-1,end_anchor+int(round(3*args.fps)))
    if post:end_hi=min(end_hi,min(int(a["refined_start_frame"]) for a in post))
    task_end=valley(activity,end_anchor,end_hi) if end_hi>end_anchor else end_anchor
    bridge=int(round(args.phase_bridge_gap_sec*args.fps))
    phase_spans=[]
    for p in valid_phases:
        spans=[(by[i]["refined_start_frame"],min(by[i]["refined_end_frame"],task_end)) for i in p["action_ids"] if by[i]["refined_start_frame"]<=task_end]
        phase_spans.append(merge_spans(spans,bridge))
    phase_spans=resolve_cross_phase_overlaps(phase_spans,activity)
    out_phases=[]; all_spans=[]
    for p,spans in zip(valid_phases,phase_spans):
        ser=[]
        for s,e in spans:
            if e<s:continue
            z={"raw_start_frame":int(s),"raw_end_frame":int(e)}; z.update(lr_span(int(s),int(e),args)); ser.append(z); all_spans.append([int(s),int(e)])
        if ser: out_phases.append({"phase_id":p["phase_id"],"instruction":p.get("instruction",""),"action_ids":p["action_ids"],"confidence":p.get("confidence"),"spans":ser})
    coarse=complement_segments(all_spans,0,int(task_end))
    coarse_out=[]
    for s,e in coarse:
        z={"raw_start_frame":s,"raw_end_frame":e,"label_policy":"coarse_task_only"}; z.update(lr_span(s,e,args)); coarse_out.append(z)
    suffix=[int(task_end+1),int(T-1)] if task_end<T-1 else None
    result={"annotation_version":"v2.2","task_goal":h.get("task_goal"),"trajectory_instruction":h.get("trajectory_instruction"),"task_end_raw_frame":int(task_end),"lerobot_episode_index":args.lerobot_episode,
      "semantic_phases":out_phases,"action_intervals":actions,"coarse_only_segments":coarse_out,"post_task_or_unassigned_suffix_raw_frames":suffix,"uncertainties":h.get("uncertainties",[]),
      "notes":["State transitions override conflicting free-form action captions.","Action boundaries combine HDF5 activity valleys with action-type minimum-duration priors.","Semantic phases may contain multiple fine-label spans; gaps are intentionally left coarse-task-only.","reverse/off_task/post_task actions are excluded from current-task fine phases."]}
    (out/"hierarchical_annotations.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    t=d["time"]-d["time"][0]; fig,ax=plt.subplots(figsize=(16,5)); ax.plot(t,activity,label="activity")
    for a in actions:
        s=min(a["refined_start_frame"],T-1); e=min(a["refined_end_frame"],T-1); ax.axvspan(t[s],t[e],alpha=.07); ax.text(t[s],np.percentile(activity,96),f"A{a['action_id']}",rotation=90,va="top",fontsize=8)
    for p in out_phases:
        for sp in p["spans"]:
            s=min(sp["raw_start_frame"],T-1); ax.axvline(t[s],linestyle="--",alpha=.5); ax.text(t[s],np.percentile(activity,84),f"P{p['phase_id']}",rotation=90,va="top")
    ax.axvline(t[min(task_end,T-1)],linestyle="-.",linewidth=2,label="task end"); ax.set_xlabel("time (s)"); ax.legend(); fig.tight_layout(); fig.savefig(out/"boundaries.png",dpi=160); plt.close(fig)
    print("saved",out/"hierarchical_annotations.json"); print("task_end_raw_frame =",task_end); print("coarse_only_segments =",coarse)
if __name__=="__main__": main()
