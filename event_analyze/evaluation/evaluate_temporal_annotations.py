#!/usr/bin/env python3
"""Unified semantic + temporal evaluation for hierarchical robot annotations."""
import argparse, json
from pathlib import Path

def slen(s,e): return max(0,int(e)-int(s)+1)
def inter(a0,a1,b0,b1): return max(0,min(int(a1),int(b1))-max(int(a0),int(b0))+1)
def iou(a0,a1,b0,b1):
    x=inter(a0,a1,b0,b1); u=slen(a0,a1)+slen(b0,b1)-x
    return x/u if u else 0.0
def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def edit(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        n=[i]
        for j,y in enumerate(b,1):
            n.append(min(n[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=n
    return d[-1]

def align(pred,gt):
    n,m=len(pred),len(gt)
    dp=[[(0,0.0,[]) for _ in range(m+1)] for _ in range(n+1)]
    def better(a,b):
        if a[0]!=b[0]: return a if a[0]>b[0] else b
        if abs(a[1]-b[1])>1e-12: return a if a[1]>b[1] else b
        return a
    for i in range(1,n+1):
        for j in range(1,m+1):
            z=better(dp[i-1][j],dp[i][j-1])
            if pred[i-1]["skill_type"]==gt[j-1]["skill_type"]:
                q=iou(pred[i-1]["raw_start_frame"],pred[i-1]["raw_end_frame"],
                      gt[j-1]["raw_start_frame"],gt[j-1]["raw_end_frame"])
                p=dp[i-1][j-1]
                z=better(z,(p[0]+1,p[1]+q,p[2]+[(i-1,j-1)]))
            dp[i][j]=z
    return dp[n][m][2]

def boundary_f1(pred,gt,fps,tol_sec,core=False):
    ps,pe=("provisional_start_frame","provisional_end_frame") if core else ("raw_start_frame","raw_end_frame")
    gs,ge=("core_start_frame","core_end_frame") if core else ("raw_start_frame","raw_end_frame")
    pb=[]; gb=[]
    for i,x in enumerate(pred):
        pb += [(i,x["skill_type"],"start",int(x.get(ps,x["raw_start_frame"]))),
               (i,x["skill_type"],"end",int(x.get(pe,x["raw_end_frame"])))]
    for j,x in enumerate(gt):
        gb += [(j,x["skill_type"],"start",int(x[gs])),
               (j,x["skill_type"],"end",int(x[ge]))]
    tol=int(round(tol_sec*fps)); cand=[]
    for pi,p in enumerate(pb):
        for gi,g in enumerate(gb):
            if p[1]==g[1] and p[2]==g[2]:
                e=abs(p[3]-g[3])
                if e<=tol: cand.append((e,pi,gi))
    cand.sort(); up=set(); ug=set(); tp=0
    for _,pi,gi in cand:
        if pi not in up and gi not in ug:
            up.add(pi); ug.add(gi); tp+=1
    fp=len(pb)-tp; fn=len(gb)-tp
    p=tp/(tp+fp) if tp+fp else 1.0
    r=tp/(tp+fn) if tp+fn else 1.0
    f=2*p*r/(p+r) if p+r else 0.0
    return {"tolerance_sec":tol_sec,"tolerance_frames":tol,"tp":tp,"fp":fp,"fn":fn,
            "precision":p,"recall":r,"f1":f}

def union_len(xs,T):
    ys=[]
    for s,e in xs:
        s=max(0,min(int(s),T-1)); e=max(0,min(int(e),T-1))
        if e>=s: ys.append((s,e))
    if not ys:return 0
    ys.sort(); cs,ce=ys[0]; n=0
    for s,e in ys[1:]:
        if s<=ce+1: ce=max(ce,e)
        else: n+=ce-cs+1; cs,ce=s,e
    return n+ce-cs+1

def evaluate(pred_doc,gt_doc,tols):
    pred=list(pred_doc.get("semantic_phases",[])); gt=list(gt_doc["steps"])
    for x in pred:
        x.setdefault("provisional_start_frame",x["raw_start_frame"])
        x.setdefault("provisional_end_frame",x["raw_end_frame"])
    fps=float(gt_doc.get("fps",30)); T=int(gt_doc["total_raw_frames"])
    pairs=align(pred,gt); mp={i for i,_ in pairs}; mg={j for _,j in pairs}
    ps=[x["skill_type"] for x in pred]; gs=[x["skill_type"] for x in gt]
    n=len(pairs); prec=n/len(pred) if pred else (1.0 if not gt else 0.0); rec=n/len(gt) if gt else 1.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    rows=[]; si=[]; ci=[]; ss=[]; se=[]; cs=[]; ce=[]; sem_inter=0; core_inter=0
    for pi,gi in pairs:
        p,g=pred[pi],gt[gi]
        sv=iou(p["raw_start_frame"],p["raw_end_frame"],g["raw_start_frame"],g["raw_end_frame"])
        cv=iou(p["provisional_start_frame"],p["provisional_end_frame"],g["core_start_frame"],g["core_end_frame"])
        a=int(p["raw_start_frame"])-int(g["raw_start_frame"]); b=int(p["raw_end_frame"])-int(g["raw_end_frame"])
        c=int(p["provisional_start_frame"])-int(g["core_start_frame"]); d=int(p["provisional_end_frame"])-int(g["core_end_frame"])
        si.append(sv); ci.append(cv); ss.append(abs(a)); se.append(abs(b)); cs.append(abs(c)); ce.append(abs(d))
        sem_inter+=inter(p["raw_start_frame"],p["raw_end_frame"],g["raw_start_frame"],g["raw_end_frame"])
        core_inter+=inter(p["provisional_start_frame"],p["provisional_end_frame"],g["core_start_frame"],g["core_end_frame"])
        tol=int(g.get("boundary_tolerance_frames",round(fps)))
        rows.append({
            "pred_index":pi,"gt_phase_id":g.get("phase_id",gi),"skill_type":g["skill_type"],
            "semantic_gt":[g["raw_start_frame"],g["raw_end_frame"]],
            "semantic_pred":[p["raw_start_frame"],p["raw_end_frame"]],"semantic_iou":sv,
            "semantic_start_error_frames":a,"semantic_end_error_frames":b,
            "semantic_start_error_sec":a/fps,"semantic_end_error_sec":b/fps,
            "core_gt":[g["core_start_frame"],g["core_end_frame"]],
            "core_pred":[p["provisional_start_frame"],p["provisional_end_frame"]],"core_iou":cv,
            "core_start_error_frames":c,"core_end_error_frames":d,
            "core_start_error_sec":c/fps,"core_end_error_sec":d/fps,
            "manual_boundary_tolerance_frames":tol,
            "semantic_start_within_manual_tolerance":abs(a)<=tol,
            "semantic_end_within_manual_tolerance":abs(b)<=tol,
            "core_start_within_manual_tolerance":abs(c)<=tol,
            "core_end_within_manual_tolerance":abs(d)<=tol
        })
    sem_gt=sum(slen(x["raw_start_frame"],x["raw_end_frame"]) for x in gt)
    sem_pr=sum(slen(x["raw_start_frame"],x["raw_end_frame"]) for x in pred)
    cor_gt=sum(slen(x["core_start_frame"],x["core_end_frame"]) for x in gt)
    cor_pr=sum(slen(x["provisional_start_frame"],x["provisional_end_frame"]) for x in pred)
    coarse=pred_doc.get("coarse_only_segments",[])
    mt_sem=sum(int(x["semantic_start_within_manual_tolerance"])+int(x["semantic_end_within_manual_tolerance"]) for x in rows)
    mt_core=sum(int(x["core_start_within_manual_tolerance"])+int(x["core_end_within_manual_tolerance"]) for x in rows)
    den=2*len(gt)
    return {
      "metadata":{"annotation_version":pred_doc.get("annotation_version"),"episode":gt_doc.get("episode"),
                  "fps":fps,"total_raw_frames":T,"gt_status":gt_doc.get("status"),
                  "matching_policy":"monotonic same-skill alignment: maximize matches, then semantic IoU"},
      "sequence":{"predicted":ps,"ground_truth":gs,"matched_phases":n,"precision":prec,"recall":rec,"f1":f1,
                  "lcs":n,"edit_distance":edit(ps,gs),"exact_sequence_match":ps==gs,
                  "unmatched_predictions":[{"pred_index":i,"skill_type":x["skill_type"],
                    "semantic_pred":[x["raw_start_frame"],x["raw_end_frame"]]} for i,x in enumerate(pred) if i not in mp],
                  "missed_gt_phases":[{"gt_phase_id":x.get("phase_id",j),"skill_type":x["skill_type"],
                    "semantic_gt":[x["raw_start_frame"],x["raw_end_frame"]]} for j,x in enumerate(gt) if j not in mg]},
      "semantic_temporal":{"matched_mean_iou":mean(si),"penalized_mean_iou":sum(si)/max(len(pred),len(gt),1),
                  "start_mae_frames":mean(ss),"end_mae_frames":mean(se),"boundary_mae_frames":mean(ss+se),
                  "start_mae_sec":mean(ss)/fps,"end_mae_sec":mean(se)/fps,"boundary_mae_sec":mean(ss+se)/fps,
                  "boundary_f1":[boundary_f1(pred,gt,fps,t,False) for t in tols],
                  "manual_tolerance_boundary_accuracy":mt_sem/den if den else 1.0},
      "core_temporal":{"matched_mean_iou":mean(ci),"penalized_mean_iou":sum(ci)/max(len(pred),len(gt),1),
                  "start_mae_frames":mean(cs),"end_mae_frames":mean(ce),"boundary_mae_frames":mean(cs+ce),
                  "start_mae_sec":mean(cs)/fps,"end_mae_sec":mean(ce)/fps,"boundary_mae_sec":mean(cs+ce)/fps,
                  "boundary_f1":[boundary_f1(pred,gt,fps,t,True) for t in tols],
                  "manual_tolerance_boundary_accuracy":mt_core/den if den else 1.0},
      "coverage":{
          "gt_fine_union_frames":union_len([(x["raw_start_frame"],x["raw_end_frame"]) for x in gt],T),
          "pred_fine_union_frames":union_len([(x["raw_start_frame"],x["raw_end_frame"]) for x in pred],T),
          "gt_fine_coverage_ratio":union_len([(x["raw_start_frame"],x["raw_end_frame"]) for x in gt],T)/T,
          "pred_fine_coverage_ratio":union_len([(x["raw_start_frame"],x["raw_end_frame"]) for x in pred],T)/T,
          "pred_coarse_only_ratio":union_len([(x["raw_start_frame"],x["raw_end_frame"]) for x in coarse],T)/T if coarse else 0.0,
          "semantic_correct_intersection_frames":sem_inter,
          "semantic_correct_gt_coverage":sem_inter/sem_gt if sem_gt else 1.0,
          "semantic_false_positive_duration_frames":sem_pr-sem_inter,
          "semantic_false_positive_duration_ratio":(sem_pr-sem_inter)/sem_pr if sem_pr else 0.0,
          "semantic_missed_gt_duration_frames":sem_gt-sem_inter,
          "semantic_missed_gt_duration_ratio":(sem_gt-sem_inter)/sem_gt if sem_gt else 0.0,
          "core_correct_intersection_frames":core_inter,
          "core_false_positive_duration_frames":cor_pr-core_inter,
          "core_false_positive_duration_ratio":(cor_pr-core_inter)/cor_pr if cor_pr else 0.0,
          "core_missed_gt_duration_frames":cor_gt-core_inter,
          "core_missed_gt_duration_ratio":(cor_gt-core_inter)/cor_gt if cor_gt else 0.0
      },
      "per_phase":rows
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("ground_truth")
    ap.add_argument("--output",default=None)
    ap.add_argument("--tolerances",default="0.5,1.0,2.0")
    a=ap.parse_args()
    pred=json.load(open(a.annotations,encoding="utf-8")); gt=json.load(open(a.ground_truth,encoding="utf-8"))
    r=evaluate(pred,gt,[float(x) for x in a.tolerances.split(",") if x.strip()])
    s=json.dumps(r,ensure_ascii=False,indent=2); print(s)
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        Path(a.output).write_text(s+"\n",encoding="utf-8")
if __name__=="__main__": main()
