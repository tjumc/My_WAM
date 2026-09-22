#!/usr/bin/env python3
"""V3.3.2 proposal generation.

Low-level gripper transitions are weak proposal cues, not semantic boundaries.
Rapid repeated gripper transitions are clustered as one retry episode and only
promoted when supported by broader state/action change.
"""
import argparse, json, math
from pathlib import Path
import h5py, numpy as np, cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-12, None)


def quat_step_angle(q):
    q = quat_normalize(q)
    # q assumed xyzw. Relative angle can be obtained from |dot|.
    dots = np.sum(q[1:] * q[:-1], axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    angle = 2.0 * np.arccos(dots)
    return np.r_[0.0, angle]


def quat_pair_angle(q1, q2):
    q1 = quat_normalize(q1); q2 = quat_normalize(q2)
    dots = np.sum(q1 * q2, axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def safe_dt(t):
    d = np.diff(t, prepend=t[0])
    med = np.median(np.diff(t))
    d[0] = med
    return np.clip(d, max(1e-4, med * 0.2), med * 5.0)


def robust_z(x, clip=6.0):
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    scale = q75 - q25
    # Fallback for near-constant channels.
    mad = np.nanmedian(np.abs(x - med), axis=0, keepdims=True) * 1.4826
    std = np.nanstd(x, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0)))
    z = (x - med) / scale
    return np.clip(z, -clip, clip)


def smooth_1d(x, n):
    n = max(1, int(n))
    if n <= 1: return np.asarray(x, dtype=float)
    k = np.ones(n, dtype=float) / n
    return np.convolve(np.asarray(x, dtype=float), k, mode='same')


def multiscale_change(features, windows):
    T, D = features.shape
    out = []
    for w in windows:
        s = np.zeros(T, dtype=float)
        for i in range(w, T-w):
            left = features[i-w:i].mean(axis=0)
            right = features[i:i+w].mean(axis=0)
            s[i] = np.sqrt(np.mean((right-left)**2))
        out.append(s)
    M = np.stack(out, axis=1)
    # favor a boundary that is supported at any scale, but reward multi-scale agreement
    return 0.55*np.max(M, axis=1) + 0.45*np.mean(M, axis=1), M


def load_episode(path):
    with h5py.File(path, 'r') as f:
        t = f['time'][:].astype(float)
        data = {
            'time': t,
            'jvel': f['joints_dict/joints_velocity_state'][:].astype(float),
            'left_pose': f['poses_dict/astribot_arm_left'][:].astype(float),
            'right_pose': f['poses_dict/astribot_arm_right'][:].astype(float),
            'left_cmd_pose': f['command_poses_dict/astribot_arm_left'][:].astype(float),
            'right_cmd_pose': f['command_poses_dict/astribot_arm_right'][:].astype(float),
            'left_grip': f['poses_dict/astribot_gripper_left'][:,0].astype(float),
            'right_grip': f['poses_dict/astribot_gripper_right'][:,0].astype(float),
            'left_grip_cmd': f['command_poses_dict/astribot_gripper_left'][:,0].astype(float),
            'right_grip_cmd': f['command_poses_dict/astribot_gripper_right'][:,0].astype(float),
            'left_wrench': f['endpoint_wrench_dict/astribot_arm_left'][:].astype(float),
            'right_wrench': f['endpoint_wrench_dict/astribot_arm_right'][:].astype(float),
        }
    return data


def build_features(d):
    t = d['time']; dt = safe_dt(t)
    lp, rp = d['left_pose'], d['right_pose']
    lcp, rcp = d['left_cmd_pose'], d['right_cmd_pose']
    def pos_speed(p):
        return np.linalg.norm(np.diff(p[:,:3], axis=0, prepend=p[:1,:3]), axis=1) / dt
    def ang_speed(p):
        return quat_step_angle(p[:,3:7]) / dt
    def force_norm(w): return np.linalg.norm(w[:,:3],axis=1)
    def torque_norm(w): return np.linalg.norm(w[:,3:6],axis=1)
    fL, fR = force_norm(d['left_wrench']), force_norm(d['right_wrench'])
    tauL, tauR = torque_norm(d['left_wrench']), torque_norm(d['right_wrench'])
    # state/command mismatch (same task-space convention in this HDF5)
    l_pos_err = np.linalg.norm(lcp[:,:3] - lp[:,:3], axis=1)
    r_pos_err = np.linalg.norm(rcp[:,:3] - rp[:,:3], axis=1)
    l_rot_err = quat_pair_angle(lcp[:,3:7], lp[:,3:7])
    r_rot_err = quat_pair_angle(rcp[:,3:7], rp[:,3:7])
    # instantaneous derivatives
    def abs_rate(x): return np.abs(np.diff(x, prepend=x[0])) / dt
    names = [
        'base_speed','base_yaw_rate','torso_joint_speed',
        'left_ee_speed','right_ee_speed','left_ee_ang_speed','right_ee_ang_speed',
        'left_gripper_cmd','right_gripper_cmd',
        'left_force','right_force','left_torque','right_torque',
        'left_pos_track_err','right_pos_track_err','left_rot_track_err','right_rot_track_err',
        'left_force_change','right_force_change'
    ]
    X = np.stack([
        np.linalg.norm(d['jvel'][:,:2],axis=1), np.abs(d['jvel'][:,2]),
        np.linalg.norm(d['jvel'][:,3:7],axis=1),
        pos_speed(lp), pos_speed(rp), ang_speed(lp), ang_speed(rp),
        d['left_grip_cmd']/100.0, d['right_grip_cmd']/100.0,
        fL, fR, tauL, tauR,
        l_pos_err, r_pos_err, l_rot_err, r_rot_err,
        abs_rate(fL), abs_rate(fR)
    ], axis=1)
    # Mild smoothing of noisy continuous channels; leave gripper command largely intact.
    sm = X.copy()
    for j,nm in enumerate(names):
        win = 3 if 'gripper_cmd' in nm else (7 if 'force_change' in nm else 5)
        sm[:,j] = smooth_1d(X[:,j], win)
    return X, sm, names


def decode_frame(path, camera, idx):
    with h5py.File(path,'r') as f:
        sizes = f[f'images_dict/{camera}/rgb_size'][:].astype(np.int64)
        raw = f[f'images_dict/{camera}/rgb']
        offsets = np.r_[0, np.cumsum(sizes)]
        idx = int(np.clip(idx,0,len(sizes)-1))
        b = np.asarray(raw[offsets[idx]:offsets[idx+1]], dtype=np.uint8)
        im = cv2.imdecode(b, cv2.IMREAD_COLOR)
    return im


def make_contact_sheet(path, center, outpath, fps=30.0, offsets_sec=(-1.5,-0.75,0,0.75,1.5)):
    cams = ['head','left','right']
    indices = [int(round(center + s*fps)) for s in offsets_sec]
    rows=[]
    cell_w, cell_h = 320, 180
    for cam in cams:
        cells=[]
        for sec,idx in zip(offsets_sec,indices):
            im=decode_frame(path,cam,idx)
            im=cv2.resize(im,(cell_w,cell_h),interpolation=cv2.INTER_AREA)
            cv2.putText(im, f'{cam}  {sec:+.2f}s  f={max(0,idx)}', (8,20), cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
            cells.append(im)
        rows.append(np.concatenate(cells,axis=1))
    sheet=np.concatenate(rows,axis=0)
    cv2.imwrite(str(outpath),sheet)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('hdf5')
    ap.add_argument('--output',default=None)
    ap.add_argument('--fps',type=float,default=30.0)
    ap.add_argument('--top-k',type=int,default=24)
    ap.add_argument('--gripper-retry-sec',type=float,default=1.0)
    ap.add_argument('--gripper-context-thr',type=float,default=0.8)
    ap.add_argument('--max-gap-sec',type=float,default=6.0)
    ap.add_argument('--lerobot-episode',type=int,default=None)
    ap.add_argument('--lerobot-max-raw-frame',type=int,default=None)
    args=ap.parse_args()
    path=Path(args.hdf5)
    out=Path(args.output or (path.stem+'_analysis'))
    (out/'contact_sheets').mkdir(parents=True,exist_ok=True)
    d=load_episode(path)
    T=len(d['time']); rel=d['time']-d['time'][0]
    X, Xs, names=build_features(d)
    Z=robust_z(Xs)
    windows=[max(3,int(round(args.fps*s))) for s in (0.3,1.0,2.0)]
    cp, cp_scales=multiscale_change(Z,windows)
    # Event cues. Gripper transitions are deliberately weak: accidental
    # open/close actions and failed re-grasps must not become semantic boundaries.
    rg=(d['right_grip_cmd']>50).astype(float); lg=(d['left_grip_cmd']>50).astype(float)
    grip_imp=np.abs(np.diff(rg,prepend=rg[0]))+np.abs(np.diff(lg,prepend=lg[0]))
    force_change = Xs[:, names.index('left_force_change')] + Xs[:, names.index('right_force_change')]
    force_imp=np.clip(robust_z(force_change[:,None])[:,0],0,None)
    cp_z = robust_z(cp[:,None])[:,0]
    cue = smooth_1d(0.8*grip_imp + 0.18*force_imp, 5)
    score = smooth_1d(cp_z + cue, 5)

    peaks, props = find_peaks(score, distance=max(10,int(0.65*args.fps)), prominence=0.35)
    edge=int(1.5*args.fps)
    peaks=np.array([p for p in peaks if edge <= p < T-edge],dtype=int)
    allcand=list(peaks)

    # Cluster rapid gripper transitions into one retry episode. Only add a
    # cluster representative if non-gripper context also supports an event.
    grip_edges=[int(x) for x in np.flatnonzero(grip_imp>0.5) if edge <= x < T-edge]
    retry_gap=max(1,int(round(args.gripper_retry_sec*args.fps)))
    clusters=[]
    for g in grip_edges:
        if not clusters or g-clusters[-1][-1] > retry_gap:
            clusters.append([g])
        else:
            clusters[-1].append(g)
    gripper_clusters={}
    for cluster in clusters:
        best=max(cluster,key=lambda i: float(max(0.0,cp_z[i]) + 0.35*force_imp[i]))
        context=float(max(0.0,cp_z[best]) + 0.35*force_imp[best])
        near_peak=any(abs(best-p)<=int(round(0.30*args.fps)) for p in peaks)
        if context >= args.gripper_context_thr or near_peak:
            allcand.append(int(best))
            gripper_clusters[int(best)]={
                'size':len(cluster),
                'start_frame':int(cluster[0]),
                'end_frame':int(cluster[-1]),
                'context_score':round(context,4),
            }

    # Rank + NMS. Gripper edges receive no extra ranking bonus.
    allcand=sorted(set(allcand), key=lambda i: float(score[i]), reverse=True)
    selected=[]
    for i in allcand:
        if all(abs(i-j)>=int(0.5*args.fps) for j in selected):
            selected.append(i)
        if len(selected)>=args.top_k: break

    # Coverage guard: do not let top-k ranking completely miss an early action
    # or a long semantic interval. Add the strongest state/action-change point
    # in uncovered gaps, independent of raw gripper edges.
    selected=sorted(selected)
    max_gap=max(1,int(round(args.max_gap_sec*args.fps)))
    boundaries=[edge]+selected+[T-edge-1]
    extra=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if b-a > max_gap:
            lo=max(edge,a+int(round(0.5*args.fps)))
            hi=min(T-edge,b-int(round(0.5*args.fps)))
            if hi>lo:
                extra.append(int(lo+np.argmax(score[lo:hi+1])))
    selected=sorted(set(selected+extra))
    events=[]
    for eid,i in enumerate(selected):
        # explain strongest feature changes around 1s scale
        w=windows[1]
        left=Z[max(0,i-w):i].mean(axis=0)
        right=Z[i:min(T,i+w)].mean(axis=0)
        delta=np.abs(right-left)
        top_idx=np.argsort(delta)[::-1][:4]
        signals=[names[j] for j in top_idx if delta[j] > 0.35]
        cluster_meta=gripper_clusters.get(int(i))
        if cluster_meta:
            signals.insert(0,'gripper_retry_cluster' if cluster_meta['size']>1 else 'gripper_transition_supported')
        lerobot_frame=None
        if args.lerobot_episode is not None and (args.lerobot_max_raw_frame is None or i<=args.lerobot_max_raw_frame):
            lerobot_frame=int(i)
        e={
            'event_id':eid,'raw_frame':int(i),'time_sec':round(float(rel[i]),3),
            'score':round(float(score[i]),4),'signals':signals,
            'right_gripper_command':float(d['right_grip_cmd'][i]),
            'left_gripper_command':float(d['left_grip_cmd'][i]),
            'gripper_cluster':cluster_meta,
        }
        if args.lerobot_episode is not None:
            e['lerobot_episode_index']=int(args.lerobot_episode)
            e['lerobot_frame_index']=lerobot_frame
        events.append(e)
        make_contact_sheet(path,i,out/'contact_sheets'/f'event_{eid:02d}_f{i:04d}.jpg',args.fps)
    report={
        'source_hdf5':path.name,'num_raw_frames':T,'duration_sec':round(float(rel[-1]),3),
        'fps_nominal':args.fps,'feature_names':names,
        'change_windows_frames':windows,'candidate_events':events,
        'proposal_version':'v3.3.2',
        'note':'Gripper edges are weak cues; rapid retry/re-grasp transitions are clustered and require non-gripper context. Semantic verification is still performed downstream.'
    }
    with open(out/'candidate_events.json','w',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    # signals plot
    fig,axs=plt.subplots(5,1,figsize=(16,12),sharex=True)
    axs[0].plot(rel, Xs[:,names.index('base_speed')],label='base speed')
    axs[0].plot(rel, Xs[:,names.index('base_yaw_rate')],label='base yaw rate',alpha=.8); axs[0].legend()
    axs[1].plot(rel,Xs[:,names.index('left_ee_speed')],label='left EE speed')
    axs[1].plot(rel,Xs[:,names.index('right_ee_speed')],label='right EE speed'); axs[1].legend()
    axs[2].plot(rel,d['left_grip_cmd']/100,label='left gripper cmd')
    axs[2].plot(rel,d['right_grip_cmd']/100,label='right gripper cmd'); axs[2].legend()
    axs[3].plot(rel,Xs[:,names.index('left_force')],label='left force')
    axs[3].plot(rel,Xs[:,names.index('right_force')],label='right force'); axs[3].legend()
    axs[4].plot(rel,score,label='event score',lw=1.3)
    for e in events:
        for ax in axs: ax.axvline(e['time_sec'],ls='--',lw=.6,alpha=.35)
    axs[4].scatter([e['time_sec'] for e in events],[score[e['raw_frame']] for e in events],s=16)
    axs[4].legend(); axs[4].set_xlabel('time (s)')
    fig.suptitle(path.name)
    fig.tight_layout()
    fig.savefig(out/'signals.png',dpi=160)
    plt.close(fig)
    # make overview montage of contact sheets
    imgs=[]
    for e in events:
        fp=out/'contact_sheets'/f"event_{e['event_id']:02d}_f{e['raw_frame']:04d}.jpg"
        im=cv2.imread(str(fp))
        # downscale overview
        w=800; h=int(im.shape[0]*w/im.shape[1])
        im=cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA)
        canvas=np.zeros((h+28,w,3),np.uint8)
        canvas[28:]=im
        cv2.putText(canvas,f"event {e['event_id']:02d}  t={e['time_sec']:.2f}s  frame={e['raw_frame']}  {','.join(e['signals'][:2])}",(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        imgs.append(canvas)
    if imgs:
        cols=2; rows=math.ceil(len(imgs)/cols); hh=max(im.shape[0] for im in imgs); ww=max(im.shape[1] for im in imgs)
        blank=np.zeros_like(imgs[0]); imgs += [blank]*(rows*cols-len(imgs))
        grid=[]
        for r in range(rows): grid.append(np.concatenate(imgs[r*cols:(r+1)*cols],axis=1))
        cv2.imwrite(str(out/'events_overview.jpg'),np.concatenate(grid,axis=0))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': 
    main()
