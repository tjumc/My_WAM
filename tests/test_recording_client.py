"""Fake-SDK end-to-end tests: compare control calls with recording on/off."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/astribot'))
from rollout_recorder import RolloutRecorder, RecorderConfig


def load_client(monkeypatch):
    monkeypatch.setitem(sys.modules,'portal',SimpleNamespace())
    spec=importlib.util.spec_from_file_location('wam_record_test_client', ROOT/'experiments/astribot/real_robot_client.py')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_case(monkeypatch, recorder, *, blend=0, mode='cmd_absolute_joint_vel', failure=None, source_path=None):
    m=load_client(monkeypatch)
    if source_path is not None:
        # Optional local parity check against the exact pre-change GitHub file.
        exec(compile(Path(source_path).read_text(), str(source_path), 'exec'), m.__dict__)
    calls=[]
    images=[np.full((4,6,3),i,dtype=np.uint8) for i in (1,2,3)]
    groups=[np.arange(d,dtype=np.float32)/10 for d in (3,4,7,1,7,1,2)]
    class Clock:
        t=100.
        @classmethod
        def time(cls): return cls.t
        @classmethod
        def perf_counter(cls): return cls.t
        @classmethod
        def sleep(cls,s): cls.t+=s; calls.append(('sleep',s))
    class Future:
        def __init__(self,value): self.value=value
        def result(self,**kwargs): return self.value
    class RPC:
        connected=True
        attempts=0
        returns=0
        def get_model_info(self):
            return Future(dict(action_mode=mode,chunk_size=4,n_action_steps=2,action_dim=25,state_dim=25,use_effort=False))
        def infer(self,t,*inputs):
            self.attempts+=1
            calls.append(('infer_inputs', [a.copy().tolist() for a in inputs[:4]]))
            Clock.t+=.1
            if failure=='rpc' and self.attempts==1: raise RuntimeError('rpc failure')
            if failure=='none' and self.attempts==1: return Future(None)
            return Future(np.arange(4*25,dtype=np.float32).reshape(4,25)/100)
        def close(self): calls.append(('rpc_close',))
    rpc=RPC()
    class Robot:
        chassis_name='base'; torso_name='torso'; arm_left_name='left';effector_left_name='lg'
        arm_right_name='right';effector_right_name='rg'
        def __init__(self,**kwargs): calls.append(('robot_init',kwargs))
        def move_to_home(self): calls.append(('home',))
        def get_cameras_info(self): return {k:{'activate':True} for k in m.CAMERAS}
        def register_image_callback(self,*args,**kw): return None
        def set_head_follow_effector(self,**kw): calls.append(('head',kw))
        def get_desired_joints_position(self,**kw): return [np.array([.1,.2,.3])]
        def get_current_joints_position(self): calls.append(('read_position',));return [a.copy() for a in groups]
        def get_current_joints_velocity(self,*args): calls.append(('read_velocity',));return [np.array([.02,.03,.04])]
        def move_joints_position(self,names,wp): calls.append(('initial_position',names,wp))
        def move_joints_waypoints(self,names,wp,times,**kw):
            calls.append(('waypoints',names,wp,times,kw));Clock.t+=.2
            if failure=='sdk': raise RuntimeError('sdk failure')
            if failure=='interrupt': raise KeyboardInterrupt()
            for i,a in enumerate(wp[-1]): groups[i]=np.array(a,dtype=np.float32)
            rpc.returns+=1
            if rpc.returns>=2: rpc.connected=False
    module=ModuleType('astribot_sdk.core.astribot_api.astribot_client');module.Astribot=Robot
    for name in ('astribot_sdk','astribot_sdk.core','astribot_sdk.core.astribot_api'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    monkeypatch.setitem(sys.modules,'astribot_sdk.core.astribot_api.astribot_client',module)
    m.portal=SimpleNamespace(Client=lambda endpoint:rpc)
    m.time=Clock
    def get_images(*,return_metadata=False):
        if return_metadata:
            return tuple(images), [dict(camera=k,received_wall_time_s=Clock.t,
                received_monotonic_s=Clock.t,age_at_selection_s=0.) for k in m.CAMERAS]
        return tuple(images)
    m._wait_for_fresh_images=get_images
    args=SimpleNamespace(server_host='localhost',server_port=2222,execute_steps=2,
        control_hz=50.,action_dt=1/30,boundary_blend_steps=blend,max_runtime=1000.,record_dir=None)
    if source_path is not None:
        m.parse_args=lambda:args
        m.main()
    else:
        m._run_client(args,recorder)
    return calls


@pytest.mark.parametrize('mode',['cmd_absolute_joint_vel','cmd_absolute_joint'])
@pytest.mark.parametrize('blend',[0,2])
def test_same_sdk_calls_and_observations_recording_on_off(tmp_path,monkeypatch,mode,blend):
    off=run_case(monkeypatch,None,mode=mode,blend=blend)
    r=RolloutRecorder(RecorderConfig(root=str(tmp_path),queue_size=64,min_free_bytes=0),{})
    try: on=run_case(monkeypatch,r,mode=mode,blend=blend)
    finally: result=r.close()
    assert on==off
    assert not result['writer_still_alive'] and result['counts']['attempted']==result['counts']['written']
    ev=[json.loads(line) for line in (r.path/'events.jsonl').read_text().splitlines()]
    observations=[e for e in ev if e['event']=='observation']
    assert [e['metadata']['feedback_after_request'] for e in observations]==[None,1,2]
    returns=[e for e in ev if e['event']=='sdk_waypoints_return']
    assert len(returns)==2 and all(not e['metadata']['physical_completion_verified'] for e in returns)
    prediction=next(e for e in ev if e['event']=='prediction')
    with np.load(r.path/prediction['arrays_file'],allow_pickle=False) as a:
        assert a['actions_raw25'].shape==(4,25)
    command=next(e for e in ev if e['event']=='sdk_waypoints_attempt')
    with np.load(r.path/command['arrays_file'],allow_pickle=False) as a:
        assert a['selected_actions_after_blend25'].shape==(2,25)
        assert a['sdk_waypoints_position23'].shape==(2,23)
    assert next(e for e in ev if e['event']=='loop_end')['metadata']['reason']=='server_disconnected'


@pytest.mark.parametrize('failure',['rpc','sdk','interrupt','none'])
def test_error_interrupt_and_none_are_explicit(tmp_path,monkeypatch,failure):
    r=RolloutRecorder(RecorderConfig(root=str(tmp_path),queue_size=64,min_free_bytes=0),{})
    try:
        if failure in ('rpc','sdk'):
            with pytest.raises(RuntimeError): run_case(monkeypatch,r,failure=failure)
        else: run_case(monkeypatch,r,failure=failure)
    finally: r.close()
    ev=[json.loads(line) for line in (r.path/'events.jsonl').read_text().splitlines()]
    names={e['event'] for e in ev}
    assert 'loop_end' in names
    expected={'rpc':'loop_error','sdk':'loop_error','interrupt':'loop_interrupted','none':'rpc_none'}
    assert expected[failure] in names
    if failure in ('sdk','interrupt'):
        assert 'sdk_waypoints_return' not in names
        assert next(e for e in ev if e['event']=='loop_end')['metadata']['command_without_next_observation']==1


def test_preflight_failure_happens_before_sdk(monkeypatch,tmp_path):
    m=load_client(monkeypatch)
    monkeypatch.setattr(sys,'argv',['client','--record_dir',str(tmp_path),'--record_queue_size','0'])
    with pytest.raises(ValueError):m.main()


def test_image_times_belong_to_same_snapshot(monkeypatch):
    m=load_client(monkeypatch)
    now=m.time.time()
    for i,k in enumerate(m.CAMERAS):
        m.images_dict[k]={'data':np.full((2,2,3),i),'timestamp':now,'received_monotonic_s':i+1}
    imgs,meta=m._wait_for_fresh_images(return_metadata=True)
    assert [int(a[0,0,0]) for a in imgs]==[0,1,2]
    assert [a['received_monotonic_s'] for a in meta]==[1,2,3]


def test_default_recording_disabled(monkeypatch):
    m=load_client(monkeypatch)
    monkeypatch.setattr(sys,'argv',['client'])
    assert m.parse_args().record_dir is None
