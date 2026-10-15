"""Synthetic catalogue/numeric/preview tests, not server data validation."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_astribot_catalog as m


def catalog(tmp, roots):
    path=tmp/'catalog with spaces.sh'
    tasks=list(dict.fromkeys(t for t,_ in roots))
    groups={t:[str(r) for task,r in roots if task==t] for t in tasks}
    text="declare -ag ASTRIBOT_DATASET_TASKS=("+' '.join(tasks)+")\n"
    text+="DATASET_ROOTPATH='/old/data'\nBENCHMARK_ROOTPATH='/old/benchmark'\ndeclare -Ag ASTRIBOT_DATASET_CATALOG=()\n"
    for task, paths in groups.items():
        text+=f"ASTRIBOT_DATASET_CATALOG[{task}]='"+'\n'.join(paths)+"'\n"
    path.write_text(text)
    return path


def dataset(tmp, name='wash_fx_20260101_compressed',dim=25,length=10):
    root=tmp/name
    (root/'meta').mkdir(parents=True)
    info=dict(codebase_version='v3.0',fps=30,total_frames=length,total_episodes=1,
              features={k:dict(dtype='float32',shape=[dim]) for k in m.FIELDS},
              video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    (root/'meta/info.json').write_text(json.dumps(info))
    ep=dict(episode_index=7,length=length)
    (root/'meta/episodes.jsonl').write_text(json.dumps(ep)+'\n')
    return root,info,ep


def batch(ep=7,dim=25,start=0,stop=10):
    f=np.arange(start,stop)
    s=np.repeat(f[:,None],dim,axis=1)*.02
    return dict(episode_index=[ep]*len(f),frame_index=f.tolist(),timestamp=(f/30).tolist(),
                **{'observation.state':s.tolist(),'action':(s+.1).tolist(),'task_index':[0]*len(f)})


def add(scan, directory, b, epsilon=1e-4):
    scan.add(np.array(b['frame_index']),np.array(b['timestamp']),
             {k:np.array(b[k]) for k in m.FIELDS},b.get('task_index'),directory,epsilon,.002)


def args(**kw):
    d=dict(numeric=True,preview_episodes=0,seed=42,batch_size=4,
           low_change_epsilon=1e-4,timestamp_tolerance=.002,min_free_gb=0,
           max_cameras=3,preview_frames=5,thumb_width=120)
    return SimpleNamespace(**(d|kw))


def entry(root,task='wash'):
    return dict(task=task,root=str(root.resolve()),catalog_path=str(root),dataset_id=m.sha(root/'meta/info.json')[:16])


def test_catalog_all_tasks_order_and_source_unchanged(tmp_path):
    cat=catalog(tmp_path,[('wash','/old/data/b'),('wash','/old/data/a'),('blocks','/old/benchmark/c')])
    original=cat.read_bytes()
    rows=m.read_catalog(cat)
    assert [r['task'] for r in rows]==['wash','wash','blocks']
    assert [Path(r['root']).name for r in rows]==['b','a','c']
    assert cat.read_bytes()==original


def test_catalog_filter_and_separate_mounts(tmp_path):
    cat=catalog(tmp_path,[('wash','/old/data/x'),('blocks','/old/benchmark/y')])
    rows=m.read_catalog(cat,data_root=tmp_path/'new data',benchmark_root=tmp_path/'new benchmark')
    assert rows[0]['root']==str(tmp_path/'new data/x')
    assert rows[1]['root']==str(tmp_path/'new benchmark/y')
    assert m.read_catalog(cat,['blocks'])[0]['root']=='/old/benchmark/y'
    with pytest.raises(ValueError,match='Unknown'): m.read_catalog(cat,['absent'])


def test_folder_names_only_hints():
    assert m.name_hints('fridge_shrimp_fx_20260424_2_compressed')['collector_hint']=='fx'
    assert m.name_hints('task01_sort_the_blocks_state_cmd')['date_hint']==''
    assert 'subtask' not in m.name_hints('wash_clothes_pickup_bxx_20260617_compressed')


def test_metadata_nonconsecutive_episode_ids(tmp_path):
    root,_,_=dataset(tmp_path)
    info,eps,_,sources=m.read_metadata(root)
    assert list(eps)==[7] and info['fps']==30
    assert sources[0].name=='episodes.jsonl'


def test_metadata_duplicate_ids_rejected(tmp_path):
    root,_,_=dataset(tmp_path)
    path=root/'meta/episodes.jsonl'
    path.write_text(path.read_text()*2)
    with pytest.raises(ValueError,match='duplicate'): m.read_metadata(root)


def test_metadata_tasks_json_without_stats_file(tmp_path):
    root,_,_=dataset(tmp_path)
    (root/'meta/tasks.jsonl').write_text('{"task_index":0,"task":"pick up"}\n')
    _,_,texts,_=m.read_metadata(root)
    assert texts[0]['task']=='pick up'
    assert not (root/'meta/stats.json').exists()


def test_moments_matches_numpy():
    x=np.random.default_rng(4).normal(size=(31,3))
    acc=m.Moments(3)
    acc.add(x[:10],np.diff(x[:10],axis=0),1e-4)
    acc.add(x[10:],np.diff(x[9:],axis=0),1e-4)
    np.testing.assert_allclose(acc.mean,x.mean(0))
    np.testing.assert_allclose(np.sqrt(acc.m2/acc.n),x.std(0))
    assert acc.steps==30


def test_numeric_hash_independent_of_batch_boundaries_and_episode_id():
    a=m.EpisodeScan(7,10,30)
    b=m.EpisodeScan(111,10,30)
    add(a,{},batch())
    d={}
    add(b,d,batch(ep=111,stop=4));add(b,d,batch(ep=111,start=4))
    assert a.summary()['numeric_valid'] and b.summary()['numeric_valid']
    assert a.summary()['numeric_sequence_sha256']==b.summary()['numeric_sequence_sha256']
    assert d['action'].steps==9


def test_hash_detects_changed_numeric_content():
    a=m.EpisodeScan(7,10,30);b=m.EpisodeScan(7,10,30)
    bb=batch();bb['action'][4][0]+=.1
    add(a,{},batch());add(b,{},bb)
    assert a.summary()['numeric_sequence_sha256']!=b.summary()['numeric_sequence_sha256']


@pytest.mark.parametrize('kind',['missing','duplicate','out_of_range','nonfinite','bad_time'])
def test_invalid_sequences_not_certified(kind):
    b=batch()
    if kind=='missing':
        for k in b:b[k].pop(3)
    if kind=='duplicate':b['frame_index'][3]=2
    if kind=='out_of_range':b['frame_index'][-1]=100
    if kind=='nonfinite':b['action'][2][0]=float('nan')
    if kind=='bad_time':b['timestamp'][3]+=.05
    scan=m.EpisodeScan(7,10,30)
    add(scan,{},b)
    assert not scan.summary()['numeric_valid']
    assert scan.summary()['numeric_sequence_sha256'] is None


def test_cross_batch_reverse_order_flagged_not_silently_sorted():
    scan=m.EpisodeScan(7,10,30);directory={}
    add(scan,directory,batch(start=5));add(scan,directory,batch(stop=5))
    assert not scan.summary()['ordered_in_storage']
    assert scan.summary()['numeric_sequence_sha256'] is None


def test_no_cross_episode_motion_differences():
    directory={}
    for ep in [7,9]:
        scan=m.EpisodeScan(ep,10,30)
        add(scan,directory,batch(ep))
    assert directory['action'].steps==18


def test_constant_velocity_is_only_low_change_not_stop_label():
    b=batch(dim=3)
    b['observation.state']=[[.5,.0,.0]]*10
    scan=m.EpisodeScan(7,10,30);add(scan,{},b)
    summary=scan.summary()
    assert summary['longest_low_state_change_seconds']==pytest.approx(9/30)
    assert not any(k in summary for k in ['stopped','success','completed'])


def test_dimension_agnostic_names():
    assert m.field_names({},'action',32)==[f'dim_{i:03d}' for i in range(32)]
    assert m.field_names({'features':{'action':{'names':['x','y']}}},'action',2)==['x','y']


def test_preview_sampling_stable_and_never_exceeds_available():
    eps={i:{} for i in range(20)}
    assert m.choose_previews(eps,'abc',3,42)==m.choose_previews(eps,'abc',3,42)
    assert len(m.choose_previews({9:{}},'abc',3,42))==1
    assert m.choose_previews(eps,'abc',0,42)==[]


def test_camera_alias_order_and_metadata_missing_feature(tmp_path):
    root,info,ep=dataset(tmp_path)
    info['features'].update({f'observation.images.{k}':{'dtype':'video'} for k in ['right','left','head']})
    ep['videos/extra/from_timestamp']=0
    assert m.camera_keys(root,info,{7:ep})[:3]==[f'observation.images.{k}' for k in ['head','left','right']]
    assert 'extra' in m.camera_keys(root,info,{7:ep})


def test_v3_video_timestamp_offset_and_no_zero_fallback(tmp_path):
    root,info,ep=dataset(tmp_path)
    camera='observation.images.head'
    prefix=f'videos/{camera}/'
    ep.update({prefix+'from_timestamp':100.,prefix+'to_timestamp':101.,prefix+'chunk_index':1,prefix+'file_index':2})
    path=root/f'videos/{camera}/chunk-001/file-002.mp4';path.parent.mkdir(parents=True);path.touch()
    assert m.video_query(root,info,ep,camera,.2)==(path,100.2)
    with pytest.raises(ValueError):m.video_query(root,info,ep,camera,2.)
    del ep[prefix+'from_timestamp']
    with pytest.raises(ValueError,match='Missing v3'):m.video_query(root,info,ep,camera,.2)


def test_v2_video_path_and_boundary(tmp_path):
    root,info,ep=dataset(tmp_path)
    info.update(codebase_version='v2.1',chunks_size=1000,
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    path=root/'videos/chunk-000/head/episode_000007.mp4';path.parent.mkdir(parents=True);path.touch()
    assert m.video_query(root,info,ep,'head',.2)==(path,.2)


def test_pyav_seek_selects_closest_and_enforces_tolerance(monkeypatch):
    class Frame:
        time_base=1/30
        def __init__(self,n):self.pts=n
        def to_image(self):return Image.new('RGB',(80,60),(self.pts,0,0))
    class Reader:
        streams=SimpleNamespace(video=[SimpleNamespace(time_base=1/30)])
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def seek(self,*a,**kw):pass
        def decode(self,*a):return iter([Frame(i) for i in range(10)])
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=lambda p:Reader()))
    assert m.decode_nearest('x',.1,.002).getpixel((0,0))==(3,0,0)
    with pytest.raises(ValueError,match='mismatch'):m.decode_nearest('x',.11,.002)


def test_contact_sheet_uses_real_timestamps_and_errors_are_visible(tmp_path,monkeypatch):
    root,info,ep=dataset(tmp_path)
    calls=[]
    monkeypatch.setattr(m,'video_query',lambda root,info,ep,c,t: (Path('x'),t+100))
    monkeypatch.setattr(m,'decode_nearest',lambda p,t,tol: (calls.append(t) or Image.new('RGB',(64,48))))
    problems=[]
    result=m.make_sheet(root,info,ep,{0:0.,4:4/30,9:.3},['head'],tmp_path/'sheet.jpg',3,120,.002,lambda *x:problems.append(x))
    assert result==3 and calls==[100.,100+4/30,100.3] and not problems
    assert Image.open(tmp_path/'sheet.jpg').width==360


def test_previews_not_guessed_if_timestamps_missing(tmp_path,monkeypatch):
    root,info,ep=dataset(tmp_path);problems=[]
    count=m.make_sheet(root,info,ep,{},['head'],tmp_path/'sheet.jpg',3,120,.002,lambda *x:problems.append(x))
    assert count==0 and len(problems)==3


def test_output_protection_and_git_ignored_details(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    with pytest.raises(ValueError):m.initialize_output(raw/'new',{raw})
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    out=m.initialize_output(repo/'audit',set())
    (out/'report.md').write_text('report')
    (out/'_local/a.jpg').touch();(out/'run.log').touch()
    check=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=repo,capture_output=True,text=True,check=True).stdout
    assert 'report.md' in check and '.gitignore' in check
    assert '_local' not in check and 'run.log' not in check
    with pytest.raises(FileExistsError):m.initialize_output(out,set())


def test_full_mocked_numeric_audit_25_and_32_dims(tmp_path,monkeypatch):
    root,_,_=dataset(tmp_path,dim=32)
    out=m.initialize_output(tmp_path/'audit',set())
    monkeypatch.setattr(m,'iter_numeric_batches',lambda r,b:iter([(r/'data/f.parquet',batch(dim=32))]))
    row,episodes,numeric,signatures,_,issues=m.audit_directory(entry(root),args(),out)
    assert row['status']=='ok' and not issues
    assert row['state_dim_observed']==32
    assert row['observed_numeric_rows']==10
    assert episodes[0]['numeric_valid'] and len(numeric)==64 and len(signatures)==1


def test_numeric_failure_retains_partial_and_continues(tmp_path,monkeypatch):
    root,_,_=dataset(tmp_path)
    out=m.initialize_output(tmp_path/'audit',set())
    def failing(r,b):
        yield r/'data/f.parquet',batch(stop=5)
        raise OSError('bad second file')
    monkeypatch.setattr(m,'iter_numeric_batches',failing)
    row,eps,numeric,signatures,_,issues=m.audit_directory(entry(root),args(),out)
    assert row['numeric_status']=='partial' and not eps[0]['numeric_valid'] and not signatures
    assert any('bad second file' in i['message'] for i in issues)
    assert all(r['scan_status']=='partial' for r in numeric)


def test_duplicate_candidates_not_claimed_as_duplicate_videos():
    sig=[dict(task='wash',dataset_id=s,episode=7,frames=10,numeric_sequence_sha256='abc') for s in ['a','b']]
    result=m.duplicate_candidates(sig)
    assert len(result)==2 and result[0]['group_size']==2
    assert 'RGB not verified' in result[0]['scope']


def test_main_catalogue_all_dirs_missing_and_present(tmp_path,monkeypatch):
    root,_,_=dataset(tmp_path)
    cat=catalog(tmp_path,[('wash',root),('wash',tmp_path/'missing'),('blocks',root)])
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(m,'iter_numeric_batches',lambda r,b:iter([(r/'data/f.parquet',batch())]))
    out=tmp_path/'audit'
    code=m.main(['--catalog',str(cat),'--output-dir',str(out),'--preview-episodes','0'])
    assert code==2
    complete=json.loads((out/'COMPLETE.json').read_text())
    assert complete['directories_scanned']==2 and complete['catalog_entries']==3
    assert not complete['split_created'] and not complete['model_loaded']
    assert all(p.read_bytes()==b for p,b in before.items())
    assert (out/'_local/index.html').is_file()
    text=(out/'report.md').read_text()
    assert '不是' in text and '恒定非零底盘速度' in text


def test_metadata_only_does_not_read_parquet_data(tmp_path,monkeypatch):
    root,_,_=dataset(tmp_path)
    cat=catalog(tmp_path,[('wash',root)])
    monkeypatch.setattr(m,'iter_numeric_batches',lambda *x:pytest.fail('numeric data should not be read'))
    assert m.main(['--catalog',str(cat),'--output-dir',str(tmp_path/'out'),'--no-numeric','--preview-episodes','0'])==0


def test_launcher_syntax_defaults_and_unchanged_source_paths():
    script=Path(__file__).resolve().parents[1]/'scripts/audit_astribot_catalog.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    text=script.read_text()
    assert 'TASKS=()' in text and 'PREVIEW_EPISODES_PER_DIR=3' in text
    assert '\ngit push ' not in text and '\ngit reset ' not in text
    assert 'CATALOG="configs/data/astribot_dataset_catalog.sh"' in text


def test_real_parquet_end_to_end_when_available(tmp_path):
    pa=pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    root,_,_=dataset(tmp_path)
    (root/'data/chunk-000').mkdir(parents=True)
    pq.write_table(pa.table(batch()),root/'data/chunk-000/file-000.parquet')
    out=m.initialize_output(tmp_path/'out',set())
    row,_,_,_,_,issues=m.audit_directory(entry(root),args(),out)
    assert row['status']=='ok' and not issues


def fake_parquet(monkeypatch, parquet_module):
    pa=ModuleType('pyarrow');pa.__path__=[];pa.parquet=parquet_module
    monkeypatch.setitem(sys.modules,'pyarrow',pa)
    monkeypatch.setitem(sys.modules,'pyarrow.parquet',parquet_module)


def test_v3_parquet_metadata_filters_stats_columns(tmp_path,monkeypatch):
    root,_,ep=dataset(tmp_path)
    (root/'meta/episodes.jsonl').unlink()
    location=root/'meta/episodes/chunk-000/file-000.parquet'
    location.parent.mkdir(parents=True);location.touch()
    calls=[]
    class File:
        schema_arrow=SimpleNamespace(names=['episode_index','length','videos/head/from_timestamp','stats/action/mean'])
        def __init__(self,path):pass
        def read(self,columns):
            calls.append(columns)
            return SimpleNamespace(to_pylist=lambda:[ep|{'videos/head/from_timestamp':15.}])
    pq=ModuleType('pyarrow.parquet');pq.ParquetFile=File
    fake_parquet(monkeypatch,pq)
    _,eps,_,_=m.read_metadata(root)
    assert eps[7]['videos/head/from_timestamp']==15
    assert calls==[['episode_index','length','videos/head/from_timestamp']]


def test_iter_numeric_batches_does_not_read_video_columns(tmp_path,monkeypatch):
    path=tmp_path/'data/chunk-000/file-000.parquet'
    path.parent.mkdir(parents=True);path.touch()
    calls=[]
    class File:
        schema_arrow=SimpleNamespace(names=list(batch())+['observation.images.head'])
        def __init__(self,path):pass
        def iter_batches(self,**kw):
            calls.append(kw)
            yield SimpleNamespace(to_pydict=lambda:batch())
    pq=ModuleType('pyarrow.parquet');pq.ParquetFile=File
    fake_parquet(monkeypatch,pq)
    assert len(list(m.iter_numeric_batches(tmp_path,4)))==1
    assert 'observation.images.head' not in calls[0]['columns']
    assert calls[0]['batch_size']==4


def test_tasks_parquet_takes_priority_over_jsonl(tmp_path,monkeypatch):
    root,_,_=dataset(tmp_path)
    (root/'meta/tasks.jsonl').write_text('{"task":"old"}\n')
    (root/'meta/tasks.parquet').touch()
    pq=ModuleType('pyarrow.parquet')
    pq.read_table=lambda p:SimpleNamespace(to_pylist=lambda:[{'task_index':0,'__index_level_0__':'new'}])
    fake_parquet(monkeypatch,pq)
    assert m.read_metadata(root)[2][0]['__index_level_0__']=='new'


def test_tolerance_not_loosened_for_high_fps(tmp_path):
    root,info,_=dataset(tmp_path)
    info['fps']=1000
    (root/'meta/info.json').write_text(json.dumps(info))
    out=m.initialize_output(tmp_path/'out',set())
    row,_,_,_,_,issues=m.audit_directory(entry(root),args(),out)
    assert row['status']=='error' and 'half a data frame' in issues[0]['message']


def test_full_shell_launcher_metadata_smoke(tmp_path):
    import shutil
    root,_,_=dataset(tmp_path)
    cat=catalog(tmp_path,[('wash',root)])
    repo=tmp_path/'repo'
    (repo/'scripts').mkdir(parents=True);(repo/'tools').mkdir()
    original=Path(__file__).resolve().parents[1]
    shutil.copy2(original/'tools/audit_astribot_catalog.py',repo/'tools')
    script=(original/'scripts/audit_astribot_catalog.sh').read_text()
    script=script.replace('CATALOG="configs/data/astribot_dataset_catalog.sh"',f'CATALOG="{cat}"')
    script=script.replace('PYTHON_BIN="python"',f'PYTHON_BIN="{sys.executable}"')
    script=script.replace('NUMERIC_SCAN=true','NUMERIC_SCAN=false').replace('PREVIEW_EPISODES_PER_DIR=3','PREVIEW_EPISODES_PER_DIR=0')
    path=repo/'scripts/audit_astribot_catalog.sh';path.write_text(script)
    result=subprocess.run(['bash',str(path)],capture_output=True,text=True,timeout=20)
    assert result.returncode==0, result.stderr+result.stdout
    complete=list(repo.glob('ambiguity_outputs/dataset_audit/*/COMPLETE.json'))
    assert len(complete)==1 and json.loads(complete[0].read_text())['status']=='complete'
    assert (complete[0].parent/'launcher.sh').is_file()
