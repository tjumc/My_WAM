#!/usr/bin/env python3
"""Read-only catalogue audit. No WAM, normalization, training, split or robot I/O.

Source the trusted repository Bash catalogue, inspect local LeRobot v2/v3
metadata, stream numerical Parquet columns, and optionally seek a few images.
Detailed episode/joint records and contact sheets are LOCAL and Git-ignored.
Numeric hashes detect exact numeric sequences only, not duplicate RGB videos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

VERSION = 1
FIELDS = ('observation.state', 'action')
CATALOG_BASH = r'''
set -euo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { echo 'Bash >= 4 required' >&2; exit 2; }
source "$1" >/dev/null
printf '%s\0' 'WAM_CATALOG_V1' "${DATASET_ROOTPATH-}" "${BENCHMARK_ROOTPATH-}"
for task in "${ASTRIBOT_DATASET_TASKS[@]}"; do
    while IFS= read -r directory; do
        [[ -z "$directory" ]] || printf '%s\0%s\0' "$task" "$directory"
    done <<< "${ASTRIBOT_DATASET_CATALOG[$task]}"
done
'''
UPLOAD_FILES = ('.gitignore', 'report.md', 'directories.csv', 'tasks.csv',
                'numeric_summary.csv', 'duplicate_candidates.csv', 'issues.csv',
                'review.csv', 'manifest.json', 'COMPLETE.json', 'FAILED.json')


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path: Path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row)) or ['note']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def sha(path):
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def read_catalog(path, tasks=(), data_root=None, benchmark_root=None):
    """Execute only the user-selected, trusted .sh. Paths are argv, never eval."""
    path = Path(path).expanduser().resolve(strict=True)
    proc = subprocess.run(['bash', '--noprofile', '--norc', '-c', CATALOG_BASH,
                           'catalog-audit', str(path)], capture_output=True, check=True, timeout=30)
    parts = proc.stdout.decode('utf-8').split('\0')
    if parts[-1] != '' or len(parts) < 4 or parts[0] != 'WAM_CATALOG_V1' or (len(parts)-4) % 2:
        raise ValueError('Catalogue output malformed; expected task/path NUL records')
    old_data, old_benchmark = parts[1:3]
    pairs = list(zip(parts[3:-1:2], parts[4:-1:2]))
    known = list(dict.fromkeys(t for t, _ in pairs))
    if set(tasks) - set(known):
        raise ValueError(f'Unknown tasks {set(tasks)-set(known)}; available: {known}')
    rows = []
    for task, root in pairs:
        if tasks and task not in tasks:
            continue
        original = Path(root).expanduser().resolve()
        resolved = original
        for old, new in ((old_data, data_root), (old_benchmark, benchmark_root)):
            if old and new:
                try:
                    relative = original.relative_to(Path(old).expanduser().resolve())
                except ValueError:
                    continue
                resolved = Path(new).expanduser().resolve() / relative
                break
        dataset_id = hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
        rows.append(dict(task=task, dataset_id=dataset_id, catalog_path=root, root=str(resolved)))
    if not rows:
        raise ValueError('Catalogue selection is empty')
    return rows


def name_hints(name):
    # Hints ONLY: neither a verified operator ID nor an inferred task subtype.
    match = re.search(r'_(?P<person>[A-Za-z]+)_(?P<date>\d{8})(?:_\d+)?(?:_compressed)?$', name)
    return {'collector_hint': match['person'] if match else '',
            'date_hint': match['date'] if match else '', 'directory_name': name}


def read_metadata(root):
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    fps = float(info.get('fps', 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Missing/invalid metadata fps')
    meta_json = root / 'meta/episodes.jsonl'
    sources = []
    if meta_json.is_file():
        sources = [meta_json]
        records = [json.loads(s) for s in meta_json.read_text(encoding='utf-8').splitlines() if s.strip()]
    else:
        sources = sorted((root / 'meta/episodes').rglob('*.parquet'))
        if not sources:
            raise FileNotFoundError('No meta/episodes.jsonl or meta/episodes/**/*.parquet')
        import pyarrow.parquet as pq
        records = []
        for path in sources:
            # v3 episode metadata can contain large per-episode statistics.
            pf = pq.ParquetFile(path)
            columns = [k for k in pf.schema_arrow.names if not k.startswith('stats/')]
            records.extend(pf.read(columns=columns).to_pylist())
    episodes = {}
    for record in records:
        ep = int(record['episode_index'])
        if ep < 0 or ep in episodes or int(record['length']) <= 0:
            raise ValueError('Invalid/duplicate episode_index or nonpositive length')
        episodes[ep] = record
    if not episodes:
        raise ValueError('Empty episode metadata')
    instructions = []
    # Match repository priority: tasks.parquet first, then tasks.jsonl.
    if (root / 'meta/tasks.parquet').is_file():
        import pyarrow.parquet as pq
        instructions = pq.read_table(root / 'meta/tasks.parquet').to_pylist()
        sources.append(root / 'meta/tasks.parquet')
    elif (root / 'meta/tasks.jsonl').is_file():
        instructions = [json.loads(s) for s in (root / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        sources.append(root / 'meta/tasks.jsonl')
    return info, episodes, instructions, sources


def iter_numeric_batches(root, batch_size):
    """Each numeric file read once; never load an image column into the audit."""
    import pyarrow.parquet as pq
    paths = sorted((root / 'data').rglob('*.parquet'))
    if not paths:
        raise FileNotFoundError('No data/**/*.parquet')
    required = ['episode_index', 'frame_index', 'timestamp', *FIELDS]
    for path in paths:
        pf = pq.ParquetFile(path)
        missing = set(required) - set(pf.schema_arrow.names)
        if missing:
            raise ValueError(f'{path.name}: missing numeric columns {sorted(missing)}')
        columns = required + (['task_index'] if 'task_index' in pf.schema_arrow.names else [])
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield path, batch.to_pydict()


class Moments:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.lo = np.full(dim, np.inf)
        self.hi = np.full(dim, -np.inf)
        self.steps = 0
        self.step_sum = np.zeros(dim)
        self.step_max = np.zeros(dim)
        self.low_steps = np.zeros(dim, dtype=np.int64)

    def add(self, x, differences, epsilon):
        # Reject nonfinite input before calling; do not silently drop individual dimensions.
        if len(x):
            n = len(x)
            mean = x.mean(axis=0)
            delta = mean - self.mean
            total = self.n + n
            self.m2 += ((x - mean)**2).sum(axis=0) + delta**2 * self.n*n/total
            self.mean += delta*n/total
            self.n = total
            self.lo = np.minimum(self.lo, x.min(axis=0))
            self.hi = np.maximum(self.hi, x.max(axis=0))
        if len(differences):
            self.steps += len(differences)
            d = np.abs(differences)
            self.step_sum += d.sum(axis=0)
            self.step_max = np.maximum(self.step_max, d.max(axis=0))
            self.low_steps += (d <= epsilon).sum(axis=0)

    def rows(self, common, names):
        return [{**common, 'dim': i, 'field_name': names[i], 'samples': self.n,
                 'min_raw': float(self.lo[i]), 'max_raw': float(self.hi[i]),
                 'mean_raw': float(self.mean[i]), 'std_raw': float(np.sqrt(max(self.m2[i]/self.n, 0))),
                 'adjacent_steps': self.steps,
                 'mean_abs_step_raw': float(self.step_sum[i]/self.steps) if self.steps else None,
                 'max_abs_step_raw': float(self.step_max[i]) if self.steps else None,
                 'low_change_step_fraction': float(self.low_steps[i]/self.steps) if self.steps else None}
                for i in range(len(names))] if self.n else []


class EpisodeScan:
    def __init__(self, episode, length, fps, keep_timestamps=False):
        self.ep, self.length, self.fps = episode, length, fps
        self.count = 0
        self.seen = np.zeros(length, dtype=bool)
        self.last = None
        self.ordered = True
        self.duplicates = self.bad_frames = self.bad_timing = self.bad_numeric = 0
        self.moments = {}
        self.hash = hashlib.sha256(b'WAM_NUMERIC_SEQUENCE_V1\0')
        self.times = {} if keep_timestamps else None
        self.task_ids = set()
        self.first_time = self.last_time = None
        self.quiet_length = self.longest_quiet = 0

    def add(self, frames, times, values, task_ids, directory_moments, epsilon, tolerance):
        order = np.argsort(frames, kind='stable')
        frames, times = frames[order], times[order]
        values = {k: x[order] for k, x in values.items()}
        self.count += len(frames)
        if task_ids is not None:
            self.task_ids.update(int(x) for x in np.asarray(task_ids)[order])
        for k, x in values.items():
            if x.ndim != 2 or x.shape[1] == 0 or len(x) != len(frames):
                raise ValueError(f'Invalid {k} shape {x.shape}')
            if k not in self.moments:
                self.moments[k] = Moments(x.shape[1])
                self.hash.update(k.encode() + np.asarray([x.shape[1]], dtype='<i8').tobytes())
            if len(self.moments[k].mean) != x.shape[1]:
                raise ValueError(f'Variable {k} dimension inside episode')
            if k not in directory_moments:
                directory_moments[k] = Moments(x.shape[1])
            if len(directory_moments[k].mean) != x.shape[1]:
                raise ValueError(f'Variable {k} dimension inside directory')
        difference_lists = {k: [] for k in values}
        finite = np.isfinite(times)
        for x in values.values():
            finite &= np.isfinite(x).all(axis=1)
        self.bad_numeric += int((~finite).sum())
        for i, (frame, timestamp) in enumerate(zip(frames, times)):
            frame, timestamp = int(frame), float(timestamp)
            valid_frame = 0 <= frame < self.length
            if not valid_frame:
                self.bad_frames += 1
            elif self.seen[frame]:
                self.duplicates += 1
            else:
                self.seen[frame] = True
            if finite[i]:
                self.first_time = timestamp if self.first_time is None else min(self.first_time, timestamp)
                self.last_time = timestamp if self.last_time is None else max(self.last_time, timestamp)
                if self.times is not None:
                    self.times[frame] = timestamp
            adjacent = self.last is not None and frame == self.last[0] + 1
            if self.last is not None and frame <= self.last[0]:
                self.ordered = False
            dt_ok = adjacent and finite[i] and self.last[3] and abs(timestamp-self.last[1]-1/self.fps) <= tolerance
            if adjacent and not dt_ok:
                self.bad_timing += 1
            if dt_ok:
                for k in values:
                    difference_lists[k].append(values[k][i] - self.last[2][k])
                low = np.max(np.abs(values['observation.state'][i]-self.last[2]['observation.state'])) <= epsilon
                self.quiet_length = self.quiet_length+1 if low else 0
                self.longest_quiet = max(self.longest_quiet, self.quiet_length)
            else:
                self.quiet_length = 0
            if finite[i]:
                self.hash.update(np.asarray([frame], dtype='<i8').tobytes())
                self.hash.update(np.asarray([timestamp], dtype='<f8').tobytes())
                for k in FIELDS:
                    self.hash.update(np.asarray(values[k][i], dtype='<f8').tobytes())
            self.last = (frame, timestamp, {k: x[i].copy() for k, x in values.items()}, bool(finite[i]))
        for k, x in values.items():
            diff = np.asarray(difference_lists[k]).reshape(-1, x.shape[1])
            for accumulator in (self.moments[k], directory_moments[k]):
                accumulator.add(x[finite], diff, epsilon)

    def summary(self):
        missing = int((~self.seen).sum())
        valid = not (missing or self.duplicates or self.bad_frames or self.bad_numeric or self.bad_timing) and self.ordered
        return {'episode': self.ep, 'expected_frames': self.length, 'observed_rows': self.count,
                'missing_frames': missing, 'duplicate_frames': self.duplicates, 'out_of_range_frames': self.bad_frames,
                'nonfinite_rows': self.bad_numeric, 'bad_timestamp_steps': self.bad_timing,
                'ordered_in_storage': self.ordered, 'numeric_valid': valid,
                'duration_observed_seconds': self.last_time-self.first_time if self.first_time is not None else None,
                'duration_nominal_seconds': (self.length-1)/self.fps,
                'longest_low_state_change_seconds': self.longest_quiet/self.fps if valid else None,
                'task_indices': sorted(self.task_ids),
                'numeric_sequence_sha256': self.hash.hexdigest() if valid else None}


def field_names(info, key, dim):
    names = info.get('features', {}).get(key, {}).get('names')
    if isinstance(names, list) and len(names) == dim and all(isinstance(s, str) for s in names):
        return names
    return [f'dim_{i:03d}' for i in range(dim)]


def camera_keys(root, info, episodes):
    keys = {k for k, v in info.get('features', {}).items()
            if str(v.get('dtype')) in ('video', 'image') or k.startswith('observation.images.')}
    for ep in episodes.values():
        for k in ep:
            if k.startswith('videos/') and len(k.split('/')) == 3:
                keys.add(k.split('/')[1])
    # Some converted datasets omit video features but keep real camera folders.
    if (root / 'videos').is_dir():
        keys.update(p.name for p in (root/'videos').iterdir() if p.is_dir() and p.name.startswith('observation.images.'))
    def priority(k):
        suffix = k.rsplit('.', 1)[-1]
        order = {'head':0, 'head_camera':0, 'cam_high':0, 'left':1, 'left_wrist':1, 'cam_left_wrist':1,
                 'right':2, 'right_wrist':2, 'cam_right_wrist':2}
        return order.get(suffix, 3), k
    return sorted(keys, key=priority)


def video_query(root, info, ep, camera, timestamp):
    template = info.get('video_path')
    if not template:
        raise ValueError('No video_path template (embedded images not decoded in v1)')
    major = int(str(info.get('codebase_version', '')).lstrip('v').split('.')[0])
    if major >= 3:
        prefix = f'videos/{camera}/'
        required = ['from_timestamp', 'chunk_index', 'file_index']
        if any(prefix+k not in ep for k in required):
            raise ValueError(f'Missing v3 video location/offset for {camera}')
        offset = float(ep[prefix+'from_timestamp'])
        end = ep.get(prefix+'to_timestamp')
        query = offset + timestamp
        if not math.isfinite(offset) or offset < 0 or query < offset or (end is not None and query >= float(end)):
            raise ValueError('Query leaves recorded episode video interval')
        relative = template.format(video_key=camera, chunk_index=int(ep[prefix+'chunk_index']),
                                   file_index=int(ep[prefix+'file_index']))
    else:
        query = timestamp
        relative = template.format(video_key=camera, episode_index=int(ep['episode_index']),
                                   episode_chunk=int(ep['episode_index'])//int(info.get('chunks_size', 1000)))
    file = (root/relative).resolve()
    if root.resolve() not in file.parents:
        raise ValueError('Video path outside dataset root')
    if not file.is_file():
        raise FileNotFoundError(file)
    return file, query


def decode_nearest(path, seconds, tolerance):
    import av  # PyAV already used by the user's WAM video pipeline.
    best, distance = None, float('inf')
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((seconds - .1) / float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            delta = abs(timestamp-seconds)
            if delta < distance:
                best, distance = frame, delta
            if timestamp >= seconds:
                break
        if best is None or distance > tolerance:
            raise ValueError(f'Video timestamp mismatch: nearest difference={distance:.6f}s')
        return best.to_image().convert('RGB')


def choose_previews(episodes, dataset_id, count, seed):
    ids = sorted(episodes)
    stable_seed = int(hashlib.sha256(f'{seed}:{dataset_id}'.encode()).hexdigest()[:16], 16)
    return sorted(np.random.default_rng(stable_seed).choice(ids, size=min(count, len(ids)), replace=False).tolist())


def make_sheet(root, info, ep, timestamps, cameras, path, nframes, width, tolerance, issue):
    from PIL import Image, ImageDraw, ImageOps
    frames = sorted(set(np.rint(np.linspace(0, int(ep['length'])-1, nframes)).astype(int).tolist()))
    cell_h = width*3//4 + 38
    canvas = Image.new('RGB', (width*len(frames), 42+cell_h*len(cameras)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{root.name[:75]} / episode {ep['episode_index']} (uniform time samples, NOT phases)", fill='black')
    count = 0
    for row, camera in enumerate(cameras):
        for col, frame in enumerate(frames):
            x, y = col*width, 42+row*cell_h
            draw.text((x+3,y+2), f'{camera.rsplit(".",1)[-1]} f={frame}', fill='black')
            try:
                if frame not in timestamps:
                    raise ValueError('No verified Parquet timestamp for preview frame')
                timestamp = timestamps[frame]
                video, query = video_query(root, info, ep, camera, timestamp)
                image = decode_nearest(video, query, tolerance)
                contained = ImageOps.contain(image, (width, cell_h-38))
                canvas.paste(contained, (x+(width-contained.width)//2, y+34))
                draw.text((x+3,y+16), f't={timestamp:.3f}s', fill='black')
                count += 1
            except Exception as exc:
                draw.text((x+5,y+50), 'UNAVAILABLE', fill='black')
                issue('preview', f'ep={ep["episode_index"]} f={frame} {camera}: {type(exc).__name__}: {exc}')
    canvas.save(path)
    return count


def initialize_output(path, roots):
    path = path.expanduser().resolve()
    if any(path == root or root in path.parents for root in roots):
        raise ValueError('Audit output must not be inside a raw dataset')
    path.mkdir(parents=True, exist_ok=True)
    if any(p.name not in ('run.log', 'launcher.sh', '.gitignore') for p in path.iterdir()):
        raise FileExistsError('Output contains a previous run; use a new directory')
    # Whitelist: no arrays, HTML, preview images, per-episode tables or logs in Git.
    ignore = '*\n' + ''.join('!/'+s+'\n' for s in UPLOAD_FILES)
    if (path/'.gitignore').exists() and (path/'.gitignore').read_text(encoding='utf-8') != ignore:
        raise FileExistsError('Refusing to replace an unrelated .gitignore')
    (path/'.gitignore').write_text(ignore, encoding='utf-8')
    (path/'_local').mkdir()
    return path


def audit_directory(entry, args, output):
    root = Path(entry['root'])
    common = {k:entry[k] for k in ('task','dataset_id')}
    row = {**entry, **name_hints(root.name), 'status':'error', 'numeric_status':'not_scanned'}
    issues, episode_rows, numeric_rows, signatures, pictures = [], [], [], [], []
    local = output/'_local'/entry['dataset_id']
    local.mkdir(exist_ok=True)
    def issue(stage, message):
        issues.append({**common, 'stage':stage, 'message':str(message)[:1600]})
    try:
        info, episodes, instructions, sources = read_metadata(root)
        lengths = np.array([int(ep['length']) for ep in episodes.values()])
        fps = float(info['fps'])
        if args.timestamp_tolerance >= .5/fps:
            raise ValueError('Timestamp tolerance must be below half a data frame')
        durations = (lengths-1)/fps
        cameras = camera_keys(root, info, episodes)
        shapes = {k:info.get('features',{}).get(k,{}) for k in (*FIELDS,*cameras)}
        dump(local/'metadata.json', {'info':info, 'instructions':instructions,
                                    'metadata_sha256':{str(p.relative_to(root)):sha(p) for p in [root/'meta/info.json',*sources]}})
        row.update(status='ok', fps=fps, codebase_version=info.get('codebase_version'),
                   metadata_episodes=len(episodes), metadata_frames=int(lengths.sum()),
                   duration_min_s=float(durations.min()), duration_median_s=float(np.median(durations)),
                   duration_p90_s=float(np.quantile(durations,.9)), duration_max_s=float(durations.max()),
                   under_5s_count=int((durations<5).sum()), camera_keys=cameras,
                   feature_schema=shapes, stored_task_texts=instructions)
        for key, actual in (('total_episodes',len(episodes)), ('total_frames',int(lengths.sum()))):
            if info.get(key) is not None and int(info[key]) != actual:
                issue('metadata', f'{key}={info[key]} differs from episode metadata sum={actual}')
        if (root/'meta/episodes.jsonl').is_file() and any((root/'meta/episodes').rglob('*.parquet')):
            issue('metadata', 'Both JSONL and Parquet episode metadata exist; JSONL priority matches repository. Check stale copies.')
        preview_ids = choose_previews(episodes, entry['dataset_id'], args.preview_episodes, args.seed)
        row['preview_episode_ids'] = preview_ids
        accumulators, directory_moments = {}, {}
        numerical_error = False
        if args.numeric:
            try:
                processed_rows = 0
                next_log = 50000
                for file, batch in iter_numeric_batches(root, args.batch_size):
                    ids = np.asarray(batch['episode_index'], dtype=np.int64)
                    for ep_id in np.unique(ids):
                        ep_id = int(ep_id)
                        if ep_id not in episodes:
                            raise ValueError(f'{file.name}: data episode {ep_id} absent from metadata')
                        selected = np.flatnonzero(ids == ep_id)
                        frames = np.asarray(batch['frame_index'],dtype=np.int64)[selected]
                        times = np.asarray(batch['timestamp'],dtype=np.float64)[selected]
                        values = {k:np.asarray([batch[k][i] for i in selected],dtype=np.float64) for k in FIELDS}
                        tasks = np.asarray(batch['task_index'])[selected] if 'task_index' in batch else None
                        if ep_id not in accumulators:
                            accumulators[ep_id] = EpisodeScan(ep_id, int(episodes[ep_id]['length']), fps, ep_id in preview_ids)
                        accumulators[ep_id].add(frames,times,values,tasks,directory_moments,args.low_change_epsilon,args.timestamp_tolerance)
                    processed_rows += len(ids)
                    if processed_rows >= next_log:
                        print(f'  [numeric] rows={processed_rows} / metadata={int(lengths.sum())}', flush=True)
                        next_log = processed_rows + 50000
                row['numeric_status'] = 'complete'
            except Exception as exc:
                issue('numeric', f'{type(exc).__name__}: {exc}')
                numerical_error = True
                row['numeric_status'] = 'partial'
        for ep_id, ep in sorted(episodes.items()):
            scan = accumulators.get(ep_id)
            result = scan.summary() if scan else {'episode':ep_id,'expected_frames':int(ep['length']),
                                                'numeric_valid':False,'numeric_sequence_sha256':None}
            result.update(**common)
            result['scan_status'] = 'partial' if numerical_error else ('scanned' if scan else 'not_scanned')
            if numerical_error:  # Do not certify hashes/coverage after an aborted stream.
                result['numeric_sequence_sha256'] = None
                result['numeric_valid'] = False
            episode_rows.append(result)
            if scan and not result['numeric_valid']:
                issue('episode', f'ep={ep_id}: invalid/incomplete/unordered numeric sequence; see _local/episodes.csv')
            if not scan and args.numeric:
                issue('episode', f'ep={ep_id}: no data read')
            if result.get('numeric_sequence_sha256'):
                signatures.append({**common, 'episode':ep_id, 'frames':int(ep['length']),
                                   'numeric_sequence_sha256':result['numeric_sequence_sha256']})
            if scan:
                detail = []
                for k,m in scan.moments.items():
                    detail.extend(m.rows({**common,'episode':ep_id,'kind':k}, field_names(info,k,len(m.mean))))
                csv_write(local/f'ep_{ep_id:06d}_numeric.csv', detail)
        for k,m in directory_moments.items():
            expected = info.get('features',{}).get(k,{}).get('shape')
            if expected != [len(m.mean)]:
                issue('schema', f'{k} observed D={len(m.mean)}, metadata shape={expected}; generic numeric statistics only')
            numeric_rows.extend(m.rows({**common, 'kind':k, 'scan_status':row['numeric_status']}, field_names(info,k,len(m.mean))))
        row['observed_numeric_rows'] = sum(a.count for a in accumulators.values()) if args.numeric else None
        row['numeric_valid_episodes'] = sum(bool(r['numeric_valid']) for r in episode_rows) if args.numeric else None
        row['state_dim_observed'] = len(directory_moments[FIELDS[0]].mean) if FIELDS[0] in directory_moments else None
        row['action_dim_observed'] = len(directory_moments[FIELDS[1]].mean) if FIELDS[1] in directory_moments else None
        csv_write(local/'episodes.csv', episode_rows)
        row['preview_images_decoded'] = 0
        if args.preview_episodes and args.numeric:
            try:
                import av  # Check once, not once per camera/frame.
                from PIL import Image
                del av, Image
                for ep_id in preview_ids:
                    if shutil.disk_usage(output).free < args.min_free_gb * 2**30:
                        raise OSError('Disk free space below preview reserve; no images written')
                    scan = accumulators.get(ep_id)
                    if not scan or not scan.summary()['numeric_valid'] or numerical_error:
                        issue('preview', f'ep={ep_id}: no complete verified numeric timeline; skip')
                        continue
                    if not cameras:
                        issue('preview', 'No camera keys discovered')
                        break
                    image = local/f'ep_{ep_id:06d}.jpg'
                    count = make_sheet(root,info,episodes[ep_id],scan.times,cameras[:args.max_cameras],
                                       image,args.preview_frames,args.thumb_width,args.timestamp_tolerance,issue)
                    row['preview_images_decoded'] += count
                    pictures.append({'episode':ep_id,'image':str(image.relative_to(output/'_local'))})
            except Exception as exc:
                issue('preview', f'{type(exc).__name__}: {exc}')
        elif args.preview_episodes:
            issue('preview', 'Numeric scan disabled: no verified timestamps; previews not attempted')
    except Exception as exc:
        issue('directory', f'{type(exc).__name__}: {exc}')
    if row['status'] == 'ok' and issues:
        row['status'] = 'needs_review'
    row['issue_count'] = len(issues)
    return row, episode_rows, numeric_rows, signatures, pictures, issues


def duplicate_candidates(signatures):
    grouped = defaultdict(list)
    for row in signatures:
        grouped[row['numeric_sequence_sha256']].append(row)
    rows = []
    for digest, matches in sorted(grouped.items()):
        if len(matches) > 1:
            for match in matches:
                rows.append({**match, 'candidate_group':digest[:16], 'group_size':len(matches),
                             'scope':'exact state+action+timestamp sequence; RGB not verified'})
    return rows


def local_index(path, rows, pictures):
    parts = ['<!doctype html><meta charset="utf-8"><title>Dataset audit</title>',
             '<style>body{font:16px sans-serif;margin:32px}img{max-width:100%;border:1px solid #ccc}section{margin:40px 0}</style>',
             '<h1>数据概览（仅本地）</h1><p>随机选轨迹，均匀选时刻；不是事件分割，不保证覆盖所有场景。未知处保留 unknown。</p>']
    for row in rows:
        parts.append('<section><h2>'+html.escape(row['task']+' / '+row['directory_name'])+'</h2>')
        parts.append('<p>'+html.escape(f"状态: {row['status']} | issues={row['issue_count']}")+'</p>')
        for image in pictures.get(row['dataset_id'], []):
            parts.append(f'<p>Episode {image["episode"]}</p><img loading="lazy" src="{html.escape(image["image"],quote=True)}">')
        parts.append('</section>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--catalog',default='configs/data/astribot_dataset_catalog.sh')
    p.add_argument('--tasks',nargs='+',default=[])
    p.add_argument('--data-root',default=None)
    p.add_argument('--benchmark-root',default=None)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--numeric',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--preview-episodes',type=int,default=3)
    p.add_argument('--preview-frames',type=int,default=5)
    p.add_argument('--max-cameras',type=int,default=3)
    p.add_argument('--thumb-width',type=int,default=240)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--timestamp-tolerance',type=float,default=.002)
    p.add_argument('--low-change-epsilon',type=float,default=1e-4)
    p.add_argument('--min-free-gb',type=float,default=2.)
    args = p.parse_args(argv)
    if args.preview_episodes < 0 or args.preview_frames < 2 or args.max_cameras < 1 or args.thumb_width < 80 or args.batch_size < 1:
        raise ValueError('Invalid preview/batch size')
    if any(not math.isfinite(x) or x < 0 for x in (args.timestamp_tolerance,args.low_change_epsilon,args.min_free_gb)) or args.timestamp_tolerance == 0:
        raise ValueError('Invalid tolerances/reserve')
    entries = read_catalog(args.catalog,args.tasks,args.data_root,args.benchmark_root)
    roots = {Path(e['root']).resolve() for e in entries}
    output = initialize_output(Path(args.output_dir), roots)
    try:
        git_head = subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        git_head = None
    dump(output/'manifest.json', {'audit_version':VERSION,'arguments':vars(args), 'git_head':git_head,
                                 'catalog_sha256':sha(args.catalog),'script_sha256':sha(__file__),
                                 'entries':entries,'note':'No split, training, normalization, model or robot. Folder labels are hints only.'})
    rows, numeric, signatures, issues, images = [], [], [], [], {}
    seen = {}
    for n,entry in enumerate(entries,1):
        print(f'[audit] {n}/{len(entries)} task={entry["task"]} directory={entry["root"]}',flush=True)
        if entry['root'] in seen:
            issues.append({'task':entry['task'],'dataset_id':entry['dataset_id'],'stage':'catalog',
                           'message':f'Repeated resolved root, already scanned under {seen[entry["root"]]}; not double-counted'})
            continue
        seen[entry['root']] = entry['task']
        row,_,nr,sig,pics,problems = audit_directory(entry,args,output)
        rows.append(row); numeric.extend(nr); signatures.extend(sig); issues.extend(problems)
        images[entry['dataset_id']] = pics
        csv_write(output/'directories.csv',rows)
        csv_write(output/'issues.csv',issues,['task','dataset_id','stage','message'])
        print(f'  status={row["status"]} episodes={row.get("metadata_episodes","?")} numeric={row["numeric_status"]} issues={len(problems)}',flush=True)
    duplicates = duplicate_candidates(signatures)
    csv_write(output/'duplicate_candidates.csv',duplicates,
              ['task','dataset_id','episode','frames','numeric_sequence_sha256','candidate_group','group_size','scope'])
    csv_write(output/'numeric_summary.csv',numeric)
    tasks = []
    for task in dict.fromkeys(e['task'] for e in entries):
        rr = [r for r in rows if r['task']==task]
        tasks.append({'task':task,'catalog_directories':sum(e['task']==task for e in entries),
                      'unique_directories_scanned':len(rr),'directories_with_metadata':sum('metadata_episodes' in r for r in rr),
                      'directories_needing_review':sum(r['status']!='ok' for r in rr),
                      'episodes_from_read_metadata':sum(r.get('metadata_episodes',0) for r in rr),
                      'numeric_valid_episodes':sum(r.get('numeric_valid_episodes') or 0 for r in rr),
                      'observed_state_dims':sorted({r['state_dim_observed'] for r in rr if r.get('state_dim_observed') is not None}),
                      'observed_action_dims':sorted({r['action_dim_observed'] for r in rr if r.get('action_dim_observed') is not None})})
    csv_write(output/'tasks.csv',tasks)
    csv_write(output/'review.csv',[{k:r[k] for k in ('task','dataset_id','directory_name')} |
               {'full_or_partial_task':'unknown','object_types':'','object_counts':'','scene_changes':'',
                'initial_conditions':'','notes':''} for r in rows])
    local_index(output/'_local/index.html',rows,images)
    status = 'complete_with_issues' if issues else 'complete'
    report = ['# Astribot 数据盘点',f'\n状态：{status}；清单 {len(entries)} 项，读取 {len(rows)} 个唯一目录。',
              '\n本次只盘点。没有创建 train/val/test，没有训练模型，也没有更改原始数据或已有输出路径。',
              '\n|任务|清单目录|读到元信息的目录|元信息轨迹数|数值有效轨迹|需复核目录|',
              '|---|---:|---:|---:|---:|---:|']
    report += [f'|{t["task"]}|{t["catalog_directories"]}|{t["directories_with_metadata"]}|{t["episodes_from_read_metadata"]}|{t["numeric_valid_episodes"]}|{t["directories_needing_review"]}|' for t in tasks]
    report += ['\n## 先看什么',
               '先检查 issues.csv 的缺失/读取/时间戳问题；再比较 directories.csv 的数量、时长、相机、维度和存储指令。',
               '打开本机 _local/index.html 看每目录随机轨迹概览，再填写 review.csv。时间均匀采样不等于阶段标签。',
               '\n## 不能由这些指标推出什么',
               '目录名中的人员、日期、pickup 等只是来源线索；unknown 格式不会强行套用 Astribot25D。',
               'under_5s 仅是短记录提示，不代表坏数据。低变化阈值采用原始数值，不代表物理静止或任务成功。',
               '尤其恒定非零底盘速度仍在移动；本工具不自动把全维低变化当停止，不自动赋予夹爪事件。',
               '数值哈希仅检查完整、有序的 state/action/timestamp 精确相同序列；不比较视频，不查近重复，不自动去重。',
               f'精确数值重复候选组：{len({r["candidate_group"] for r in duplicates})}。无候选不代表没有重复。',
               '扫描失败或缺失时统计仅覆盖已读部分；未扫描不是零差异。元数据名义时长与真实时间戳时长分开。',
               '\n## 存储',
               '只提交本目录白名单内的报告、摘要CSV和manifest/状态。_local、run.log、launcher.sh默认Git忽略。',
               '详细轨迹/逐维统计和概览图仅保留本机，不使用 git add -f。完成清单不代表数据完全有效。']
    (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    dump(output/'COMPLETE.json',{'status':status,'catalog_entries':len(entries),'directories_scanned':len(rows),
                                'issue_count':len(issues),'numeric_scan_requested':args.numeric,
                                'split_created':False,'model_loaded':False})
    print(f'[audit] {status}: {output}\nRead report.md / issues.csv; local overview: _local/index.html',flush=True)
    return 2 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
