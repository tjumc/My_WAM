# Event Analyze V2

V2 separates semantics from temporal boundaries.

## Stage A — conservative local verification
```bash
python qwen_verify_events_v2.py ANALYSIS_DIR \
  --hdf5 episode.hdf5 \
  --task "put the dish into the dishwasher"
```
Output: `event_semantics_v2.jsonl`

## Stage B — deterministic dedup + semantic phase grouping
```bash
python qwen_compose_phases_v2.py ANALYSIS_DIR \
  --task "put the dish into the dishwasher" \
  --min-confidence 0.60 \
  --duplicate-sec 2.0
```
Outputs:
- `atomic_events_v2.json`
- `semantic_hierarchy_v2.json`

Qwen only groups atomic IDs and names phases. It cannot choose frame boundaries.

## Stage C — signal-grounded boundary refinement
```bash
python refine_boundaries_v2.py ANALYSIS_DIR \
  --hdf5 episode.hdf5 \
  --lerobot-episode 11 \
  --lerobot-max-raw-frame 1645
```
Outputs:
- `hierarchical_annotations_v2.json`
- `boundaries_v2.png`

## One-command run
```bash
bash run_v2.sh episode.hdf5 ANALYSIS_DIR "put the dish into the dishwasher"
```

## What to inspect first
1. How many V1 candidates became false in `event_semantics_v2.jsonl`.
2. Whether repeated grasp/release proposals collapsed in `atomic_events_v2.json`.
3. Whether `semantic_hierarchy_v2.json` contains only event IDs, not invented frames.
4. Whether `hierarchical_annotations_v2.json` phase intervals are monotonic and non-overlapping.
5. Whether frames after task completion are excluded from the original task supervision.
