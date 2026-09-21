# Event Analyze V2.1

V2.1 fixes the V2 failure mode where every candidate was rejected because the
candidate center was not an exact action boundary.

## Core change

A candidate frame is now only an **attention anchor**.

```text
candidate point
    ↓
±1.5 s multi-view semantic window
    ↓
semantic action evidence
    ↓ temporal clustering
atomic action interval
    ↓ Qwen semantic composition
semantic phase
    ↓ state/action refinement
numerical phase boundaries
```

Qwen never chooses frame numbers.

## Stage A — semantic-window verification

```bash
python qwen_verify_windows_v2_1.py ANALYSIS_DIR \
  --hdf5 episode.hdf5 \
  --task "put the dish into the dishwasher"
```

Output: `window_semantics_v2_1.jsonl`

A window can be `contains_semantic_action=true` even when the candidate center is
in the middle of a continuous `open`, `transport`, `place`, or `retract` action.

## Stage B — temporal action clustering + phase composition

```bash
python qwen_cluster_actions_v2_1.py ANALYSIS_DIR \
  --task "put the dish into the dishwasher" \
  --min-confidence 0.55 \
  --max-gap-sec 1.0
```

Outputs:
- `action_intervals_v2_1.json`
- `semantic_hierarchy_v2_1.json`

Overlapping/nearby windows with compatible action semantics are merged into one
action interval. Qwen then groups action IDs into phases, without frame numbers.

## Stage C — signal-grounded action and phase boundaries

```bash
python refine_boundaries_v2_1.py ANALYSIS_DIR \
  --hdf5 episode.hdf5 \
  --lerobot-episode 11 \
  --lerobot-max-raw-frame 1645
```

Outputs:
- `hierarchical_annotations_v2_1.json`
- `boundaries_v2_1.png`

The output explicitly records unassigned prefix/suffix frames instead of forcing
an incorrect fine-grained phase label over the entire trajectory.

## One-command run

```bash
bash run_v2_1.sh \
  episode.hdf5 \
  ANALYSIS_DIR \
  "put the dish into the dishwasher" \
  11 \
  1645
```

## Expected diagnostic behavior

For the current dishwasher example, V1 produced 24/24 semantic events and V2
produced 0/24. V2.1 should land between those extremes:

```text
24 candidate windows
    ↓
meaningful semantic windows (some true, some false)
    ↓
fewer action intervals after temporal clustering
    ↓
even fewer semantic phases
```

Do not optimize toward a fixed count. Inspect whether continuous action windows are
retained, duplicate windows are merged, and unrelated/post-task motion is excluded.
