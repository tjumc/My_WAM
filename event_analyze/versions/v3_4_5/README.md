# V3.4.5 — selective re-observation, global consistency, compact runtime

V3.4.5 is driven by the frozen six-trajectory V3.4.4 validation result rather
than by another episode26/27-specific fix.

V3.4.4 validation established three systematic problems:

1. mean policy recall was only 0.500 (mean F1 0.593);
2. every validation trajectory consumed the full 8-query targeted budget;
3. internally inconsistent final phases could survive boundary refinement.

V3.4.5 addresses these without reading manual GT during inference.

## Method changes

### 1. Schema-driven gripper/contact bridge

For articulated/container manipulation, missing VLM hand-contact evidence may
be bridged only when **all** of the following agree:

- a schema-declared gripper interaction signature;
- a direction-consistent visual entity-state transition;
- robot interaction signal above the configured threshold.

The gripper signal alone never creates an action.

For the dishwasher task, the one-time schema onboarding records that door/rack
contact manipulation normally uses the right gripper in the closed state.

### 2. Selective ambiguity ranking

V3.4.4 tiled long lifecycle search horizons and frequently saturated the query
budget. V3.4.5 instead creates at most one focused request per semantic issue.

Requests are ranked by:

- semantic impact;
- existing robot/interaction support;
- temporal overlap deduplication.

The configured `max_requests` is now a ceiling, not a target. An accepted
transition that merely lacks extra directional confirmation no longer consumes
targeted-query budget.

### 3. Final global consistency gate

After numerical boundary refinement:

- phases entirely after the task horizon are removed;
- raw and provisional semantic spans are clamped to the valid trajectory/task horizon, while original out-of-horizon values remain recorded in diagnostics;
- unresolved duration-conflict phases shorter than the skill minimum are
  removed;
- schema-inconsistent articulated lifecycle transitions are removed;
- unmet schema `expected_final_state` is reported but no missing action is
  invented.

The gate writes:

```text
diagnostics/final_consistency.json
```

### 4. Compact pass1 -> pass2 reasoning trace

V3.4.5 no longer permanently saves complete pass1 copies of tracker,
ownership, candidate, and validation JSON files.

Instead it records:

```text
diagnostics/reasoning_trace.json
```

with:

- pass1 sequence;
- rejected-candidate reasons;
- ambiguity requests;
- targeted-query/merge counts;
- pass1 -> pass2 sequence delta;
- pass2 -> final consistency delta.

## Runtime layout

By default, after a successful run:

```text
v3_4_5/
├── hierarchical_annotations.json
├── evaluation_temporal.json        # only when frame-level GT exists
├── diagnostics/
│   ├── reasoning_trace.json
│   └── final_consistency.json
└── cache/
    ├── proposal/
    ├── entity_observations_base.jsonl
    ├── entity_observations.jsonl
    ├── targeted_observations.jsonl
    ├── targeted_temporal_strips/
    ├── vlm_raw_targeted/
    └── boundaries.png
```

Pure intermediate reasoning JSON files are removed after export.

For full debugging:

```bash
KEEP_DEBUG=1 bash versions/v3_4_5/run.sh ...
```

This preserves the complete runtime chain.

## Single-trajectory run

```bash
bash versions/v3_4_5/run.sh \
  /path/to/episode.hdf5 \
  output/<episode>_analysis \
  "put the dish into the dishwasher"
```

Only frame-level GT should be passed through the stable `GT` environment
variable.

## Frozen validation run

Do not redefine the existing split.

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --version v3_4_5 \
  --split validation
```

The six validation trajectories have sequence-level GT and therefore obtain
policy sequence P/R/F1/Edit/Exact metrics without being passed to the temporal
evaluator.

## Controlled comparison to run

Compare V3.4.5 against V3.4.4 on exactly the same frozen validation split:

- mean policy Precision / Recall / F1;
- mean Edit Distance / Exact Rate;
- targeted queries per trajectory;
- merged targeted observations;
- final-consistency repairs / unresolved violations.

V3.4.5 is implemented; improvement claims are pending this controlled run.
