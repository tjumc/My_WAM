# V3.1 — Trajectory-Level Entity State Tracking

V3.1 keeps the V3 constrained entity observer unchanged and replaces
independent-window skill inference with trajectory-level tracking.

Pipeline:

```text
state/action proposals
  -> constrained multi-view entity observations
  -> trajectory-level entity state tracking
  -> deterministic state-transition skill discovery
  -> HDF5 signal-grounded boundary refinement
  -> hierarchical annotations + coarse-only gaps
```

## What changed from V3

1. Confidence-weighted full-trajectory tracking with state persistence.
2. Graph-constrained physical transitions for door / dish rack / cutlery basket.
3. Unknown and occluded windows no longer erase the last plausible state.
4. Cross-entity hand/object exclusivity helps recover object trajectories.
5. Missing motion states can be bridged, e.g. `in -> out` still implies a pull.
6. Weak affordance-compatible completion can bridge an occluded object release;
   such states are marked `interpolated` and down-weighted.
7. Repeated `open -> closing` windows produce one terminal close action instead
   of duplicate close skills.

The manual regression fixture is never read by the annotation pipeline.

## Controlled comparison

By default `run.sh` reuses:

`output/<episode>_analysis/versions/v3/entity_observations.jsonl`

when it exists. This isolates the gain from trajectory-level tracking without
changing the VLM observations. Set `V31_REOBSERVE=1` to force a fresh observation pass.

## Run

```bash
bash versions/v3_1/run.sh \
  /path/to/episode.hdf5 \
  output/<episode>_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

Outputs:

- `entity_observations.jsonl`
- `tracked_entity_states.json`
- `skill_candidates.json`
- `hierarchical_annotations.json`
- `boundaries.png`

Evaluate against the manual regression fixture:

```bash
python versions/v3/evaluate_manual_gt.py \
  output/<episode>_analysis/versions/v3_1/hierarchical_annotations.json \
  regression/dishwasher_episode27_manual_gt.json
```
