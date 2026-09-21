# V3.2 — Targeted Re-observation + Interaction-Grounded Validation

V3.2 keeps the V3/V3.1 architecture and addresses the two dominant V3.1 errors:

- **false positive container transitions** from visual state jitter;
- **missing cutlery-basket transitions** because the generic VLM observer rarely saw the basket.

Pipeline:

```text
shared state/action proposals
  -> generic constrained entity observations (reuse V3 by default)
  -> targeted cutlery-basket re-observation
  -> trajectory-level entity state tracking
  -> deterministic skill candidates
  -> interaction-grounded transition validation
  -> HDF5 signal-grounded boundary refinement
  -> hierarchical annotations + coarse-only gaps
```

## Precision controls

A container state transition is not accepted solely because the tracked visual
state changes. Validation also uses:

- entity-specific hand contact;
- right-arm motion;
- force-change signal;
- gripper-change signal;
- short reverse-pair rejection;
- a one-cycle container lifecycle by default.

Object-placement skills require an actually **observed** held state. A purely
interpolated held state cannot create a placement skill.

## Recall recovery

The generic VLM observer is not rerun. V3.2 selects a small set of windows
around knife/fork interactions and asks a focused question about only the
`cutlery_basket`. These targeted observations are merged into the generic
observations before tracking.

This keeps the experiment interpretable:

- V3 -> V3.1: effect of trajectory-level tracking;
- V3.1 -> V3.2: effect of targeted observation + interaction validation.

## Run

```bash
bash versions/v3_2/run.sh \
  /path/to/episode.hdf5 \
  output/<episode>_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

Useful switches:

```bash
V32_SKIP_TARGETED=1   # ablate targeted re-observation
V32_REOBSERVE_ALL=1  # rerun the generic VLM observer instead of reusing V3
```

Outputs:

- `entity_observations_base.jsonl`
- `targeted_cutlery_observations.jsonl`
- `entity_observations.jsonl` (augmented)
- `tracked_entity_states.json`
- `skill_candidates.json`
- `skill_candidates_validated.json`
- `interaction_validation_report.json`
- `hierarchical_annotations.json`
- `boundaries.png`

The manual regression fixture is never read by the annotation pipeline.
