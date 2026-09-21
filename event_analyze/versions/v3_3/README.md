# V3.3 — Dense Targeted Temporal Observation

V3.3 keeps V3.2's high-precision interaction validation and replaces the failed
sparse cutlery-basket re-observation with **dense relative-motion observation**.

## Why V3.3

V3.2 reached high sequence precision, but the two missing skills were both
cutlery-basket transitions. Sparse 5-time-point contact sheets repeatedly
classified the basket as `in -> in` or `uncertain`, even when the full video
contains the interaction.

The issue is representation, not another prompt-tuning problem.

## Pipeline

```text
generic entity observations (fixed V3 baseline)
  -> automatic search regions around utensil manipulation
  -> decode dense head + right frames directly from HDF5
  -> dense temporal strips
  -> VLM relative-motion judgment:
       cutlery basket vs dish rack
  -> high-confidence dense observations as extra tracker anchors
  -> trajectory-level entity tracking
  -> deterministic skills
  -> interaction-grounded validation
  -> HDF5 boundary refinement
```

## Dense observation

Default:
- 4.5 s temporal window
- 2.0 s stride
- 10 synchronized time points
- head + right camera
- direct raw HDF5 JPEG decoding

The VLM is asked for **relative motion**, not merely `in/out`:
- `moving_out`
- `moving_in`
- `stationary`
- `uncertain`

Dense relative-motion evidence is weighted more strongly than generic absolute
state guesses, but only for the targeted `cutlery_basket` entity.

## Run

```bash
bash versions/v3_3/run.sh \
  /path/to/episode.hdf5 \
  output/<episode>_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

Useful ablation switches:

```bash
V33_SKIP_DENSE=1
V33_REOBSERVE_ALL=1
```

## Generalization boundary

V3.3 is still a **known-ontology dishwasher pipeline**. The tracking,
interaction-validation and boundary-refinement ideas are reusable, but entity
names, state graphs and skill mappings are currently scene/task specific.

Do not claim full cross-scene generalization from V3.3. See
`event_analyze/GENERALIZATION.md` for the proposed task-agnostic direction.
