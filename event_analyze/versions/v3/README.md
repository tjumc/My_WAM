# V3 — Entity-State-Grounded Skill Discovery

V3 removes free-form action/phase generation from the core annotation path.

Pipeline:

```text
state/action proposal
  -> constrained multi-view entity-state observation
  -> deterministic entity state machines
  -> skill sequence
  -> HDF5 signal-based boundary refinement
  -> hierarchical annotations + coarse-only gaps
```

## Ontology

Environment:
- dishwasher door: closed / opening / open / closing
- dish rack: in / moving_out / out / moving_in
- cutlery basket: in / moving_out / out / moving_in

Objects:
- knife: tabletop / hand / cutlery_basket
- fork: tabletop / hand / cutlery_basket
- plate: tabletop / hand / dish_rack

Robot:
- right/left hand relations are explicitly observed; uncertain identity remains uncertain.

## Deterministic skills

- navigate_to_station
- open_dishwasher_door
- pull_out_dish_rack
- pull_out_cutlery_basket
- place_knife_in_cutlery_basket
- place_fork_in_cutlery_basket
- push_in_cutlery_basket
- place_plate_in_dish_rack
- push_in_dish_rack
- close_dishwasher_door

The list above defines the ontology, not an enforced sequence. Only observed state transitions create skills.

## Run

```bash
bash versions/v3/run.sh \
  /path/to/episode.hdf5 \
  output/<episode>_analysis \
  "put the dish into the dishwasher" \
  11 1645
```

Outputs are written directly to `output/<episode>_analysis/versions/v3/`.

## Regression evaluation

The manual fixture for the current episode is stored under `event_analyze/regression/` and is **never read by the annotation pipeline**. It is only used to quantify sequence errors while developing the method.
