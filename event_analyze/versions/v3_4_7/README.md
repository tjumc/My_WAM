# V3.4.7 — schema-driven latent state implication

V3.4.7 is a focused reasoning update on top of V3.4.6. It does not change
perception, ambiguity detection, targeted-query selection, conservative
assimilation, or boundary refinement.

V3.4.6 improved validation mean policy F1 from 0.653 to 0.691 with exactly the
same base observations, ambiguity requests, and targeted observations as
V3.4.5. It also validated conservative pass1-anchor recovery and an independent
task-completion frontier.

The remaining failure exposed by episode19 is more fundamental:

- a placement into a receptacle was accepted;
- the corresponding receptacle pull-out skill was missed;
- therefore the final decoder still believed the receptacle remained in its
  initial state;
- a later valid push-in transition was rejected as lifecycle-inconsistent.

## Core idea

An accepted action can imply a latent world state that must have been true for
that action to succeed.

V3.4.7 unifies two schema-driven implication families:

```text
accepted dependent use
    -> prerequisite state must hold

accepted placement into receptacle
    -> receptacle usage_state must hold
```

Examples under the current schema:

```text
accepted pull_out_dish_rack
    -> door=open

accepted place_plate_in_dish_rack
    -> dish_rack=out

accepted place_utensil_in_cutlery_basket
    -> cutlery_basket=out
```

Only hidden lifecycle state is updated. V3.4.7 never fabricates a missing
`open` / `pull_out` action label.

## Why this matters

Without usage-state implication, a system can miss an entire receptacle
lifecycle and still incorrectly appear internally consistent:

```text
expected initial dish_rack = in
pull_out_dish_rack missed
place_plate_in_dish_rack accepted
push_in_dish_rack missed
final decoder still says dish_rack = in
expected final dish_rack = in
=> false "consistent"
```

V3.4.7 instead reasons:

```text
place_plate_in_dish_rack accepted
=> dish_rack must have been out during use
=> if no later push_in is observed, final state remains out
=> expected_final_state=in is unresolved
```

This makes consistency checking informative even when explicit transition
skills are missing.

## Controlled inheritance

V3.4.7 deliberately inherits:

- V3.4.5 base perception / tracking / validation;
- V3.4.6 conservative pass1-anchor assimilation;
- V3.4.6 ambiguity requests and targeted-observation reuse policy;
- V3.4.6 dependency-aware task-completion frontier.

The only methodological change is the generalized latent state implication in
the final task-graph decoder.

For a clean comparison, V3.4.7 reuses V3.4.6 targeted observations only when
the request signature matches exactly.

## Diagnostics

`final_consistency.json` now reports:

```text
latent_state_inferences
latent_prerequisite_state_inferences
latent_usage_state_inferences
```

A placement-generated implication is recorded with:

```text
reason = accepted_placement_requires_receptacle_usage_state
```

Repeated final-state recomputation no longer writes duplicate latent inference
metadata into the semantic phase.

## Primary validation questions

1. Does episode19 keep the V3.4.6/V3.4.5 recovered `push_in_dish_rack`
   instead of rejecting it?
2. Does episode6 stop reporting a false internally-consistent dish-rack
   lifecycle after accepting plate placement without rack transitions?
3. Do missing cutlery-basket push-in phases become explicit final-state
   violations rather than silently disappearing?
4. Are the V3.4.6 gains on episode9, episode21, and episode23 preserved?
5. Does policy precision remain high while recall/F1 improve?
6. Do base/ambiguity/targeted evidence fingerprints remain unchanged from
   V3.4.6?

## Validation run

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --version v3_4_7 \
  --split validation
```

Do not open the held-out split yet.

V3.4.7 is an implementation hypothesis until the frozen validation run is
completed.
