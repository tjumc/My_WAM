# V3.4.8 — implication-aware candidate rehabilitation

V3.4.8 adds two reasoning changes on top of the V3.4.7 pipeline. It remains an
implementation hypothesis until the development regression is run.

## Candidate rehabilitation

Before pass2, V3.4.8 saves robot-supported container-transition candidates that
pass1 rejected. After targeted perception and pass2, it returns a saved
candidate to the existing interaction validator only when all of these agree:

- a selected targeted request was linked to the same entity and skill;
- the accepted targeted observation shows the schema-declared transition
  direction with confidence on both endpoints;
- an already accepted placement into that receptacle implies the lifecycle
  state that makes the candidate necessary;
- the original robot-interaction support remains above the query threshold.

The candidate is then rechecked by the existing interaction, robot-signal, and
lifecycle gates. A task-graph implication alone never creates a phase. The
validator also propagates placement usage-state implications in time order, so
a directly supported return-to-final-state candidate is not rejected just
because its pull-out phase was missed.

Each decision is recorded in
`diagnostics/candidate_rehabilitation.json`, including the rejected candidate,
robot support, targeted state transition, implication gap, and validator
outcome.

## Anchor-preserving boundary arbitration

The boundary refiner matches reconciled skills against validated pass1 anchors.
When a new phase overlaps an anchor, it allocates enough duration to keep the
anchor intact whenever possible. If the shared interval cannot fit both
minimum durations, the new candidate receives the conflict and fails closed;
the validated anchor is retained. Pair-level conflicts remain visible in the
boundary diagnostics.

## Controlled inheritance

V3.4.8 reuses V3.4.7 proposal and base-observation artifacts when available.
Targeted observations are reused only when every request signature matches.
Task-specific legacy dense perception remains disabled. Perception, proposal
generation, and query policy otherwise follow V3.4.7.

## Development regression

Run only the frozen development split, which contains episodes 26 and 27:

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --version v3_4_8 \
  --split development
```

Compare against both `v3_4_7` and `v3_4_7_dense_ablation`. Check whether episode27
recovers a basket lifecycle phase with direct evidence, and whether episode26
keeps its validated open-door anchor when the basket phase is inserted. Do not
run validation or held-out until this development regression is reviewed.

## Status

Implementation and documentation are complete. No experiment has been run for
V3.4.8 yet. The code path preserves the existing output layout under
`versions/v3_4_8/` and adds `candidate_rehabilitation.json` to compact
diagnostics.
