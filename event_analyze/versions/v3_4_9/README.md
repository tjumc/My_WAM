# V3.4.9 — post-placement lifecycle search and duration-safe boundaries

V3.4.8 development improved mean policy F1 from 0.619 to 0.759, but episode26
regressed from 5/10 to 4/10 and episode27 still missed `push_in_cutlery_basket`.
This version addresses those two observed failure paths. It is an untested
development hypothesis; validation and held-out remain closed.

## Lifecycle search

Accepted placement into a receptacle implies its usage state when computing
the expected final lifecycle. If the final state is still missing, the query
planner searches after the last accepted placement for a robot proposal from
the schema-declared gripper arm. The focused request includes the placement
endpoint and the strongest robot event. This avoids spending the query on an
earlier rejected basket transition when the missing close action is later.

After targeted observation, one new transition candidate may be proposed from
that robot event. It requires the selected request, a high-confidence visual
transition in the schema direction, and accepted placements that imply the
source state. The unchanged interaction validator then rechecks robot signal,
contact or schema gripper bridge, and lifecycle ordering. Every gate and
validator outcome is recorded in `diagnostics/candidate_rehabilitation.json`.

## Boundary arbitration

The refiner first checks whether both neighboring phases can fit their minimum
durations inside their raw envelopes. It preserves the full pass1 anchor span
when compatible. When that preference would collapse a valid new phase, it
chooses a cut that preserves both duration floors and records
`anchor_floor_fallback`. Only a genuinely infeasible pair is marked as a
duration conflict.

## Controlled development run

V3.4.9 uses a separate output directory and reuses V3.4.8 proposal and base
observations when available. Targeted observations are reused only if all
request signatures match. Run only the frozen development split:

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --version v3_4_9 \
  --split development
```

Review episode26 door/rack spans and episode27 post-placement basket request,
candidate audit, final sequence, and temporal metrics before any broader run.
