# V3.4.4 — Ambiguity-Triggered Targeted Re-Observation

V3.4.4 adds a closed-loop perception/reasoning stage.

```text
base structured observations
        ↓
reasoning pass 1
        ↓
generic ambiguity detection
        ↓
schema-driven targeted VLM queries
        ↓
merge only sufficiently confident new evidence
        ↓
reasoning pass 2
        ↓
final annotation
```

## What triggers a query

The detector does not read manual GT. It creates targeted requests only when
additional visual evidence can change semantic or temporal reasoning:

- unresolved joint hand-object ownership;
- a candidate rejected because visual interaction evidence is missing or
  inconsistent;
- an articulated entity does not reach a schema-declared expected final state;
- an accepted articulated transition has contact evidence but no direct visual
  direction evidence.

Fine identity uncertainty that is already safely handled by semantic backoff
does **not** trigger a query.

## Task completion state is schema knowledge

For this task family the schema declares:

```text
door            -> expected final state: closed
dish_rack       -> expected final state: in
cutlery_basket  -> expected final state: in
```

These are one-time task-family constraints, not per-trajectory labels. If an
accepted trajectory ends with an unresolved lifecycle, the system searches only
the relevant post-use interval for additional evidence.

## Schema-driven targeted prompt

The targeted VLM prompt is generated from:

- entity display name;
- observation key;
- entity type;
- allowed state space;
- contact tokens;
- portable-object holding tokens.

It does not contain hard-coded dishwasher action order and is never told which
state transition is expected. It only observes before/after state and hand
relation.

## Legacy dense pass

The old cutlery-specific dense pass is disabled by default.

For controlled ablation only:

```bash
V344_USE_LEGACY_DENSE=1 bash versions/v3_4_4/run.sh ...
```

The default V3.4.4 path therefore evaluates whether generic ambiguity-triggered
targeting can replace fixed task-specific dense querying.

## GT

Use the stable version-independent variable:

```bash
GT=regression/dishwasher_episode27_manual_gt.json \
bash versions/v3_4_4/run.sh ...
```

## New outputs

```text
ambiguity_requests.json
targeted_observations.jsonl
tracked_entity_states_pass1.json
hand_object_ownership_pass1.json
skill_candidates_validated_pass1.json
interaction_validation_report_pass1.json
```

Heavy targeted strips and raw VLM responses remain under runtime output and are
not intended for Git.
