# V3.4.2 — Joint Hand-Object Temporal Ownership

V3.4.2 addresses the main temporal-grounding failure observed after V3.4.1:
portable-object tracks are individually plausible but can compete for the same
hand/time interval.

## Motivation

Examples from the development trajectories:

```text
episode26:
knife-like utensil [707,772]
fork-like utensil  [707,896]
```

and:

```text
episode27:
plate track enters right_hand before the directly observed plate-holding window,
causing plate to start before the cutlery-basket push.
```

Independent object tracking has no global resource constraint, so these
conflicts can survive into skill inference.

## New stage

```text
independent entity tracks
        ↓
joint hand-object ownership arbitration
        ↓
ownership-resolved tracks
        ↓
completion-aware manipulation episodes
```

The resolver uses:

- direct structured hand identity evidence;
- object-location evidence;
- non-portable hand interaction blockers;
- task-schema concurrency policy;
- source/target trajectory continuity;
- direct object-context switches.

## Evidence ownership

A key correction relative to V3.4.1 is that sibling evidence cannot be reused
to validate two competing object actions.

For example, an observed `holding_fork` interval is not simultaneously used as
the held-state evidence for a knife-like candidate.

A coarse policy-equivalent label can still be accepted under occlusion when:

1. the candidate has a uniquely assigned ownership interval;
2. its source and target states are directly observed;
3. robot interaction evidence supports the transfer.

Thus semantic backoff remains reliability-first without duplicating evidence.

## Direct identity priority

When a structured observation contains conflicting fields, explicit hand
identity has priority over object-location classification.

Example:

```text
right_hand = holding_plate
knife_location = right_hand
```

is treated as an ambiguity in `knife_location`, not as evidence that the right
hand owns both plate and knife.

## Task schema

The schema now contains:

```json
"manipulation_policy": {
  "max_concurrent_portable_objects": 1,
  "allow_inferred_parent_ownership": true
}
```

This is task-family configuration rather than a universal assumption. A
genuinely concurrent/bimanual task may configure a larger concurrency limit.

## Outputs

V3.4.2 adds:

```text
hand_object_ownership.json
tracked_entity_states_owned.json
```

The ownership report records direct ownership intervals, blockers, arbitration
decisions, and unresolved conflicts. Unresolved conflicts are intentionally
preserved so a later ambiguity-triggered perception stage can revisit them.

## Controlled comparison

V3.4.2 reuses V3.4.1 perception by default. Therefore changes relative to
V3.4.1 can be attributed to ownership arbitration rather than a fresh Qwen run.
