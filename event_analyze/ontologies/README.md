# Task Ontologies

This directory is the planned boundary between generic inference algorithms and task-family knowledge.

For a new task family, a human may define the ontology/schema once:
- relevant entities and their types;
- meaningful states;
- legal transitions;
- accessibility and affordance relations;
- task-relevant semantic classes;
- optional parent classes for reliability-aware label backoff.

After that schema is fixed, per-trajectory inference should be automatic.

The first draft, `dishwasher_loading.json`, introduces:

```text
knife / fork -> utensil -> portable_object
```

This lets the annotation system prefer a correct coarse label over an unreliable fine-grained label without asking a human to resolve each trajectory.

V3.3.3 remains unchanged for reproducibility. V3.4.0 is the first runtime that consumes the ontology for tracking, skill inference, lifecycle, accessibility, and affordance reasoning. Perception schema generation is the next migration step.


## Reliability-first supervision

A semantic class can declare `training_label_policy: "parent"` when its members
are policy-equivalent and fine identity is not worth unreliable supervision.
V3.4.1 consumes this policy. Fine identity remains available as diagnostic
metadata, while the training label uses the configured parent class.


## Manipulation concurrency policy

A task schema may declare how many portable objects can be actively manipulated
at once. For the current sequential dishwasher demonstrations:

```json
"manipulation_policy": {
  "max_concurrent_portable_objects": 1
}
```

This is task-family configuration, not a universal robotics assumption.
Concurrent/bimanual tasks should use a different value and ownership policy.


## Task completion and targeted perception

V3.4.4 adds task-schema fields that describe task-family completion and usage
semantics without labeling individual trajectories:

- `usage_state`: the state in which a receptacle is usable for object placement;
- `expected_final_state`: the expected state at task completion;
- `targeted_perception`: generic ambiguity-query budget and confidence policy.

For the dishwasher family, the two receptacles use `out` as their usage state
and return to `in` at completion; the door returns to `closed`.

These declarations are used only to detect reasoning inconsistencies and choose
where to re-observe. The targeted VLM prompt is not told which transition it is
supposed to find.
