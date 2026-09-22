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
