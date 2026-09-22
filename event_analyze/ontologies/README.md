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

At this stage these files document the target architecture. V3.3.3 runtime is intentionally unchanged so existing regression results remain reproducible.
