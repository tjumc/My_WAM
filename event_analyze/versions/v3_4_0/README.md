# V3.4.0 — Schema-Driven Reasoning

V3.4.0 is the first architecture-cleanup version. It does **not** aim to add
more dishwasher-specific heuristics.

## Goal

Separate:

```text
generic annotation algorithms
        +
human-configurable task-family schema
```

A human may define a schema once for a new task family. After that, individual
trajectories should be processed automatically.

## Schema-driven in V3.4.0

The following reasoning components now read the task schema:

- tracker entity/state definitions;
- legal state-transition graphs;
- motion source / endpoint relations;
- container transition-to-skill mappings;
- portable-object source / hand / target states;
- object target receptacles;
- object placement skill mappings;
- container lifecycle transitions;
- accessibility prerequisites;
- receptacle-placement affordance relations;
- skill language labels.

Default schema:

```text
event_analyze/ontologies/dishwasher_loading.json
```

## Still task-specific in this stage

Perception is intentionally held fixed for controlled comparison:

- V3.3.3 generic VLM prompt still explicitly names dishwasher entities;
- dense re-observation still specifically targets cutlery_basket.

These are the next components to make schema-driven after reasoning equivalence is verified.

## Reliability-first ontology

The schema includes:

```text
knife / fork -> utensil -> portable_object
```

This is preparatory metadata for automatic semantic backoff. V3.4.0 does not yet
change final labels based on this hierarchy; first we verify that schema-driven
reasoning reproduces V3.3.3 behavior.

## Run

```bash
bash versions/v3_4_0/run.sh \
  /path/to/episode.hdf5 \
  output/episode_analysis \
  "put the dish into the dishwasher"
```

Optional custom schema is the sixth positional argument.

When V3.3.3 outputs exist, perception artifacts are reused automatically so any
difference is attributable to the schema-driven reasoning refactor.
