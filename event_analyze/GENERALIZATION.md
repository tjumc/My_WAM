# Generalization Analysis

## Human involvement boundary

The intended deployment model is **human-assisted task onboarding, automatic per-trajectory inference**.

For a new task family, a human may provide the task-relevant schema once, including:
- important entities / semantic classes;
- meaningful state variables;
- legal or useful transitions;
- accessibility / affordance relations;
- which fine object distinctions matter for policy learning;
- parent classes used for reliability-aware semantic backoff.

After that schema is fixed, individual trajectories should not require manual decisions.
Human inspection remains appropriate for development, debugging, ground-truth construction, and evaluation.

This is different from requiring a fully open-world task ontology. The near-term goal is:
1. configurable task-family transfer;
2. automatic processing of many unseen trajectories within each configured task family;
3. progressive reduction of task-specific code into declarative schemas.

## What the current V3.x pipeline can plausibly generalize to

The current system is best described as **within-ontology generalization**.

For a known dishwasher ontology, the architecture is designed to tolerate:
- different initial robot/base positions;
- different object positions;
- timing and execution-speed variation;
- missing proposal points;
- partial occlusion;
- noisy VLM state estimates;
- trajectory-specific variation in when a skill occurs.

This is the right scope for evaluating unseen initial-condition combinations
within the same task family.

However, this capability must still be demonstrated empirically across multiple
episodes. It should not be claimed solely from the single episode regression.

## What it does NOT currently generalize to

A completely different scene/task, such as:
- loading a washing machine;
- opening a refrigerator and retrieving food;
- setting a table;
- microwave manipulation;
- drawer/cabinet organization;

will break important V3.x assumptions because the following are hard-coded:
- entity ontology: door, dish_rack, cutlery_basket, knife, fork, plate;
- entity state spaces;
- state-transition graphs;
- skill names and transition-to-skill mapping;
- entity-specific contact tokens;
- some lifecycle priors, including the one-cycle container default.

Therefore V3.x is not task-agnostic even though several internal algorithms are.

## Which components are reusable

Mostly task-agnostic:
1. state/action proposal generation;
2. uncertainty-triggered targeted observation;
3. dense temporal re-observation;
4. trajectory-level state smoothing/tracking;
5. interaction-grounded validation;
6. HDF5/state-action boundary refinement;
7. coarse-only gaps for uncertain intervals.

Task-specific today:
1. ontology construction;
2. affordance/state definition;
3. transition graph;
4. semantic skill realization.

## Recommended task-agnostic architecture

A future V4 should add a **scene-schema induction** stage before tracking:

```text
coarse task + sparse multi-view trajectory samples
  -> open-vocabulary scene/entity discovery
  -> typed affordance schema
  -> generic state variables
  -> generic trajectory-level tracking
  -> generic transition templates
  -> language realization
```

Instead of hard-coding entity names, use typed roles such as:

- Openable(entity): closed / opening / open / closing
- Slider(entity): in / moving_out / out / moving_in
- Portable(object): support-A / hand / container-B
- Receptacle(entity): contains(object)
- RobotHand: free / approaching / contact / holding(object)

Then generic transition templates become:

```text
Openable: closed -> open
  => open(entity)

Slider: in -> out
  => pull_out(entity)

Portable: support_A -> hand -> receptacle_B
  => move/place(object, receptacle_B)
```

Natural-language labels can be generated after the symbolic transition is
identified.

## Recommended evaluation claims

For the current paper version, separate:

1. **Known-ontology, unseen-configuration generalization**
   - same task family;
   - unseen object positions / robot starts / timing;
   - evaluate automatic annotation accuracy and downstream policy generalization.

2. **Cross-scene ontology transfer**
   - future or additional experiment;
   - requires schema induction or manually supplied per-scene schemas.

This distinction keeps the scientific claim precise: V3.x can be a strong
annotation engine without pretending that a dishwasher-specific ontology is a
general household ontology.
