# V3.3.3 Pipeline Audit and Clean-Pipeline Principles

## Core principles

1. **Task onboarding may use human input**
   - For a new task family, a human may define the task schema once: relevant entities, useful state variables, task-relevant semantic classes, accessibility/affordance relations, and optional label hierarchy.
   - This is configuration / ontology design, not trajectory labeling.

2. **Per-trajectory inference must be automatic**
   - After the task schema is fixed, every trajectory in that task family should run from raw trajectory + coarse task instruction to hierarchical labels without human intervention.
   - Human inspection is reserved for development, debugging, GT construction, and evaluation.

3. **Reliability before semantic detail**
   - A finer label is used only when perception supports it reliably.
   - If fine identity is ambiguous, automatically back off to a task-relevant parent class.
   - Wrong fine labels are worse than correct coarse labels.

4. **Task relevance**
   - Distinctions that do not change the required manipulation policy should not be mandatory labels.
   - Example: knife and fork may be treated as `utensil` if both share the same manipulation policy and fine identity is unreliable.

5. **No episode-specific rules**
   - No frame numbers, episode IDs, known phase counts, or trajectory-specific hand-written order are allowed in inference.

## Current V3.3.3 pipeline

```text
raw HDF5 + coarse instruction + task-family schema
  -> state/action proposal generation
  -> generic multi-view VLM entity-state observations
  -> targeted dense observations
  -> trajectory-level entity tracking
  -> state-transition / manipulation-episode skill inference
  -> interaction + lifecycle + affordance validation
  -> HDF5 boundary refinement
  -> hierarchical annotations
```

## Audit: reusable vs task-specific

### A. Mostly task-agnostic mechanisms to retain

- HDF5 state/action change-point proposal generation.
- Weak treatment of raw gripper edges.
- Multi-view temporal observation rather than direct action naming.
- Confidence-weighted, graph-constrained temporal tracking.
- Motion-source and motion-endpoint temporal priors.
- Object-centric manipulation episodes.
- Interaction-grounded validation using robot motion / force / contact evidence.
- Short-reversal suppression.
- Continuous-signal boundary refinement.
- Coarse-only fallback for unresolved intervals.
- Regression evaluation.

### B. Task-family knowledge that should become configuration

Currently hard-coded in Python:

- entity names;
- state spaces;
- transition graphs;
- transition-to-skill mappings;
- contact tokens;
- object destination mappings;
- receptacle-affordance relations;
- physical accessibility relations;
- label hierarchy and language realization.

These may be defined manually once when a new task family is introduced, then reused automatically across all trajectories of that task.

### C. Over-specialized logic that should be generalized

1. Dense re-observation currently targets only `cutlery_basket`.
   It should become ambiguity-triggered targeted observation for any entity or relation.

2. Dense search windows currently depend on dishwasher-specific object identities.
   They should instead be triggered by generic uncertainty or inconsistent transitions.

3. One-container-cycle is a useful weak prior for this dataset, but should not be a universal algorithmic rule.

4. The door-open prerequisite is currently explicit dishwasher code.
   It should become a schema relation such as `requires_state(parent_entity, state)`.

5. Knife/fork identity is currently mandatory.
   It should become optional fine-grained supervision with automatic fallback to `utensil`.

## Important finding

There are currently **no episode-specific frame constants or episode-ID branches** in V3.3.3 inference.
The main generalization risk is task-family hard-coding, not episode memorization.

## Reliability-aware semantic hierarchy

```text
knife / fork
    -> utensil
       -> portable_object
```

Automatic backoff example:

```text
fine identity reliable:
  place_knife_in_cutlery_basket

fine identity unreliable, parent class reliable:
  place_utensil_in_cutlery_basket

object class unreliable, manipulation relation reliable:
  place_object_in_receptacle
```

## Recommended clean pipeline

```text
0. one-time task-family schema definition (human-assisted if needed)
1. generic proposal generation
2. broad VLM state observation
3. trajectory tracking
4. ambiguity detection
5. targeted re-observation of only ambiguous entities/relations
6. reliability-aware semantic backoff
7. generic manipulation-episode construction
8. generic physical / interaction validation using schema relations
9. boundary refinement
10. hierarchical annotation output
```

## Development protocol

For every proposed fix ask:

1. Does it encode an episode-specific fact?
2. Is it a reusable algorithmic mechanism or a task-schema relation?
3. After the schema is fixed, can it run on unseen trajectories without human intervention?
4. If perception is uncertain, can the label become coarser instead of guessing?
5. Does the semantic distinction change the robot's required policy?

A change should not enter the main inference pipeline unless it passes these checks.
