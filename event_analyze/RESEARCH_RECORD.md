# Research Record: Problems, Solutions, and Contribution Candidates

> This document records the methodological evolution of `event_analyze`.
> It is intentionally different from a software changelog: the goal is to
> preserve the **scientific problem -> diagnosis -> general solution -> evidence**
> chain, and to distinguish paper-level ideas from local engineering fixes.
>
> Last updated: V3.4.4.

---

## 1. Research objective

The project starts from long-horizon humanoid household demonstrations that
contain multi-view video, robot state/action trajectories, and only a coarse
task-level instruction.

The target is to automatically produce temporally grounded hierarchical
language supervision:

```text
L0 task goal
L1 trajectory-specific instruction
L2 semantic phases / subtasks
L3 atomic actions when reliable
```

The central research question is:

> How can coarse task-level demonstrations be converted into reliable,
> temporally aligned, task-relevant hierarchical annotations without requiring
> manual labeling of every trajectory?

The current operating principle is:

> **Human-assisted task onboarding, automatic per-trajectory inference.**

A human may define a task-family schema once. After the schema is fixed,
individual trajectories in that task family should be processed automatically.

A second hard principle is:

> **Reliability has priority over semantic detail.**

A correct coarse label is preferred to a detailed but unreliable label.

---

## 2. Scientific principles established during development

### 2.1 Perception is not phase generation

The VLM should primarily answer structured state questions:

- what state is an articulated object in?
- where is a portable object?
- what is the hand interacting with?
- is an object visibly held?
- has a receptacle moved in/out?

It should not be responsible for independently generating the final semantic
phase sequence from each local window.

Current formulation:

> Qwen is responsible for seeing; trajectory-level reasoning organizes local,
> noisy, and potentially inconsistent visual observations into temporally
> coherent skills.

### 2.2 Robot control changes are proposals, not semantic boundaries

Low-level control signals can contain:

- accidental gripper open/close commands;
- failed grasp attempts;
- re-grasps;
- corrective movements;
- transient force/contact changes.

Therefore a raw action change is not itself a semantic phase transition.

Current principle:

> State/action signals propose where to inspect. Semantic phases are inferred
> from persistent object-state transitions and hand-object interaction.

### 2.3 Task semantics should be task-relevant, not maximally detailed

If two visual categories require the same robot policy, fine distinction is not
mandatory supervision.

Dishwasher example:

```text
knife ----\
          -> utensil -> cutlery_basket
fork  ----/
```

The final training label may be:

```text
place_utensil_in_cutlery_basket
```

even when optional diagnostic metadata keeps a knife/fork hypothesis.

---

# 3. Problems encountered and what we learned

## P1. Coarse task instruction does not describe long-horizon internal structure

### Observation

A demonstration may only have:

```text
put the dish into the dishwasher
```

while the actual trajectory contains navigation, door manipulation, rack
manipulation, multiple object transfers, receptacle closing, etc.

### Why it matters

A policy trained with only one fixed instruction receives little explicit
supervision about temporal task structure. The same task instruction covers
very different robot states and required actions.

### Solution direction

Build hierarchical temporally aligned phase annotations automatically from
video + state/action trajectories.

### Status

**Core research problem; pipeline exists, accuracy still being improved.**

### Contribution potential

**High.** This is the overall problem formulation.

---

## P2. Raw gripper edges are semantically unreliable

### Observation

Manual inspection showed that arm motion may accidentally trigger gripper
opening/closing. Knife/fork manipulation may also contain failed grasps and
re-grasps.

A naive method that treats each gripper edge as a phase boundary therefore
over-segments the trajectory.

### Diagnosis

Low-level actuation events are not equivalent to semantic actions.

### Solution

- weaken gripper events during proposal generation;
- cluster nearby retries;
- treat state/action changes as event proposals rather than final boundaries;
- require persistent object-state / interaction evidence for semantic skills.

### Status

**Largely solved at the design level.**

The pipeline no longer defines semantic phases directly from raw gripper edges.

### Contribution potential

**Medium as an isolated idea; high as part of the complete method.**

The useful paper message is not "we tuned gripper thresholds", but:

> low-level robot control events are used only as event proposals, while
> semantic skills are inferred from persistent state transitions and
> interaction evidence.

---

## P3. Independent local VLM judgments are temporally inconsistent

### Observation

The same entity can be judged differently in neighboring windows because of:

- occlusion;
- camera viewpoint;
- transient motion;
- ambiguous object identity;
- VLM stochasticity.

Examples included container states flipping between `in/out` or
`open/closed`, and portable-object locations jumping between tabletop, hand,
target, and other.

### Diagnosis

Independent before/after classification lacks trajectory-level temporal
coherence.

### Solution

Trajectory-level graph-constrained state tracking:

- confidence-weighted emissions;
- persistence;
- legal state-transition graphs;
- motion source priors;
- motion endpoint priors;
- direct-vs-interpolated support metadata.

### Important failure discovered

A Viterbi path can incorrectly propagate a later stable state backward to the
trajectory start. For example, a door may appear `open` at the beginning of
the decoded track even when a later directly observed `opening` event implies
that it had to be `closed` immediately beforehand.

### General fix

Use causal motion-source evidence:

```text
opening    -> source must be closed
closing    -> source must be open
moving_out -> source must be in
moving_in  -> source must be out
```

and infer initial lifecycle from early **direct observations**, not from the
smoothed tracker output.

### Status

**Substantially solved, but perception errors remain upstream.**

### Contribution potential

**High as a component.**

This supports the argument for trajectory-level state reasoning rather than
window-wise action captioning.

---

## P4. Directional motion evidence is useful even when absolute state is uncertain

### Observation

Dense VLM inspection may reliably say:

```text
the basket is moving outward
```

while being less reliable about whether the endpoint is absolutely `in` or
`out`.

### Initial mistake

Treating weak dense observations as absolute state labels can create false
state anchors.

### Solution

Represent dense motion as directional transition evidence:

```text
moving_out -> weak endpoint prior: out
moving_in  -> weak endpoint prior: in
```

rather than pretending that the endpoint was directly observed.

### Status

**Implemented.**

### Contribution potential

**Medium.**

Useful as part of the state-transition grounding mechanism rather than a
standalone paper contribution.

---

## P5. Visual state changes alone can hallucinate semantic actions

### Observation

A tracker can produce a plausible transition even when there is little evidence
that the robot actually interacted with that entity.

### Diagnosis

Pure vision/state tracking can confuse:

- occlusion-induced state changes;
- camera ambiguity;
- unrelated scene motion;
- physically impossible transition ordering.

### Solution

Interaction-grounded validation using:

- arm motion;
- force/wrench change;
- gripper change;
- hand-entity contact relations;
- direction-consistent visual transition evidence.

Additionally use physical lifecycle / affordance constraints, e.g.:

```text
container must be accessible before manipulating its contents
receptacle pull-out should precede use
receptacle push-in should follow use
```

### Status

**Implemented; still imperfect when contact perception is noisy.**

### Contribution potential

**High as part of the framework.**

A strong paper framing is:

> semantic transitions are accepted only when jointly supported by visual
> entity-state change and robot interaction / physical consistency.

---

## P6. Task-specific physical knowledge was hard-coded in Python

### Observation

Earlier versions directly encoded names and rules such as:

- door;
- dish_rack;
- cutlery_basket;
- knife/fork/plate;
- transition-to-skill mappings;
- accessibility relations.

This made it unclear whether the method could transfer beyond one task.

### Diagnosis

The algorithms contained both:

1. reusable inference logic; and
2. task-family knowledge.

These two layers were mixed together.

### Solution: V3.4.0 schema-driven reasoning

Separate:

```text
generic reasoning algorithms
        +
human-configurable task-family schema
```

The schema now defines:

- entities;
- state spaces;
- legal transitions;
- motion source / endpoint relations;
- object target receptacles;
- accessibility constraints;
- transition-to-skill mappings;
- language labels;
- semantic hierarchy.

### Evidence

Using the same perception evidence:

- V3.4.0 reproduced V3.3.3 exactly on episode26;
- V3.4.0 reproduced V3.3.3 exactly on episode27;

including skill sequence, boundaries, and confidence values.

This controlled comparison shows that dishwasher-specific reasoning knowledge
can be moved from code into configuration without changing behavior.

### Status

**Reasoning layer solved.**

Perception prompts are still partly dishwasher-specific, so full task-schema
driven perception is not finished.

### Contribution potential

**High if cross-task experiments validate it.**

Potential framing:

> A human-configurable task schema specifies task-relevant entities, state
> variables and physical relations once, after which trajectory annotation is
> automatic.

This is stronger and more realistic than claiming fully open-world task
understanding.

---

## P7. Fine object identity can be less reliable than the action semantics

### Observation

Knife and fork are visually similar and may be confused, especially during
grasping and occlusion.

However, for the current robot policy, both require the same behavior:

```text
pick utensil from source
-> put utensil into cutlery basket
```

### Diagnosis

Forcing the annotator to choose knife vs fork can create incorrect supervision
without adding control-relevant information.

### Solution: reliability-first semantic backoff

V3.4.1 introduces task-relevant semantic abstraction:

```text
knife / fork -> utensil
```

Final training supervision uses:

```text
place_utensil_in_cutlery_basket
```

while optional fields preserve:

- `fine_entity`;
- `fine_skill_type`;
- `semantic_backoff` provenance.

If exact fine held evidence is missing but a policy-equivalent sibling provides
reliable held evidence, the coarse parent action may be accepted without
claiming the fine identity was correct.

### Evidence

Episode26 changed from missing the knife phase to two utensil placements.

At policy-normalized sequence level:

```text
episode26 V3.4.1:
Precision = 1.000
Recall    = 0.900
F1        = 0.947
Edit      = 1
```

The only missing policy-level semantic phase is currently
`push_in_cutlery_basket`.

Episode27 policy-normalized sequence remains:

```text
Precision = 0.900
Recall    = 0.900
F1        = 0.900
Edit      = 2
```

### Status

**Implemented and showing useful behavior.**

### Contribution potential

**Very high.**

Potential paper concept:

> **Reliability-aware task-relevant semantic granularity**: the system emits the
> finest label that is both task-relevant and supported by perception, backing
> off to a policy-equivalent parent semantic class instead of guessing.

This idea is broader than knife/fork and can generalize to mug/cup, bottle/can,
etc., when fine identity does not change the required policy.

---

## P8. Re-grasp and object adjustment caused duplicate semantic phases

### Observation

Episode26 plate tracking contained:

```text
hand -> dish_rack -> hand -> dish_rack -> hand -> dish_rack
```

Earlier logic used a fixed target dwell threshold. Because intermediate target
contacts lasted around the threshold, one true plate placement was emitted as
three separate semantic skills.

### Diagnosis

Target dwell duration alone is not a semantic completion criterion.

A robot can place, adjust, and re-grasp the same object without starting a new
task-level action.

### Solution: completion-aware object-centric manipulation episode

V3.4.1 closes an object manipulation only when the object reaches the target
and there is no later re-grasp before a clear object-context switch.

Thus:

```text
hand -> target -> hand -> target
```

remains one semantic episode when the robot is still working on the same
object.

### Evidence

Episode26:

```text
V3.4.0:
place_plate
place_plate
place_plate

V3.4.1:
place_plate   (one episode, retry_collapsed=true)
```

The resulting plate episode spans the whole adjustment sequence rather than
three artificial phases.

### Status

**Solved for the observed retry/adjustment pattern.**

### Contribution potential

**Very high.**

Potential framing:

> Completion-aware object-centric manipulation episodes distinguish transient
> placement/re-grasp adjustments from true semantic completion.

This is substantially stronger than a hand-tuned retry time threshold.

---

## P9. Missing fine held-state evidence can cause a true action to be rejected

### Observation

Episode26 produced a plausible knife transfer:

```text
tabletop -> other -> left_hand(interpolated) -> cutlery_basket
```

but the validator rejected the fine knife action because no sufficiently
confident **observed** knife-held state existed.

At the same time, a policy-equivalent utensil held-state could be visible.

### Initial behavior

Reject the action entirely:

```text
no_observed_held_state
```

### Current solution

At policy-equivalent parent level, accept a coarse utensil placement if there is
reliable held evidence for a member of the same semantic parent class.

### Status

**Partially solved by semantic backoff.**

The broader problem of recovering the correct identity under ambiguity remains.

### Contribution potential

Part of **reliability-aware semantic backoff**.

A future extension is ambiguity-triggered targeted re-observation.

---

## P10. Fixed fine-label evaluation becomes misleading after intentional semantic backoff

### Observation

If GT says:

```text
knife, fork
```

and the method intentionally outputs:

```text
utensil, utensil
```

ordinary exact-label evaluation counts two semantic errors even though the
coarser labels are the intended supervision granularity.

### Solution

Evaluation now supports two views:

1. raw / fine-label metrics;
2. policy-normalized metrics using the task schema.

Episode27 V3.4.1 demonstrates the distinction:

```text
raw sequence F1    = 0.70
policy sequence F1 = 0.90
```

The policy-normalized temporal metrics on episode27 are currently approximately:

```text
semantic matched mIoU = 0.392
semantic boundary MAE = 1.94 s
core matched mIoU     = 0.374
core boundary MAE     = 1.86 s
```

### Status

**Implemented.**

### Contribution potential

**Medium/high as an evaluation principle** supporting reliability-aware
hierarchical semantics.

---

## P11. Receptacle push-in remains difficult

### Observation

Episode26 still misses:

```text
push_in_cutlery_basket
```

even after the object semantics improved.

Previous analysis showed multiple noisy cutlery-basket transitions and candidate
push/pull events. Candidates may be rejected because of:

- short reverse/jitter filtering;
- inaccurate placement-end timing;
- insufficient entity-specific contact evidence;
- noisy basket state tracking;
- affordance ordering constraints.

### Status

**Unresolved.**

### Scientific importance

This is evidence that container-state perception and interaction validation are
still not robust enough.

### Contribution potential

The specific missing action is **not** itself a paper contribution.

A general solution may become a contribution if it yields:

> robust ambiguity-triggered re-observation or joint physical-temporal
> inference for articulated receptacle interactions.

---

## P12. Object temporal ownership is still inaccurate

### Observation A: episode26 utensil overlap

V3.4.1 recovers two utensil transfers, but their provisional spans overlap
strongly:

```text
utensil-1: 707 -> 772
utensil-2: 707 -> 896
```

Final boundary refinement is then forced to create non-overlapping labels and
compresses the first raw phase to only a few frames.

### Observation B: episode27 plate begins too early

The plate candidate starts around the fork/basket interaction region. This makes
the predicted order:

```text
fork -> plate -> push basket
```

while manual GT is:

```text
fork -> push basket -> plate
```

The predicted push-basket timing itself is near the correct region; the larger
problem is that plate temporal ownership begins too early.

### Diagnosis

Object tracks are still inferred mostly independently. When multiple object
identities compete for the same hand/time region, there is no sufficiently
strong joint assignment mechanism.

### Status

**Implemented in V3.4.2 with initial controlled positive evidence.**

V3.4.2 introduces a separate ownership-arbitration stage before manipulation
episode inference. It enforces task-configured portable-object concurrency,
prioritizes explicit hand identity over conflicting object-location fields,
trims speculative held intervals at direct object-context switches, and
prevents sibling objects from reusing the same held evidence.

Using exactly the same generic and dense perception evidence as V3.4.1:

- episode27 plate provisional start moved from 931 to 1044;
- the predicted order changed from
  `utensil -> plate -> push basket` to
  `utensil -> push basket -> plate`;
- episode27 policy-normalized sequence improved from
  `F1=0.90, Edit=2` to `F1=1.00, Edit=0`;
- policy semantic mIoU improved from approximately 0.392 to 0.423;
- policy semantic boundary MAE improved from approximately 1.94 s to 1.51 s;
- episode26 overlapping utensil provisional spans changed from
  `[707,772]` and `[707,896]` to approximately
  `[707,739]` and `[739,896]`.

The core mIoU on episode27 decreased slightly (approximately 0.374 -> 0.354),
so the result should be reported as a sequence / semantic-boundary improvement
with a small core-IoU trade-off, not as universal improvement on every metric.

### Implemented solution direction

Joint hand-object temporal ownership:

```text
one hand -> at most one actively held tracked object
```

combined with:

- trajectory continuity;
- source/target state consistency;
- object-context switch reasoning;
- confidence;
- optional targeted VLM re-observation only in ambiguous intervals.

### Contribution potential

**High if solved generically.**

Potential framing:

> Joint temporal hand-object assignment resolves conflicting object identities
> and assigns each manipulation interval to a physically consistent object
> track.

---

## P13. Dense targeted perception is currently too task-specific

### Observation

The current dense pass is specialized around cutlery-basket transitions.

This improves a known difficult entity but does not provide a generic mechanism
for new tasks.

### Desired general solution

Automatically detect ambiguity such as:

- source -> target with no reliable held state;
- conflicting object identities for one hand;
- target -> hand -> target adjustment;
- physically inconsistent transition;
- uncertain container direction.

Then automatically create a focused re-observation request for only that
interval/entity/relation.

### Status

**Implemented in V3.4.4; controlled evaluation pending.**

V3.4.4 adds a two-pass closed loop:

```text
reasoning pass 1
-> generic ambiguity detection
-> schema-driven targeted VLM observation
-> merge confident evidence
-> reasoning pass 2
```

The default V3.4.4 path disables the old cutlery-specific dense pass. The new
detector can trigger on unresolved ownership, visually rejected candidates,
receptacle use without a ready lifecycle, unfinished expected final lifecycle,
or accepted articulated transitions lacking directional visual evidence.

### Contribution potential

**Very high if controlled experiments confirm useful error recovery at lower VLM query cost than uniform dense re-observation.**

Potential framing:

> Ambiguity-triggered perception refinement selectively revisits temporally
> inconsistent entity relations rather than densely querying the full
> trajectory.

This may reduce both VLM cost and error rate.

---

## P14. Perception itself is not fully schema-driven yet

### Observation

V3.4.0 successfully made tracking/inference/validation schema-driven, but the
base VLM prompt still explicitly asks about dishwasher-specific entities.

### Status

**Partially solved in V3.4.4.**

Targeted re-observation prompts are now generated from the task schema, including
entity names, observation keys, state spaces, contact tokens, holding tokens,
usage states and completion expectations.

The broad/base observation prompt is still V3.3.3-compatible and therefore
dishwasher-specific.

### Next direction

Generate the broad structured VLM observation prompt automatically from the task schema:

```text
schema
-> entity/state question templates
-> broad observation
-> ambiguity-triggered focused observation
```

### Contribution potential

**High if cross-task transfer is demonstrated.**

This would complete the separation between generic algorithms and task-family
configuration.

---

## P15. Generalization evidence is still too small

### Current development set

- episode26: manually inspected sequence-only provisional GT;
- episode27: frame-level manual GT.

These trajectories have been repeatedly inspected and used for debugging.

### Consequence

They are development/regression trajectories, not convincing held-out evidence.

### Required future experiments

At minimum:

1. more unseen dishwasher trajectories;
2. varying object initial positions / heights / robot starts;
3. failure/retry-heavy trajectories;
4. at least one different task family with a separately configured schema;
5. downstream policy training comparison using:
   - coarse instruction only;
   - fine labels from the proposed pipeline;
   - possibly manual oracle labels.

### Status

**Major unresolved experimental requirement.**

### Contribution potential

Not a method contribution, but essential for validating all claimed
contributions.

---

## P16. Boundary refinement can invalidate upstream semantic ownership

### Observation

After V3.4.2 successfully separated episode26's two utensil provisional spans,
the first utensil was still reduced by final raw-boundary refinement to only a
few frames.

The failure was caused by pairwise overlap resolution being applied
left-to-right. A middle phase could first have its start pushed forward by the
left neighbor, then have its end pushed backward by the right neighbor.

### Diagnosis

Boundary refinement was acting as if it could independently redefine semantic
ownership. It should instead be a numerical calibration stage downstream of
semantic reasoning.

### V3.4.3 solution

- treat provisional semantic intervals as ownership priors;
- resolve overlap right-to-left so the right boundary is known before the left
  boundary is allowed to move;
- enforce skill-specific minimum duration during overlap arbitration whenever
  feasible;
- explicitly expose infeasible duration conflicts.

### Status

**Implemented and validated on the current regression trajectories.**

Episode26's first utensil raw span changed from the V3.4.2 two-frame collapse
`[744,745]` to `[714,738]` (25 frames), while the two utensil phases remained
separate and the policy sequence stayed unchanged.

### Contribution potential

**Medium as a method principle.**

The broader formulation is useful:

> low-level continuous signals refine semantic boundaries but must not silently
> invalidate higher-level temporal ownership constraints.

---

## P17. Boundary relocation can make validation evidence stale

### Observation

A container candidate could be validated using robot signal at one interval and
then be relocated to a distant hand-contact anchor afterward.

Episode27 exposed this clearly: the final push-basket interval was near frame
999 while the stored signal window used for acceptance was centered much later.

### Diagnosis

Changing the candidate's temporal hypothesis after validation breaks
evidence-boundary consistency.

### V3.4.3 solution

```text
candidate
-> contact evidence
-> optional contact-anchor relocation
-> recompute robot signal at relocated interval
-> final validation
```

A relocated candidate must pass robot-signal validation at its final anchor.

### Status

**Implemented and validated on the current regression trajectories.**

For episode27, contact-anchor-adjusted container skills now store robot signal
evidence around their adjusted provisional intervals. In particular,
`push_in_cutlery_basket` is revalidated with signal window approximately
`[979,1020]` around final provisional interval `[999,1000]`, instead of
reusing the earlier stale signal window around frame 1112.

### Contribution potential

**Medium as part of interaction-grounded temporal validation.**

---

# 4. Version evolution and what each version taught us

## V3.3.1

Key result:

- episode27 recognized most phases;
- episode26 generalized poorly and missed several actions.

Episode26 output contained only six phases, exposing that successful performance
on one trajectory did not imply within-task generalization.

Main lesson:

> the pipeline needed stronger trajectory-level state reasoning and more robust
> treatment of retries / initial lifecycle.

---

## V3.3.2

Major experiments:

- weakened gripper proposals;
- retry clustering;
- dense cutlery directional evidence;
- initial lifecycle logic;
- affordance ordering.

What it taught us:

- direction-only dense evidence is safer than fabricated absolute anchors;
- lifecycle initialization from a smoothed Viterbi track is unsafe;
- adjacent-only duplicate removal is insufficient;
- object placement should be modeled as a manipulation episode, not a list of
  independent state transitions.

---

## V3.3.3

Major additions:

- object-centric manipulation episodes;
- dense motion endpoint constraints;
- causal motion-source priors;
- early direct-observation lifecycle initialization;
- accessibility prerequisites;
- physical/interaction validation improvements.

Important result:

- episode26 sequence quality improved substantially relative to V3.3.1;
- episode27 recovered all ten skill types, although plate/push-basket ordering
  remained wrong.

Main lesson:

> Most remaining failures were no longer simple missing phases; they were
> identity, semantic-completion, and temporal-ownership failures.

---

## V3.4.0

Purpose:

**architectural cleanup, not accuracy improvement.**

Moved task-family reasoning knowledge into
`ontologies/dishwasher_loading.json`.

Controlled evidence:

- V3.4.0 == V3.3.3 on episode26;
- V3.4.0 == V3.3.3 on episode27;

under reused perception evidence.

Main lesson:

> task-specific knowledge can be separated from generic reasoning without
> changing current behavior.

---

## V3.4.1

Major additions:

1. reliability-first policy-level semantic backoff;
2. completion-aware object manipulation episodes;
3. policy-normalized evaluation.

Observed improvements:

### Episode26

V3.4.0:

```text
... fork, plate, plate, plate, push rack ...
```

V3.4.1:

```text
... utensil, utensil, plate, push rack ...
```

Policy-level:

```text
P=1.00, R=0.90, F1=0.947, Edit=1
```

Remaining missing phase:

```text
push_in_cutlery_basket
```

### Episode27

Policy-level remains:

```text
P=0.90, R=0.90, F1=0.90, Edit=2
```

Remaining ordering issue:

```text
pred: fork/utensil -> plate -> push basket
GT:   fork/utensil -> push basket -> plate
```

Main lesson:

> reliability-aware abstraction and semantic completion are useful, but the
> next bottleneck is **temporal ownership among competing object tracks**.

---

## V3.4.2

Purpose:

**joint temporal ownership before object-skill inference.**

Major additions:

1. direct hand-object ownership extraction from structured observations;
2. task-configured portable-object concurrency;
3. non-portable interaction blockers;
4. unique evidence ownership across competing object candidates;
5. ownership-aware context-switch boundaries;
6. explicit ambiguous-conflict output for future targeted re-observation.

Controlled regression results with the same V3.4.1 perception evidence:

### Episode26

- policy sequence remains `P=1.00, R=0.90, F1=0.947, Edit=1`;
- the two utensil provisional intervals are separated at the next object's
  direct ownership boundary;
- `push_in_cutlery_basket` is still missing.

### Episode27

- plate provisional start moves from 931 to 1044;
- `push_in_cutlery_basket` now precedes plate placement;
- policy-normalized sequence reaches
  `P=1.00, R=1.00, F1=1.00, Edit=0`;
- semantic mIoU improves from about 0.392 to 0.423;
- semantic boundary MAE improves from about 1.94 s to 1.51 s.

Main lesson:

> Joint ownership can correct action ordering using the same perception
> evidence, but final boundary realization must preserve the ownership intervals
> instead of independently collapsing them.

---

## V3.4.3

Purpose:

**make final temporal boundaries consistent with upstream semantic ownership and
with the evidence used to validate them.**

Major additions:

1. right-to-left duration-constrained overlap arbitration;
2. provisional semantic intervals treated as ownership priors;
3. explicit boundary duration-conflict metadata;
4. contact-anchor relocation before final robot-signal decision;
5. robot-signal recomputation after any container boundary relocation.

Controlled regression results:

### Episode26

- first utensil raw span changes from `[744,745]` to `[714,738]`;
- second utensil remains separate at `[739,895]`;
- plate remains one completion-aware episode;
- policy sequence remains `P=1.00, R=0.90, F1=0.947, Edit=1`;
- `push_in_cutlery_basket` is still missing.

### Episode27

- policy-normalized sequence remains exact:
  `P=1.00, R=1.00, F1=1.00, Edit=0`;
- plate remains after `push_in_cutlery_basket`;
- semantic mIoU improves slightly from about 0.423 to 0.430;
- semantic boundary MAE improves slightly from about 1.51 s to 1.48 s;
- core mIoU and core boundary MAE remain approximately unchanged from V3.4.2;
- every contact-anchor-adjusted container action is revalidated using signal
  evidence around its adjusted interval.

Main lesson:

> Continuous-signal refinement should calibrate an already inferred semantic
> interval, while validation evidence must be recomputed whenever that interval
> is relocated.

---

## V3.4.4

Purpose:

**replace fixed task-specific dense perception with reasoning-triggered,
schema-driven targeted re-observation.**

Major additions:

1. first-pass ambiguity detection from reasoning outputs;
2. generic ownership-conflict trigger;
3. receptacle-use / lifecycle-consistency triggers;
4. schema-declared `usage_state` and `expected_final_state`;
5. schema-generated targeted observation prompts;
6. confidence-gated targeted evidence merge;
7. second reasoning pass after new evidence;
8. compact logging of targeted-query count and merge count.

Default behavior:

- broad/base observations may be reused from V3.4.3;
- legacy cutlery-specific dense evidence is disabled by default;
- `V344_USE_LEGACY_DENSE=1` is reserved for ablation;
- manual GT is not visible to ambiguity detection or targeted perception.

Key pending checks:

### Episode26

- does the generic loop detect the unresolved cutlery-basket lifecycle?
- can targeted observation recover `push_in_cutlery_basket` without the old
  hard-coded dense cutlery search?
- are existing utensil and plate phases preserved?

### Episode27

- does the method avoid unnecessary queries when reasoning is already coherent?
- is the exact 10-step policy-normalized sequence preserved?
- what is the targeted-query cost relative to the old uniform dense pass?

These are implementation hypotheses until V3.4.4 regression results are
committed.

---

# 5. Current problem status summary

| Problem | Status | Current interpretation |
|---|---|---|
| coarse task instruction lacks internal phases | active core problem | main research motivation |
| raw gripper edges cause false boundaries | mostly solved | proposal cue only |
| local VLM observations inconsistent over time | substantially solved | graph-constrained tracking |
| Viterbi backward lifecycle leakage | solved | causal motion-source + early direct state |
| weak directional evidence treated as absolute state | solved | transition endpoint prior |
| vision-only transitions create false skills | partially solved | interaction + physical validation |
| task knowledge hard-coded in reasoning | solved in reasoning layer | schema-driven V3.4.0 |
| knife/fork fine identity unreliable | solved at policy-label level | utensil backoff |
| re-grasp creates duplicate plate skills | solved for current pattern | completion-aware episodes |
| fine-vs-coarse evaluation mismatch | solved | policy-normalized metrics |
| push-in cutlery basket missing | V3.4.4 targeted recovery pending eval | generic lifecycle-triggered re-observation |
| overlapping utensil temporal ownership | provisionally solved in V3.4.2 | joint hand-object ownership |
| plate starts too early in episode27 | solved on regression episode27 | ownership/context-switch reasoning |
| final boundary can collapse a valid phase | solved on current regression cases | duration-constrained semantic boundary arbitration |
| relocated boundary may use stale robot signal | solved on current regression cases | post-anchor signal revalidation |
| targeted dense perception is task-specific | V3.4.4 implemented, pending eval | generic ambiguity-triggered schema-driven targeting |
| base VLM perception prompt is task-specific | partially unresolved | targeted prompt is schema-driven; broad prompt still task-specific |
| held-out / cross-task experimental evidence | unresolved | required for paper |

---

# 6. Candidate paper contributions

The following list is intentionally conservative. A candidate should only become
a claimed contribution after sufficient experiments.

## Contribution Candidate A — State-action-grounded trajectory-level skill discovery

### Core idea

Use low-level robot state/action only to propose events, then infer semantic
skills from persistent entity-state transitions and hand-object interactions
over the whole trajectory.

### Includes

- multi-view structured state observation;
- graph-constrained temporal state tracking;
- state-transition-based skill inference;
- robot interaction validation;
- physical lifecycle / affordance consistency;
- continuous-signal boundary refinement.

### Current maturity

**Implemented.**

### Evidence still needed

Large-scale held-out trajectory evaluation and comparisons against:

- direct VLM phase captioning;
- action-change segmentation;
- vision-only state-transition reasoning.

### Paper potential

**Primary contribution candidate.**

---

## Contribution Candidate B — Reliability-aware task-relevant semantic granularity

### Core idea

Do not force the finest visual category when that distinction is unreliable or
irrelevant to control.

Emit the finest **reliable and policy-relevant** semantic class.

Example:

```text
knife / fork -> utensil
```

### Key property

Fine identity is preserved as optional diagnostic metadata, while training
supervision backs off to a policy-equivalent parent class.

### Current maturity

**Implemented in V3.4.1 with initial evidence.**

### Paper potential

**Primary contribution candidate.**

This is more general and scientifically stronger than merely merging knife and
fork manually.

---

## Contribution Candidate C — Completion-aware object-centric manipulation episodes

### Core idea

A semantic object action is not completed at the first target contact.

Re-grasp / adjustment remains part of the same skill until there is evidence of
true completion or a context switch.

### Current maturity

**Implemented; episode26 duplicate plate phases collapsed from three to one.**

### Paper potential

**Primary or strong secondary contribution candidate.**

Particularly relevant to real robot demonstrations containing failed grasps,
retries, and corrective manipulation.

---

## Contribution Candidate D — Human-configurable task schema with automatic trajectory annotation

### Core idea

Separate reusable inference logic from task-family knowledge.

Human intervention is allowed once at task onboarding to specify:

- entities;
- meaningful states;
- transitions;
- task-relevant semantic classes;
- physical prerequisites / affordances.

Afterward, all trajectories in the task family are processed automatically.

### Current maturity

**Reasoning layer implemented and behavior-equivalent to the earlier hard-coded
pipeline.**

### Missing piece

Perception prompt generation is not yet fully schema-driven.

### Paper potential

**High if demonstrated on multiple task families.**

Without cross-task experiments it should be presented as an architectural
design, not a proven generalization result.

---

## Contribution Candidate E — Ambiguity-triggered targeted perception refinement

### Core idea

Only re-query the VLM when temporal reasoning detects a specific inconsistency.

Potential triggers:

- object reaches target without reliable held evidence;
- one hand is assigned to multiple objects;
- abrupt object identity switch;
- uncertain container motion direction;
- physically inconsistent state sequence.

### Current maturity

**Implemented in V3.4.4; controlled evaluation pending.**

The implementation records both the ambiguity requests and the accepted targeted
observations, allowing direct analysis of query count, merge rate and downstream
annotation change.

### Paper potential

**Very high contribution candidate pending controlled evidence.**

Would provide a principled bridge between perception uncertainty and reasoning,
and may reduce VLM cost compared with uniform dense querying.

---

## Contribution Candidate F — Joint hand-object temporal ownership

### Core idea

Infer portable-object tracks jointly rather than independently.

Constraint:

```text
one hand cannot reliably hold two different tracked objects at the same time
```

Use:

- source state;
- target state;
- hand evidence;
- temporal continuity;
- physical exclusivity;
- context switches;
- optional targeted re-observation.

### Current maturity

**Implemented in V3.4.2 with initial controlled validation.**

Without changing the perception evidence, V3.4.2 corrected episode27's early
plate ownership and achieved an exact 10-step policy-normalized sequence.
Episode26 provisional utensil ownership overlap was also separated.

The implementation outputs unresolved ownership conflicts explicitly, which can
serve as automatic triggers for later targeted re-observation.

### Paper potential

**High contribution candidate with initial evidence.** Broader held-out
validation is still required before making a paper-level generalization claim.

---

# 7. What should NOT be presented as a paper contribution

The following are useful engineering decisions but should not be overclaimed:

- changing a fixed threshold from one value to another;
- episode-specific frame corrections;
- rules that mention episode26/27;
- adding a dishwasher-only if/else;
- moving files to `results/`;
- Git storage cleanup;
- a specific cutlery-basket workaround without a generic formulation;
- merely using Qwen or a VLM.

These may support the system but are not the scientific core.

---

# 8. Current strongest paper story

A coherent current method story is:

```text
coarse task + robot trajectory + multi-view video
        ↓
state/action event proposals
        ↓
structured VLM entity-state observations
        ↓
trajectory-level graph-constrained state tracking
        ↓
task-schema physical / affordance reasoning
        ↓
completion-aware object-centric manipulation episodes
        ↓
interaction-grounded validation
        ↓
reliability-aware semantic backoff
        ↓
continuous-signal boundary refinement
        ↓
hierarchical language annotations
```

A concise method statement is:

> We automatically convert coarse task-level labels of long-horizon humanoid
> household demonstrations into state-action-grounded, temporally aligned,
> hierarchical language annotations. The method combines trajectory-level
> entity-state reasoning, physical interaction validation, completion-aware
> object manipulation episodes, and reliability-aware semantic abstraction.

---

# 9. Immediate research priorities

Priority order should be:

1. **Evaluate V3.4.4 closed-loop targeted perception**
   - recover unresolved lifecycle errors without GT;
   - measure query count / merge rate / annotation change;
   - compare against the legacy fixed dense pass.

2. **Schema-driven broad perception**
   - remove the remaining dishwasher-specific base VLM prompt;
   - derive all entity/state questions from the task schema.

3. **Container interaction robustness**
   - verify that lifecycle recovery generalizes beyond the cutlery basket.

4. **Generalization experiments**
   - multiple unseen trajectories of the same task;
   - at least one additional task family;
   - downstream policy-learning benefit.

---

# 10. Experimental discipline going forward

For every proposed modification, record:

1. **Observed failure**
2. **General cause**
3. **General algorithmic change**
4. **Why it is not episode-specific**
5. **Controlled comparison**
6. **Which trajectories improve / regress**
7. **Whether the change affects**
   - perception;
   - temporal reasoning;
   - semantic granularity;
   - validation;
   - boundary refinement.
8. **Whether it is**
   - engineering;
   - method component;
   - contribution candidate.

Whenever possible, reuse the same perception evidence across versions when
testing reasoning changes. This isolates the effect of the algorithm from VLM
stochasticity.

Episode26/27 should remain development/regression examples. Final paper claims
must rely on unseen trajectories that were not repeatedly inspected during
method development.
