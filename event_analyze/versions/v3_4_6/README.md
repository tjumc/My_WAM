# V3.4.6 — conservative closed-loop reasoning

V3.4.6 is a controlled reasoning update built on V3.4.5. It addresses two
validation failures without changing the frozen validation split or reading GT
during inference.

V3.4.5 improved mean policy F1 from 0.593 to 0.653 and reduced targeted queries
from 8.00 to 6.83 per trajectory, but exposed two structural problems:

1. targeted observations could erase correct pass1 skills (episode9);
2. a late predicted phase could extend the task horizon that was then used to
   validate that same phase (episode23).

## Method changes

### 1. Conservative semantic anchors

Validated pass1 skills are retained as semantic anchors by default.

After targeted perception, raw pass2 reasoning is still computed normally.
V3.4.6 then reconciles pass1 and pass2:

- pass2 additions are allowed;
- a pass1 skill already represented by pass2 is unchanged;
- a pass1 skill missing from pass2 is restored unless strong targeted evidence
  directly shows the reverse transition for the same entity in the same time
  interval;
- targeted evidence from another semantic goal cannot silently delete an
  unrelated pass1 skill.

This is implemented in:

```text
conservative_reasoning.py
```

and summarized in:

```text
diagnostics/conservative_update.json
```

### 2. Controlled targeted-evidence reuse

For a clean V3.4.5 -> V3.4.6 reasoning comparison, V3.4.6 reuses V3.4.5
targeted observations only when every request has exactly the same:

- ambiguity kind;
- target entities;
- start frame;
- end frame.

If any request differs, V3.4.6 performs fresh targeted VLM calls instead.

### 3. Schema initial-state prior

The task-family schema may optionally declare:

```json
"expected_initial_state": "closed"
```

for articulated entities.

This is task-onboarding knowledge, not per-trajectory GT. For the current
dishwasher family the configured initial states are:

```text
door            closed
dish_rack       in
cutlery_basket  in
```

The final decoder uses these states as lifecycle priors. It does not fabricate
missing transition skills.

### 4. Latent prerequisite state inference

If an already accepted downstream action has a schema prerequisite, that action
is evidence that the prerequisite state must have held at that moment.

For example, if:

```text
dish_rack requires door=open
```

then an accepted dish-rack interaction can update the hidden door state to
`open` even if the explicit open-door skill was missed.

This only updates hidden state. It never inserts an unobserved action label.

### 5. Independent task-completion frontier

The task horizon is no longer defined as the end of the last predicted phase.

The schema declares a task-completion policy. For the dishwasher task the outer
door is the frontier entity:

```text
first door=closed transition
after the last dependent rack/basket use
```

is the terminal completion frontier.

Therefore a later spurious phase cannot extend its own validity horizon.

### 6. Gratutitous lifecycle-cycle pruning

For an articulated entity, once it reaches its expected final state after its
last dependent use, later cycles with no task-enabling role are removed.

This is designed to suppress patterns such as:

```text
close -> open -> close
```

after all downstream work has already finished.

The rule is dependency-driven and schema-based; it is not specific to the
dishwasher-door action names.

## Controlled inheritance

V3.4.6 intentionally inherits V3.4.5:

- base perception;
- entity tracking;
- hand-object ownership;
- skill inference;
- interaction validation;
- ambiguity detection;
- targeted VLM prompt;
- boundary refinement.

The new behavior is concentrated in:

```text
conservative_reasoning.py
finalize_annotations.py
reasoning_trace.py
reuse_targeted.py
```

This keeps the V3.4.5 -> V3.4.6 comparison focused on reasoning.

## Diagnostics

The compact reasoning trace now shows:

```text
pass1
  ↓
targeted perception
  ↓
raw pass2
  ↓
conservative assimilation
  ↓
reconciled pass2
  ↓
task-graph final decoding
```

Default retained diagnostics:

```text
diagnostics/
├── reasoning_trace.json
├── conservative_update.json
└── final_consistency.json
```

Temporary full pass1/pass2 reconciliation artifacts are deleted after a
successful run unless:

```bash
KEEP_DEBUG=1
```

is set.

## Validation run

Use the existing frozen manifest:

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --version v3_4_6 \
  --split validation
```

Do not run the held-out split yet.

## Evaluation questions

The controlled V3.4.5 -> V3.4.6 comparison should answer:

1. Does episode9 keep its correct pass1 open-door / pull-rack anchors?
2. Does episode23 terminate at the first task-relevant door close and remove the
   later gratuitous reopen cycle?
3. Does mean validation F1 improve without lowering the high precision of
   V3.4.5?
4. How many pass1 anchors are restored or directly contradicted?
5. How often can V3.4.5 targeted evidence be reused exactly?
6. Does the new completion frontier reduce unresolved internal-consistency
   cases?

V3.4.6 is implemented as a reasoning hypothesis. Performance claims require the
frozen validation run.
