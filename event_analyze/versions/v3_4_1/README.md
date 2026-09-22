# V3.4.1 — Reliability-First Object Semantics

V3.4.1 keeps V3.4.0's schema-driven reasoning and adds two general mechanisms.

## 1. Policy-level semantic backoff

The task schema may declare fine object classes as policy-equivalent.

For dishwasher loading:

```text
knife / fork -> utensil
```

The final training label becomes:

```text
place_utensil_in_cutlery_basket
```

while `fine_skill_type` and `fine_entity` are retained for diagnostics.

If the exact fine held-state is missing but a policy-equivalent sibling is
reliably observed in-hand during the transfer, the coarse parent label may be
validated. The system does **not** claim that the fine identity was correct.

## 2. Completion-aware manipulation episodes

A placement no longer ends merely because the object remains at the target for
a fixed duration.

```text
hand -> target -> hand -> target
```

is one manipulation episode unless another portable object is clearly
manipulated between the target contact and re-grasp.

This replaces the previous fixed target-dwell completion heuristic.

## Evaluation

Fine labels and policy-level labels are both retained in evaluation.

`regression_matrix.py` reports both raw and policy-normalized sequence metrics.
`evaluate_temporal_annotations.py --schema ...` adds a
`policy_normalized` evaluation section.

## Controlled comparison

V3.4.1 reuses V3.4.0 perception artifacts by default, so changes are attributable
to semantic backoff and object-episode reasoning rather than a fresh Qwen run.
