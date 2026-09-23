# V3.4.7 dense-evidence diagnostic ablation

This is **not** a candidate production version.

Purpose:

> diagnose whether the current articulated/container lifecycle recall bottleneck
> is primarily caused by missing temporal visual evidence or by downstream
> candidate validation/reasoning.

The ablation keeps the V3.4.7 pipeline and adds the old task-specific
cutlery-basket dense relative-motion observer before pass1.

Controlled parts:

- proposal artifacts are reused from the generic V3.4.7 run when available;
- base Qwen observations are reused from the generic V3.4.7 run when available;
- V3.4.7 tracking / ownership / inference / validation are unchanged;
- V3.4.7 conservative assimilation and final state-implication decoder are
  unchanged;
- targeted observations are reused from generic V3.4.7 only when the ambiguity
  request signatures remain exactly identical after dense evidence is added.

Changed part:

- `v3_3_3/dense_targeted_reobserve.py` is explicitly run to create
  cutlery-basket directional motion evidence.

Outputs are isolated under:

```text
output/.../versions/v3_4_7_dense_ablation/
results/.../v3_4_7_dense_ablation/
```

so the generic V3.4.7 baseline is not overwritten.

Run only the development split:

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --version v3_4_7_dense_ablation \
  --split development
```

Interpretation:

- large recovery toward the historical V3.4.3 sequence means temporal
  articulated-state perception is the dominant bottleneck;
- little recovery means candidate validation/reasoning remains the dominant
  bottleneck;
- this ablation must not be used as the final generic method because the dense
  observer is cutlery-basket specific.

Do not run held-out with this ablation.

## Development result

The completed development batch did not change aggregate sequence metrics from
generic V3.4.7:

```text
Precision = 1.000
Recall    = 0.450
F1        = 0.619
Edit      = 5.5
Exact     = 0 / 2
```

Episode27 remained at 4/10 with the same four policy labels. Episode26's raw
pass2 recovered `pull_out_cutlery_basket`, but final boundary arbitration
compressed the validated `open_dishwasher_door` anchor to frames 118–127
(10 frames, below its 21-frame minimum), so the consistency gate removed it.
The final sequence length stayed at five phases. Dense evidence had local value
but did not solve the recurring missed push-in candidate or improve aggregate
sequence quality.

This points to two downstream changes for V3.4.8: evidence-gated rehabilitation
of an existing rejected candidate, and boundary arbitration that preserves a
validated pass1 anchor when a new phase competes for the same frames.
