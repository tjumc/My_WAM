# Qwen observation audit

## Interface and image-delivery probe

From `event_analyze/`, make two small calls on an existing development image:

```bash
IMAGE=output/dishwasher_2_fx_20260529_episode_27_analysis/versions/v3_4_9/cache/proposal/contact_sheets/event_00_f0071.jpg
python evaluation/probe_qwen_interface.py "$IMAGE"
python evaluation/probe_qwen_interface.py "$IMAGE" --without-image
```

The output contains the requested endpoint/model, response model/IDs when the
service returns them, and a short answer. It never prints the API key. The
plate in this image is pink; the no-image control should say `NO_IMAGE`.
This comparison checks whether the endpoint appears to use image content. It
cannot establish the actual backend model: a gateway may echo the requested
model alias. Give the response IDs to the service owner and ask for the actual
deployment/model and image-decoding result in the server log.

## Question

How often does the current Qwen observation step correctly report visible entity
states and state changes in the images it was actually shown? This checks Qwen
perception separately from proposal coverage and downstream phase reasoning.

The unit is an **entity at the early or late end of one queried image window**.
Base and targeted queries are reported separately: targeted queries were selected
because the pipeline found ambiguity and are not a random sample.

## Blind development pilot

From `event_analyze/`, prepare a packet from the existing V3.4.9 development
cache. The command makes no model calls and reads no HDF5 files:

```bash
python evaluation/qwen_observation_audit.py prepare \
  --cache-dir output/dishwasher_2_fx_20260529_episode_26_analysis/versions/v3_4_9/cache \
  --cache-dir output/dishwasher_2_fx_20260529_episode_27_analysis/versions/v3_4_9/cache \
  --output-dir output/qwen_audit_v349_development
```

The packet contains `images/`, `human_labels.csv`, and `predictions.csv`.
Do not open `predictions.csv` until human labeling is complete. The sampled base
windows span each trajectory; all targeted windows are included. Record the
chosen episode IDs and seed when using this on new trajectories. Pass only
explicit cache directories; the script never scans the frozen held-out split.

For each row in `human_labels.csv`, inspect the image named in `image`. The
three rows are head, left, and right cameras. Columns advance from early to
late time. Fill `human_before` from the first column and `human_after` from the
last column. Use the same state vocabulary as the Qwen observation prompt in
`versions/v3_3_3/observe_entities.py`. Use `not_visible` when an entity cannot
be seen; use `uncertain` when it is visible but its state cannot be determined.
Do not infer a state from the task script or a later image. Add a note for
occlusion, image quality, or disputed identity. Leave a cell empty only when it
has not been reviewed.

After labeling, run:

```bash
python evaluation/qwen_observation_audit.py score output/qwen_audit_v349_development \
  > output/qwen_audit_v349_development/score.json
```

The score separates exact state accuracy, precision/recall/coverage for useful
concrete states, transition-pair accuracy,
high-confidence errors, and cases where Qwen asserts a state for an entity the
human cannot see. Read per-episode rows before pooled counts. Overlapping
windows in one episode are correlated; pooled endpoint counts are not
independent trajectories or a generalization estimate.

## Decision sequence

1. Use the development packet to settle the human labeling rules and find
   obvious visual failure modes. No accuracy claim about new trajectories is
   justified from these two repeatedly inspected episodes.
2. Freeze a separate sample of new `fx` trajectories after confirming their
   task and format. Run the same Qwen prompts, create the packet, and label
   before opening the predictions. Include `hcc` only after confirming that
   its task and visual setup match the scope being tested.
3. If Qwen is often wrong on the exact images it saw, improve visual
   observation or use a coarser state vocabulary. If Qwen is mostly right but
   the final phase is missing or misplaced, inspect proposal coverage and
   downstream reasoning separately. Never attribute an unqueried transition
   to a Qwen recognition error.

This audit estimates correctness **conditional on a Qwen query**. It does not
measure accuracy on every frame of a trajectory or the quality of training
window labels.
