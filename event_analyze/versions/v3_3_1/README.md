# V3.3.1 — Entity-Specific Dense Fusion + Affordance Consistency

V3.3.1 is a controlled fix of V3.3. It does **not** change the overall method.

## Why

V3.3 showed that dense temporal observation can see the missing cutlery-basket
motion, but its pseudo observations polluted the shared Viterbi timeline. That
changed otherwise-correct dish-rack and object tracks and reduced overall
precision.

## Fixes

1. **Entity-specific targeted timeline**
   - generic observations keep the original shared baseline timeline;
   - dense cutlery frames are added only to the `cutlery_basket` tracker;
   - dish-rack / knife / fork / plate tracks therefore remain comparable to V3.2.

2. **Bounded confidence**
   - confidence is always clamped to `[0, 1]`;
   - dense reliability is represented as a separate `source_weight`;
   - no more values such as `confidence=1.71`.

3. **Directional dense evidence only**
   - `moving_out` / `moving_in` windows can become tracker anchors;
   - `stationary` dense windows remain diagnostic output but are not injected,
     because overlapping stationary windows caused false in/out reversals.

4. **Receptacle-accessibility temporal consistency**
   - `pull_out(receptacle)` should precede the first `place(*, receptacle)`;
   - `push_in(receptacle)` should follow the last `place(*, receptacle)`;
   - this is an affordance-role constraint rather than a hard-coded knife/fork action order.

5. **Direction-aware dense contact validation**
   - a dense `moving_out` window cannot be reused as generic contact evidence
     for a `push_in` candidate.

## Controlled rerun

By default V3.3.1 reuses the already-computed V3.3
`dense_cutlery_observations.jsonl`, so the comparison isolates fusion and
validation changes without another stochastic VLM call.

```bash
bash versions/v3_3_1/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

Force a fresh dense VLM pass:

```bash
V331_REDENSE=1 bash versions/v3_3_1/run.sh ...
```

Run the unified frame-level regression automatically:

```bash
V331_GT=regression/dishwasher_episode27_manual_gt.json \
bash versions/v3_3_1/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

The GT is used only after annotation generation for evaluation and never feeds
the annotation pipeline.
