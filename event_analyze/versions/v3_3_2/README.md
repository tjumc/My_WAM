# V3.3.2 — Retry-Aware Proposals + Directional Dense Evidence

V3.3.2 targets failure modes exposed by held-out episode 26 without reading any
manual GT during inference.

## Changes from V3.3.1

1. **Gripper edges are weak proposal cues**
   - raw gripper open/close is no longer force-inserted as a high-value event;
   - rapid open/close transitions are clustered as one retry/re-grasp episode;
   - a gripper cluster is promoted only when broader state/action or force
     context supports it;
   - long uncovered temporal gaps receive a state/action-change coverage anchor.

2. **Retry-aware semantic grouping**
   - repeated grasp attempts for the same object within a short window are
     collapsed into one semantic placement skill;
   - low-level retries remain execution details rather than extra phases.

3. **Trajectory-conditioned container lifecycle**
   - door / dish rack / cutlery basket lifecycle states are initialized from the
     first reliable tracked stable state;
   - trajectories are no longer assumed to start from closed / in / in.

4. **Dense cutlery evidence is direction-only**
   - dense Qwen observations remain `moving_out` / `moving_in` evidence;
   - they are not converted into synthetic absolute `in -> out` or
     `out -> in` observations;
   - dense frames affect only the cutlery-basket tracker.

5. **Local jitter is resolved before affordance filtering**
   - short opposite-direction pull/push pairs are handled first;
   - long-horizon receptacle accessibility constraints are applied afterwards.

## Run

```bash
bash versions/v3_3_2/run.sh \
  /path/to/episode.hdf5 \
  output/episode_analysis \
  "put the dish into the dishwasher"
```

Optional LeRobot mapping:

```bash
bash versions/v3_3_2/run.sh \
  /path/to/episode.hdf5 \
  output/episode_analysis \
  "put the dish into the dishwasher" \
  LEROBOT_EPISODE_ID \
  MAX_RAW_FRAME
```

Force individual stages to rerun with:
`V332_REPROPOSE=1`, `V332_REOBSERVE=1`, or `V332_REDENSE=1`.

For regression only, after a manual GT exists:

```bash
V332_GT=regression/<gt>.json bash versions/v3_3_2/run.sh ...
```

GT is read only by the final evaluator.
