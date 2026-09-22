# V3.4.3 — Boundary-Validation Consistency

V3.4.3 keeps V3.4.2 joint hand-object ownership unchanged and fixes two
temporal-consistency problems exposed by the controlled regression.

## 1. Semantic-ownership-preserving boundary refinement

Earlier boundary refinement could expand neighboring phases around HDF5
activity valleys and then resolve overlap left-to-right. A middle phase could
therefore be trimmed from both sides and collapse to only a few frames even
when its upstream provisional semantic interval was reasonable.

V3.4.3 changes overlap arbitration to:

- process overlaps right-to-left;
- treat provisional intervals as semantic ownership priors;
- use activity valleys only as numerical refinement cues;
- enforce skill-specific minimum duration during overlap arbitration whenever
  feasible;
- explicitly record infeasible duration conflicts instead of silently
  collapsing a phase.

This is a general refinement constraint, not an episode-specific boundary fix.

## 2. Post-anchor robot-signal revalidation

Container validation may relocate a sparse tracker boundary to a nearby direct
hand-contact anchor. In V3.4.2, the candidate could pass robot-signal validation
at its old interval and then be moved afterward.

V3.4.3 uses:

```text
candidate
  -> entity-contact evidence
  -> optional contact-anchor relocation
  -> recompute robot signal at relocated interval
  -> final accept / reject
```

Therefore the final semantic interval and the robot signal used to validate it
are temporally consistent.

## Controlled comparison

Perception and joint ownership are reused from V3.4.2 by default. The intended
test is therefore isolated to boundary realization and post-relocation
validation.

Expected checks:

- episode26: the first utensil raw phase should no longer collapse to a
  two-frame interval;
- episode27: the correct policy-level action order should remain unchanged;
- every `interaction_anchor_adjusted=true` phase should carry signal evidence
  recomputed at the adjusted provisional interval.


## Stable GT environment variable

From V3.4.3 onward, use the version-independent environment variable:

```bash
GT=regression/dishwasher_episode27_manual_gt.json \
bash versions/v3_4_3/run.sh ...
```

Future versions should keep `GT` unchanged so experiment commands can be
reused across versions. `V343_GT` is accepted only as a backward-compatible
fallback in this version.
