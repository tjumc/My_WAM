# Single-task expert vs shared multi-task WAM (offline)

Status: complete

This is an on-training-demonstrations fit probe, not a held-out validation/test
or real-robot closed-loop success measurement. A difference is not proof of
causal positive/negative transfer; training budgets and prompt sources can differ.

Metrics: 32 future action steps, true raw 25D action, inverse-normalized
with each checkpoint's own stats. Scoring scale for both variants is
the task expert's fixed training-action global_std (with documented floor).
Mean is equal-weighted over episodes; windows and diffusion seeds
within an episode are not independent validation cases.

| Task | Episodes | Expert | Shared | Shared - Expert | Shared better episode % |
|---|---:|---:|---:|---:|---:|
| oven_put | 3 | 0.07219 | 0.15973 | +0.08754 | 0.0% |
| fridge | 3 | 0.06135 | 0.09579 | +0.03444 | 33.3% |
| dishwasher | 3 | 0.06265 | 19.49464 | +19.43199 | 0.0% |
| oven_takeout | 3 | 0.04125 | 0.08765 | +0.04640 | 0.0% |
| collect_clothes | 3 | 0.02786 | 0.05725 | +0.02939 | 0.0% |
| dry_clothes | 3 | 0.04491 | 0.86931 | +0.82441 | 0.0% |
| wash_clothes | 3 | 0.04263 | 0.07794 | +0.03530 | 33.3% |

Task-equal macro expert RMSE: 0.05041
Task-equal macro shared RMSE: 2.97747

## Pairing and coverage

- oven_put: 9/9 paired calls (100.0%), 3 paired episodes.
- fridge: 9/9 paired calls (100.0%), 3 paired episodes.
- dishwasher: 9/9 paired calls (100.0%), 3 paired episodes.
- oven_takeout: 9/9 paired calls (100.0%), 3 paired episodes.
- collect_clothes: 9/9 paired calls (100.0%), 3 paired episodes.
- dry_clothes: 9/9 paired calls (100.0%), 3 paired episodes.
- wash_clothes: 9/9 paired calls (100.0%), 3 paired episodes.

## Code provenance

Sample preparation git HEAD: `65b678f791c844ae7f765156365de3b6d3fac02d`.
- Scored with git HEAD `68534efabad5f3bf233cdcf1a92f87496ecb7e73`, tool SHA-256 `3756c8d2042b27d151ed344301c9fe91cc97d63f8c89c5b17e278883abdb6765`, GT reader `episode-range-mapped-v1`.

## Prompt conditions (not forced identical)

- oven_put: expert=["put the plate into the oven"], shared=["Put the plate into the oven"]
- fridge: expert=["Move to the fridge, take out the plate with food inside from the opened drawer, then move to the desk and put the plate on the desk"], shared=["Move to the fridge, take out the plate with food inside from the opened drawer, then move to the desk and put the plate on the desk"]
- dishwasher: expert=["put the dish into the dishwasher"], shared=["Put the dish into the dishwasher"]
- oven_takeout: expert=["take the plate out of the oven"], shared=["Put the plate into the oven"]
- collect_clothes: expert=["Gather the clothes into the laundry basket"], shared=["Gather the clothes into the laundry basket"]
- dry_clothes: expert=["transfer the clothes to the dryer"], shared=["Transfer the clothes to the dryer"]
- wash_clothes: expert=["put the clothes into the washing machine"], shared=["Put the clothes into the washing machine"]

See groups.csv and chunks.csv for arm/gripper and within-chunk effects.
Source directories are retained in samples.csv but not classified by appearance.
When coverage is incomplete, summary contains only matched calls and must not
be treated as the planned full-sample comparison.
