# Batch Evaluation

Batch evaluation is an orchestration layer around the existing per-episode
versioned pipeline. It does not reimplement annotation logic.

## 1. Discover and freeze the dataset split

The tracked configuration is:

```text
batches/dishwasher_v1.json
```

It scans:

```text
/efs/share/1919650160032350208/efs_backup/nas-backup/compressed_data/astribot/dishwasher_2_fx_20260529_compressed/
```

for `*.hdf5`.

Generate the frozen manifest once:

```bash
python evaluation/discover_batch.py
```

Discovery uses the HDF5 files actually present under the configured directory.
The scanned file count is recorded in the frozen manifest; it is not hard-coded
because the local compressed-data directory may contain only a subset of the
larger dataset metadata.

**Do not pass `batches/dishwasher_v1.json` to `run_batch.py`.** That file is
only the discovery configuration. The runner requires the generated frozen
manifest `batches/dishwasher_v1_manifest.json` and now rejects config files
explicitly.

This creates:

```text
batches/dishwasher_v1_manifest.json
```

Default split policy:

```text
development = episode26, episode27
validation  = 6 deterministic unseen episodes
heldout     = 6 deterministic unseen episodes
reserve     = all remaining episodes
```

The non-development episodes are shuffled with a fixed seed
(`20260923`) before allocation. The generated manifest contains a dataset
fingerprint and absolute HDF5 paths.

The discovery script refuses to overwrite an existing manifest unless
`--force` is supplied. This is intentional: after the split is frozen, later
versions must reuse it rather than silently changing the test set.

**Commit the generated manifest after inspecting it.**

## 2. Dry-run before spending VLM calls

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --split validation \
  --dry-run
```

No episode pipeline or VLM request is executed.

## 3. Run a split

First verify regression behavior:

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --split development
```

Then run validation. For a version comparison, always pass the version
explicitly so the frozen manifest itself does not need to change:

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --version v3_4_5 \
  --split validation
```

The runner calls exactly the existing interface:

```text
versions/<version>/run.sh HDF5 OUTPUT TASK ...
```

It inherits API/environment configuration from the shell. Ground truth is split
into two roles:

- `sequence_gt`: sequence-only semantic evaluation used by the batch summarizer;
- `temporal_gt`: frame-level GT passed to the per-episode runner through the
  stable `GT` environment variable.

Sequence-only GT is never passed to the temporal evaluator. This avoids treating
manual skill order as fabricated frame boundaries.

Default execution is serial:

```text
--workers 1
```

If API and storage capacity permit, episode-level parallelism can be enabled:

```bash
--workers 2
```

Do not increase concurrency merely to reduce wall-clock time; V3.4.4+ performs
real targeted VLM requests.

## 4. Resume behavior

Compact results that already exist under:

```text
results/<episode>/<version>/hierarchical_annotations.json
```

are skipped by default.

To intentionally rerun every selected episode:

```bash
--force
```

To retry only episodes recorded as failed in the same split:

```bash
--retry-failed
```

A failure in one episode does not stop the remaining episodes. The command
returns nonzero after finishing the batch if any selected episode failed.

## 5. Logs and status

Runtime logs remain local:

```text
output/batch_logs/<batch>/<version>/<split>/
├── <episode>.log
└── batch_status.json
```

No credentials or API keys are written into the status file.

## 6. Automatic summary

After a successful batch run, the runner automatically executes
`summarize_batch.py`.

Compact aggregate results are written to:

```text
results/batches/<batch>/<version>/<split>/
├── batch_summary.json
├── episode_metrics.json
└── failure_cases.json
```

The summary supports mixed GT granularity:

- sequence-only GT -> policy sequence P/R/F1/Edit/Exact;
- frame-level GT -> sequence metrics plus temporal metrics;
- no GT -> predicted sequence and closed-loop query-cost statistics only.

The six frozen validation trajectories share a manually reviewed policy-level
10-step sequence. Knife/fork order may differ at the fine level, but both map to
the policy-equivalent `utensil` label.

For V3.4.4+, closed-loop statistics include:

- number of ambiguity requests;
- number of targeted VLM queries;
- number of merged targeted observations;
- ambiguity trigger kinds.

These fields are intended for accuracy-vs-perception-cost analysis.

For V3.4.5+, the batch summary also aggregates final-consistency behavior:
dropped invalid phases, clamped spans, lifecycle violations, unresolved
expected-final-state violations, and the number of unresolved episodes.

## Experimental discipline

Episode26/27 are development/regression examples and may be inspected freely.

Validation trajectories may be used for aggregate model development, but avoid
adding per-episode rules after inspecting individual failures.

Held-out trajectories should not be inspected or used to change the method
until the method version and evaluation protocol are frozen.
