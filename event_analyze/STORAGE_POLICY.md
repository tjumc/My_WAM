# Repository Storage Policy

This repository stores **methods, schemas, ground truth, and compact experiment
results**. Large or reproducible runtime caches should remain outside Git.

## Tracked in Git

- source code;
- task ontologies / schemas;
- human ground truth;
- evaluation code;
- final `hierarchical_annotations.json`;
- `evaluation_temporal.json` when available;
- compact `summary.json`;
- reproducibility `manifest.json`;
- compact V3.4.5+ diagnostics such as `reasoning_trace.json`,
  `conservative_update.json` (V3.4.6+), and `final_consistency.json`.

## Local-only runtime cache

The following belong under `event_analyze/output/` and should not be newly
committed:

- proposal contact sheets;
- dense temporal strips;
- overview / diagnostic images;
- raw VLM responses;
- copied perception caches;
- intermediate tracker / proposal visualizations;
- other artifacts that can be regenerated from the trajectory and schema.

## Why keep both output/ and results/

`output/` is the working directory for debugging and development.

`results/` is the compact experiment record intended for Git history.

A normal run may produce hundreds of runtime files, but only a few small files
should enter `results/`.

## Existing historical outputs

Older `event_analyze/output/` files are intentionally not removed or history-
rewritten in this change. Git ignore rules do not stop tracking files that are
already tracked.

This avoids destructive cleanup while changing the policy for future versions.
If repository growth later becomes material, old generated outputs can be
removed from the current tree in a dedicated migration; rewriting Git history
should only be considered separately and with a backup.

## Export

V3.4.0 and later can call:

```bash
python common/export_results.py \
  --episode-dir <episode_analysis_dir> \
  --output-dir <version_output_dir> \
  --version <version> \
  --task "<task>" \
  --schema <schema.json> \
  --hdf5 <episode.hdf5> \
  --event-root <event_analyze>
```

The exporter creates:

```text
results/<episode>/<version>/
├── hierarchical_annotations.json
├── evaluation_temporal.json   # if available
├── summary.json
├── manifest.json
└── diagnostics/               # V3.4.5+, when available
    ├── reasoning_trace.json
    ├── conservative_update.json   # V3.4.6+
    └── final_consistency.json
```

The manifest fingerprints important evidence and records closed-loop query
statistics. V3.4.5+ compacts the completed local runtime directory: reusable
perception/proposal artifacts move under `cache/`, while pure intermediate
reasoning snapshots are removed unless `KEEP_DEBUG=1`. V3.4.6 keeps only the
compact conservative-assimilation delta instead of permanent full pass1/pass2
copies.
