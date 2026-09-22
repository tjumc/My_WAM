# Compact Experiment Results

This directory is the Git-tracked home for **small, reproducible experiment
summaries**. Heavy runtime artifacts stay under `event_analyze/output/` and are
ignored for new files.

Each run is exported as:

```text
results/
└── <episode>/
    └── <version>/
        ├── hierarchical_annotations.json
        ├── evaluation_temporal.json   # when GT evaluation was run
        ├── summary.json
        └── manifest.json
```

## What belongs here

Keep:
- final hierarchical annotations;
- evaluation metrics;
- compact phase summaries;
- reproducibility metadata / hashes.

Do not keep:
- contact sheets;
- dense temporal strips;
- overview images;
- raw VLM responses;
- proposal visualizations;
- other reproducible intermediate caches.

## Why

The repository should store methods, schemas, ground truth, and comparable final
results—not every generated image and cache from every run.

## Existing tracked outputs

Older files already committed under `event_analyze/output/` remain in Git
history and may still be tracked in the current tree. The new ignore rule is
intentionally non-destructive: it prevents new untracked runtime artifacts from
being added, but it does not rewrite history or delete old experiments.
