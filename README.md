# My_WAM

Stable codebase for Fast-WAM training, offline evaluation, and Astribot real-robot inference.

本仓库的 `main` 只维护可复用的训练、推理、数据加载和评估代码；具体研究实验放在独立分支，避免将生成数据和中间结果混入核心代码。

## Code layout

| Path | Purpose |
| --- | --- |
| `src/fastwam/` | Model, data loading, training, and inference components |
| `configs/` | Model, dataset, and training configurations |
| `scripts/` | General training, preprocessing, and evaluation scripts |
| `task_train_scripts/` | Astribot single-task and multi-task training launchers |
| `experiments/astribot/` | Astribot server, client, and offline evaluation |
| `experiments/libero/`, `experiments/robotwin/` | Benchmark evaluation |
| `third_party/` | Bundled third-party code |

The Astribot task/dataset directory catalog is `configs/data/astribot_dataset_catalog.sh`. Update machine-specific dataset and checkpoint paths in the relevant launch/config scripts before running.

For Astribot real-robot inference, see [scripts/real_robot/README.md](scripts/real_robot/README.md). **Robot execution requires the correct SDK, network settings, and an appropriate safety review.**

For upstream Fast-WAM model documentation and setup instructions, see [English](docs/README.md) / [中文](docs/README_zh.md).

## Branch policy

- `main`: reusable, stable code; no experiment outputs.
- `archive/research-diagnostics-2026-10-16`: frozen snapshot of history-ambiguity, input-swap, recording, dataset-audit diagnostics and their previously committed outputs.
- `research/single-vs-multitask`: single-task expert vs multi-task shared-model comparison.
- Future topics: create a new `research/<topic>` branch before making changes. Only merge a tested, reusable change into `main` deliberately.

Generated datasets, model weights, videos, raw predictions, logs, and run-specific results should stay on experiment storage. The `.gitignore` includes common output locations; tracked files in a research branch are not removed retroactively.
