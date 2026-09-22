# Event Analyze

目录按“共享分析 + 版本流水线”组织。

```text
event_analyze/
├── README.md
├── GENERALIZATION.md
├── common/
├── regression/
├── versions/
│   ├── v1/
│   ├── v2/
│   ├── v2_1/
│   ├── v2_2/
│   ├── v3/
│   ├── v3_1/
│   ├── v3_2/
│   ├── v3_3/
│   ├── v3_3_1/
│   ├── v3_3_2/
│   ├── v3_3_3/
│   └── v3_4_0/      # current active: schema-driven reasoning
└── output/
```

当前推荐运行 V3.4.0：

```bash
bash versions/v3_4_0/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

V3.4.0 将 tracking / skill inference / validation 的 task-family 知识迁移到 `ontologies/dishwasher_loading.json`；新任务允许一次性人工配置 schema，但单条轨迹推理必须自动化。当前 perception 暂时保持 V3.3.3-compatible，用于验证 schema-driven reasoning 的等价性。对全新的 HDF5，入口脚本会自动先生成 candidate_events.json 和 contact_sheets/。

泛化边界与未来 task-agnostic 设计见 `GENERALIZATION.md`。


## 统一评估

已有 frame-level manual GT 后，版本比较不再只看阶段序列。统一运行：

```bash
python evaluation/evaluate_temporal_annotations.py \
  output/dishwasher_2_fx_20260529_episode_27_analysis/versions/v3_2/hierarchical_annotations.json \
  regression/dishwasher_episode27_manual_gt.json \
  --output output/dishwasher_2_fx_20260529_episode_27_analysis/versions/v3_2/evaluation_temporal.json
```

报告包括 sequence Precision/Recall/F1、Edit Distance、semantic/core mIoU、Boundary MAE、Boundary F1@0.5/1/2s、fine-label coverage、false-positive duration 和 missed-GT duration。详见 `evaluation/README.md`。


## 序列回归矩阵

episode26 目前只有人工确认的 sequence-only provisional GT；episode27 有 frame-level GT。可先统一检查版本是否发生语义序列退化：

```bash
python evaluation/regression_matrix.py
```

默认比较 `v3_3_1,v3_3_2,v3_3_3,v3_4_0` 在 episode26/27 上的 sequence Precision/Recall/F1、LCS/Edit Distance。
