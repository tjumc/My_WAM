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
│   └── v3_3/        # current active: dense targeted temporal observation
└── output/
```

当前推荐运行 V3.3：

```bash
bash versions/v3_3/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

V3.3 保留 V3.2 的 interaction-grounded validation，并对低可观测的小型
articulated entity 使用从 HDF5 直接解码的 dense temporal observation。

泛化边界与未来 task-agnostic 设计见 `GENERALIZATION.md`。
