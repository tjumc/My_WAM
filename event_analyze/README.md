# Event Analyze

目录按“共享分析 + 版本流水线”组织。

```text
event_analyze/
├── README.md
├── common/
│   ├── analyze_episode.py
│   └── run_analyze.sh
├── regression/
│   └── dishwasher_episode27_manual_gt.json
├── versions/
│   ├── v1/
│   ├── v2/
│   ├── v2_1/
│   ├── v2_2/
│   ├── v3/          # archived independent-window entity state machine
│   └── v3_1/        # current active trajectory-level entity tracker
└── output/
    └── <episode>_analysis/
        ├── candidate_events.json
        ├── contact_sheets/
        ├── events_overview.jpg
        ├── signals.png
        └── versions/
            ├── v1/
            ├── v2/
            ├── v2_1/
            ├── v2_2/
            ├── v3/
            └── v3_1/
```

共享 proposal 只生成一次。不同版本只把自己的结果写入
`output/<episode>_analysis/versions/<version>/`。

当前推荐运行 V3.1：

```bash
bash versions/v3_1/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

V3.1 的核心原则：
- VLM 仍只输出受限实体状态，不生成 action/phase；
- 对 door / dish rack / cutlery basket / knife / fork / plate 做全轨迹状态跟踪；
- 使用状态持久性、物理转移图、遮挡插值和 hand-object 互斥恢复连续实体轨迹；
- skill 由确定性状态机从完整状态轨迹推导；
- 不确定区间保持 coarse-task-only；
- `regression/` 的人工标注只用于评估，流水线不会读取。

默认会复用 V3 已生成的 `entity_observations.jsonl`，从而把 V3→V3.1
提升归因到 trajectory-level tracking，而不是新的 VLM 调用。
