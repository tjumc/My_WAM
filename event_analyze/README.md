# Event Analyze

目录按“共享分析 + 版本流水线”组织，避免后续版本继续堆在根目录。

```text
event_analyze/
├── README.md
├── common/
│   ├── analyze_episode.py
│   └── run_analyze.sh
├── regression/
│   └── dishwasher_episode27_manual_gt.json
├── versions/
│   ├── v1/          # archived baseline
│   ├── v2/          # archived conservative-event version
│   ├── v2_1/        # archived semantic-window version
│   ├── v2_2/        # archived state-transition version
│   └── v3/          # current active entity-state-machine version
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
            └── v3/
```

共享 proposal 只生成一次。不同版本只把自己的结果写入 `output/<episode>_analysis/versions/<version>/`。

当前推荐运行 V3：

```bash
bash versions/v3/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

V3 的核心原则：
- VLM 只输出受限实体状态，不生成 action/phase；
- door / dish rack / cutlery basket / knife / fork / plate 分开建模；
- skill 由确定性状态机从状态转移推导；
- 不确定区间保持 coarse-task-only，不强行赋细粒度标签；
- `regression/` 中的人工标注只用于评估，不被自动标注流水线读取。

以后新增版本统一放到 `versions/vX_Y/` 或 `versions/vN/`，不要再往 `event_analyze/` 根目录增加版本文件。
