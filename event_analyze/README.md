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
│   ├── v3/
│   ├── v3_1/        # trajectory-level entity tracking
│   └── v3_2/        # current active: targeted observation + interaction validation
└── output/
    └── <episode>_analysis/
        ├── candidate_events.json
        ├── contact_sheets/
        ├── events_overview.jpg
        ├── signals.png
        └── versions/
            └── ...
```

当前推荐运行 V3.2：

```bash
bash versions/v3_2/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

V3.2：
- 复用 V3 generic entity observations；
- 对低可观测的 cutlery basket 做 targeted re-observation；
- V3.1 trajectory-level tracking 恢复连续实体轨迹；
- container transition 必须通过 interaction-grounded validation；
- object placement 必须有真实 observed held state；
- 不确定区间保持 coarse-task-only；
- regression 人工 GT 只用于离线评估，流水线不会读取。
