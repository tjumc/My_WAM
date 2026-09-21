# Event Analyze

目录按“共享分析 + 版本流水线”组织，避免后续版本继续堆在根目录。

```text
event_analyze/
├── README.md
├── common/
│   ├── analyze_episode.py
│   └── run_analyze.sh
├── versions/
│   ├── v1/          # archived baseline
│   ├── v2/          # archived conservative-event version
│   ├── v2_1/        # archived semantic-window version
│   └── v2_2/        # current active version
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
            └── v2_2/
```

共享 proposal 只生成一次。不同版本只把自己的结果写入 `output/<episode>_analysis/versions/<version>/`。

当前推荐运行 V2.2：

```bash
bash versions/v2_2/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

以后新增版本统一放到 `versions/vX_Y/`，不要再往 `event_analyze/` 根目录增加版本文件。
