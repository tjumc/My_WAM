# V2.1

当前使用版本：semantic-window action localization。

候选 frame 只是 attention anchor，不要求中心帧恰好是动作边界。

流程：

```text
candidate window
  -> Qwen semantic-window verification
  -> temporal action clustering
  -> semantic phase composition
  -> HDF5 signal-based boundary refinement
```

运行：

```bash
bash versions/v2_1/run.sh \
  /path/to/episode.hdf5 \
  output/<episode>_analysis \
  "put the dish into the dishwasher" \
  11 1645
```

结果会自动归档到：

`output/<episode>_analysis/versions/v2_1/`
