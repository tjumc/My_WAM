# V2.2

当前 active 版本。V2.2 在 V2.1 的 semantic-window 框架上解决四类问题：

1. **状态变化优先于自由 caption**：先识别 dishwasher door / rack / hand-object / object location 的前后状态，再由代码推导 open/close/pull/push/grasp/release/transport。
2. **轨迹一致性约束**：孤立左右手跳变会被降权；中途出现的 `post_task` 若后面又恢复 relevant，会自动降级为 `off_task`。
3. **带动作时长先验的边界细化**：避免 transport/open/push 等持续动作被 activity valley 压缩成几帧。
4. **允许 coarse-only gaps**：semantic phase 不再强制连续覆盖整条轨迹，未被可靠细标签覆盖的区间保持 coarse task supervision。

运行：

```bash
bash versions/v2_2/run.sh \
  /path/to/episode.hdf5 \
  output/<episode>_analysis \
  "put the dish into the dishwasher" \
  11 1645
```

输出目录：

`output/<episode>_analysis/versions/v2_2/`

主要文件：

- `window_semantics.jsonl`
- `window_semantics_consistent.jsonl`
- `action_intervals.json`
- `semantic_hierarchy.json`
- `hierarchical_annotations.json`
- `boundaries.png`
- `vlm_raw/`

训练时建议：
- fine span 内使用 `Task: <L0>. Current stage: <L2>.`
- `coarse_only_segments` 仅使用 L0
- 跨 fine span 边界的训练窗口优先 skip
