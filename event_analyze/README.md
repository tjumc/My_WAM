# Event Analyze

目录按“共享分析 + 版本流水线”组织。

```text
event_analyze/
├── README.md
├── GENERALIZATION.md
├── common/
├── regression/
├── results/          # compact Git-tracked experiment records
├── output/           # local runtime/debug cache; new files ignored by Git
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
│   ├── v3_4_0/
│   ├── v3_4_1/
│   ├── v3_4_2/
│   └── v3_4_3/      # current active: boundary-validation consistency
```

当前推荐运行 V3.4.3：

```bash
bash versions/v3_4_3/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

V3.4.3 保留 V3.4.2 的 joint hand-object temporal ownership，并进一步约束最终时序：boundary refinement 不得无约束地压缩上游已确认的 semantic ownership interval；container boundary 若被 contact anchor 重定位，必须在新位置重新计算 robot signal 后再验证。perception 默认继续复用 V3.4.1，以便做受控 reasoning 对比。当前 perception 暂时保持 V3.3.3-compatible，用于验证 schema-driven reasoning 的等价性。对全新的 HDF5，入口脚本会自动先生成 candidate_events.json 和 contact_sheets/。

泛化边界与未来 task-agnostic 设计见 `GENERALIZATION.md`。截至当前版本的问题演化、已解决问题、未解决问题与论文贡献候选统一记录在 `RESEARCH_RECORD.md`。


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

默认比较 `v3_3_1,v3_3_2,v3_3_3,v3_4_0,v3_4_1,v3_4_2,v3_4_3` 在 episode26/27 上的 sequence Precision/Recall/F1、LCS/Edit Distance。


## 实验文件保存规范

从 V3.4.0 开始，`output/` 作为本地运行与调试缓存；新的运行产物默认不再作为 Git 长期实验记录。运行结束会自动将精简结果导出到：

```text
results/<episode>/<version>/
```

其中长期保留 final annotation、evaluation、summary 和 manifest。contact sheets、dense strips、VLM raw response、overview image 等可重建中间文件留在本地即可。详见 `STORAGE_POLICY.md`。

旧版本已经提交的 `output/` 文件暂时不做破坏性清理，也不重写 Git 历史。


## 统一 GT 环境变量

从 V3.4.3 开始，所有后续版本统一使用 `GT` 指定 ground truth，不再把版本号写进变量名：

```bash
GT=regression/dishwasher_episode27_manual_gt.json \
bash versions/v3_4_3/run.sh ...
```

后续版本保持 `GT` 不变，这样同一条实验命令只需要替换版本目录即可。
