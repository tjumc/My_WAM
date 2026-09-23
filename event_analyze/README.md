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
│   ├── v3_4_3/
│   ├── v3_4_4/
│   ├── v3_4_5/
│   ├── v3_4_6/
│   └── v3_4_7/      # current active: schema-driven latent state implication
```

当前推荐运行 V3.4.7：

```bash
bash versions/v3_4_7/run.sh \
  /path/to/episode.hdf5 \
  output/dishwasher_2_fx_20260529_episode_27_analysis \
  "put the dish into the dishwasher" \
  11 \
  1645
```

V3.4.7 在 V3.4.6 conservative closed-loop reasoning 之上继续补齐 schema-driven latent state implication：已接受的 dependent action 可证明 prerequisite state，已接受的 object placement 可证明目标 receptacle 当时处于 usage_state。系统只更新隐藏生命周期状态，不会凭空补造 missing skill。这样能够避免“漏掉 pull-out 后把真实 push-in 判成非法”，也能避免“整个 receptacle lifecycle 都漏掉却错误显示 final state 一致”。

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

默认比较到 `v3_4_7` 为止的版本，在 episode26/27 上检查 sequence Precision/Recall/F1、LCS/Edit Distance。


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
bash versions/v3_4_7/run.sh ...
```

后续版本保持 `GT` 不变，这样同一条实验命令只需要替换版本目录即可。


## 批量多轨迹评估

批量工具只负责调度，单条轨迹仍然调用现有
`versions/<version>/run.sh`，不会复制一套分析逻辑。

首先在数据所在服务器上扫描并冻结 dataset split：

```bash
python evaluation/discover_batch.py
```

默认扫描：

```text
/efs/share/1919650160032350208/efs_backup/nas-backup/compressed_data/astribot/dishwasher_2_fx_20260529_compressed/
```

生成：

```text
batches/dishwasher_v1_manifest.json
```

默认划分：

```text
development: episode26, episode27
validation:  6 条
heldout:     6 条
reserve:     其余全部
```

生成后应检查并提交该 manifest。discovery 以配置目录下实际存在的 HDF5 为准，
并把实际扫描数量写入 frozen manifest；不再把外部数据集元信息中的 episode 总数
作为本地目录强制校验。后续版本必须复用同一 manifest，不要根据模型结果重新划分。

注意：`batches/dishwasher_v1.json` 只是扫描配置，不能直接传给
`run_batch.py`；批量运行必须使用生成后的
`batches/dishwasher_v1_manifest.json`。

先 dry-run：

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --split validation \
  --dry-run
```

实际运行：

```bash
python evaluation/run_batch.py \
  batches/dishwasher_v1_manifest.json \
  --split validation
```

默认串行执行。批量运行结束后会自动汇总 compact results，包括
policy sequence 指标、已有 frame-level GT 的 temporal 指标，以及
V3.4.4+ 的 ambiguity / targeted-query / merged-observation 成本。

完整说明见 `batches/README.md`。


## V3.4.7 runtime artifact layout

V3.4.7 默认在成功导出后收敛本地运行目录：

```text
versions/v3_4_7/
├── hierarchical_annotations.json
├── evaluation_temporal.json   # only with frame-level GT
├── diagnostics/
│   ├── reasoning_trace.json
│   ├── conservative_update.json
│   └── final_consistency.json
└── cache/
    └── reproducible perception/proposal artifacts
```

完整中间 tracker / ownership / validation 文件仅在 `KEEP_DEBUG=1` 时长期保留。
这属于工程存储优化，不作为论文贡献。详见 `versions/v3_4_7/README.md`。
