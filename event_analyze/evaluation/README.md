# Temporal Annotation Evaluation

统一评估自动层次标注与 frame-level manual ground truth。

## 输入

- prediction: `hierarchical_annotations.json`
- GT: `regression/dishwasher_episode27_manual_gt.json`

GT 使用两套边界：
- `raw_start/end_frame`: semantic span，用于阶段级 segmentation；
- `core_start/end_frame`: strict physical-interaction span。

预测对应：
- `raw_start/end_frame` -> semantic span；
- `provisional_start/end_frame` -> core interaction estimate。

## 指标

### Sequence
- phase precision / recall / F1
- LCS
- edit distance
- exact sequence match

匹配采用 **单调同类 phase 对齐**：先最大化匹配阶段数，再用 semantic IoU 解决重复同类预测的配对问题。漏检和重复预测不会被 nearest-neighbor 隐藏。

### Semantic temporal
- matched mIoU
- penalized mIoU（未匹配 phase 按 0 计入）
- start/end/boundary MAE
- Boundary F1 @ ±0.5 / ±1 / ±2 s
- manual boundary tolerance accuracy

### Core temporal
使用预测 `provisional_*` 对 GT `core_*`，计算相同的 IoU / MAE / Boundary F1。

### Coverage
- predicted fine-label coverage
- coarse-only ratio
- correctly covered GT duration
- false-positive duration ratio
- missed-GT duration ratio

## 使用

```bash
python evaluation/evaluate_temporal_annotations.py \
  output/<episode>_analysis/versions/v3_2/hierarchical_annotations.json \
  regression/dishwasher_episode27_manual_gt.json \
  --output output/<episode>_analysis/versions/v3_2/evaluation_temporal.json
```

建议以后每个版本都保存一份 `evaluation_temporal.json`，作为 regression test。
