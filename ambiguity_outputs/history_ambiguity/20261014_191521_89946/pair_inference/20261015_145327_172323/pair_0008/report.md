# 第 8 对：WAM 离线推理

每侧 10 个 seed；每次 32 步。只输入当前图像、当前 proprio、相同任务文本。
评分只统计真实 25 维；GT 是示范，不是所有正确动作的集合。

| 侧别 | 对本侧 GT 的 RMSE（均值±seed标准差） | 对另一侧 GT 的 RMSE | 更接近另一侧 GT 的 seed 数 |
|---|---:|---:|---:|
| A | 0.04287 ± 0.00275 | 0.66400 | 0/10 |
| B | 0.03748 ± 0.00476 | 0.66883 | 0/10 |

GT 两侧距离：0.67114；同 seed 两侧预测距离均值：0.66165。

## 解读边界
- 更接近另一侧 GT 只是分支混淆线索，不等于真机失败；当前关节差异也可能区分阶段。
- 预测距离较小不自动证明动作坍缩；GT 本来相似时尤其不能这样解释。
- 完成态应与真实减速/收尾 GT 比较，不能把任何非零动作视为错误。
- 未自动划分测试集，不代表未见轨迹的泛化结果，也没有证明记忆缺失是原因。
- 原始 GT 保留未裁剪值；预测反归一化未裁剪。存在训练归一化裁剪时，两种空间误差可能不同。
- joint_curves 的虚线是多 seed 均值，并非实际执行轨迹；逐 seed 数值见 CSV/NPZ。
- 当前不进行连续 rollout，因此不能断言未来几秒会重新开门或长期保持。

文件：seed_metrics.csv、paired_seed_metrics.csv、group_metrics.csv、actions.npz、
policy_inputs.npz、predictions_raw.csv、current_policy_inputs.png、joint_curves/。
