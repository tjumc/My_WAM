# 云端当前输入交换诊断

运行 `bash scripts/probe_ambiguity_inputs.sh`。先使用已能运行离线 WAM 的环境，
修改新 sh 顶部的路径和实际任务指令；无需 Astribot 环境，不启动真机或 RPC 服务。
默认第 1、2、9、16、43 对，10 个 seed，四组合共 200 次推理，模型只加载一次。
这是固定探索集合，不是随机抽样评估。所有轨迹按用户说明标注为训练集已见。

| 名称 | 当前三相机图像 | 当前 proprio | 角色 |
|---|---|---|---|
| AA | A | A | 原始基线 |
| BB | B | B | 原始基线 |
| BA | B | A | 保持 A 的 proprio，只换图像 |
| AB | A | B | 保持 B 的 proprio，只换图像 |

四组合用同一个原始任务文本和相同 seed 集合，均重新推理。第一版不复用历史预测，
避免 checkpoint、预处理、文本缓存或归一化配置变化使对照失配。
复用 `infer_ambiguity_pairs.py` 的数据读取、实际 episode/frame 定位、GT独立校验、
当前帧提取和 `infer_action`；模型不会收到未来视频、GT、阶段或完成标签。

## 指标

`pair_summary.csv` 是 `group_summary.csv` 的 all25 行。指标在归一化动作空间计算，
排除32维中补齐的7维；原始单位预测另存 `_raw/`。

- `image_at_proprio_A_mean` = 跨seed平均 RMSE(BA,AA)。
- `image_at_proprio_B_mean` = 跨seed平均 RMSE(AB,BB)。
- `proprio_at_image_A_mean` = 跨seed平均 RMSE(AB,AA)。
- `proprio_at_image_B_mean` = 跨seed平均 RMSE(BA,BB)。
- `image_effect_mean` / `proprio_effect_mean` 为对应两条边效应的平均。
- `interaction_rms_mean` = 跨seed平均 RMS(AA+BB-BA-AB)，描述非加性。
- `AA_own_gt_rmse_mean` / `BB_own_gt_rmse_mean` 是原始基线示范拟合误差。
- `*_seed_std` 为效应在seed之间的标准差；`*_seed_spread_rms` 为各组合输出的采样散布。

分组保留底盘、躯干、左臂、左夹爪、右臂、右夹爪、头部，避免全维平均掩盖局部变化。
不把seed当独立任务试验。共同seed控制采样噪声，不等于跨硬件严格确定性保证。

**混合输入可能物理上不一致，没有对应GT动作。** `_raw/*_reference_distances.csv`
中的距离只能定位预测靠近哪个参考，不能解释成混合条件的错误率或真机失败率。
proprio变化大不意味着视觉被忽略：绝对关节位置命令必然受当前姿态影响。
图像和proprio的干预幅度没有匹配，因此这不是可量化的“模态重要性排名”。
这个实验也没有操纵历史，不能用于证明记忆缺失或世界模型闭环能力。

## 存储与上传

只新增 `pairs.json` 同级目录的 `input_swap/本次时间戳/`，不改旧目录。
`SWAP_ROOT` 可在新 sh 顶部配置。每次使用新目录；拒绝覆盖已有结果或原始数据。

```
input_swap/本次时间戳/
  report.md                # 先读中文摘要
  pair_summary.csv         # 每对总体效应
  group_summary.csv        # 每对分组效应与seed标准差
  manifest.json            # 配置、代码/缓存哈希、checkpoint大小/mtime及输入来源
  COMPLETE.json            # 全部完成才生成；失败则FAILED.json
  _raw/                    # 本机详细数组和逐seed表，Git默认忽略
  run.log                  # 本机日志，默认忽略
  launcher.sh              # 本次启动参数副本，默认忽略
  .gitignore               # 只允许精简报告进入Git
```

仅上传报告、两张摘要CSV、manifest、状态文件和 `.gitignore`。不要用 `git add -f`
绕过忽略规则。无批量曲线、截图或视频输出；完整数组在服务器另行备份。
NPZ通过 `np.load(path, allow_pickle=False)` 读取，condition轴顺序固定AA、BB、BA、AB。
模型输出反归一化不裁剪；GT原始值保留，训练归一化的裁剪计数写入每对原始元数据。
检查点身份仅记录路径/大小/修改时间，不声称是完整权重哈希。

运行失败时会保留已导出的pair并报告缺项；不能把缺失结果解释为无敏感性。

## 无权重测试

`python -m pytest tests/test_probe_ambiguity_inputs.py -q`

测试使用合成输入/模拟模型，不代表真实权重已经跑通，也不验证混合输入物理可行性。
