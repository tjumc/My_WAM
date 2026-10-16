# 单任务专家 vs 多任务 WAM：同轨迹离线对照

本实验只比较**现有 checkpoint 在已见训练示范上的动作预测拟合**，不重训、不操作真机、不做人工子类标注。它不能证明未见轨迹泛化、任务间因果迁移、长程闭环成功，也不自动推断哪个任务帮助/伤害了另一个任务。

新增：
- `scripts/compare_expert_multitask.sh`：把需要经常修改的参数集中在文件顶部。
- `tools/compare_expert_multitask.py`：生成固定样本、推理、配对汇总。
- `tests/test_compare_expert_multitask.py`：可在无模型/GPU环境运行的单元测试。

## 工作方式

使用 `configs/data/astribot_dataset_catalog.sh` 已配置的 7 个多任务训练任务（不含 sort_blocks）。每任务先覆盖不同目录，再从剩余轨迹随机抽取；正式建议 30 条 episode × 8 个窗口 × 3 个 seed；启动脚本默认采用小规模 pilot 3×3×1。每个窗口的源目录、episode 和当前帧固定记录在 `samples.csv`，其后 32 帧 GT 只用于评分。

两模型使用同一帧原始 RGB、同一机器人状态和同一 seed；各自按自己的 `stats.json` 做输入归一化、输出反归一化。动作评分只使用原始 25D，采用**每个任务专家模型的训练 action global_std** 作共同逐维尺度，过小维度设下限 `max(1e-4, 1% × 正标准差中位数)`，同时导出部位的原始单位 RMSE。主指标为按 episode 等权的归一化 RMSE。

指令保持实际训练条件：
- 单任务专家：训练时的固定 override 指令，默认来自已有单任务训练脚本，详见启动脚本。
- 多任务模型：直接使用源 LeRobot 数据记录中的任务文本；如果大小写不同也不强迫一致，报告分别记录文本。

多任务权重只加载一次，跨任务按需更换预计算的文本 embedding。模型推理沿用 Astribot FastWAM Portal policy 的 preprocess/infer/postprocess；从 LeRobot 读取的 RGB 帧先转换为其 SDK BGR 输入约定，以免颜色通道颠倒。训练 dataloader 中的图像插值方式和 Portal 路径可能有细微差异；这是部署侧的一致离线比较，不能当作与训练 loss 完全相同的评价。

## 运行步骤

在 `research/single-vs-multitask` 分支，原有 WAM 环境中：

1. 编辑 `scripts/compare_expert_multitask.sh` 顶部的 `RUN_NAME`、`TASKS`、样本量。首次推荐小规模验证：设置 `EPISODES_PER_TASK=3`、`WINDOWS_PER_EPISODE=3`、`NUM_SEEDS=1`，与正式结果使用不同 `RUN_NAME`。
2. 运行准备（无需 GPU/权重）：

   ```bash
   bash scripts/compare_expert_multitask.sh prepare
   ```

3. 在启动脚本顶部填写**准确**的 `MULTITASK_CHECKPOINT` 和每个任务 `EXPERT_*_CHECKPOINT`；脚本不自动搜索“最新/最好” checkpoint。默认单任务 `runs/astribot_<task>_posttrain32/train/config.yaml`，多任务 `runs/astribot_multitask_posttrain32/train/config.yaml`，如果真实模型的配置、stats 或 cache 存在别处，需先调整启动脚本对应路径。检查固定提示词和当时训练日志一致。然后：

   ```bash
   bash scripts/compare_expert_multitask.sh run
   bash scripts/compare_expert_multitask.sh summarize
   ```

4. 查看 `comparison_reports/<RUN_NAME>/report.md` 和 `tasks.csv`、`groups.csv`、`chunks.csv`。如果 `coverage.csv` 不是 100%，报告会醒目标示 INCOMPLETE，不能只看有结果的部分做结论。

**成本提醒**：正式全量配置（7 任务 × 30 episode × 8 window × 3 seed × 2 模型）约 10,080 次模型推理；启动脚本默认只做 3×3×1 的环境校验。单个模型任务的结果逐条写入 `_raw/*.jsonl`，命令中断后可按相同配置恢复，不会把失败样本随机替换成其他样本。

## 输出及 Git 规则

每轮单独命名 `comparison_reports/<RUN_NAME>/`；目录内的 `.gitignore` 只允许简洁报告和样本、配置清单提交到研究分支，`_raw/` 中的模型配置表、目录原始清单、逐次指标不上传。不要 `git add -f` 提交原始预测/视频。

| 文件 | 含义 |
|---|---|
| `samples.csv` | 固定配对窗口清单，路径+episode+起始帧 |
| `run_manifest.json` | 抽样参数、源码/数据清单摘要 |
| `model_plan.json` | checkpoint/归一化/文本缓存的冻结参数与评分设置 |
| `tasks.csv` | 各任务专家 vs 联合模型的 episode 等权 RMSE 和差值 |
| `groups.csv` | 底盘、躯干、左右臂、夹爪、头的差异 |
| `chunks.csv` | 预测 chunk 的前8/中16/后8步 |
| `coverage.csv` | 成功配对样本数量、缺失率 |
| `report.md` | 结果解读及提示词条件 |

## 解释边界与安全检查

- 所有采样来自训练时涉及的目录，**即使轨迹从未进入前面的离线诊断，也不能称为独立测试集**。
- 单任务与多任务训练采用不同数据规模、总更新数和可能不同的学习配置；观测到的差异不能直接证明知识迁移或梯度冲突。
- 多任务默认直接使用原始任务指令，专家用 override；指令差异是潜在混杂因素，应保留记录。
- 模型会返回一段 32×25 的动作，不等于机器人实际执行的完整轨迹；没有闭环状态演化。
- 会核对每个窗口的 episode、frame 和当前 GT action，禁止数据加载器意外随机换样本。任意一个推理报错即停止，允许修复后续跑。
- 需要 `numpy`、`pyarrow`、原有 WAM/HF/torch、视频解码器及已生成的文本缓存。首次真实服务器集成验证仍需要执行小规模 pilot。

测试：

```bash
python -m unittest discover -s tests -p 'test_compare_expert_multitask.py' -v
```
