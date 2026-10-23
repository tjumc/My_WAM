# Astribot：chunk 抖动与指令条件诊断

第一轮只采集正确 dishwasher 指令的稳定执行作为基线，再把记录的同一观测用于原指令、同义改写、fridge 指令的离线比较。分别使用专家模型和混合模型，各自保持训练对应的 stats。这里的指标不能证明任务成功、语义拒绝或正确路由。

本工具不提供自主导航、planner、任务完成判断或控制平滑修复。先确定误指令下的预测变化和 chunk 接续问题，再决定门控与控制修改。

## 1. 在真机电脑生成诊断 client

支持已核对的 `pi0_astribot_ymn/evaluate/infer_client_vel.py`，源文件 SHA256：

```
a71c7853ede3e1ef64e9a2ced71bfabe469ba0c3d970df3173c39d97b9309d4a
```

从 WAM 仓库运行，替换路径为真机电脑上的路径：

```bash
python experiments/astribot/instrument_robot_client.py \
  --client /path/to/pi0_astribot_ymn/evaluate/infer_client_vel.py \
  --output /path/to/pi0_astribot_ymn/evaluate/infer_client_vel_diagnostics.py
```

安装器只生成代码，不导入 SDK、不执行机器人动作；原文件保留。版本不符时停止，需检查差异后适配，不能直接绕过 SHA 检查。生成的 `infer_client_vel_diagnostics.py` 和同目录 `wam_diagnostics.py` 需要一起部署。

执行时保留原有工作目录、Python 环境及所有控制参数，只把入口改为诊断副本，并追加：

```bash
python infer_client_vel_diagnostics.py \
  --trace-dir /path/to/traces/dishwasher_correct_shared \
  --capture-every 10 --trace-joint-hz 20
```

上面的命令未列出你原来的控制参数，实际运行须保留。诊断不修改动作、速度积分、waypoint 时间、执行步数或停止规则。开启记录会增加数组复制、磁盘写入和 SDK 读取负担：磁盘写入使用有界后台队列，拥堵时丢记录并报告；`--trace-joint-hz 0` 可关闭周期采样，但开启记录时仍在下发前后读关节状态。先核对开启记录后的正确指令基线表现。SDK 读取可能产生延迟，工具不能承诺零开销。

默认不传 `--trace-dir` 时不记录。每次启用都会新建 session 目录，不覆盖已有实验。正常 Ctrl+C 后检查 `summary.json`；异常退出、写入错误或丢记录必须在分析中注明。

## 2. 可选：server 记录

client 单侧记录已经包含 RPC 输入、预测动作和执行时序。若还需要分离推理耗时，在使用本分支 `fastwam_portal_server.py` 的原启动命令中追加：

```bash
--trace-dir /path/to/traces/server --capture-every 10
```

也可设置 `WAM_TRACE_DIR` 和 `WAM_CAPTURE_EVERY`，适用于调用该 server 的启动脚本。其他部署目录的旧 server 不会自动获得此功能，需同步代码；不要仅设置变量就以为已经记录。

client/server 以 RPC 的 `request_id` 关联。它取自 client 的 wall clock，仅用于关联；两台机器的 monotonic 时间不能直接相减，wall clock 差也不能直接解释为网络延迟。

session 中：

- `manifest.json`：配置、模型信息、源码/配置摘要及控制时序。
- `events.jsonl`：请求、推理返回、下发、SDK 返回、实际关节采样及错误。
- `sample_*.npz`：无损图像/状态或预测动作；不使用 pickle。
- `summary.json`：写入数量、丢记录数、写入错误和关闭状态。

图片采样间隔由 `--capture-every` 控制，0 关闭图片保存；预测动作仍逐次记录。图像较大，应按实验长度预留磁盘空间。不要把观测、权重、trace 提交到 GitHub。

## 3. 画出 chunk 接续曲线

仅需 NumPy；画图另需 Matplotlib。可在 Linux 分析电脑运行：

```bash
python experiments/astribot/analyze_trace.py /path/to/client_SESSION \
  --output /path/to/analysis/new_run
```

输出 `boundaries.csv`、左右臂关节曲线和 `summary.json`。`--no-plots` 只输出表格。输出目录必须不存在。

重点看：

- 新 chunk 第一帧目标与下发前实际关节位置的差值。
- 新 chunk 第一帧与上个 **已下发** chunk 最后一帧的差值；不能把后者当作已执行位置。
- 观测状态读取开始至下发的时间、RPC 时间、SDK 调用返回时间。
- chunk 下发边界附近实测曲线是否出现跳变或反向运动。

首个 chunk 含原有初始定位，CSV 单独标记，分析接续时排除。图中虚线是目标的名义时间表；SDK 返回时间不等于动作完成时间，实际执行语义尚待确认。关节采样是主机读取时间，并非控制器时间。观测延迟以读取关节开始为基准，不是相机曝光时间；图像回调时间另外记录。底盘速度与关节角不合并计算角度误差。

## 4. 同观测、不同指令的离线对照

在可运行 WAM 的环境中进行，不启动 Portal 服务、不连接机器人。选择含 `img0/img1/img2/state` 的 NPZ：在 `events.jsonl` 中找到 `request`（client）或 `infer_result`（server）的 `artifact`。不要把仅含 actions 的 NPZ 传入。建议选任务开始、中段、接近完成时的多帧观测。

准备 `prompts.json`，第一项是基线。以下为格式占位，务必替换为数据中的精确训练指令和对应改写：

```json
[
  {"id": "dishwasher_train", "text": "<精确 dishwasher 训练指令>"},
  {"id": "dishwasher_paraphrase", "text": "<保持动作目标不变的同义改写>"},
  {"id": "fridge_train", "text": "<精确 fridge 训练指令>"}
]
```

每条指令必须已有与训练模板、context 长度一致的 T5 cache。工具复用 server 的缓存加载逻辑，缺失任何 cache 会在加载权重前报错；不会静默用旧指令 embedding。同义改写和纠正后的 oven 标签可能需要先补 cache。

```bash
python experiments/astribot/compare_prompts.py \
  --observations /path/to/obs_start.npz /path/to/obs_middle.npz \
  --prompts /path/to/prompts.json --seeds 0 1 2 \
  --config /path/to/config.yaml \
  --checkpoint /path/to/model.pt \
  --dataset-stats /path/to/model_stats.json \
  --text-cache-dir /path/to/text_cache \
  --action-mode '<与部署 server 完全一致的 action_mode>' \
  --num-inference-steps 10 \
  --output /path/to/comparison/expert
```

其他推理参数也需与部署匹配，例如 `--action-horizon`、`--sigma-shift`、`--vae-device-mode`、`--mixed-precision`。预训练资产沿用部署环境变量。加 `--validate-only` 可在无 GPU 环境校验输入文件和路径；此模式不验证模型或 embedding cache。

然后保持同一批观测/指令/种子，用混合模型及它自己的配置与 stats 再跑一次，输出到另一目录。结果含所有动作 NPZ、`metrics.csv` 和带状态的 manifest。异常时不输出伪造分数，已完成样本保留。

指标按躯干、左右臂分别计算同种子下相对基线的预测差。通过不同 seed 的原始动作比较随机波动；可额外加入与基线文本完全相同但 ID 不同的条目检查重复推理差异。配对 seed 不保证所有 GPU 算子完全确定。改指令后动作变化只能表明输出对条件敏感，是否做对任务仍需与任务目标/人工标注结合。

固定观测对照不能复现闭环轨迹随错误动作不断偏离的全过程，尤其不能单凭此证明 chunk 抖动成因。应结合正确指令的执行 trace、已有错指令测试记录，再决定是否需要额外受控实验。

## 验证

```bash
python3 -m unittest discover -s tests -p test_astribot_diagnostics.py -v
```

CPU 测试覆盖有界记录、输入校验、指标含义、种子配对和 server 推理返回兼容性。它们不替代真机 SDK 时序验证或实际 GPU 权重推理。
