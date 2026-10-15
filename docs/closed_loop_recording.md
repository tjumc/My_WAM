# 真机闭环记录与首次异常诊断

本模块记录客户端**实际读到的观测、完整 RPC 动作响应、截取/混合后的动作、传给 SDK 的 waypoint 参数**。
它不训练、不加载额外模型、不改任务指令、不改动作参数，也不是安全控制器。

## 启用

先更新仓库。在机器人客户端机器的 SDK Python 环境中，检查 `scripts/real_robot/client_record.sh` 顶部：

```bash
RECORD_ENABLED=true
RECORD_ROOT="${HOME}/wam_rollouts"
RECORD_QUEUE_SIZE=8
RECORD_MAX_GB=5
RECORD_MIN_FREE_GB=2
RECORD_IMAGE_EVERY=1
RECORD_CLOSE_TIMEOUT=3
PYTHON_BIN="python"
```

**会启动真实机器人，并执行原来已有的 move_to_home / 首次位置移动。**
先遵循实验室现有安全流程，保持物理急停可用。不要把 Ctrl+C 或停止记录当成硬件急停。
服务端按原方式启动；客户端改用：

```bash
bash scripts/real_robot/client_record.sh
```

任务指令、服务器地址和所有执行参数仍读取原 `scripts/real_robot/config.sh`。
原 `client.sh` 默认不启用记录；新启动器设 `RECORD_ENABLED=false` 也可禁用。
记录开始于主控制循环；启动期的 move_to_home 等并非完整录制范围。

`RECORD_ROOT` 是**客户端本机**路径，不是推理服务器路径。推荐客户端可写的本地实验磁盘，尽量避免慢网络盘。
旧的 mining / inference / replay 输出路径完全不变。每次创建独立 `rollout_日期_随机ID/`，不覆盖。
大小参数单位 GiB；达到容量上限/空间保留线后停用的是**记录**，不是机器人。

## 记录格式

```
rollout_.../
  .gitignore          # 默认忽略目录内所有原始产物
  manifest.json       # 客户端参数、源码哈希、字段含义
  events.jsonl        # 递增 event_id、request_id、墙钟/单调时钟、数组相对路径
  records/*.npz       # 三路原图、状态、预测与命令的无损数值快照
  CLOSED.json         # 有界排空后生成，含丢记录/写盘错误/复制耗时
```

运行中保留 `RUNNING.json`。结束后若没有 `CLOSED.json`，说明可能被强制中断、写盘失败或超过排空期限，不能当完整记录。
`closed` 仅表示记录关闭，不表示任务成功；`closed_with_gaps` 明确存在缺失。

主要事件：

- `observation`：当前三相机、原始25维 policy state、SDK实际关节位置；`feedback_after_request` 指向上一条尝试下发的请求。
- `prediction`：服务器返回的完整反归一化动作（截取和混合前）、RPC往返耗时。**不是GT**。
- `sdk_initial_position_attempt/return`：原客户端首次单点位置移动；此后原来的1秒等待未改。
- `sdk_waypoints_attempt`：截取并可选混合后的动作、**实际23维SDK位置waypoints**、时间表、group names 与调用参数。
- `sdk_waypoints_return`：方法返回时刻和调用耗时；返回值为简单标量时保留，复杂对象仅记录类型。
- `rpc_none`、`loop_error`、`loop_interrupted`、`loop_end`：无有效预测、异常、中断、退出原因；不新增自动成功标签。

速度模式下模型前三维是底盘速度，SDK下发的前三维是积分后位置，两者分别保存，不能混为一谈。
`selected_actions_after_blend25` 仍含预测head两维，但SDK23维不含它们；原head-follow机制未改。
读取 NPZ 使用 `np.load(path, allow_pickle=False)`，图像 `image_0/1/2` 顺序为 head/left/right，保留回调原始dtype/数值，不做颜色空间假设。

## 时间与反馈限制

所有时钟来自**客户端**；RPC往返不是纯模型推理耗时，SDK调用耗时不一定是物理执行时长。
相机记录的是回调接收时间，不是采集时间；三相机和关节读取不保证同步。保留状态读取起止时刻。
下一轮自然观测作为反馈，不增加后台 SDK 查询、不额外等待运动完成。
最后一轮退出后可能没有反馈，明确保留缺失，不补零、不拿预测充当真实状态。
每次推理保存快照，**不是30 Hz连续录像或连续执行轨迹**。跨chunk的短暂滑落等仍可能需要实验室外部录像。

磁盘I/O、压缩在线程中；队列容量也包含正在写的事件。队列满时在复制前丢弃本条，并计数。
数组快照复制在主线程，仍有开销；`submit_max_ms`/`submit_total_ms` 记录这一开销，但不是整个控制系统的实时性保证。
硬退出、电源故障可能留下部分索引/孤立NPZ；分析前核对完整性。

## 输出可上传的小报告

运行结束后，在 `scripts/summarize_robot_rollout.sh` 顶部填写终端打印的 `RUN_DIR`，执行：

```bash
bash scripts/summarize_robot_rollout.sh
```

只生成 `report.md`、`summary.json`、`requests.csv`，不复制图片/NPZ。可把这个新报告目录提交 GitHub。
它包含参数和服务器报告的checkpoint路径等，提交前检查是否需要去除敏感路径。
原始目录中所有文件默认被 `.gitignore` 忽略；不要 `git add -f` 强行上传。完整数据在实验存储另行备份。

先看记录完整性，再从第一轮开始填写 `requests.csv` 的阶段、预期行为、真实结果。
只在首次异常处标记 `local_execution / outcome_verification / phase_selection / runtime_timing / unknown`，没有证据就保留unknown。
SDK返回、模型输出非零动作、轨迹结束或max_runtime都不能自动解释为任务成功/失败。

## 无硬件测试

在带 numpy/pytest 的 Python 环境执行，测试用模拟SDK，不需要真实机器人或权重：

```bash
python -m pytest tests/test_rollout_recorder.py tests/test_recording_client.py -q
```

真实硬件尚需实测记录开销、磁盘速度及实际SDK返回/完成时序；本模块不据此改变执行频率或安全参数。
