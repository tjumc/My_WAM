# Fast-WAM Astribot 真机推理

真机 client 已经收进 `My_WAM`，不再依赖 `pi0_astribot/evaluate/infer_client_vel.py`。WAM 真机运行使用本仓库的 `experiments/astribot/real_robot_client.py`。

## WAM 的 25D state/action 语义

Dishwasher 数据的 joint 表示为 25D：

```text
0:3    base_vx, base_vy, base_omega
3:7    torso joints (4)
7:14   left arm joints (7)
14     left gripper
15:22  right arm joints (7)
22     right gripper
23:25  head joints (2)
```

当 server 报告的 `action_mode` 包含 `vel` 时，client 使用当前 chassis velocity 作为 state 的前 3 维，并将预测的前三维 chassis velocity 按 `ACTION_DT` 积分成 chassis position waypoints。其他身体关节直接按 position waypoint 执行。

当前 client 保留原 Astribot 真机行为：开启 head-follow，因此预测的 `action[23:25]` 暂不显式下发。

## 1. 本机路径配置

```bash
cp wam_local_paths.example.sh wam_local_paths.sh
```

编辑 `wam_local_paths.sh`，至少配置模型资产路径；如果需要由脚本自动 source Astribot SDK，也可以设置 `ASTRIBOT_SDK_ENV`。

## 2. 启动 WAM Portal Server

终端 1（WAM 环境）：

```bash
cd /path/to/My_WAM
WAM_CONDA_ENV=wam_infer bash scripts/real_robot_wam_server.sh
```

建议显式指定当前实验配置：

```bash
CONFIG=/path/to/train/config.yaml \
CHECKPOINT=/path/to/checkpoints/weights/step_xxxxxx.pt \
DATASET_STATS=/path/to/stats.json \
TEXT_CACHE_DIR=/path/to/text_cache \
TASK_PROMPT=dishwasher \
ACTION_MODE=cmd_absolute_joint_vel \
bash scripts/real_robot_wam_server.sh
```

## 3. 启动机器人 Client

终端 2（Astribot SDK 环境）：

```bash
source ~/Workspace/astribot_sdk/install/env.sh
cd /path/to/My_WAM
bash scripts/real_robot_wam_client_vel.sh
```

如果 server 在其他机器：

```bash
SERVER_HOST=<WAM_SERVER_IP> SERVER_PORT=2222 bash scripts/real_robot_wam_client_vel.sh
```

第一次上真机默认只执行 1 个 action 后重新规划：

```text
EXECUTE_STEPS=1
```

确认动作语义、方向和尺度正确后再增加，例如：

```bash
EXECUTE_STEPS=8 bash scripts/real_robot_wam_client_vel.sh
```

Dishwasher 数据为 30 Hz，因此默认 `ACTION_DT=1/30 s`。如训练数据频率改变，需要同步修改。

## 4. 运行前检查

client 会从 server 的 `get_model_info()` 校验 `state_dim` 和 `action_dim`。当前 WAM client 只接受 25D/25D，避免误把 Pi0 的 28D deployment interface 接到 WAM checkpoint 上。
