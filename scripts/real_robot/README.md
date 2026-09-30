# Fast-WAM Astribot 真机推理

真机启动只使用本目录下三个脚本：

```text
scripts/real_robot/
├── config.sh   # 每次实验只修改这里
├── server.sh   # 启动 Fast-WAM Portal server
├── client.sh   # 启动 Astribot client
└── README.md
```

## 1. 修改配置

先打开：

```bash
vim scripts/real_robot/config.sh
```

最常修改的是：

```text
TASK_PROMPT
CONFIG
CHECKPOINT
DATASET_STATS
TEXT_CACHE_DIR
DIFFSYNTH_MODEL_BASE_PATH

SERVER_HOST
SERVER_PORT
DEVICE
VAE_DEVICE_MODE
NUM_INFERENCE_STEPS

ASTRIBOT_SDK_ENV
EXECUTE_STEPS
ACTION_DT
```

Dishwasher 当前 WAM 使用 25D state/action：

```text
0:3    base_vx, base_vy, base_omega
3:7    torso joints
7:14   left arm
14     left gripper
15:22  right arm
22     right gripper
23:25  head
```

## 2. 启动 Server

在 WAM 推理机器上：

```bash
bash scripts/real_robot/server.sh
```

## 3. 启动 Client

在连接 Astribot 的机器上：

```bash
bash scripts/real_robot/client.sh
```

第一次真机测试建议保持：

```bash
EXECUTE_STEPS="1"
```

确认底盘速度方向、尺度以及 torso/双臂动作正确后，再在 `config.sh` 中增大 `EXECUTE_STEPS`。

## 配置原则

模型结构、state/action 维度和数据处理方式来自训练 `config.yaml`；真机部署时经常变化的 checkpoint、prompt、server 地址和执行参数全部集中在本目录的 `config.sh`。

真机启动不再依赖 `wam_local_paths.sh`，也不再依赖 `pi0_astribot` 的 client。
