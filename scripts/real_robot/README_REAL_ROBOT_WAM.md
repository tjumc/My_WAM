# WAM 真机推理入口

这些入口放在 WAM 项目下，避免改动或新增 `pi0_astribot/evaluate` 里的文件。

先在仓库根目录配置不受 Git 跟踪的 `wam_local_paths.sh`：

```bash
cp wam_local_paths.example.sh wam_local_paths.sh
# 编辑当前服务器需要加载的外部模型和任务路径
```

## 1. 启动 WAM Portal Server

终端 1，使用 WAM 环境：

```bash
cd /path/to/wam_repo
WAM_CONDA_ENV=wam_infer bash scripts/real_robot_wam_server.sh
```

如果已经手动进入 WAM 环境：

```bash
conda activate wam_infer
cd /path/to/wam_repo
bash scripts/real_robot_wam_server.sh
```

默认端口是 `2222`，默认 `ACTION_MODE=cmd_absolute_joint_vel`。

## 2. 启动机器人控制 Client

终端 2，使用原来的 Astribot SDK 环境：

```bash
source ~/Workspace/astribot_sdk/install/env.sh
cd /path/to/wam_repo
bash scripts/real_robot_wam_client_vel.sh
```

这个脚本会调用：

```text
${PI0_ROOT}/evaluate/infer_client_vel.py
```

但不会在 `pi0_astribot` 里新增文件。

## 3. 常用覆盖

如果路径不一样：

```bash
PI0_ROOT=/path/to/pi0_astribot bash scripts/real_robot_wam_client_vel.sh
```

如果 WAM server 不在机器人本机：

```bash
SERVER_HOST=<WAM_SERVER_IP> SERVER_PORT=2222 bash scripts/real_robot_wam_client_vel.sh
```

第一次上真机默认保守：

```text
EXECUTE_STEPS=1
LOCK_CHASSIS_YAW=current
```

确认动作正常后可以放开：

```bash
EXECUTE_STEPS=8 LOCK_CHASSIS_YAW=off bash scripts/real_robot_wam_client_vel.sh
```
