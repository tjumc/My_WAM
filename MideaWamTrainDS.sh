#!/bin/bash
kill -9 python
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
# =========================================================================
# 1. 网络与多机环境配置 (复用之前的 Volcano 环境配置)
# =========================================================================
# NCCL 环境变量：确保 IB/RoCE 网络通信正常，避免通信超时
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0,en0
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=10000 
export NCCL_SOCKET_TIMEOUT_MS=360000

# === 获取 Master 节点信息 ===
# 从 /etc/volcano/worker.host 读取主节点 IP
export MASTER_ADDR=$(cat /etc/volcano/worker.host | head -n 1 | awk '{print $1}')
export MASTER_PORT=29604  # 指定一个固定的空闲端口

# === 节点与 GPU 计数配置 ===
export NNODES=2                   # 总机器数
export GPUS_PER_NODE=8             # 单机卡数
export TOTAL_GPUS=$((NNODES * GPUS_PER_NODE)) # 总进程数: 3 * 8 = 24

# 关键修复：将 Volcano 节点索引映射为 accelerate/deepspeed 需要的 NODE_RANK

# 调试信息
echo "------------------------------------------------"
echo "Master Node IP: ${MASTER_ADDR}"
echo "Master Port   : ${MASTER_PORT}"
echo "This Node Rank: ${NODE_RANK} (Total Nodes: ${NNODES})"
echo "Total GPUs    : ${TOTAL_GPUS}"
echo "------------------------------------------------"

# =====================================================
# 4. 启动训练
# =====================================================
# =====================================================
# 4. 启动训练
# =====================================================
cd /data/share/1919650160032350208/wzy/FastWAM
source /root/YES/etc/profile.d/conda.sh
conda activate ./envs/fastwam

# ====================================================
# 启动训练
# train_zero1.sh 第一个参数传总 GPU 数
# 2 台 x 8 卡 = 16
# ====================================================
bash scripts/train_zero1ds.sh 8 task=pick_place_1e-4   resume=./checkpoints/step_055000.pt