#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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
export NCCL_NVLS_ENABLE=0

# === 获取 Master 节点信息 ===
# 从 /etc/volcano/worker.host 读取主节点 IP
export MASTER_ADDR=$(cat /etc/volcano/worker.host | head -n 1 | awk '{print $1}')
export MASTER_PORT=29604  # 指定一个固定的空闲端口

# === 节点与 GPU 计数配置 ===
export NNODES=16                   # 总机器数
export GPUS_PER_NODE=8             # 单机卡数
export TOTAL_GPUS=$((NNODES * GPUS_PER_NODE)) # 总进程数: 2 * 8 = 16

# 关键修复：将 Volcano 节点索引映射为 accelerate/deepspeed 需要的 NODE_RANK
export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-${MACHINE_RANK:-0}}}"

# === 训练配置 ===
CONFIG="configs/train/pretrain.yaml"
ACCELERATE_CONFIG="scripts/accelerate_configs/accelerate_zero1_ds.yaml"
TASK_NAME="pretrain"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${TASK_NAME}/dim16}"

mkdir -p "${OUTPUT_DIR}"

# 调试信息
echo "------------------------------------------------"
echo "Master Node IP: ${MASTER_ADDR}"
echo "Master Port   : ${MASTER_PORT}"
echo "This Node Rank: ${NODE_RANK} (Total Nodes: ${NNODES})"
echo "Total GPUs    : ${TOTAL_GPUS}"
echo "Config        : ${CONFIG}"
echo "Output Dir    : ${OUTPUT_DIR}"
echo "Entry         : scripts/train.py"
echo "------------------------------------------------"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train.py \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 12 \
  --num_workers 16 \
  --learning_rate 2.0e-4 \
  --weight_decay 1.0e-2 \
  --num_epochs 1 \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --lr_scheduler_type cosine \
  --seed 42 \
  --max_grad_norm 1.0 \
  --log_every 20 \
  --save_every 10000 \
  --eval_every 0 \
  --eval_num_inference_steps 10 \
  --resume "/data/private/2026/FastWAM/runs/pretrain/dim16/checkpoints/state/step_030000"
  "$@"
