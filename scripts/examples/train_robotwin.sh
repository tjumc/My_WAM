#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Distributed training.
export NNODES=2
export GPUS_PER_NODE=8
export TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-${MACHINE_RANK:-0}}}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -f /etc/volcano/worker.host ]]; then
    MASTER_ADDR="$(awk 'NF {print $1; exit}' /etc/volcano/worker.host)"
  elif ((NNODES == 1)); then
    MASTER_ADDR=127.0.0.1
  else
    echo "ERROR: MASTER_ADDR is required for multi-node training." >&2
    exit 1
  fi
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-${WAM_MASTER_PORT:-29604}}"

# Known-working Volcano/NCCL defaults; all can be overridden by the platform.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
if [[ -n "${WAM_NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME="${WAM_NCCL_SOCKET_IFNAME}"
fi

# Wan2.2 base model and ActionDiT initialization.
export DIFFSYNTH_MODEL_BASE_PATH="/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints"
CONFIG="configs/train/robotwin.yaml"
OUTPUT_DIR="runs/robotwin"
mkdir -p "${OUTPUT_DIR}"

echo "RoboTwin: node ${NODE_RANK}/${NNODES}, GPUs ${GPUS_PER_NODE}, master ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

exec accelerate launch \
  --config_file "scripts/accelerate_configs/accelerate_zero1_ds.yaml" \
  --num_processes "$TOTAL_GPUS" \
  --num_machines "$NNODES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  scripts/train.py \
  --config "$CONFIG" \
  --batch_size 12 \
  --num_workers 16 \
  --learning_rate 2e-4 \
  --weight_decay 1e-2 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --output_dir "$OUTPUT_DIR" \
  "$@"
