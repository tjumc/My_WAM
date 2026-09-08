#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Wan2.2 base model and ActionDiT initialization.
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints}"

# Distributed training.
export GPUS_PER_NODE="${GPUS_PER_NODE:-${WAM_GPUS_PER_NODE:-8}}"
if [[ -z "${NNODES:-}" ]]; then
  if [[ -n "${WAM_NNODES:-}" ]]; then
    NNODES="${WAM_NNODES}"
  elif [[ -f /etc/volcano/worker.host ]]; then
    NNODES="$(awk 'NF {count++} END {print count+0}' /etc/volcano/worker.host)"
  elif [[ -n "${WORLD_SIZE:-}" ]]; then
    NNODES=$((WORLD_SIZE / GPUS_PER_NODE))
  else
    NNODES=1
  fi
fi
export NNODES
if ((NNODES < 1 || GPUS_PER_NODE < 1)); then
  echo "ERROR: NNODES and GPUS_PER_NODE must both be positive." >&2
  exit 1
fi
export TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-${MACHINE_RANK:-0}}}"
if ((NODE_RANK < 0 || NODE_RANK >= NNODES)); then
  echo "ERROR: NODE_RANK=${NODE_RANK} is outside [0, $((NNODES - 1))]." >&2
  exit 1
fi

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

CONFIG="${WAM_CONFIG:-configs/train/robotwin.yaml}"
OUTPUT_DIR="${WAM_OUTPUT_DIR:-runs/robotwin/test}"
mkdir -p "${OUTPUT_DIR}"

echo "RoboTwin: node ${NODE_RANK}/${NNODES}, GPUs ${GPUS_PER_NODE}, master ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

exec accelerate launch \
  --config_file "${WAM_ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}" \
  --num_processes "$TOTAL_GPUS" \
  --num_machines "$NNODES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  scripts/train.py \
  --config "$CONFIG" \
  --batch_size "${WAM_BATCH_SIZE:-1}" \
  --num_workers "${WAM_NUM_WORKERS:-4}" \
  --learning_rate "${WAM_LEARNING_RATE:-2e-4}" \
  --weight_decay "${WAM_WEIGHT_DECAY:-1e-2}" \
  --num_epochs "${WAM_NUM_EPOCHS:-5}" \
  --gradient_accumulation_steps "${WAM_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --mixed_precision "${WAM_MIXED_PRECISION:-bf16}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
