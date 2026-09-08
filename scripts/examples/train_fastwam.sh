#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Model, dataset preparation, and output paths.
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/path/to/checkpoints}"
export ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-/path/to/ActionDiT_linear_interp_Wan22.pt}"
export WAM_STATS_PATH="${WAM_STATS_PATH:-/path/to/dataset_stats.json}"
export WAM_TEXT_CACHE_DIR="${WAM_TEXT_CACHE_DIR:-/path/to/text_embeds_cache}"
export WAM_PRETRAIN_CKPT="${WAM_PRETRAIN_CKPT:-/path/to/pretrained_wam.pt}"
export WAM_OUTPUT_DIR="${WAM_OUTPUT_DIR:-runs/fastwam/collect_clothes}"

# Task and dataset catalog.
export WAM_TASK_NAME="${WAM_TASK_NAME:-collect_clothes}"
export WAM_TASK_INSTRUCTION="${WAM_TASK_INSTRUCTION:-collect the clothes}"
source configs/data/astribot_dataset_catalog.sh
astribot_select_dataset_task "${WAM_TASK_NAME}"
printf -v WAM_DATASET_DIRS_ENV '%s\n' "${WAM_DATASET_DIRS[@]}"
export WAM_DATASET_DIRS_ENV

# Managed multi-node topology.
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

mkdir -p "$(dirname "${WAM_STATS_PATH}")" "${WAM_TEXT_CACHE_DIR}" "${WAM_OUTPUT_DIR}"
PREPARE_DONE="${WAM_TEXT_CACHE_DIR}/.prepare_complete"

has_text_cache() {
  find "${WAM_TEXT_CACHE_DIR}" -type f -name '*.pt' -print -quit 2>/dev/null | grep -q .
}

# Only rank 0 prepares shared artifacts; all other nodes wait for completion.
if [[ ! -f "${WAM_STATS_PATH}" ]] || ! has_text_cache || [[ ! -f "${PREPARE_DONE}" ]]; then
  if ((NODE_RANK == 0)); then
    rm -f "${PREPARE_DONE}"
    python scripts/precompute_stats_optimize.py \
      --dataset-yaml configs/data/astribot_posttrain32.yaml \
      --output "${WAM_STATS_PATH}" \
      --num-workers "${WAM_PREPARE_NUM_WORKERS:-8}" \
      --skip-quantile

    env WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
      MASTER_PORT="${WAM_TEXT_MASTER_PORT:-29605}" \
      python scripts/precompute_text_embeds_direct.py \
      --dataset-dir "${WAM_DATASET_DIRS[@]}" \
      --cache-dir "${WAM_TEXT_CACHE_DIR}" \
      --override-instruction "${WAM_TASK_INSTRUCTION}" \
      --context-len 128 \
      --batch-size "${WAM_TEXT_BATCH_SIZE:-16}" \
      --model-id Wan-AI/Wan2.2-TI2V-5B \
      --tokenizer-model-id Wan-AI/Wan2.1-T2V-1.3B \
      --no-redirect-common-files \
      --skip-existing
    touch "${PREPARE_DONE}"
  else
    wait_seconds=0
    wait_timeout="${WAM_PREPARE_WAIT_TIMEOUT:-7200}"
    until [[ -f "${WAM_STATS_PATH}" ]] && has_text_cache && [[ -f "${PREPARE_DONE}" ]]; do
      if ((wait_seconds >= wait_timeout)); then
        echo "ERROR: timed out waiting for rank 0 to prepare stats and text cache." >&2
        exit 1
      fi
      sleep 10
      wait_seconds=$((wait_seconds + 10))
    done
  fi
fi

echo "FastWAM: node ${NODE_RANK}/${NNODES}, GPUs ${GPUS_PER_NODE}, master ${MASTER_ADDR}:${MASTER_PORT}"
echo "Output: ${WAM_OUTPUT_DIR}"

exec accelerate launch \
  --config_file "${WAM_ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}" \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train.py \
  --config "${WAM_CONFIG:-configs/train/astribot_posttrain32.yaml}" \
  --batch_size "${WAM_BATCH_SIZE:-1}" \
  --num_workers "${WAM_NUM_WORKERS:-4}" \
  --learning_rate "${WAM_LEARNING_RATE:-2e-4}" \
  --weight_decay "${WAM_WEIGHT_DECAY:-1e-2}" \
  --num_epochs "${WAM_NUM_EPOCHS:-5}" \
  --gradient_accumulation_steps "${WAM_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --mixed_precision "${WAM_MIXED_PRECISION:-bf16}" \
  --lr_scheduler_type "${WAM_LR_SCHEDULER_TYPE:-cosine}" \
  --seed "${WAM_SEED:-42}" \
  --max_grad_norm "${WAM_MAX_GRAD_NORM:-1.0}" \
  --log_every "${WAM_LOG_EVERY:-10}" \
  --save_every "${WAM_SAVE_EVERY:-10000}" \
  --eval_every "${WAM_EVAL_EVERY:-10000}" \
  --eval_num_inference_steps "${WAM_EVAL_NUM_INFERENCE_STEPS:-10}" \
  --resume "${WAM_PRETRAIN_CKPT}" \
  "$@"
