#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src"

# Model, dataset preparation, and output paths.
export DIFFSYNTH_MODEL_BASE_PATH="/path/to/checkpoints"
export ACTION_DIT_PRETRAINED_PATH="/path/to/ActionDiT_linear_interp_Wan22.pt"
export WAM_STATS_PATH="/path/to/dataset_stats.json"
export WAM_TEXT_CACHE_DIR="/path/to/text_embeds_cache"
export WAM_PRETRAIN_CKPT="/path/to/pretrained_wam.pt"
export WAM_OUTPUT_DIR="runs/fastwam/collect_clothes"

# Task and dataset catalog.
export WAM_TASK_NAME="collect_clothes"
export WAM_TASK_INSTRUCTION="collect the clothes"
source configs/data/astribot_dataset_catalog.sh
astribot_select_dataset_task "${WAM_TASK_NAME}"
printf -v WAM_DATASET_DIRS_ENV '%s\n' "${WAM_DATASET_DIRS[@]}"
export WAM_DATASET_DIRS_ENV

# Managed multi-node topology.
export NNODES=2
export GPUS_PER_NODE=8
export TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
export NODE_RANK="${VC_TASK_INDEX}"
export MASTER_ADDR="$(awk 'NF {print $1; exit}' /etc/volcano/worker.host)"
export MASTER_PORT=29604

# Volcano/NCCL settings.
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export NCCL_NVLS_ENABLE=0

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
      --num-workers 8 \
      --skip-quantile

    env WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
      MASTER_PORT=29605 \
      python scripts/precompute_text_embeds_direct.py \
      --dataset-dir "${WAM_DATASET_DIRS[@]}" \
      --cache-dir "${WAM_TEXT_CACHE_DIR}" \
      --override-instruction "${WAM_TASK_INSTRUCTION}" \
      --context-len 128 \
      --batch-size 16 \
      --model-id Wan-AI/Wan2.2-TI2V-5B \
      --tokenizer-model-id Wan-AI/Wan2.1-T2V-1.3B \
      --no-redirect-common-files \
      --skip-existing
    touch "${PREPARE_DONE}"
  else
    wait_seconds=0
    wait_timeout=7200
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
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train.py \
  --config configs/train/astribot_posttrain32.yaml \
  --batch_size 1 \
  --num_workers 4 \
  --learning_rate 2e-4 \
  --weight_decay 1e-2 \
  --num_epochs 5 \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --lr_scheduler_type cosine \
  --seed 42 \
  --max_grad_norm 1.0 \
  --log_every 10 \
  --save_every 10000 \
  --eval_every 10000 \
  --eval_num_inference_steps 10 \
  --resume "${WAM_PRETRAIN_CKPT}" \
  "$@"
