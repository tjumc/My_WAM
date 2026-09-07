#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source scripts/load_wam_local_paths.sh
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

: "${WAM_TASK_NAME:?Set WAM_TASK_NAME in wam_local_paths.sh}"
: "${WAM_TASK_INSTRUCTION:?Set WAM_TASK_INSTRUCTION in wam_local_paths.sh}"
: "${WAM_PRETRAIN_CKPT:?Set WAM_PRETRAIN_CKPT in wam_local_paths.sh}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH in wam_local_paths.sh}"
: "${ACTION_DIT_PRETRAINED_PATH:?Set ACTION_DIT_PRETRAINED_PATH in wam_local_paths.sh}"
if ! declare -p WAM_DATASET_DIRS >/dev/null 2>&1 || [[ "${#WAM_DATASET_DIRS[@]}" -eq 0 ]]; then
  echo "ERROR: WAM_DATASET_DIRS must contain at least one absolute dataset path." >&2
  exit 1
fi

if [[ ! "${WAM_TASK_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "ERROR: WAM_TASK_NAME may only contain letters, numbers, '_' and '-'." >&2
  exit 1
fi

RUN_NAME="${WAM_RUN_NAME:-astribot_${WAM_TASK_NAME}_posttrain32}"
CONFIG="${WAM_CONFIG:-configs/train/astribot_posttrain32.yaml}"
DATASET_YAML="${WAM_DATASET_YAML:-configs/data/astribot_posttrain32.yaml}"
ACCELERATE_CONFIG="${WAM_ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

export WAM_STATS_PATH="${WAM_STATS_PATH:-runs/${RUN_NAME}/stats.json}"
export WAM_TEXT_CACHE_DIR="${WAM_TEXT_CACHE_DIR:-posttrain/text_embeds_cache_${RUN_NAME}}"
export WAM_OUTPUT_DIR="${WAM_OUTPUT_DIR:-./runs/${RUN_NAME}/train}"

: "${DIFFSYNTH_SKIP_DOWNLOAD:=true}"
: "${WAN22_MODEL_ID:=Wan-AI/Wan2.2-TI2V-5B}"
: "${WAN22_TOKENIZER_MODEL_ID:=Wan-AI/Wan2.2-TI2V-5B}"
: "${WAN22_REDIRECT_COMMON_FILES:=false}"
export DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_SKIP_DOWNLOAD
export WAN22_MODEL_ID WAN22_TOKENIZER_MODEL_ID WAN22_REDIRECT_COMMON_FILES

if [[ ! -d "${DIFFSYNTH_MODEL_BASE_PATH}" ]]; then
  echo "ERROR: base model directory not found: ${DIFFSYNTH_MODEL_BASE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
  echo "ERROR: ActionDiT checkpoint not found: ${ACTION_DIT_PRETRAINED_PATH}" >&2
  exit 1
fi
# if [[ ! -f "${WAM_PRETRAIN_CKPT}" ]]; then
#   echo "ERROR: pretrain checkpoint not found: ${WAM_PRETRAIN_CKPT}" >&2
#   exit 1
# fi
for dataset_dir in "${WAM_DATASET_DIRS[@]}"; do
  if [[ ! -d "${dataset_dir}" ]]; then
    echo "ERROR: dataset directory not found: ${dataset_dir}" >&2
    exit 1
  fi
done

if [[ ! -e "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" && \
      -d "${DIFFSYNTH_MODEL_BASE_PATH}/Wan2.2-TI2V-5B" ]]; then
  mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI"
  ln -sfn ../Wan2.2-TI2V-5B "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
fi

# Keep the known-working Volcano/NCCL defaults while allowing environment overrides.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# Let the platform/NCCL plugin select the socket interface unless this server
# explicitly provides one; interface names differ across managed environments.
if [[ -n "${WAM_NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME="${WAM_NCCL_SOCKET_IFNAME}"
fi
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -f /etc/volcano/worker.host ]]; then
    MASTER_ADDR="$(awk 'NR==1 {print $1}' /etc/volcano/worker.host)"
  else
    MASTER_ADDR="127.0.0.1"
  fi
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-${WAM_MASTER_PORT:-29604}}"
export NNODES="${NNODES:-${WAM_NNODES:-1}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-${WAM_GPUS_PER_NODE:-1}}"
export TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-${MACHINE_RANK:-0}}}"

mkdir -p "$(dirname "${WAM_STATS_PATH}")" "${WAM_TEXT_CACHE_DIR}" "${WAM_OUTPUT_DIR}"

has_text_cache() {
  find "${WAM_TEXT_CACHE_DIR}" -type f -name '*.pt' -print -quit 2>/dev/null | grep -q .
}

RUN_PREPARE="${RUN_PREPARE:-${WAM_RUN_PREPARE:-auto}}"
should_prepare=0
case "${RUN_PREPARE}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    should_prepare=1
    ;;
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    should_prepare=0
    ;;
  auto|Auto|AUTO)
    if [[ ! -f "${WAM_STATS_PATH}" ]] || ! has_text_cache; then
      should_prepare=1
    fi
    ;;
  *)
    echo "ERROR: unsupported WAM_RUN_PREPARE=${RUN_PREPARE}; use auto/true/false." >&2
    exit 1
    ;;
esac

echo "------------------------------------------------"
echo "Astribot generic post-training"
echo "Task name      : ${WAM_TASK_NAME}"
echo "Instruction    : ${WAM_TASK_INSTRUCTION}"
echo "Datasets       : ${#WAM_DATASET_DIRS[@]}"
echo "Pretrain ckpt  : ${WAM_PRETRAIN_CKPT}"
echo "Model base     : ${DIFFSYNTH_MODEL_BASE_PATH}"
echo "ActionDiT      : ${ACTION_DIT_PRETRAINED_PATH}"
echo "Stats          : ${WAM_STATS_PATH}"
echo "Text cache     : ${WAM_TEXT_CACHE_DIR}"
echo "Output         : ${WAM_OUTPUT_DIR}"
echo "Nodes x GPUs   : ${NNODES} x ${GPUS_PER_NODE}"
echo "Node rank      : ${NODE_RANK}"
echo "Master         : ${MASTER_ADDR}:${MASTER_PORT}"
echo "NCCL socket IF : ${NCCL_SOCKET_IFNAME:-<auto>}"
echo "Run prepare    : ${RUN_PREPARE} -> ${should_prepare}"
echo "------------------------------------------------"

if [[ "${should_prepare}" -eq 1 ]]; then
  if [[ "${NODE_RANK}" -eq 0 ]]; then
    python scripts/precompute_stats_optimize.py \
      --dataset-yaml "${DATASET_YAML}" \
      --output "${WAM_STATS_PATH}" \
      --num-workers "${WAM_PREPARE_NUM_WORKERS:-8}" \
      --skip-quantile \
      --profile \
      --profile-interval 30

    TEXT_REDIRECT_ARGS=()
    case "${WAN22_REDIRECT_COMMON_FILES}" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF)
        TEXT_REDIRECT_ARGS+=(--no-redirect-common-files)
        ;;
    esac

    TEXT_EMBED_ARGS=(
      scripts/precompute_text_embeds_direct.py
      --dataset-yaml "${DATASET_YAML}"
      --cache-dir "${WAM_TEXT_CACHE_DIR}"
      --override-instruction "${WAM_TASK_INSTRUCTION}"
      --context-len 128
      --batch-size "${WAM_TEXT_BATCH_SIZE:-16}"
      --model-id "${WAN22_MODEL_ID}"
      --tokenizer-model-id "${WAN22_TOKENIZER_MODEL_ID}"
      "${TEXT_REDIRECT_ARGS[@]}"
      --skip-existing
    )

    text_gpus="${WAM_TEXT_GPUS:-1}"
    text_master_port="${WAM_TEXT_MASTER_PORT:-29605}"
    if ((text_gpus <= 1)); then
      # Managed jobs may export an outer WORLD_SIZE; isolate rank-0-only preprocessing from it.
      env WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="${text_master_port}" \
        python "${TEXT_EMBED_ARGS[@]}"
    else
      env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
        -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u TORCHELASTIC_RUN_ID \
        torchrun \
          --nnodes=1 \
          --node_rank=0 \
          --nproc_per_node="${text_gpus}" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="127.0.0.1:${text_master_port}" \
          --rdzv_id=wam-text-precompute \
          "${TEXT_EMBED_ARGS[@]}"
    fi
  else
    wait_seconds=0
    wait_timeout="${WAM_PREPARE_WAIT_TIMEOUT:-7200}"
    until [[ -f "${WAM_STATS_PATH}" ]] && has_text_cache; do
      if ((wait_seconds >= wait_timeout)); then
        echo "ERROR: timed out waiting for rank 0 to finish data preparation." >&2
        exit 1
      fi
      sleep 10
      wait_seconds=$((wait_seconds + 10))
    done
  fi
fi

case "${WAM_PREPARE_ONLY:-false}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    echo "[INFO] Preparation complete; WAM_PREPARE_ONLY=true, training skipped."
    exit 0
    ;;
esac

TRAIN_ARGS=(
  --config "${CONFIG}"
  --output_dir "${WAM_OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE:-${WAM_BATCH_SIZE:-12}}"
  --num_workers "${NUM_WORKERS:-${WAM_NUM_WORKERS:-16}}"
  --learning_rate "${LEARNING_RATE:-${WAM_LEARNING_RATE:-2.0e-4}}"
  --weight_decay "${WEIGHT_DECAY:-${WAM_WEIGHT_DECAY:-1.0e-2}}"
  --num_epochs "${NUM_EPOCHS:-${WAM_NUM_EPOCHS:-5}}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-${WAM_GRADIENT_ACCUMULATION_STEPS:-1}}"
  --mixed_precision "${MIXED_PRECISION:-${WAM_MIXED_PRECISION:-bf16}}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-${WAM_LR_SCHEDULER_TYPE:-cosine}}"
  --seed "${SEED:-${WAM_SEED:-42}}"
  --max_grad_norm "${MAX_GRAD_NORM:-${WAM_MAX_GRAD_NORM:-1.0}}"
  --log_every "${LOG_EVERY:-${WAM_LOG_EVERY:-10}}"
  --save_every "${SAVE_EVERY:-${WAM_SAVE_EVERY:-10000}}"
  --eval_every "${EVAL_EVERY:-${WAM_EVAL_EVERY:-20000}}"
  --eval_num_inference_steps "${EVAL_NUM_INFERENCE_STEPS:-${WAM_EVAL_NUM_INFERENCE_STEPS:-10}}"
  --resume "${WAM_PRETRAIN_CKPT}"
)

if [[ -n "${MAX_STEPS:-${WAM_MAX_STEPS:-}}" ]]; then
  TRAIN_ARGS+=(--max_steps "${MAX_STEPS:-${WAM_MAX_STEPS}}")
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${TOTAL_GPUS}" \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train.py \
  "${TRAIN_ARGS[@]}" \
  "$@"
