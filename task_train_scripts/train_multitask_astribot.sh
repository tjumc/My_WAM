#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# =========================
# Tasks
# =========================
TASKS=(
    oven_put
    fridge
    dishwasher
    oven_takeout
    collect_clothes
    dry_clothes
    wash_clothes
)

# =========================
# Paths
# =========================
BASE_CONFIG=configs/train/astribot_posttrain32.yaml
DATASET_CONFIG=configs/data/astribot_posttrain32.yaml
ACCELERATE_CONFIG=scripts/accelerate_configs/accelerate_zero1_ds.yaml

RUN_NAME=astribot_multitask_posttrain32
MULTITASK_CONFIG=runs/$RUN_NAME/astribot_multitask_posttrain32.yaml

export DIFFSYNTH_MODEL_BASE_PATH=/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints
export ACTION_DIT_PRETRAINED_PATH=""
export WAM_PRETRAIN_CKPT=""

export WAM_STATS_PATH=runs/$RUN_NAME/stats.json
export WAM_TEXT_CACHE_DIR=posttrain/text_embeds_cache_$RUN_NAME
export WAM_OUTPUT_DIR=runs/$RUN_NAME/train

# =========================
# Training
# =========================
BATCH_SIZE=12
NUM_WORKERS=16
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-2
NUM_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
SAVE_EVERY=10000
EVAL_EVERY=10000

PREPARE_DATA=true

# =========================
# Distributed
# =========================
NNODES=1
GPUS_PER_NODE=8
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29604
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

export PYTHONPATH=$ROOT/src:$PYTHONPATH
export DIFFSYNTH_SKIP_DOWNLOAD=true

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export NCCL_NVLS_ENABLE=0

# =========================
# Collect multi-task datasets
# =========================
source configs/data/astribot_dataset_catalog.sh

ALL_DATASET_DIRS=()

for task in "${TASKS[@]}"; do
    mapfile -t TASK_DIRS <<< "${ASTRIBOT_DATASET_CATALOG[$task]}"
    ALL_DATASET_DIRS+=("${TASK_DIRS[@]}")
done

printf -v WAM_DATASET_DIRS_ENV '%s\n' "${ALL_DATASET_DIRS[@]}"
export WAM_DATASET_DIRS_ENV

mkdir -p runs/$RUN_NAME "$WAM_TEXT_CACHE_DIR" "$WAM_OUTPUT_DIR"

# Disable global instruction override for multi-task training.
python task_train_scripts/make_multitask_config.py \
    "$BASE_CONFIG" \
    "$MULTITASK_CONFIG"

# =========================
# Preprocess
# =========================
if $PREPARE_DATA; then
    python scripts/precompute_stats_optimize.py \
        --dataset-yaml "$DATASET_CONFIG" \
        --output "$WAM_STATS_PATH" \
        --num-workers 8 \
        --skip-quantile

    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29605 \
    python scripts/precompute_text_embeds_direct.py \
        --dataset-yaml "$DATASET_CONFIG" \
        --cache-dir "$WAM_TEXT_CACHE_DIR" \
        --context-len 128 \
        --batch-size 16 \
        --model-id Wan-AI/Wan2.2-TI2V-5B \
        --tokenizer-model-id Wan-AI/Wan2.2-TI2V-5B \
        --no-redirect-common-files \
        --skip-existing
fi

# =========================
# Train one shared model
# =========================
RESUME_ARGS=()
if [[ -n "$WAM_PRETRAIN_CKPT" ]]; then
    RESUME_ARGS=(--resume "$WAM_PRETRAIN_CKPT")
fi

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$TOTAL_GPUS" \
    --num_machines "$NNODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    scripts/train.py \
    --config "$MULTITASK_CONFIG" \
    --output_dir "$WAM_OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save_every "$SAVE_EVERY" \
    --eval_every "$EVAL_EVERY" \
    "${RESUME_ARGS[@]}"
