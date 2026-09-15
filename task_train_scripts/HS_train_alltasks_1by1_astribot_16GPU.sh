#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# Paths.
export DIFFSYNTH_MODEL_BASE_PATH="/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints"
export ACTION_DIT_PRETRAINED_PATH="/efs/share/1919650160032350208/users/machong14/wam/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
export WAM_PRETRAIN_CKPT=""
CONFIG="configs/train/astribot_posttrain32.yaml"
DATASET_CONFIG="configs/data/astribot_posttrain32.yaml"
ACCELERATE_CONFIG="scripts/accelerate_configs/accelerate_zero1_ds.yaml"

# Training.
BATCH_SIZE=12
NUM_WORKERS=16
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-2
NUM_EPOCHS=10
GRADIENT_ACCUMULATION_STEPS=1
SAVE_EVERY=10000
EVAL_EVERY=10000
PREPARE_DATA=true #是否需要预处理

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

export PYTHONPATH="$ROOT/src"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export NNODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3 NCCL_NSOCKS_PERTHREAD=1 NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=10000 NCCL_SOCKET_TIMEOUT_MS=360000 NCCL_NVLS_ENABLE=0

declare -A TASK_INSTRUCTIONS=(
  [fridge]="Move to the fridge, take out the plate with food inside from the opened drawer, then move to the desk and put the plate on the desk"
  [dishwasher]="put the dish into the dishwasher"
  [oven_put]="put the plate into the oven"
  [oven_takeout]="take the plate out of the oven"
  [collect_clothes]="Gather the clothes into the laundry basket"
  [dry_clothes]="transfer the clothes to the dryer"
  [wash_clothes]="put the clothes into the washing machine"
  [sort_blocks]="Put the blue block on the table into the right plate, and the red block into the left plate"
)

TASKS=(fridge collect_clothes dry_clothes wash_clothes)
if (( $# )); then TASKS=("$@"); fi

source configs/data/astribot_dataset_catalog.sh

for task in "${TASKS[@]}"; do
  export WAM_TASK_NAME="$task"
  export WAM_TASK_INSTRUCTION="${TASK_INSTRUCTIONS[$task]}"
  astribot_select_dataset_task "$task"
  printf -v WAM_DATASET_DIRS_ENV '%s\n' "${WAM_DATASET_DIRS[@]}"
  export WAM_DATASET_DIRS_ENV

  RUN_NAME=astribot_"$task"_posttrain32
  export WAM_STATS_PATH=runs/"$RUN_NAME"/stats.json
  export WAM_TEXT_CACHE_DIR=posttrain/text_embeds_cache_"$RUN_NAME"
  export WAM_OUTPUT_DIR=runs/"$RUN_NAME"/train
  mkdir -p "$(dirname "$WAM_STATS_PATH")" "$WAM_TEXT_CACHE_DIR" "$WAM_OUTPUT_DIR"

  if $PREPARE_DATA; then
    python scripts/precompute_stats_optimize.py \
      --dataset-yaml "$DATASET_CONFIG" \
      --output "$WAM_STATS_PATH" \
      --num-workers 8 \
      --skip-quantile
  
    env WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29605 \
      python scripts/precompute_text_embeds_direct.py \
      --dataset-yaml "$DATASET_CONFIG" \
      --cache-dir "$WAM_TEXT_CACHE_DIR" \
      --override-instruction "$WAM_TASK_INSTRUCTION" \
      --context-len 128 \
      --batch-size 16 \
      --model-id Wan-AI/Wan2.2-TI2V-5B \
      --tokenizer-model-id Wan-AI/Wan2.2-TI2V-5B \
      --no-redirect-common-files \
      --skip-existing
  fi

  echo "Astribot: $task ($NNODES node x $GPUS_PER_NODE GPU)"
  accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$TOTAL_GPUS" \
    --num_machines "$NNODES" \
    --machine_rank "$NODE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    scripts/train.py \
    --config "$CONFIG" \
    --output_dir "$WAM_OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save_every "$SAVE_EVERY" \
    --eval_every "$EVAL_EVERY" \
    --resume "$WAM_PRETRAIN_CKPT"
done
