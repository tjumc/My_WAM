#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

# Wan2.2 base model and ActionDiT initialization.
export DIFFSYNTH_MODEL_BASE_PATH="/path/to/checkpoints"
export ACTION_DIT_PRETRAINED_PATH="/path/to/ActionDiT_linear_interp_Wan22.pt"

# Dataset statistics, text cache, pretrained WAM checkpoint, and output.
export WAM_STATS_PATH="/path/to/dataset_stats.json"
export WAM_TEXT_CACHE_DIR="/path/to/text_embeds_cache"
export WAM_PRETRAIN_CKPT="/path/to/pretrained_wam.pt"
export WAM_OUTPUT_DIR="runs/fastwam/collect_clothes"

# Task and dataset catalog.
export WAM_TASK_NAME="collect_clothes"
export WAM_TASK_INSTRUCTION="collect the clothes"
source configs/data/astribot_dataset_catalog.sh
astribot_select_dataset_task "$WAM_TASK_NAME"
printf -v WAM_DATASET_DIRS_ENV '%s\n' "${WAM_DATASET_DIRS[@]}"
export WAM_DATASET_DIRS_ENV

# Prepare normalization statistics and cached T5 context before training.
python scripts/precompute_stats_optimize.py \
  --dataset-yaml configs/data/astribot_posttrain32.yaml \
  --output "$WAM_STATS_PATH" \
  --num-workers 8 \
  --skip-quantile

python scripts/precompute_text_embeds_direct.py \
  --dataset-dir "${WAM_DATASET_DIRS[@]}" \
  --cache-dir "$WAM_TEXT_CACHE_DIR" \
  --override-instruction "$WAM_TASK_INSTRUCTION" \
  --context-len 128 \
  --batch-size 16 \
  --model-id Wan-AI/Wan2.2-TI2V-5B \
  --tokenizer-model-id Wan-AI/Wan2.1-T2V-1.3B \
  --no-redirect-common-files \
  --skip-existing

# Distributed training.
export NNODES=1
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "$((NNODES * GPUS_PER_NODE))" \
  --num_machines "$NNODES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  scripts/train.py \
  --config configs/train/astribot_posttrain32.yaml \
  --batch_size 12 \
  --num_workers 16 \
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
  --resume "$WAM_PRETRAIN_CKPT"
