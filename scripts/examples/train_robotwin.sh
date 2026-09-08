#!/usr/bin/env bash
set -e

# Wan2.2 base model and ActionDiT initialization.
export DIFFSYNTH_MODEL_BASE_PATH="/path/to/checkpoints"

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
  --config configs/train/robotwin.yaml \
  --batch_size 12 \
  --num_workers 16 \
  --learning_rate 2e-4 \
  --weight_decay 1e-2 \
  --num_epochs 5