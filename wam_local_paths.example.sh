#!/usr/bin/env bash

# Copy this file to wam_local_paths.sh. This ignored file selects the active
# task and contains all machine/run-specific values.

export DIFFSYNTH_MODEL_BASE_PATH="/absolute/path/to/foundation_model"
export ACTION_DIT_PRETRAINED_PATH="/absolute/path/to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
export WAM_PRETRAIN_CKPT="/absolute/path/to/pretrain_checkpoint.pt"

export WAM_TASK_NAME="collect_clothes"
export WAM_TASK_INSTRUCTION="collect the clothes"

# By default, WAM_TASK_NAME selects directories from the tracked Astribot
# catalog. Set WAM_DATASET_CATALOG to another catalog, or uncomment this array
# to override the selected directories for the current machine/run.
# export WAM_DATASET_CATALOG="/absolute/path/to/astribot_dataset_catalog.sh"
# WAM_DATASET_DIRS=(
#   "/absolute/path/to/dataset_1"
#   "/absolute/path/to/dataset_2"
# )

# Frequently changed launch parameters. Other hyperparameters have stable
# defaults in scripts/train_astribot_posttrain32.sh and remain overridable.
export WAM_NNODES=2
export WAM_GPUS_PER_NODE=16
export WAM_BATCH_SIZE=12
export WAM_NUM_WORKERS=16
export WAM_NUM_EPOCHS=5
export WAM_RUN_PREPARE=auto
