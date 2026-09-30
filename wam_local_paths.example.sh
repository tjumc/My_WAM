#!/usr/bin/env bash
# Copy to wam_local_paths.sh and edit for the current machine.
# wam_local_paths.sh is intentionally ignored by Git.

export WAM_TASK_NAME="dishwasher"
export DIFFSYNTH_MODEL_BASE_PATH="/path/to/model_assets"
export ASTRIBOT_SDK_ENV="$HOME/Workspace/astribot_sdk/install/env.sh"

# Optional real-robot server defaults.
# export CONFIG="/path/to/run/train/config.yaml"
# export CHECKPOINT="/path/to/run/train/checkpoints/weights/step_xxxxxx.pt"
# export DATASET_STATS="/path/to/run/stats.json"
# export TEXT_CACHE_DIR="/path/to/text_embedding_cache"
# export TASK_PROMPT="dishwasher"
