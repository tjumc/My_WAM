#!/usr/bin/env bash
# ============================================================
#              FAST-WAM REAL-ROBOT CONFIG
#              EDIT THIS FILE BEFORE RUNNING
# ============================================================

# -------------------- Experiment --------------------
TASK_PROMPT="dishwasher"

CONFIG="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/train/config.yaml"
CHECKPOINT="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/train/checkpoints/weights/step_275430.pt"
DATASET_STATS="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/stats.json"
TEXT_CACHE_DIR="posttrain/text_embeds_cache_astribot_dishwasher_posttrain32"

# Wan2.2 model assets root. This path must exist on the server machine.
DIFFSYNTH_MODEL_BASE_PATH="/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints"

# -------------------- Server --------------------
SERVER_HOST="127.0.0.1"
SERVER_PORT="2222"

DEVICE="cuda"
VAE_DEVICE_MODE="cpu"
NUM_INFERENCE_STEPS="10"
ACTION_MODE="cmd_absolute_joint_vel"

# Optional: leave empty if the environment is already activated.
WAM_CONDA_ENV=""

# -------------------- Client / Robot --------------------
ASTRIBOT_SDK_ENV="$HOME/Workspace/astribot_sdk/install/env.sh"

# Start conservatively with 1. Increase after the motion is verified.
EXECUTE_STEPS="1"
CONTROL_HZ="50"
ACTION_DT="0.03333333333333333"
BOUNDARY_BLEND_STEPS="0"
MAX_RUNTIME="1800"

# -------------------- Rarely changed --------------------
DIFFSYNTH_SKIP_DOWNLOAD="true"
WAN22_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
WAN22_TOKENIZER_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
WAN22_REDIRECT_COMMON_FILES="true"
