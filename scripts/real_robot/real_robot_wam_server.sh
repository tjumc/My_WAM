#!/usr/bin/env bash
set -euo pipefail

# Real-robot WAM Portal server launcher.
# Keep this file in the WAM repo so pi0_astribot can stay unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WAM_ROOT}"

source scripts/load_wam_local_paths.sh

if [[ -n "${WAM_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${WAM_CONDA_ENV}"
fi

export PORTAL_PORT="${PORTAL_PORT:-2222}"
export DEVICE="${DEVICE:-cuda}"
export ACTION_MODE="${ACTION_MODE:-cmd_absolute_joint_vel}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export WAN22_MODEL_ID="${WAN22_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
export WAN22_TOKENIZER_MODEL_ID="${WAN22_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
export WAN22_REDIRECT_COMMON_FILES="${WAN22_REDIRECT_COMMON_FILES:-true}"

RUN_DIR="${WAM_ROOT}/runs/astribot_washclothes_posttrain32_from_step10k"
export CONFIG="${CONFIG:-${RUN_DIR}/train/config.yaml}"
export CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/train/checkpoints/weights/step_050000.pt}"
export DATASET_STATS="${DATASET_STATS:-${RUN_DIR}/stats.json}"
export TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-${WAM_ROOT}/posttrain/text_embeds_cache_astribot_washclothes_posttrain32}"

# Safer default after OOM; override with VAE_DEVICE_MODE=gpu if there is enough memory.
export VAE_DEVICE_MODE="${VAE_DEVICE_MODE:-cpu}"

echo "------------------------------------------------"
echo "Starting WAM Portal server"
echo "WAM_ROOT       : ${WAM_ROOT}"
echo "Portal port    : ${PORTAL_PORT}"
echo "Device         : ${DEVICE}"
echo "VAE mode       : ${VAE_DEVICE_MODE}"
echo "Action mode    : ${ACTION_MODE}"
echo "Config         : ${CONFIG}"
echo "Checkpoint     : ${CHECKPOINT}"
echo "Dataset stats  : ${DATASET_STATS}"
echo "Model base     : ${DIFFSYNTH_MODEL_BASE_PATH}"
echo "Text cache     : ${TEXT_CACHE_DIR}"
echo "------------------------------------------------"

exec bash scripts/real_robot/run_astribot_wam_infer_server.sh
