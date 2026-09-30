#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/config.sh"

if [[ -n "${WAM_CONDA_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${WAM_CONDA_ENV}"
fi

for path in "${CONFIG}" "${CHECKPOINT}" "${DATASET_STATS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: file not found: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${DIFFSYNTH_MODEL_BASE_PATH}" ]]; then
  echo "ERROR: DIFFSYNTH_MODEL_BASE_PATH not found: ${DIFFSYNTH_MODEL_BASE_PATH}" >&2
  echo "Edit scripts/real_robot/config.sh before running." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TEXT_CACHE_DIR
export DIFFSYNTH_MODEL_BASE_PATH
export DIFFSYNTH_SKIP_DOWNLOAD
export WAN22_MODEL_ID
export WAN22_TOKENIZER_MODEL_ID
export WAN22_REDIRECT_COMMON_FILES

echo "================ Fast-WAM Real Robot Server ================"
echo "Task          : ${TASK_PROMPT}"
echo "Config        : ${CONFIG}"
echo "Checkpoint    : ${CHECKPOINT}"
echo "Stats         : ${DATASET_STATS}"
echo "Text cache    : ${TEXT_CACHE_DIR}"
echo "Port          : ${SERVER_PORT}"
echo "Device        : ${DEVICE}"
echo "VAE mode      : ${VAE_DEVICE_MODE}"
echo "Infer steps   : ${NUM_INFERENCE_STEPS}"
echo "Action mode   : ${ACTION_MODE}"
echo "============================================================"

exec python experiments/astribot/fastwam_portal_server.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset_stats "${DATASET_STATS}" \
  --task "${TASK_PROMPT}" \
  --port "${SERVER_PORT}" \
  --device "${DEVICE}" \
  --action_mode "${ACTION_MODE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --vae_device_mode "${VAE_DEVICE_MODE}"
