#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source scripts/load_wam_local_paths.sh
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH in wam_local_paths.sh}"
: "${DIFFSYNTH_SKIP_DOWNLOAD:=true}"
: "${WAN22_MODEL_ID:=Wan-AI/Wan2.2-TI2V-5B}"
: "${WAN22_TOKENIZER_MODEL_ID:=Wan-AI/Wan2.2-TI2V-5B}"
: "${WAN22_REDIRECT_COMMON_FILES:=false}"
export DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_SKIP_DOWNLOAD
export WAN22_MODEL_ID WAN22_TOKENIZER_MODEL_ID WAN22_REDIRECT_COMMON_FILES

if [ ! -e "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" ] && \
   [ -d "${DIFFSYNTH_MODEL_BASE_PATH}/Wan2.2-TI2V-5B" ]; then
  mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI"
  ln -sfn ../Wan2.2-TI2V-5B "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
fi

RUN_DIR="${REPO_ROOT}/runs/astribot_washclothes_posttrain32_from_step10k"
CONFIG="${CONFIG:-${RUN_DIR}/train/config.yaml}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/train/checkpoints/weights/step_050000.pt}"
DATASET_STATS="${DATASET_STATS:-${RUN_DIR}/stats.json}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-${REPO_ROOT}/posttrain/text_embeds_cache_astribot_washclothes_posttrain32}"
export TEXT_CACHE_DIR
TASK_PROMPT="${TASK_PROMPT:-put the clothes into the washing machine}"

DEVICE="${DEVICE:-cuda}"
ACTION_MODE="${ACTION_MODE:-cmd_absolute_joint_vel}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
VAE_DEVICE_MODE="${VAE_DEVICE_MODE:-cpu}"

EXTRA_ARGS=()
if [[ -n "${INPUT_NPZ:-}" ]]; then
  EXTRA_ARGS+=(--input_npz "${INPUT_NPZ}")
fi
if [[ -n "${OUTPUT_NPZ:-}" ]]; then
  EXTRA_ARGS+=(--output_npz "${OUTPUT_NPZ}")
fi
if [[ -n "${ACTION_HORIZON:-}" ]]; then
  EXTRA_ARGS+=(--action_horizon "${ACTION_HORIZON}")
fi
if [[ -n "${N_ACTION_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--n_action_steps "${N_ACTION_STEPS}")
fi
if [[ -n "${MIXED_PRECISION:-}" ]]; then
  EXTRA_ARGS+=(--mixed_precision "${MIXED_PRECISION}")
fi
if [[ -n "${SIGMA_SHIFT:-}" ]]; then
  EXTRA_ARGS+=(--sigma_shift "${SIGMA_SHIFT}")
fi
if [[ -n "${TEXT_CFG_SCALE:-}" ]]; then
  EXTRA_ARGS+=(--text_cfg_scale "${TEXT_CFG_SCALE}")
fi
if [[ -n "${NEGATIVE_PROMPT:-}" ]]; then
  EXTRA_ARGS+=(--negative_prompt "${NEGATIVE_PROMPT}")
fi
if [[ -n "${RAND_DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--rand_device "${RAND_DEVICE}")
fi
if [[ -n "${SEED:-}" ]]; then
  EXTRA_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${FAKE_OBS_SEED:-}" ]]; then
  EXTRA_ARGS+=(--fake_obs_seed "${FAKE_OBS_SEED}")
fi
case "${TILED:-false}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    EXTRA_ARGS+=(--tiled)
    ;;
esac

echo "------------------------------------------------"
echo "Offline Astribot WAM inference"
echo "Config        : ${CONFIG}"
echo "Checkpoint    : ${CHECKPOINT}"
echo "Dataset stats : ${DATASET_STATS}"
echo "Task prompt   : ${TASK_PROMPT}"
echo "Device        : ${DEVICE}"
echo "Action mode   : ${ACTION_MODE}"
echo "Infer steps   : ${NUM_INFERENCE_STEPS}"
echo "Input npz     : ${INPUT_NPZ:-<synthetic>}"
echo "Output npz    : ${OUTPUT_NPZ:-<none>}"
echo "Model base    : ${DIFFSYNTH_MODEL_BASE_PATH}"
echo "Wan model id  : ${WAN22_MODEL_ID}"
echo "Tokenizer id  : ${WAN22_TOKENIZER_MODEL_ID}"
echo "Redirect      : ${WAN22_REDIRECT_COMMON_FILES}"
echo "Text cache    : ${TEXT_CACHE_DIR:-<from config>}"
echo "------------------------------------------------"

python experiments/astribot/offline_wam_infer.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset_stats "${DATASET_STATS}" \
  --task "${TASK_PROMPT}" \
  --device "${DEVICE}" \
  --action_mode "${ACTION_MODE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --vae_device_mode "${VAE_DEVICE_MODE}" \
  "${EXTRA_ARGS[@]}"
