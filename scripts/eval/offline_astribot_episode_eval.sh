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
: "${WAN22_REDIRECT_COMMON_FILES:=true}"
export DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_SKIP_DOWNLOAD
export WAN22_MODEL_ID WAN22_TOKENIZER_MODEL_ID WAN22_REDIRECT_COMMON_FILES

RUN_DIR="${REPO_ROOT}/runs/astribot_washclothes_posttrain32_from_step10k"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-${REPO_ROOT}/posttrain/text_embeds_cache_astribot_washclothes_posttrain32}"
export TEXT_CACHE_DIR

CONFIG="${CONFIG:-${RUN_DIR}/train/config.yaml}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/train/checkpoints/weights/step_050000.pt}"
DATASET_STATS="${DATASET_STATS:-${RUN_DIR}/stats.json}"
TASK_PROMPT="${TASK_PROMPT:-put the clothes into the washing machine}"

EPISODE_INDEX="${EPISODE_INDEX:-0}"
START_STEP="${START_STEP:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-1}"
DEVICE="${DEVICE:-cuda}"
ACTION_MODE="${ACTION_MODE:-cmd_absolute_joint_vel}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
VAE_DEVICE_MODE="${VAE_DEVICE_MODE:-cpu}"

OUTPUT_NPZ="${OUTPUT_NPZ:-${REPO_ROOT}/runs/offline_episode_eval/episode_${EPISODE_INDEX}_start_${START_STEP}.npz}"

EXTRA_ARGS=()
if [[ -n "${INPUT_NPZ:-}" ]]; then
  EXTRA_ARGS+=(--input_npz "${INPUT_NPZ}")
else
  EXTRA_ARGS+=(--episode_index "${EPISODE_INDEX}")
fi
if [[ -n "${DATASET_DIR:-}" ]]; then
  EXTRA_ARGS+=(--dataset_dir "${DATASET_DIR}")
elif [[ -n "${DATASET_DIRS:-}" ]]; then
  IFS=':' read -r -a _DATASET_DIR_ARRAY <<< "${DATASET_DIRS}"
  for _dataset_dir in "${_DATASET_DIR_ARRAY[@]}"; do
    [[ -n "${_dataset_dir}" ]] && EXTRA_ARGS+=(--dataset_dir "${_dataset_dir}")
  done
else
  if ! declare -p WAM_DATASET_DIRS >/dev/null 2>&1 || [[ "${#WAM_DATASET_DIRS[@]}" -eq 0 ]]; then
    echo "ERROR: set DATASET_DIR, DATASET_DIRS, or WAM_DATASET_DIRS in wam_local_paths.sh." >&2
    exit 1
  fi
  for _dataset_dir in "${WAM_DATASET_DIRS[@]}"; do
    EXTRA_ARGS+=(--dataset_dir "${_dataset_dir}")
  done
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${MIXED_PRECISION:-}" ]]; then
  EXTRA_ARGS+=(--mixed_precision "${MIXED_PRECISION}")
fi
if [[ -n "${ACTION_HORIZON:-}" ]]; then
  EXTRA_ARGS+=(--action_horizon "${ACTION_HORIZON}")
fi
if [[ -n "${N_ACTION_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--n_action_steps "${N_ACTION_STEPS}")
fi
if [[ -n "${SIGMA_SHIFT:-}" ]]; then
  EXTRA_ARGS+=(--sigma_shift "${SIGMA_SHIFT}")
fi
if [[ -n "${RAND_DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--rand_device "${RAND_DEVICE}")
fi
if [[ -n "${SEED:-}" ]]; then
  EXTRA_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${OUTPUT_JSON:-}" ]]; then
  EXTRA_ARGS+=(--output_json "${OUTPUT_JSON}")
fi
case "${TILED:-false}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    EXTRA_ARGS+=(--tiled)
    ;;
esac

echo "------------------------------------------------"
echo "Offline Astribot episode eval"
echo "Config        : ${CONFIG}"
echo "Checkpoint    : ${CHECKPOINT}"
echo "Dataset stats : ${DATASET_STATS}"
echo "Dataset dir   : ${DATASET_DIR:-${DATASET_DIRS:-<from config>}}"
echo "Input npz     : ${INPUT_NPZ:-<lerobot episode>}"
echo "Episode index : ${EPISODE_INDEX}"
echo "Start step    : ${START_STEP}"
echo "Max steps     : ${MAX_STEPS:-<all>}"
echo "Replan steps  : ${REPLAN_STEPS}"
echo "Task prompt   : ${TASK_PROMPT}"
echo "Device        : ${DEVICE}"
echo "Infer steps   : ${NUM_INFERENCE_STEPS}"
echo "VAE mode      : ${VAE_DEVICE_MODE}"
echo "Model base    : ${DIFFSYNTH_MODEL_BASE_PATH}"
echo "Redirect      : ${WAN22_REDIRECT_COMMON_FILES}"
echo "Text cache    : ${TEXT_CACHE_DIR:-<from config>}"
echo "Output npz    : ${OUTPUT_NPZ}"
echo "------------------------------------------------"

python experiments/astribot/offline_episode_eval.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset_stats "${DATASET_STATS}" \
  --task "${TASK_PROMPT}" \
  --start_step "${START_STEP}" \
  --replan_steps "${REPLAN_STEPS}" \
  --output_npz "${OUTPUT_NPZ}" \
  --device "${DEVICE}" \
  --action_mode "${ACTION_MODE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --vae_device_mode "${VAE_DEVICE_MODE}" \
  --model_base_path "${DIFFSYNTH_MODEL_BASE_PATH}" \
  "${EXTRA_ARGS[@]}"
