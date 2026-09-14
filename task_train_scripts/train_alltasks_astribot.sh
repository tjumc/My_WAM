#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Multi-node configuration ----------------------------------------------
export WAM_NNODES=1
export WAM_GPUS_PER_NODE=8
export WAM_BATCH_SIZE="${WAM_BATCH_SIZE:-12}"
export WAM_GRADIENT_ACCUMULATION_STEPS="${WAM_GRADIENT_ACCUMULATION_STEPS:-1}"
export WAM_NUM_WORKERS="${WAM_NUM_WORKERS:-16}"
export WAM_NUM_EPOCHS="${WAM_NUM_EPOCHS:-5}"
export WAM_RUN_PREPARE="${WAM_RUN_PREPARE:-auto}"
export WAM_SAVE_EVERY="${WAM_SAVE_EVERY:-10000}"
export WAM_EVAL_EVERY="${WAM_EVAL_EVERY:-10000}"

# Keep task instructions explicit because all data from one high-level group
# is trained with the same text condition.
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

TASKS=(
  oven_put
  fridge
  dishwasher
  oven_takeout
  collect_clothes
  dry_clothes
  wash_clothes
)

# Optional positional arguments run only the requested tasks, for example:
#   bash task_train_scripts/train_all_astribot_2x8.sh dishwasher dry_clothes
if [[ "$#" -gt 0 ]]; then
  TASKS=("$@")
fi

LOCAL_PATHS_FILE="${WAM_LOCAL_PATHS_FILE:-${REPO_ROOT}/wam_local_paths.sh}"
if [[ ! -f "${LOCAL_PATHS_FILE}" ]]; then
  echo "ERROR: local path config not found: ${LOCAL_PATHS_FILE}" >&2
  exit 1
fi

# Read machine-specific model paths once. Dataset paths are selected from the
# catalog separately for each task below.
# shellcheck source=/dev/null
source "${LOCAL_PATHS_FILE}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH in ${LOCAL_PATHS_FILE}}"
: "${ACTION_DIT_PRETRAINED_PATH:?Set ACTION_DIT_PRETRAINED_PATH in ${LOCAL_PATHS_FILE}}"
: "${WAM_PRETRAIN_CKPT:?Set WAM_PRETRAIN_CKPT in ${LOCAL_PATHS_FILE}}"

CATALOG="${WAM_DATASET_CATALOG:-${REPO_ROOT}/configs/data/astribot_dataset_catalog.sh}"
if [[ ! -f "${CATALOG}" ]]; then
  echo "ERROR: dataset catalog not found: ${CATALOG}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CATALOG}"

for task_name in "${TASKS[@]}"; do
  if [[ ! -v "ASTRIBOT_DATASET_CATALOG[${task_name}]" ]]; then
    echo "ERROR: task '${task_name}' is not present in ${CATALOG}" >&2
    exit 1
  fi
  if [[ -z "${TASK_INSTRUCTIONS[${task_name}]:-}" ]]; then
    echo "ERROR: no instruction configured for task '${task_name}'" >&2
    exit 1
  fi
done

TEMP_CONFIG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wam-all-tasks.XXXXXX")"
trap 'rm -rf "${TEMP_CONFIG_DIR}"' EXIT

write_task_config() {
  local task_name="$1"
  local run_name="astribot_${task_name}_posttrain32"
  local config_path="${TEMP_CONFIG_DIR}/${task_name}.sh"

  # The common launcher sources this temporary file, so each iteration can
  # override the task without modifying the ignored machine-local config.
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf 'export DIFFSYNTH_MODEL_BASE_PATH=%q\n' "${DIFFSYNTH_MODEL_BASE_PATH}"
    printf 'export ACTION_DIT_PRETRAINED_PATH=%q\n' "${ACTION_DIT_PRETRAINED_PATH}"
    printf 'export WAM_PRETRAIN_CKPT=%q\n' "${WAM_PRETRAIN_CKPT}"
    printf 'export WAM_TASK_NAME=%q\n' "${task_name}"
    printf 'export WAM_TASK_INSTRUCTION=%q\n' "${TASK_INSTRUCTIONS[${task_name}]}"
    printf 'export WAM_NNODES=%q\n' "${WAM_NNODES}"
    printf 'export WAM_GPUS_PER_NODE=%q\n' "${WAM_GPUS_PER_NODE}"
    printf 'export WAM_BATCH_SIZE=%q\n' "${WAM_BATCH_SIZE}"
    printf 'export WAM_GRADIENT_ACCUMULATION_STEPS=%q\n' "${WAM_GRADIENT_ACCUMULATION_STEPS}"
    printf 'export WAM_NUM_WORKERS=%q\n' "${WAM_NUM_WORKERS}"
    printf 'export WAM_NUM_EPOCHS=%q\n' "${WAM_NUM_EPOCHS}"
    printf 'export WAM_RUN_PREPARE=%q\n' "${WAM_RUN_PREPARE}"
    printf 'export WAM_SAVE_EVERY=%q\n' "${WAM_SAVE_EVERY}"
    printf 'export WAM_EVAL_EVERY=%q\n' "${WAM_EVAL_EVERY}"
    printf 'export WAM_RUN_NAME=%q\n' "${run_name}"
    printf 'export WAM_STATS_PATH=%q\n' "runs/${run_name}/stats.json"
    printf 'export WAM_TEXT_CACHE_DIR=%q\n' "posttrain/text_embeds_cache_${run_name}"
    printf 'export WAM_OUTPUT_DIR=%q\n' "./runs/${run_name}/train"
  } > "${config_path}"
  printf '%s\n' "${config_path}"
}

echo "================================================"
echo "Astribot sequential multi-task training"
echo "Tasks          : ${TASKS[*]}"
echo "Nodes x GPUs   : ${WAM_NNODES} x ${WAM_GPUS_PER_NODE}"
echo "Batch size     : ${WAM_BATCH_SIZE}"
echo "Grad accum     : ${WAM_GRADIENT_ACCUMULATION_STEPS}"
echo "Run prepare    : ${WAM_RUN_PREPARE}"
echo "================================================"

for task_name in "${TASKS[@]}"; do
  task_config="$(write_task_config "${task_name}")"
  echo ""
  echo "================================================"
  echo "Starting task: ${task_name}"
  echo "Instruction  : ${TASK_INSTRUCTIONS[${task_name}]}"
  echo "Config       : ${task_config}"
  echo "================================================"

  NNODES="${WAM_NNODES}" \
    GPUS_PER_NODE="${WAM_GPUS_PER_NODE}" \
    WAM_LOCAL_PATHS_FILE="${task_config}" \
    bash "${REPO_ROOT}/scripts/train_astribot_posttrain32.sh"

  echo "[INFO] Task completed: ${task_name}"
done

echo "================================================"
echo "All requested tasks completed successfully."
echo "================================================"