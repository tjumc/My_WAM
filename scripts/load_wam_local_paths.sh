#!/usr/bin/env bash

# Shared loader for machine-specific paths. The loaded file is intentionally
# excluded from Git so each deployment can keep its own storage layout.
WAM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAM_LOCAL_PATHS_FILE="${WAM_LOCAL_PATHS_FILE:-${WAM_REPO_ROOT}/wam_local_paths.sh}"

if [[ ! -f "${WAM_LOCAL_PATHS_FILE}" ]]; then
  echo "ERROR: WAM local path config not found: ${WAM_LOCAL_PATHS_FILE}" >&2
  echo "Create it from ${WAM_REPO_ROOT}/wam_local_paths.example.sh and fill in local paths." >&2
  return 1 2>/dev/null || exit 1
fi

export WAM_REPO_ROOT WAM_LOCAL_PATHS_FILE
# shellcheck source=/dev/null
source "${WAM_LOCAL_PATHS_FILE}"

if ! declare -p WAM_DATASET_DIRS >/dev/null 2>&1 || [[ "${#WAM_DATASET_DIRS[@]}" -eq 0 ]]; then
  WAM_DATASET_CATALOG="${WAM_DATASET_CATALOG:-${WAM_REPO_ROOT}/configs/data/astribot_dataset_catalog.sh}"
  if [[ ! -f "${WAM_DATASET_CATALOG}" ]]; then
    echo "ERROR: dataset catalog not found: ${WAM_DATASET_CATALOG}" >&2
    return 1 2>/dev/null || exit 1
  fi

  # Resolve the selected task only when local paths do not explicitly override it.
  # shellcheck source=/dev/null
  source "${WAM_DATASET_CATALOG}"
  astribot_select_dataset_task "${WAM_TASK_NAME:?Set WAM_TASK_NAME in wam_local_paths.sh}" || {
    return 1 2>/dev/null || exit 1
  }
fi

printf -v WAM_DATASET_DIRS_ENV '%s\n' "${WAM_DATASET_DIRS[@]}"
export WAM_DATASET_DIRS_ENV WAM_DATASET_CATALOG

export WAM_TASK_NAME WAM_TASK_INSTRUCTION WAM_PRETRAIN_CKPT
export DIFFSYNTH_MODEL_BASE_PATH ACTION_DIT_PRETRAINED_PATH
