#!/usr/bin/env bash

WAM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WAM_REPO_ROOT

LOCAL_PATHS_FILE="${WAM_REPO_ROOT}/wam_local_paths.sh"
if [[ -f "${LOCAL_PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_PATHS_FILE}"
fi

# Dataset selection is only needed by training/data scripts. Real-robot runtime
# can load model/config paths without defining WAM_TASK_NAME.
if ! declare -p WAM_DATASET_DIRS >/dev/null 2>&1 || [[ "${#WAM_DATASET_DIRS[@]}" -eq 0 ]]; then
  if [[ -n "${WAM_TASK_NAME:-}" ]]; then
    WAM_DATASET_CATALOG="${WAM_DATASET_CATALOG:-${WAM_REPO_ROOT}/configs/data/astribot_dataset_catalog.sh}"
    if [[ ! -f "${WAM_DATASET_CATALOG}" ]]; then
      echo "ERROR: dataset catalog not found: ${WAM_DATASET_CATALOG}" >&2
      return 1 2>/dev/null || exit 1
    fi
    # shellcheck source=/dev/null
    source "${WAM_DATASET_CATALOG}"
    astribot_select_dataset_task "${WAM_TASK_NAME}" || {
      return 1 2>/dev/null || exit 1
    }
  fi
fi

if declare -p WAM_DATASET_DIRS >/dev/null 2>&1; then
  printf -v WAM_DATASET_DIRS_ENV '%s\n' "${WAM_DATASET_DIRS[@]}"
  export WAM_DATASET_DIRS_ENV
fi
export WAM_DATASET_CATALOG
