#!/usr/bin/env bash

# Shared loader for machine-specific paths. The loaded file is intentionally
# excluded from Git so each deployment can keep its own storage layout.
WAM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WAM_REPO_ROOT
# shellcheck source=/dev/null

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
