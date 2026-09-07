#!/usr/bin/env bash
set -euo pipefail

# Real-robot client launcher for WAM inference.
# It calls the existing pi0_astribot/evaluate/infer_client_vel.py without
# adding new files to pi0_astribot.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WAM_ROOT}"

source scripts/load_wam_local_paths.sh

if [[ -z "${PI0_ROOT:-}" ]]; then
  for candidate in "${WAM_ROOT}/../pi0_astribot" "${WAM_ROOT}/../../git/pi0_astribot"; do
    if [[ -d "${candidate}" ]]; then
      PI0_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ ! -d "${PI0_ROOT:-}" ]]; then
  echo "ERROR: PI0_ROOT not found. Pass PI0_ROOT=/path/to/pi0_astribot when launching." >&2
  exit 1
fi

if [[ -n "${ASTRIBOT_SDK_ENV:-}" ]]; then
  source "${ASTRIBOT_SDK_ENV}"
fi

SERVER_HOST="${SERVER_HOST:-${VLA_SERVER_HOST:-127.0.0.1}}"
SERVER_PORT="${SERVER_PORT:-${VLA_SERVER_PORT:-2222}}"

has_arg() {
  local needle="$1"
  shift
  for arg in "$@"; do
    [[ "${arg}" == "${needle}" ]] && return 0
  done
  return 1
}

CLIENT_ARGS=(
  --server_host "${SERVER_HOST}"
  --server_port "${SERVER_PORT}"
)

# Conservative first-run defaults. Override via env vars or explicit CLI args.
if ! has_arg "--execute_steps" "$@"; then
  EXECUTE_STEPS="${EXECUTE_STEPS:-1}"
  if [[ -n "${EXECUTE_STEPS}" ]]; then
    CLIENT_ARGS+=(--execute_steps "${EXECUTE_STEPS}")
  fi
fi
if ! has_arg "--lock_chassis_yaw" "$@"; then
  LOCK_CHASSIS_YAW="${LOCK_CHASSIS_YAW:-current}"
  if [[ -n "${LOCK_CHASSIS_YAW}" ]]; then
    CLIENT_ARGS+=(--lock_chassis_yaw "${LOCK_CHASSIS_YAW}")
  fi
fi

echo "------------------------------------------------"
echo "Starting Astribot WAM client"
echo "PI0_ROOT       : ${PI0_ROOT}"
echo "WAM server     : ${SERVER_HOST}:${SERVER_PORT}"
echo "Extra defaults : execute_steps=${EXECUTE_STEPS:-<not set>} lock_chassis_yaw=${LOCK_CHASSIS_YAW:-<not set>}"
echo "User args      : $*"
echo "------------------------------------------------"

cd "${PI0_ROOT}"
exec python evaluate/infer_client_vel.py "${CLIENT_ARGS[@]}" "$@"
