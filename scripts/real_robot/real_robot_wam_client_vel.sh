#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WAM_ROOT}"

if [[ -f "${WAM_ROOT}/wam_local_paths.sh" ]]; then
  # shellcheck source=/dev/null
  source "${WAM_ROOT}/wam_local_paths.sh"
fi

if [[ -n "${ASTRIBOT_SDK_ENV:-}" ]]; then
  # shellcheck source=/dev/null
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

CLIENT_ARGS=(--server_host "${SERVER_HOST}" --server_port "${SERVER_PORT}")

if ! has_arg "--execute_steps" "$@"; then
  CLIENT_ARGS+=(--execute_steps "${EXECUTE_STEPS:-1}")
fi
if ! has_arg "--control_hz" "$@"; then
  CLIENT_ARGS+=(--control_hz "${CONTROL_HZ:-50}")
fi
if ! has_arg "--action_dt" "$@"; then
  CLIENT_ARGS+=(--action_dt "${ACTION_DT:-0.03333333333333333}")
fi
if ! has_arg "--boundary_blend_steps" "$@"; then
  CLIENT_ARGS+=(--boundary_blend_steps "${BOUNDARY_BLEND_STEPS:-0}")
fi

echo "------------------------------------------------"
echo "Starting Astribot Fast-WAM client"
echo "WAM server     : ${SERVER_HOST}:${SERVER_PORT}"
echo "Execute steps  : ${EXECUTE_STEPS:-1}"
echo "Control Hz     : ${CONTROL_HZ:-50}"
echo "Action dt      : ${ACTION_DT:-0.03333333333333333}"
echo "User args      : $*"
echo "------------------------------------------------"

exec python experiments/astribot/real_robot_client.py "${CLIENT_ARGS[@]}" "$@"
