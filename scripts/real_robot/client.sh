#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/config.sh"

if [[ ! -f "${ASTRIBOT_SDK_ENV}" ]]; then
  echo "ERROR: Astribot SDK env not found: ${ASTRIBOT_SDK_ENV}" >&2
  echo "Edit scripts/real_robot/config.sh before running." >&2
  exit 1
fi
source "${ASTRIBOT_SDK_ENV}"

echo "================ Astribot Fast-WAM Client =================="
echo "Server        : ${SERVER_HOST}:${SERVER_PORT}"
echo "Execute steps : ${EXECUTE_STEPS}"
echo "Control Hz    : ${CONTROL_HZ}"
echo "Action dt     : ${ACTION_DT}"
echo "Blend steps   : ${BOUNDARY_BLEND_STEPS}"
echo "Max runtime   : ${MAX_RUNTIME}s"
echo "============================================================"

exec python experiments/astribot/real_robot_client.py \
  --server_host "${SERVER_HOST}" \
  --server_port "${SERVER_PORT}" \
  --execute_steps "${EXECUTE_STEPS}" \
  --control_hz "${CONTROL_HZ}" \
  --action_dt "${ACTION_DT}" \
  --boundary_blend_steps "${BOUNDARY_BLEND_STEPS}" \
  --max_runtime "${MAX_RUNTIME}"
