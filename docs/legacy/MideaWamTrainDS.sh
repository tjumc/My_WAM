#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Volcano normally provides VC_TASK_INDEX and /etc/volcano/worker.host.
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-$(awk 'NR == 1 {print $1; exit}' /etc/volcano/worker.host)}"
export MASTER_PORT="${MASTER_PORT:-29604}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"

LAUNCHER="${WAM_TRAIN_LAUNCHER:-scripts/train_zero1ds.sh}"
if [[ ! -x "${LAUNCHER}" && ! -f "${LAUNCHER}" ]]; then
  echo "Missing training launcher: ${LAUNCHER}" >&2
  exit 1
fi

echo "node ${NODE_RANK}/${NNODES}, gpus=${GPUS_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"

if [[ -f /root/YES/etc/profile.d/conda.sh ]]; then
  source /root/YES/etc/profile.d/conda.sh
  conda activate "${WAM_CONDA_ENV:-./envs/fastwam}"
fi

exec bash "${LAUNCHER}" "${GPUS_PER_NODE}" \
  task="${WAM_TASK:-pick_place_1e-4}" \
  resume="${WAM_RESUME:-./checkpoints/step_055000.pt}" \
  "$@"
