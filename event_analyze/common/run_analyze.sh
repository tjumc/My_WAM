#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash common/run_analyze.sh HDF5 EPISODE_DIR [TOP_K] [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TOP_K="${3:-24}"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$HDF5" --output "$EPISODE_DIR" --top-k "$TOP_K")

if [[ -n "$LEROBOT_EPISODE" ]]; then
  ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
  ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
fi

python "$SCRIPT_DIR/analyze_episode.py" "${ARGS[@]}"
