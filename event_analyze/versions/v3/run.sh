#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash versions/v3/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$EPISODE_DIR/versions/v3"
mkdir -p "$OUT"

python "$SCRIPT_DIR/observe_entities.py" \
  "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5" --task "$TASK"

python "$SCRIPT_DIR/infer_state_machine.py" \
  "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5" --task "$TASK"

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE"); fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME"); fi
PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}" \
  python "$SCRIPT_DIR/refine_boundaries.py" "${REFINE_ARGS[@]}"

echo "V3 complete: $OUT"
echo "Inspect: entity_observations.jsonl, skill_candidates.json, hierarchical_annotations.json, boundaries.png"
