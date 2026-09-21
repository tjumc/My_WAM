#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash versions/v2_2/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$EPISODE_DIR/versions/v2_2"
mkdir -p "$OUT"

python "$SCRIPT_DIR/qwen_verify_windows.py"   "$EPISODE_DIR"   --output-dir "$OUT"   --hdf5 "$HDF5"   --task "$TASK"

python "$SCRIPT_DIR/qwen_cluster_actions.py"   "$EPISODE_DIR"   --output-dir "$OUT"   --task "$TASK"   --min-confidence 0.55   --max-gap-sec 1.0

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then
  REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
  REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
fi

PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"   python "$SCRIPT_DIR/refine_boundaries.py" "${REFINE_ARGS[@]}"

echo "V2.2 complete: $OUT"
echo "Inspect:"
echo "  $OUT/window_semantics.jsonl"
echo "  $OUT/window_semantics_consistent.jsonl"
echo "  $OUT/action_intervals.json"
echo "  $OUT/semantic_hierarchy.json"
echo "  $OUT/hierarchical_annotations.json"
echo "  $OUT/boundaries.png"
