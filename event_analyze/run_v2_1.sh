#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_v2_1.sh HDF5 ANALYSIS_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
ANALYSIS_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

python qwen_verify_windows_v2_1.py \
  "$ANALYSIS_DIR" \
  --hdf5 "$HDF5" \
  --task "$TASK"

python qwen_cluster_actions_v2_1.py \
  "$ANALYSIS_DIR" \
  --task "$TASK" \
  --min-confidence 0.55 \
  --max-gap-sec 1.0

REFINE_ARGS=(
  "$ANALYSIS_DIR"
  --hdf5 "$HDF5"
)
if [[ -n "$LEROBOT_EPISODE" ]]; then
  REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
  REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
fi
python refine_boundaries_v2_1.py "${REFINE_ARGS[@]}"

echo "V2.1 complete."
echo "Inspect:"
echo "  $ANALYSIS_DIR/window_semantics_v2_1.jsonl"
echo "  $ANALYSIS_DIR/action_intervals_v2_1.json"
echo "  $ANALYSIS_DIR/semantic_hierarchy_v2_1.json"
echo "  $ANALYSIS_DIR/hierarchical_annotations_v2_1.json"
echo "  $ANALYSIS_DIR/boundaries_v2_1.png"
