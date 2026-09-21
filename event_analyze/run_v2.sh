#!/usr/bin/env bash
set -euo pipefail

export AIGC_API_KEY="msk-57d214329a4b63bebb0617ce7dadbddb4d00c382f61be32a015d73c4e6c8d121"
export AIGC_USER="suty11"
export AIGC_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"
# Usage:
#   bash run_v2.sh HDF5 ANALYSIS_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="/efs/share/1919650160032350208/efs_backup/nas-backup/compressed_data/astribot/dishwasher_2_fx_20260529_compressed/dishwasher_2_fx_20260529_episode_27.hdf5"
ANALYSIS_DIR="output/dishwasher_2_fx_20260529_episode_27_analysis"
TASK_PROMPT="put the dish into the dishwasher"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

python qwen_verify_events_v2.py \
  "$ANALYSIS_DIR" \
  --hdf5 "$HDF5" \
  --task "$TASK_PROMPT"

python qwen_compose_phases_v2.py \
  "$ANALYSIS_DIR" \
  --task "$TASK_PROMPT" \
  --min-confidence 0.60 \
  --duplicate-sec 2.0

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
python refine_boundaries_v2.py "${REFINE_ARGS[@]}"

echo "V2 complete."
echo "Inspect:"
echo "  $ANALYSIS_DIR/event_semantics_v2.jsonl"
echo "  $ANALYSIS_DIR/atomic_events_v2.json"
echo "  $ANALYSIS_DIR/semantic_hierarchy_v2.json"
echo "  $ANALYSIS_DIR/hierarchical_annotations_v2.json"
echo "  $ANALYSIS_DIR/boundaries_v2.png"
