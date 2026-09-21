#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash versions/v3_1/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$EPISODE_DIR/versions/v3_1"
mkdir -p "$OUT"

if [[ -f "$EPISODE_DIR/versions/v3/entity_observations.jsonl" && "${V31_REOBSERVE:-0}" != "1" ]]; then
  cp "$EPISODE_DIR/versions/v3/entity_observations.jsonl" "$OUT/entity_observations.jsonl"
  echo "Reused V3 entity observations for controlled V3 -> V3.1 comparison."
else
  python "$SCRIPT_DIR/observe_entities.py"     "$EPISODE_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
fi

python "$SCRIPT_DIR/track_entity_states.py"   "$EPISODE_DIR"   --output-dir "$OUT"

python "$SCRIPT_DIR/infer_state_machine.py"   "$EPISODE_DIR"   --output-dir "$OUT"   --hdf5 "$HDF5"   --task "$TASK"

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then
  REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
  REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
fi

PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"   python "$SCRIPT_DIR/refine_boundaries.py" "${REFINE_ARGS[@]}"

echo "V3.1 complete: $OUT"
echo "Inspect:"
echo "  $OUT/tracked_entity_states.json"
echo "  $OUT/skill_candidates.json"
echo "  $OUT/hierarchical_annotations.json"
echo "  $OUT/boundaries.png"
