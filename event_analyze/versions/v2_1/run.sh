#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash versions/v2_1/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION_OUT="$EPISODE_DIR/versions/v2_1"
mkdir -p "$VERSION_OUT"

# V2.1 scripts currently consume the shared proposal files at EPISODE_DIR.
# They write temporary version-specific files there; this wrapper archives them
# into VERSION_OUT after all stages finish.
python "$SCRIPT_DIR/qwen_verify_windows_v2_1.py"   "$EPISODE_DIR"   --hdf5 "$HDF5"   --task "$TASK"

python "$SCRIPT_DIR/qwen_cluster_actions_v2_1.py"   "$EPISODE_DIR"   --task "$TASK"   --min-confidence 0.55   --max-gap-sec 1.0

REFINE_ARGS=("$EPISODE_DIR" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then
  REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
  REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
fi
PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"   python "$SCRIPT_DIR/refine_boundaries_v2_1.py" "${REFINE_ARGS[@]}"

# Archive V2.1 outputs so the episode root stays clean.
for f in   window_semantics_v2_1.jsonl   action_intervals_v2_1.json   semantic_hierarchy_v2_1.json   hierarchical_annotations_v2_1.json   boundaries_v2_1.png
do
  if [[ -e "$EPISODE_DIR/$f" ]]; then
    mv -f "$EPISODE_DIR/$f" "$VERSION_OUT/$f"
  fi
done

if [[ -d "$EPISODE_DIR/vlm_raw_v2_1" ]]; then
  rm -rf "$VERSION_OUT/vlm_raw_v2_1"
  mv "$EPISODE_DIR/vlm_raw_v2_1" "$VERSION_OUT/"
fi

echo "V2.1 complete: $VERSION_OUT"
