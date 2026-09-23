#!/usr/bin/env bash
set -euo pipefail

# Usage:
# GT=... KEEP_DEBUG=1 bash versions/v3_4_5/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME] [SCHEMA]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCHEMA="${6:-$EVENT_ROOT/ontologies/dishwasher_loading.json}"
OUT="$EPISODE_DIR/versions/v3_4_5"
PREV="$EPISODE_DIR/versions/v3_4_4"
FALLBACK="$EPISODE_DIR/versions/v3_4_3"
mkdir -p "$OUT"

echo "V3.4.5 task schema: $SCHEMA"

SOURCE="$PREV"
if [[ ! -d "$SOURCE" ]]; then
  SOURCE="$FALLBACK"
fi

source_artifact () {
  local root="$1"
  local rel="$2"
  if [[ -e "$root/$rel" ]]; then
    printf '%s\n' "$root/$rel"
    return 0
  fi
  if [[ -e "$root/cache/$rel" ]]; then
    printf '%s\n' "$root/cache/$rel"
    return 0
  fi
  return 1
}

PROPOSAL_SRC=""
if PROPOSAL_SRC="$(source_artifact "$SOURCE" proposal 2>/dev/null)" && [[ "${V345_REPROPOSE:-0}" != "1" ]]; then
  rm -rf "$OUT/proposal"
  cp -a "$PROPOSAL_SRC" "$OUT/proposal"
  echo "Reused proposal artifacts from $PROPOSAL_SRC."
else
  mkdir -p "$OUT/proposal"
  PROPOSAL_ARGS=("$HDF5" --output "$OUT/proposal")
  if [[ -n "$LEROBOT_EPISODE" ]]; then PROPOSAL_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE"); fi
  if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then PROPOSAL_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME"); fi
  python "$SCRIPT_DIR/../v3_3_3/analyze_episode.py" "${PROPOSAL_ARGS[@]}"
fi

BASE_SRC=""
if BASE_SRC="$(source_artifact "$SOURCE" entity_observations_base.jsonl 2>/dev/null)" && [[ "${V345_REOBSERVE:-0}" != "1" ]]; then
  cp "$BASE_SRC" "$OUT/entity_observations_base.jsonl"
  cp "$BASE_SRC" "$OUT/entity_observations.jsonl"
  echo "Reused base Qwen observations from $BASE_SRC."
else
  python "$SCRIPT_DIR/../v3_3_3/observe_entities.py"     "$EPISODE_DIR" --proposal-dir "$OUT/proposal" --output-dir "$OUT"     --hdf5 "$HDF5" --task "$TASK"
  cp "$OUT/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
fi

# V3.4.5 keeps the legacy task-specific dense pass disabled by default.
rm -f "$OUT/dense_cutlery_transition_evidence.jsonl" "$OUT/dense_cutlery_observations.jsonl"
if [[ "${V345_USE_LEGACY_DENSE:-0}" == "1" ]]; then
  for name in dense_cutlery_transition_evidence.jsonl dense_cutlery_observations.jsonl; do
    SRC=""
    if SRC="$(source_artifact "$SOURCE" "$name" 2>/dev/null)"; then
      cp "$SRC" "$OUT/$name"
    fi
  done
  echo "Legacy cutlery-specific dense evidence enabled for ablation."
fi

reasoning_pass () {
  python "$SCRIPT_DIR/track_entity_states.py"     "$EPISODE_DIR" --output-dir "$OUT" --schema "$SCHEMA"

  python "$SCRIPT_DIR/resolve_hand_object_ownership.py"     "$EPISODE_DIR" --output-dir "$OUT" --schema "$SCHEMA"

  python "$SCRIPT_DIR/infer_state_machine.py"     "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5"     --task "$TASK" --schema "$SCHEMA"

  python "$SCRIPT_DIR/validate_interactions.py"     "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5" --schema "$SCHEMA"
}

echo "=== V3.4.5 reasoning pass 1 ==="
reasoning_pass

python "$SCRIPT_DIR/reasoning_trace.py"   --output-dir "$OUT" --mode capture

python "$SCRIPT_DIR/detect_ambiguities.py"   --output-dir "$OUT" --schema "$SCHEMA" --hdf5 "$HDF5"

# A completed compacted V3.4.5 run may already have targeted observations in
# cache. Restore them before --reuse.
if [[ ! -f "$OUT/targeted_observations.jsonl" && -f "$OUT/cache/targeted_observations.jsonl" && "${V345_RETARGET:-0}" != "1" ]]; then
  cp "$OUT/cache/targeted_observations.jsonl" "$OUT/targeted_observations.jsonl"
fi

TARGET_ARGS=(
  --output-dir "$OUT"
  --schema "$SCHEMA"
  --hdf5 "$HDF5"
  --task "$TASK"
)
if [[ -f "$OUT/targeted_observations.jsonl" && "${V345_RETARGET:-0}" != "1" ]]; then
  TARGET_ARGS+=(--reuse)
fi
python "$SCRIPT_DIR/targeted_reobserve.py" "${TARGET_ARGS[@]}"

echo "=== V3.4.5 reasoning pass 2 ==="
reasoning_pass

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE"); fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME"); fi
PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"   python "$SCRIPT_DIR/refine_boundaries.py" "${REFINE_ARGS[@]}"

python "$SCRIPT_DIR/finalize_annotations.py"   --output-dir "$OUT" --schema "$SCHEMA" --hdf5 "$HDF5"

python "$SCRIPT_DIR/reasoning_trace.py"   --output-dir "$OUT" --mode finalize

if [[ -n "${GT:-}" ]]; then
  python "$EVENT_ROOT/evaluation/evaluate_temporal_annotations.py"     "$OUT/hierarchical_annotations.json" "$GT"     --schema "$SCHEMA"     --output "$OUT/evaluation_temporal.json"
fi

python "$EVENT_ROOT/common/export_results.py"   --episode-dir "$EPISODE_DIR" --output-dir "$OUT"   --version "v3_4_5" --task "$TASK" --schema "$SCHEMA"   --hdf5 "$HDF5" --event-root "$EVENT_ROOT"

COMPACT_ARGS=(--output-dir "$OUT")
if [[ "${KEEP_DEBUG:-0}" == "1" ]]; then
  COMPACT_ARGS+=(--keep-debug)
fi
python "$SCRIPT_DIR/compact_runtime.py" "${COMPACT_ARGS[@]}"

echo "V3.4.5 complete: $OUT"
echo "Final annotation: $OUT/hierarchical_annotations.json"
echo "Diagnostics: $OUT/diagnostics/"
echo "Reusable cache: $OUT/cache/"
echo "Compact result exported under: $EVENT_ROOT/results/"
