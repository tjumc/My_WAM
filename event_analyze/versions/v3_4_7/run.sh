#!/usr/bin/env bash
set -euo pipefail

# Usage:
# GT=... KEEP_DEBUG=1 bash versions/v3_4_7/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME] [SCHEMA]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCHEMA="${6:-$EVENT_ROOT/ontologies/dishwasher_loading.json}"
OUT="$EPISODE_DIR/versions/v3_4_7"
SOURCE="$EPISODE_DIR/versions/v3_4_6"
FALLBACK="$EPISODE_DIR/versions/v3_4_5"
mkdir -p "$OUT"

echo "V3.4.7 task schema: $SCHEMA"

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

# Keep perception/proposals controlled relative to V3.4.6 whenever possible.
PROPOSAL_SRC=""
if PROPOSAL_SRC="$(source_artifact "$SOURCE" proposal 2>/dev/null)" && [[ "${V347_REPROPOSE:-0}" != "1" ]]; then
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
if BASE_SRC="$(source_artifact "$SOURCE" entity_observations_base.jsonl 2>/dev/null)" && [[ "${V347_REOBSERVE:-0}" != "1" ]]; then
  cp "$BASE_SRC" "$OUT/entity_observations_base.jsonl"
  cp "$BASE_SRC" "$OUT/entity_observations.jsonl"
  echo "Reused base Qwen observations from $BASE_SRC."
else
  python "$SCRIPT_DIR/../v3_3_3/observe_entities.py" \
    "$EPISODE_DIR" --proposal-dir "$OUT/proposal" --output-dir "$OUT" \
    --hdf5 "$HDF5" --task "$TASK"
  cp "$OUT/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
fi

# Keep legacy task-specific dense perception disabled unless explicitly ablated.
rm -f "$OUT/dense_cutlery_transition_evidence.jsonl" "$OUT/dense_cutlery_observations.jsonl"
if [[ "${V347_USE_LEGACY_DENSE:-0}" == "1" ]]; then
  for name in dense_cutlery_transition_evidence.jsonl dense_cutlery_observations.jsonl; do
    SRC=""
    if SRC="$(source_artifact "$SOURCE" "$name" 2>/dev/null)"; then
      cp "$SRC" "$OUT/$name"
    fi
  done
  echo "Legacy cutlery-specific dense evidence enabled for ablation."
fi

# V3.4.7 deliberately inherits V3.4.6 closed-loop reasoning and V3.4.5
# perception/tracking/validation. The controlled experiment isolates only the
# new schema-driven latent state implication in the final decoder.
reasoning_pass () {
  python "$SCRIPT_DIR/../v3_4_5/track_entity_states.py" \
    "$EPISODE_DIR" --output-dir "$OUT" --schema "$SCHEMA"

  python "$SCRIPT_DIR/../v3_4_5/resolve_hand_object_ownership.py" \
    "$EPISODE_DIR" --output-dir "$OUT" --schema "$SCHEMA"

  python "$SCRIPT_DIR/../v3_4_5/infer_state_machine.py" \
    "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5" \
    --task "$TASK" --schema "$SCHEMA"

  python "$SCRIPT_DIR/../v3_4_5/validate_interactions.py" \
    "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5" --schema "$SCHEMA"
}

echo "=== V3.4.7 reasoning pass 1 ==="
reasoning_pass

python "$SCRIPT_DIR/../v3_4_6/conservative_reasoning.py" \
  --output-dir "$OUT" --mode capture

python "$SCRIPT_DIR/../v3_4_5/detect_ambiguities.py" \
  --output-dir "$OUT" --schema "$SCHEMA" --hdf5 "$HDF5"

# Reuse V3.4.6 targeted observations only when every request signature matches.
# Otherwise issue fresh targeted calls. This preserves controlled comparisons.
if [[ "${V347_RETARGET:-0}" != "1" ]]; then
  rm -f "$OUT/targeted_observations.jsonl"
  if python "$SCRIPT_DIR/../v3_4_6/reuse_targeted.py" --output-dir "$OUT" --source-dir "$SOURCE"; then
    echo "Exact V3.4.6 targeted evidence reused."
  else
    rm -f "$OUT/targeted_observations.jsonl"
  fi
fi

TARGET_ARGS=(
  --output-dir "$OUT"
  --schema "$SCHEMA"
  --hdf5 "$HDF5"
  --task "$TASK"
)
if [[ -f "$OUT/targeted_observations.jsonl" && "${V347_RETARGET:-0}" != "1" ]]; then
  TARGET_ARGS+=(--reuse)
fi
python "$SCRIPT_DIR/../v3_4_5/targeted_reobserve.py" "${TARGET_ARGS[@]}"

echo "=== V3.4.7 reasoning pass 2 ==="
reasoning_pass

python "$SCRIPT_DIR/../v3_4_6/conservative_reasoning.py" \
  --output-dir "$OUT" --mode reconcile --schema "$SCHEMA"

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE"); fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME"); fi
PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}" \
  python "$SCRIPT_DIR/../v3_4_5/refine_boundaries.py" "${REFINE_ARGS[@]}"

python "$SCRIPT_DIR/finalize_annotations.py" \
  --output-dir "$OUT" --schema "$SCHEMA" --hdf5 "$HDF5"

python "$SCRIPT_DIR/reasoning_trace.py" --output-dir "$OUT"

if [[ -n "${GT:-}" ]]; then
  python "$EVENT_ROOT/evaluation/evaluate_temporal_annotations.py" \
    "$OUT/hierarchical_annotations.json" "$GT" \
    --schema "$SCHEMA" --output "$OUT/evaluation_temporal.json"
fi

python "$EVENT_ROOT/common/export_results.py" \
  --episode-dir "$EPISODE_DIR" --output-dir "$OUT" \
  --version "v3_4_7" --task "$TASK" --schema "$SCHEMA" \
  --hdf5 "$HDF5" --event-root "$EVENT_ROOT"

COMPACT_ARGS=(--output-dir "$OUT")
if [[ "${KEEP_DEBUG:-0}" == "1" ]]; then
  COMPACT_ARGS+=(--keep-debug)
fi
python "$SCRIPT_DIR/compact_runtime.py" "${COMPACT_ARGS[@]}"

echo "V3.4.7 complete: $OUT"
echo "Final annotation: $OUT/hierarchical_annotations.json"
echo "Diagnostics: $OUT/diagnostics/"
echo "Reusable cache: $OUT/cache/"
echo "Compact result exported under: $EVENT_ROOT/results/"
