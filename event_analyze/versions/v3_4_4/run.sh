#!/usr/bin/env bash
set -euo pipefail

# Usage:
# GT=... bash versions/v3_4_4/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME] [SCHEMA]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCHEMA="${6:-$EVENT_ROOT/ontologies/dishwasher_loading.json}"
OUT="$EPISODE_DIR/versions/v3_4_4"
PREV="$EPISODE_DIR/versions/v3_4_3"
FALLBACK="$EPISODE_DIR/versions/v3_4_2"
PROPOSAL_DIR="$OUT/proposal"
mkdir -p "$OUT"

echo "V3.4.4 task schema: $SCHEMA"

SOURCE="$PREV"
if [[ ! -d "$SOURCE" ]]; then
  SOURCE="$FALLBACK"
fi

if [[ -d "$SOURCE/proposal" && "${V344_REPROPOSE:-0}" != "1" ]]; then
  rm -rf "$PROPOSAL_DIR"
  cp -a "$SOURCE/proposal" "$PROPOSAL_DIR"
  echo "Reused proposal artifacts from $(basename "$SOURCE")."
else
  mkdir -p "$PROPOSAL_DIR"
  PROPOSAL_ARGS=("$HDF5" --output "$PROPOSAL_DIR")
  if [[ -n "$LEROBOT_EPISODE" ]]; then PROPOSAL_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE"); fi
  if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then PROPOSAL_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME"); fi
  python "$SCRIPT_DIR/../v3_3_3/analyze_episode.py" "${PROPOSAL_ARGS[@]}"
fi

# Broad perception remains V3.3.3-compatible for now. Targeted perception below
# is schema-driven and reasoning-triggered.
if [[ -f "$SOURCE/entity_observations_base.jsonl" && "${V344_REOBSERVE:-0}" != "1" ]]; then
  cp "$SOURCE/entity_observations_base.jsonl" "$OUT/entity_observations_base.jsonl"
  cp "$SOURCE/entity_observations_base.jsonl" "$OUT/entity_observations.jsonl"
  echo "Reused base Qwen observations from $(basename "$SOURCE")."
else
  python "$SCRIPT_DIR/../v3_3_3/observe_entities.py"     "$EPISODE_DIR" --proposal-dir "$PROPOSAL_DIR" --output-dir "$OUT"     --hdf5 "$HDF5" --task "$TASK"
  cp "$OUT/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
fi

# Default V3.4.4 path does NOT use the old cutlery-specific dense pass.
rm -f "$OUT/dense_cutlery_transition_evidence.jsonl" "$OUT/dense_cutlery_observations.jsonl"
if [[ "${V344_USE_LEGACY_DENSE:-0}" == "1" ]]; then
  if [[ -f "$SOURCE/dense_cutlery_transition_evidence.jsonl" ]]; then
    cp "$SOURCE/dense_cutlery_transition_evidence.jsonl" "$OUT/dense_cutlery_transition_evidence.jsonl"
  fi
  if [[ -f "$SOURCE/dense_cutlery_observations.jsonl" ]]; then
    cp "$SOURCE/dense_cutlery_observations.jsonl" "$OUT/dense_cutlery_observations.jsonl"
  fi
  echo "Legacy cutlery-specific dense evidence enabled for ablation."
fi

reasoning_pass () {
  python "$SCRIPT_DIR/track_entity_states.py"     "$EPISODE_DIR" --output-dir "$OUT" --schema "$SCHEMA"

  python "$SCRIPT_DIR/resolve_hand_object_ownership.py"     "$EPISODE_DIR" --output-dir "$OUT" --schema "$SCHEMA"

  python "$SCRIPT_DIR/infer_state_machine.py"     "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5"     --task "$TASK" --schema "$SCHEMA"

  python "$SCRIPT_DIR/validate_interactions.py"     "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5" --schema "$SCHEMA"
}

echo "=== V3.4.4 reasoning pass 1 ==="
reasoning_pass

cp "$OUT/tracked_entity_states.json" "$OUT/tracked_entity_states_pass1.json"
cp "$OUT/tracked_entity_states_owned.json" "$OUT/tracked_entity_states_owned_pass1.json"
cp "$OUT/hand_object_ownership.json" "$OUT/hand_object_ownership_pass1.json"
cp "$OUT/skill_candidates.json" "$OUT/skill_candidates_pass1.json"
cp "$OUT/skill_candidates_validated.json" "$OUT/skill_candidates_validated_pass1.json"
cp "$OUT/interaction_validation_report.json" "$OUT/interaction_validation_report_pass1.json"

python "$SCRIPT_DIR/detect_ambiguities.py"   --output-dir "$OUT" --schema "$SCHEMA" --hdf5 "$HDF5"

TARGET_ARGS=(
  --output-dir "$OUT"
  --schema "$SCHEMA"
  --hdf5 "$HDF5"
  --task "$TASK"
)
if [[ -f "$OUT/targeted_observations.jsonl" && "${V344_RETARGET:-0}" != "1" ]]; then
  TARGET_ARGS+=(--reuse)
fi
python "$SCRIPT_DIR/targeted_reobserve.py" "${TARGET_ARGS[@]}"

echo "=== V3.4.4 reasoning pass 2 ==="
reasoning_pass

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE"); fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME"); fi
PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"   python "$SCRIPT_DIR/refine_boundaries.py" "${REFINE_ARGS[@]}"

if [[ -n "${GT:-}" ]]; then
  python "$EVENT_ROOT/evaluation/evaluate_temporal_annotations.py"     "$OUT/hierarchical_annotations.json" "$GT"     --schema "$SCHEMA"     --output "$OUT/evaluation_temporal.json"
fi

python "$EVENT_ROOT/common/export_results.py"   --episode-dir "$EPISODE_DIR" --output-dir "$OUT"   --version "v3_4_4" --task "$TASK" --schema "$SCHEMA"   --hdf5 "$HDF5" --event-root "$EVENT_ROOT"

echo "V3.4.4 complete: $OUT"
echo "Ambiguity requests: $OUT/ambiguity_requests.json"
echo "Targeted observations: $OUT/targeted_observations.jsonl"
echo "Compact result exported under: $EVENT_ROOT/results/"
