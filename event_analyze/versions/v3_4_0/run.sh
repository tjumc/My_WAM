#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash versions/v3_4_0/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME] [SCHEMA]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCHEMA="${6:-$EVENT_ROOT/ontologies/dishwasher_loading.json}"
OUT="$EPISODE_DIR/versions/v3_4_0"
PREV="$EPISODE_DIR/versions/v3_3_3"
PROPOSAL_DIR="$OUT/proposal"
mkdir -p "$OUT"

echo "V3.4.0 task schema: $SCHEMA"

# Keep perception fixed when V3.3.3 artifacts exist.
if [[ -d "$PREV/proposal" && "${V340_REPROPOSE:-0}" != "1" ]]; then
  rm -rf "$PROPOSAL_DIR"
  cp -a "$PREV/proposal" "$PROPOSAL_DIR"
  echo "Reused V3.3.3 proposal artifacts."
else
  mkdir -p "$PROPOSAL_DIR"
  PROPOSAL_ARGS=("$HDF5" --output "$PROPOSAL_DIR")
  if [[ -n "$LEROBOT_EPISODE" ]]; then
    PROPOSAL_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
  fi
  if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
    PROPOSAL_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
  fi
  python "$SCRIPT_DIR/../v3_3_3/analyze_episode.py" "${PROPOSAL_ARGS[@]}"
fi

if [[ -f "$PREV/entity_observations_base.jsonl" && "${V340_REOBSERVE:-0}" != "1" ]]; then
  cp "$PREV/entity_observations_base.jsonl" "$OUT/entity_observations_base.jsonl"
  cp "$PREV/entity_observations_base.jsonl" "$OUT/entity_observations.jsonl"
  echo "Reused V3.3.3 generic Qwen observations."
else
  python "$SCRIPT_DIR/../v3_3_3/observe_entities.py"     "$EPISODE_DIR"     --proposal-dir "$PROPOSAL_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
  cp "$OUT/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
fi

if [[ -f "$PREV/dense_cutlery_transition_evidence.jsonl" && "${V340_REDENSE:-0}" != "1" ]]; then
  cp "$PREV/dense_cutlery_transition_evidence.jsonl" "$OUT/dense_cutlery_transition_evidence.jsonl"
  if [[ -f "$PREV/dense_cutlery_observations.jsonl" ]]; then
    cp "$PREV/dense_cutlery_observations.jsonl" "$OUT/dense_cutlery_observations.jsonl"
  fi
  echo "Reused V3.3.3 dense observations."
else
  python "$SCRIPT_DIR/../v3_3_3/dense_targeted_reobserve.py"     "$EPISODE_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
fi

python "$SCRIPT_DIR/track_entity_states.py"   "$EPISODE_DIR" --output-dir "$OUT" --schema "$SCHEMA"

python "$SCRIPT_DIR/infer_state_machine.py"   "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5"   --task "$TASK" --schema "$SCHEMA"

python "$SCRIPT_DIR/validate_interactions.py"   "$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5" --schema "$SCHEMA"

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then
  REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
  REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
fi
PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"   python "$SCRIPT_DIR/refine_boundaries.py" "${REFINE_ARGS[@]}"

if [[ -n "${V340_GT:-}" ]]; then
  python "$EVENT_ROOT/evaluation/evaluate_temporal_annotations.py"     "$OUT/hierarchical_annotations.json" "$V340_GT"     --output "$OUT/evaluation_temporal.json"
fi

echo "V3.4.0 complete: $OUT"
echo "Reasoning is schema-driven; perception remains V3.3.3-compatible in this stage."
