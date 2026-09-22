#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash versions/v3_3_3/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$EPISODE_DIR/versions/v3_3_3"
PREV="$EPISODE_DIR/versions/v3_3_2"
PROPOSAL_DIR="$OUT/proposal"
mkdir -p "$OUT"

# Controlled comparison: reuse V3.3.2 proposal/Qwen observations by default.
if [[ -d "$PREV/proposal" && "${V333_REPROPOSE:-0}" != "1" ]]; then
  rm -rf "$PROPOSAL_DIR"
  cp -a "$PREV/proposal" "$PROPOSAL_DIR"
  echo "Reused V3.3.2 proposal artifacts."
else
  mkdir -p "$PROPOSAL_DIR"
  PROPOSAL_ARGS=("$HDF5" --output "$PROPOSAL_DIR")
  if [[ -n "$LEROBOT_EPISODE" ]]; then
    PROPOSAL_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
  fi
  if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
    PROPOSAL_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
  fi
  python "$SCRIPT_DIR/analyze_episode.py" "${PROPOSAL_ARGS[@]}"
fi

if [[ -f "$PREV/entity_observations_base.jsonl" && "${V333_REOBSERVE:-0}" != "1" ]]; then
  cp "$PREV/entity_observations_base.jsonl" "$OUT/entity_observations_base.jsonl"
  cp "$PREV/entity_observations_base.jsonl" "$OUT/entity_observations.jsonl"
  echo "Reused V3.3.2 generic Qwen observations."
else
  python "$SCRIPT_DIR/observe_entities.py"     "$EPISODE_DIR"     --proposal-dir "$PROPOSAL_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
  cp "$OUT/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
fi

if [[ -f "$PREV/dense_cutlery_transition_evidence.jsonl" && "${V333_REDENSE:-0}" != "1" ]]; then
  cp "$PREV/dense_cutlery_transition_evidence.jsonl" "$OUT/dense_cutlery_transition_evidence.jsonl"
  if [[ -f "$PREV/dense_cutlery_observations.jsonl" ]]; then
    cp "$PREV/dense_cutlery_observations.jsonl" "$OUT/dense_cutlery_observations.jsonl"
  fi
  echo "Reused V3.3.2 dense Qwen observations."
else
  python "$SCRIPT_DIR/dense_targeted_reobserve.py"     "$EPISODE_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
fi

python "$SCRIPT_DIR/track_entity_states.py" "$EPISODE_DIR" --output-dir "$OUT"

python "$SCRIPT_DIR/infer_state_machine.py"   "$EPISODE_DIR"   --output-dir "$OUT"   --hdf5 "$HDF5"   --task "$TASK"

python "$SCRIPT_DIR/validate_interactions.py"   "$EPISODE_DIR"   --output-dir "$OUT"   --hdf5 "$HDF5"

REFINE_ARGS=("$EPISODE_DIR" --output-dir "$OUT" --hdf5 "$HDF5")
if [[ -n "$LEROBOT_EPISODE" ]]; then
  REFINE_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
fi
if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
  REFINE_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
fi

PYTHONPATH="$EVENT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"   python "$SCRIPT_DIR/refine_boundaries.py" "${REFINE_ARGS[@]}"

if [[ -n "${V333_GT:-}" ]]; then
  python "$EVENT_ROOT/evaluation/evaluate_temporal_annotations.py"     "$OUT/hierarchical_annotations.json"     "$V333_GT"     --output "$OUT/evaluation_temporal.json"
fi

echo "V3.3.3 complete: $OUT"
echo "Controlled comparison defaults to reusing V3.3.2 Qwen observations."
echo "Inspect:"
echo "  $OUT/tracked_entity_states.json"
echo "  $OUT/skill_candidates.json"
echo "  $OUT/skill_candidates_validated.json"
echo "  $OUT/interaction_validation_report.json"
echo "  $OUT/hierarchical_annotations.json"
