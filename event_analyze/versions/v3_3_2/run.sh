#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash versions/v3_3_2/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$EPISODE_DIR/versions/v3_3_2"
PROPOSAL_DIR="$OUT/proposal"
mkdir -p "$OUT" "$PROPOSAL_DIR"

# V3.3.2 uses its own proposal set so gripper-retry filtering can be evaluated
# without overwriting proposal artifacts from older versions.
if [[ ! -f "$PROPOSAL_DIR/candidate_events.json" || "${V332_REPROPOSE:-0}" == "1" ]]; then
  PROPOSAL_ARGS=("$HDF5" --output "$PROPOSAL_DIR")
  if [[ -n "$LEROBOT_EPISODE" ]]; then
    PROPOSAL_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
  fi
  if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
    PROPOSAL_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
  fi
  python "$SCRIPT_DIR/analyze_episode.py" "${PROPOSAL_ARGS[@]}"
else
  echo "Reused V3.3.2 proposal artifacts."
fi

# Generic Qwen entity observations over the V3.3.2 proposal windows.
if [[ ! -f "$OUT/entity_observations_base.jsonl" || "${V332_REOBSERVE:-0}" == "1" ]]; then
  python "$SCRIPT_DIR/observe_entities.py"     "$EPISODE_DIR"     --proposal-dir "$PROPOSAL_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
  cp "$OUT/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
else
  echo "Reused V3.3.2 generic entity observations."
fi

# Dense cutlery evidence is stored separately as directional motion evidence.
if [[ ! -f "$OUT/dense_cutlery_transition_evidence.jsonl" || "${V332_REDENSE:-0}" == "1" ]]; then
  python "$SCRIPT_DIR/dense_targeted_reobserve.py"     "$EPISODE_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
else
  cp "$OUT/entity_observations_base.jsonl" "$OUT/entity_observations.jsonl"
  echo "Reused V3.3.2 dense directional evidence."
fi

python "$SCRIPT_DIR/track_entity_states.py"   "$EPISODE_DIR"   --output-dir "$OUT"

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

if [[ -n "${V332_GT:-}" ]]; then
  python "$EVENT_ROOT/evaluation/evaluate_temporal_annotations.py"     "$OUT/hierarchical_annotations.json"     "$V332_GT"     --output "$OUT/evaluation_temporal.json"
fi

echo "V3.3.2 complete: $OUT"
echo "Inspect:"
echo "  $PROPOSAL_DIR/candidate_events.json"
echo "  $OUT/entity_observations_base.jsonl"
echo "  $OUT/dense_cutlery_observations.jsonl"
echo "  $OUT/dense_cutlery_transition_evidence.jsonl"
echo "  $OUT/tracked_entity_states.json"
echo "  $OUT/skill_candidates.json"
echo "  $OUT/skill_candidates_validated.json"
echo "  $OUT/interaction_validation_report.json"
echo "  $OUT/hierarchical_annotations.json"
