#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash versions/v3_3_1/run.sh HDF5 EPISODE_DIR TASK [LEROBOT_EPISODE] [LEROBOT_MAX_RAW_FRAME]
HDF5="$1"
EPISODE_DIR="$2"
TASK="$3"
LEROBOT_EPISODE="${4:-}"
LEROBOT_MAX_RAW_FRAME="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$EPISODE_DIR/versions/v3_3_1"
mkdir -p "$OUT"

# A brand-new HDF5 has no proposal artifacts yet. Generate the shared
# state/action candidate events + multi-view contact sheets automatically.
if [[ ! -f "$EPISODE_DIR/candidate_events.json" || ! -d "$EPISODE_DIR/contact_sheets" ]]; then
  echo "No candidate_events.json/contact_sheets found; generating proposals from HDF5..."
  PROPOSAL_ARGS=("$HDF5" --output "$EPISODE_DIR")
  if [[ -n "$LEROBOT_EPISODE" ]]; then
    PROPOSAL_ARGS+=(--lerobot-episode "$LEROBOT_EPISODE")
  fi
  if [[ -n "$LEROBOT_MAX_RAW_FRAME" ]]; then
    PROPOSAL_ARGS+=(--lerobot-max-raw-frame "$LEROBOT_MAX_RAW_FRAME")
  fi
  python "$EVENT_ROOT/common/analyze_episode.py" "${PROPOSAL_ARGS[@]}"
else
  echo "Reused existing candidate events/contact sheets."
fi

# Keep generic observations fixed by default.
if [[ -f "$EPISODE_DIR/versions/v3/entity_observations.jsonl" && "${V331_REOBSERVE_ALL:-0}" != "1" ]]; then
  cp "$EPISODE_DIR/versions/v3/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
  echo "Reused V3 generic entity observations."
else
  python "$SCRIPT_DIR/observe_entities.py"     "$EPISODE_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
  cp "$OUT/entity_observations.jsonl" "$OUT/entity_observations_base.jsonl"
fi

# Controlled V3.3 -> V3.3.1 comparison: reuse already computed dense VLM
# observations and rebuild only the pseudo evidence/fusion layer.
if [[ -f "$EPISODE_DIR/versions/v3_3/dense_cutlery_observations.jsonl" && "${V331_REDENSE:-0}" != "1" ]]; then
  python "$SCRIPT_DIR/prepare_dense_evidence.py"     --base "$OUT/entity_observations_base.jsonl"     --dense "$EPISODE_DIR/versions/v3_3/dense_cutlery_observations.jsonl"     --output-dir "$OUT"
  echo "Reused V3.3 dense VLM observations; reran only V3.3.1 evidence preparation."
else
  python "$SCRIPT_DIR/dense_targeted_reobserve.py"     "$EPISODE_DIR"     --output-dir "$OUT"     --hdf5 "$HDF5"     --task "$TASK"
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

# Optional regression evaluation. This never feeds GT back into the pipeline.
if [[ -n "${V331_GT:-}" ]]; then
  python "$EVENT_ROOT/evaluation/evaluate_temporal_annotations.py"     "$OUT/hierarchical_annotations.json"     "$V331_GT"     --output "$OUT/evaluation_temporal.json"
fi

echo "V3.3.1 complete: $OUT"
echo "Inspect:"
echo "  $OUT/dense_cutlery_observations.jsonl"
echo "  $OUT/dense_cutlery_pseudo_observations.jsonl"
echo "  $OUT/tracked_entity_states.json"
echo "  $OUT/skill_candidates.json"
echo "  $OUT/skill_candidates_validated.json"
echo "  $OUT/interaction_validation_report.json"
echo "  $OUT/hierarchical_annotations.json"
echo "Optional regression: set V331_GT=/path/to/manual_gt.json"
