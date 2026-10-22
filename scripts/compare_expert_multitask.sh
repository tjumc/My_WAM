#!/usr/bin/env bash
set -euo pipefail

# ===========================================================
# Expert vs multi-task WAM: edit settings here
# Usage: bash scripts/compare_expert_multitask.sh prepare|run|summarize
# Run 'prepare' first. Set exact checkpoint filenames before 'run'.
# ===========================================================
RUN_NAME="${RUN_NAME:-pilot_01}"
TASKS=(oven_put fridge dishwasher oven_takeout collect_clothes dry_clothes wash_clothes)

# Full comparison recommendation: 30 episodes x 8 windows x 3 seeds / task.
# Small smoke run: EPISODES_PER_TASK=3 WINDOWS_PER_EPISODE=3 NUM_SEEDS=1.
EPISODES_PER_TASK="${EPISODES_PER_TASK:-3}"
WINDOWS_PER_EPISODE="${WINDOWS_PER_EPISODE:-3}"
NUM_SEEDS="${NUM_SEEDS:-1}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
DEVICE="${DEVICE:-cuda}"
VAE_DEVICE_MODE="${VAE_DEVICE_MODE:-cpu}"
VARIANT="${VARIANT:-all}"  # all / expert / shared
MIXED_PRECISION="${MIXED_PRECISION:-}"  # empty: use each checkpoint config

# Use the exact already-trained weights; do not automatically pick newest/best.
# Paths below are relative to the repository root unless absolute.
MULTITASK_CHECKPOINT="${MULTITASK_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_multitask_posttrain32/train/checkpoints/weights/step_158700.pt}"
declare -A EXPERT_CHECKPOINTS=(
  [oven_put]="${EXPERT_OVEN_PUT_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_oven_put_posttrain32/train/checkpoints/weights/step_066250.pt}"
  [fridge]="${EXPERT_FRIDGE_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_fridge_posttrain32/train/checkpoints/weights/step_143940.pt}"
  [dishwasher]="${EXPERT_DISHWASHER_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/train/checkpoints/weights/step_275430.pt}"
  [oven_takeout]="${EXPERT_OVEN_TAKEOUT_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_oven_takeout_posttrain32/train/checkpoints/weights/step_081210.pt}"
  [collect_clothes]="${EXPERT_COLLECT_CLOTHES_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_collect_clothes_posttrain32/train/checkpoints/weights/step_082310.pt}"
  [dry_clothes]="${EXPERT_DRY_CLOTHES_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dry_clothes_posttrain32/train/checkpoints/weights/step_125360.pt}"
  [wash_clothes]="${EXPERT_WASH_CLOTHES_CHECKPOINT:-/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_wash_clothes_posttrain32/train/checkpoints/weights/step_071730.pt}"
)

# Expert prompt overrides taken from single-task training scripts.
# If actual saved training prompts differ, edit them to match the checkpoints.
declare -A EXPERT_PROMPTS=(
  [fridge]="Move to the fridge, take out the plate with food inside from the opened drawer, then move to the desk and put the plate on the desk"
  [dishwasher]="put the dish into the dishwasher"
  [oven_put]="put the plate into the oven"
  [oven_takeout]="take the plate out of the oven"
  [collect_clothes]="Gather the clothes into the laundry basket"
  [dry_clothes]="transfer the clothes to the dryer"
  [wash_clothes]="put the clothes into the washing machine"
)

# Overrides for model run directories and text caches, if custom naming was used.
SINGLE_RUNS_ROOT="${SINGLE_RUNS_ROOT:-runs}"
MULTITASK_RUN="${MULTITASK_RUN:-runs/astribot_multitask_posttrain32}"
MULTITASK_CACHE="${MULTITASK_CACHE:-posttrain/text_embeds_cache_astribot_multitask_posttrain32}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export WAN22_REDIRECT_COMMON_FILES="${WAN22_REDIRECT_COMMON_FILES:-false}"

# ===========================================================
MODE="${1:-prepare}"
if [[ "$MODE" != prepare && "$MODE" != run && "$MODE" != summarize ]]; then
  echo "Usage: bash $0 prepare|run|summarize" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
OUT_DIR="$REPO_ROOT/comparison_reports/$RUN_NAME"
mkdir -p "$OUT_DIR/_raw"
CATALOG_TSV="$OUT_DIR/_raw/catalog.tsv"
MODELS_TSV="$OUT_DIR/_raw/models.tsv"

source configs/data/astribot_dataset_catalog.sh

# Freeze the catalog snapshot. Do not change samples across prepare/run.
if [[ "$MODE" == prepare ]]; then
  printf 'task\tdataset_dir\n' > "$CATALOG_TSV"
  for task in "${TASKS[@]}"; do
    [[ -n "${ASTRIBOT_DATASET_CATALOG[$task]:-}" ]] || { echo "Unknown task $task" >&2; exit 2; }
    while IFS= read -r dataset_dir; do
      [[ -n "$dataset_dir" ]] && printf '%s\t%s\n' "$task" "$dataset_dir" >> "$CATALOG_TSV"
    done <<< "${ASTRIBOT_DATASET_CATALOG[$task]}"
  done
  python tools/compare_expert_multitask.py prepare \
    --catalog "$CATALOG_TSV" --output "$OUT_DIR" \
    --episodes-per-task "$EPISODES_PER_TASK" \
    --windows-per-episode "$WINDOWS_PER_EPISODE" --seed "$SEED"
  echo "Next: set exact checkpoint paths above, then run:"
  echo "  bash scripts/compare_expert_multitask.sh run"
elif [[ "$MODE" == run ]]; then
  if [[ ! -f "$OUT_DIR/samples.csv" ]]; then
    echo "No frozen samples; execute 'prepare' first." >&2
    exit 2
  fi
  if [[ -z "$MULTITASK_CHECKPOINT" ]]; then
    echo "MULTITASK_CHECKPOINT is empty. Fill an exact .pt path at script top or export it." >&2
    exit 2
  fi
  # Read the frozen task list rather than assuming this invocation's TASKS matches 'prepare'.
  mapfile -t FROZEN_TASKS < <(python - "$OUT_DIR/samples.csv" <<'PY'
import csv,sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    print('\n'.join(dict.fromkeys(r['task'] for r in csv.DictReader(f))))
PY
  )
  printf 'kind\ttask\tcheckpoint\tconfig\tstats\ttext_cache\tprompt\n' > "$MODELS_TSV"
  for task in "${FROZEN_TASKS[@]}"; do
    run="$SINGLE_RUNS_ROOT/astribot_${task}_posttrain32"
    checkpoint="${EXPERT_CHECKPOINTS[$task]:-}"
    [[ -n "$checkpoint" ]] || {
      echo "Missing expert checkpoint for $task. Set its EXPERT_*_CHECKPOINT variable." >&2
      exit 2
    }
    printf 'expert\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$task" "$checkpoint" "$run/train/config.yaml" "$run/stats.json" \
      "posttrain/text_embeds_cache_astribot_${task}_posttrain32" \
      "${EXPERT_PROMPTS[$task]}" >> "$MODELS_TSV"
  done
  printf 'shared\t__all__\t%s\t%s\t%s\t%s\t%s\n' \
    "$MULTITASK_CHECKPOINT" "$MULTITASK_RUN/train/config.yaml" \
    "$MULTITASK_RUN/stats.json" "$MULTITASK_CACHE" "" >> "$MODELS_TSV"

  if [[ ! -d "$DIFFSYNTH_MODEL_BASE_PATH" ]]; then
    echo "Base model directory missing: $DIFFSYNTH_MODEL_BASE_PATH" >&2
    exit 2
  fi
  MP_ARGS=()
  [[ -n "$MIXED_PRECISION" ]] && MP_ARGS=(--mixed-precision "$MIXED_PRECISION")
  python tools/compare_expert_multitask.py run \
    --models "$MODELS_TSV" --output "$OUT_DIR" \
    --seed "$SEED" --num-seeds "$NUM_SEEDS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --variant "$VARIANT" --device "$DEVICE" \
    --vae-device-mode "$VAE_DEVICE_MODE" "${MP_ARGS[@]}"
  echo "After inference: bash scripts/compare_expert_multitask.sh summarize"
else
  python tools/compare_expert_multitask.py summarize --output "$OUT_DIR"
fi
