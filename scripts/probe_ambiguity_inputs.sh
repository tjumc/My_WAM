#!/usr/bin/env bash
# 云端输入交换敏感性：AA/BB/BA/AB 四组合；无训练、无真机连接。
# 在已有 WAM 环境运行：bash scripts/probe_ambiguity_inputs.sh
# 不改旧的筛选、推理、回放路径。详细数组只留服务器，不生成批量图片。

# ==================== 常用设置：只改这里 ====================
PAIRS_JSON="ambiguity_outputs/history_ambiguity/20261014_191521_89946/pairs.json"
PAIR_RANKS=(1 2 9 16 43)        # 1/9人工可区分、2开始/收尾、16较近proprio、43较高拟合误差
                               # 固定探索集合，不是随机测试集；全50对可改为({1..50})
CONFIG="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/train/config.yaml"
CHECKPOINT="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/train/checkpoints/weights/step_275430.pt"
STATS="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/stats.json"
TEXT_CACHE_DIR="posttrain/text_embeds_cache_astribot_dishwasher_posttrain32"
TASK_PROMPT="put the dish into the dishwasher"   # 沿用你的实际训练指令，两侧始终相同
MODEL_BASE_PATH="/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints"
DATASET_DIR=""                 # 留空：pairs.json中的源数据目录；挂载变化时填写
SWAP_ROOT=""                   # 留空：pairs.json所在目录/input_swap/新时间戳/
PYTHON_BIN="python"
NUM_SEEDS=10                   # 四组合均用42..51；5对共200次推理
SEED=42
NUM_INFERENCE_STEPS=10
DEVICE="cuda"
MIXED_PRECISION="bf16"
VAE_DEVICE_MODE="cpu"
SIGMA_SHIFT=""
TILED=false
VIDEO_BACKEND="pyav"
TOLERANCE_S="0.002"
WAN22_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
WAN22_TOKENIZER_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
WAN22_REDIRECT_COMMON_FILES="true"
DIFFSYNTH_SKIP_DOWNLOAD="true"
# ==================== 以下通常无需修改 ====================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
command -v "$PYTHON_BIN" >/dev/null || { echo "找不到Python，请激活WAM环境。" >&2; exit 1; }
[[ -f tools/probe_ambiguity_inputs.py ]] || { echo "缺少Python脚本，请更新仓库。" >&2; exit 1; }
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WAN22_MODEL_ID WAN22_TOKENIZER_MODEL_ID WAN22_REDIRECT_COMMON_FILES DIFFSYNTH_SKIP_DOWNLOAD
export WAM_TASK_INSTRUCTION="$TASK_PROMPT"
[[ -n "$SWAP_ROOT" ]] || SWAP_ROOT="$(dirname -- "$PAIRS_JSON")/input_swap"
OUT="${SWAP_ROOT%/}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
cp -- "$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")" "$OUT/launcher.sh"
trap 'status=$?; printf "\n交换实验未完成（退出码 %s），查看 %s/run.log。\n" "$status" "$OUT" >&2; exit "$status"' ERR
ARGS=(--pairs-json "$PAIRS_JSON" --ranks "${PAIR_RANKS[@]}"
      --checkpoint "$CHECKPOINT" --text-cache-dir "$TEXT_CACHE_DIR" --task "$TASK_PROMPT"
      --output-dir "$OUT" --num-seeds "$NUM_SEEDS" --seed "$SEED"
      --num-inference-steps "$NUM_INFERENCE_STEPS" --device "$DEVICE"
      --mixed-precision "$MIXED_PRECISION" --vae-device-mode "$VAE_DEVICE_MODE"
      --video-backend "$VIDEO_BACKEND" --tolerance-s "$TOLERANCE_S")
[[ -z "$CONFIG" ]] || ARGS+=(--config "$CONFIG")
[[ -z "$STATS" ]] || ARGS+=(--dataset-stats "$STATS")
[[ -z "$DATASET_DIR" ]] || ARGS+=(--dataset-dir "$DATASET_DIR")
[[ -z "$MODEL_BASE_PATH" ]] || ARGS+=(--model-base-path "$MODEL_BASE_PATH")
[[ -z "$SIGMA_SHIFT" ]] || ARGS+=(--sigma-shift "$SIGMA_SHIFT")
case "$TILED" in true) ARGS+=(--tiled);; false) ;; *) echo "TILED 必须是true/false" >&2; exit 1;; esac
"$PYTHON_BIN" -u tools/probe_ambiguity_inputs.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
printf '\n完成：%s\n上传report.md、pair_summary.csv、group_summary.csv、manifest.json和COMPLETE.json即可。\n' "$OUT"
printf '不要上传_raw/；不使用git add -f。原始与交换组合都重跑，不复用旧结果。\n'
