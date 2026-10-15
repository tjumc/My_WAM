#!/usr/bin/env bash
# 第2对 WAM 离线推理。只生成动作并与 GT 比较，不训练、不控制机器人。
# 在 My_WAM 仓库中：bash scripts/infer_ambiguity_pairs.sh
# 原筛选/回放/诊断脚本及输出路径不作修改；这里只创建新的推理结果。

# ==================== 常用设置：只改这里 ====================
PAIRS_JSON="ambiguity_outputs/history_ambiguity/20261014_191521_89946/pairs.json"
PAIR_RANKS=(2)                        # pairs.json 的 rank，不是 episode 编号
CONFIG="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/train/config.yaml"
CHECKPOINT="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/train/checkpoints/weights/step_275430.pt"
STATS="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/stats.json"
TEXT_CACHE_DIR="posttrain/text_embeds_cache_astribot_dishwasher_posttrain32"
TASK_PROMPT="put the dish into the dishwasher"              # 必须使用原任务指令；不要改成阶段/完成提示
MODEL_BASE_PATH="/efs/share/1919650160032350208/projects/foundation_model/FastWAM/checkpoints"
DATASET_DIR=""                       # 留空读取 source_a/source_b；挂载改变时才填写
INFERENCE_ROOT=""                    # 留空：pairs.json所在目录/pair_inference/新时间戳/
PYTHON_BIN="python"                   # 已有 WAM 环境，也可填解释器绝对路径

NUM_SEEDS=10                         # 左右使用相同 seed 集合，共 2 * NUM_SEEDS 次/pair
SEED=42                              # 默认 42..51；不是挑最好的 seed
NUM_INFERENCE_STEPS=10                # flow-matching 去噪步数，不是动作长度
DEVICE="cuda"
MIXED_PRECISION="bf16"
VAE_DEVICE_MODE="cpu"                # 与现有真机配置一致；显存充足可改 gpu
SIGMA_SHIFT=""                       # 留空沿用 checkpoint config 的 scheduler
TILED=false
SAVE_PLOTS=true                      # 25张逐维原始单位曲线和逐seed误差图
VIDEO_BACKEND="pyav"
TOLERANCE_S="0.002"                   # 时间对齐容差，不是成功/静止阈值
BASE_MODE="velocity"                 # 当前 cmd_absolute_joint_vel；不要混用底盘位置数据

# 本地 Wan 资源布局，沿用真机脚本的设置。
WAN22_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
WAN22_TOKENIZER_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
WAN22_REDIRECT_COMMON_FILES="true"
DIFFSYNTH_SKIP_DOWNLOAD="true"
# 动作长度从 CONFIG 的 num_frames-1 获取，当前为32步，不在这里另改。
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
command -v "$PYTHON_BIN" >/dev/null || { echo "找不到Python，请激活WAM环境或改PYTHON_BIN。" >&2; exit 1; }
[[ -f tools/infer_ambiguity_pairs.py ]] || { echo "缺少Python脚本，请先更新仓库。" >&2; exit 1; }
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WAN22_MODEL_ID WAN22_TOKENIZER_MODEL_ID WAN22_REDIRECT_COMMON_FILES DIFFSYNTH_SKIP_DOWNLOAD
export WAM_TASK_INSTRUCTION="$TASK_PROMPT"

if [[ -z "$INFERENCE_ROOT" ]]; then
    INFERENCE_ROOT="$(dirname -- "$PAIRS_JSON")/pair_inference"
fi
OUT="${INFERENCE_ROOT%/}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
cp -- "$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")" "$OUT/launcher.sh"
trap 'status=$?; printf "\n推理未完成（退出码 %s），查看 %s/run.log。\n" "$status" "$OUT" >&2; exit "$status"' ERR

ARGS=(--pairs-json "$PAIRS_JSON" --ranks "${PAIR_RANKS[@]}"
      --checkpoint "$CHECKPOINT" --text-cache-dir "$TEXT_CACHE_DIR" --task "$TASK_PROMPT"
      --output-dir "$OUT" --num-seeds "$NUM_SEEDS" --seed "$SEED"
      --num-inference-steps "$NUM_INFERENCE_STEPS" --device "$DEVICE"
      --mixed-precision "$MIXED_PRECISION" --vae-device-mode "$VAE_DEVICE_MODE"
      --video-backend "$VIDEO_BACKEND" --tolerance-s "$TOLERANCE_S" --base-mode "$BASE_MODE")
[[ -z "$CONFIG" ]] || ARGS+=(--config "$CONFIG")
[[ -z "$STATS" ]] || ARGS+=(--dataset-stats "$STATS")
[[ -z "$DATASET_DIR" ]] || ARGS+=(--dataset-dir "$DATASET_DIR")
[[ -z "$MODEL_BASE_PATH" ]] || ARGS+=(--model-base-path "$MODEL_BASE_PATH")
[[ -z "$SIGMA_SHIFT" ]] || ARGS+=(--sigma-shift "$SIGMA_SHIFT")
case "$TILED" in true) ARGS+=(--tiled);; false) ;; *) echo "TILED 必须是 true/false" >&2; exit 1;; esac
case "$SAVE_PLOTS" in false) ARGS+=(--no-plots);; true) ;; *) echo "SAVE_PLOTS 必须是 true/false" >&2; exit 1;; esac

"$PYTHON_BIN" -u tools/infer_ambiguity_pairs.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
printf '\n完成：%s\n先看 pair_0002/report.md。COMPLETE.json 表示本轮全部完成。\n' "$OUT"
