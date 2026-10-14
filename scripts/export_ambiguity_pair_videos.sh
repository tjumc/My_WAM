#!/usr/bin/env bash
# 候选回放：只读轨迹，导出前后视频；不训练、不推理、不控制机器人。
# 所有常用设置集中在这里。不会修改原筛选脚本或它的输出路径。

# -------------------- 修改这里 --------------------
PAIRS_JSON="ambiguity_outputs/history_ambiguity/20261014_191521_89946/pairs.json"
PAIR_RANKS=(1 9)              # 按 pairs.json 的 rank 选取，不是 episode 序号
BEFORE_SECONDS=3
AFTER_SECONDS=3
OUTPUT_FPS=10                # 回放帧率；不改变原始轨迹的时间速度
PAUSE_SECONDS=1              # 到 CURRENT 时额外停留 1 秒，方便查看
PYTHON_BIN="python"          # 使用已有 WAM 环境
VIDEO_BACKEND="pyav"
TOLERANCE_S="0.002"          # 视频时间对齐容差，不自动放宽
CAMERAS=(head left right)    # 上方、左下、右下；支持仓库内的相机别名

# 留空：读取 pairs.json 中的原始数据目录。挂载路径变化时再填写。
DATASET_DIR=""
# 留空：在 pairs.json 所在目录下新建 pair_videos/本次时间戳/。
# 这里只指定新回放文件位置，不改动原来的 mining 输出路径。
REPLAY_ROOT=""
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "请激活 WAM 环境或修改 PYTHON_BIN。" >&2; exit 1; }
[[ -f "$PAIRS_JSON" ]] || { echo "找不到 PAIRS_JSON：$PAIRS_JSON" >&2; exit 1; }
[[ -f tools/export_ambiguity_pair_videos.py ]] || { echo "请先更新仓库，缺少回放 Python 脚本。" >&2; exit 1; }
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "$REPLAY_ROOT" ]]; then
    REPLAY_ROOT="$(dirname -- "$PAIRS_JSON")/pair_videos"
fi
OUT="${REPLAY_ROOT%/}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
cp -- "$SCRIPT_PATH" "$OUT/launcher.sh"
trap 'status=$?; printf "\n回放导出失败（%s），请查看 %s/run.log\n" "$status" "$OUT" >&2; exit "$status"' ERR
ARGS=(--pairs-json "$PAIRS_JSON" --ranks "${PAIR_RANKS[@]}"
      --before-seconds "$BEFORE_SECONDS" --after-seconds "$AFTER_SECONDS"
      --fps "$OUTPUT_FPS" --pause-seconds "$PAUSE_SECONDS"
      --video-backend "$VIDEO_BACKEND" --tolerance-s "$TOLERANCE_S"
      --cameras "${CAMERAS[@]}" --output-dir "$OUT")
if [[ -n "$DATASET_DIR" ]]; then
    ARGS+=(--dataset-dir "$DATASET_DIR")
fi
"$PYTHON_BIN" -u tools/export_ambiguity_pair_videos.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log"
printf '\n结果目录：%s\n' "$OUT"
printf 'pair_XXXX.mp4：左右同步回放，CURRENT 是筛选时刻，前后是实际 GT 视频。\n'
printf 'pair_XXXX_current.png：当前帧截图；pair_XXXX.json：逐帧定位信息。\n'
printf 'review.csv：人工核查记录；manifest.json：输入与参数；run.log：运行日志。\n'
