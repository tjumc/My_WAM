#!/usr/bin/env bash
# 离线生成可上传GitHub的小报告：不启动机器人，不复制原始图片/数组。
# -------------------- 只修改这里 --------------------
RUN_DIR=""                         # 填客户端打印的完整 rollout_... 目录
REPORT_ROOT="reports/closed_loop"   # 仅用于新摘要，不改原始记录/既有输出路径
PYTHON_BIN="python"
# ==================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd -- "$SCRIPT_DIR/.." && pwd)"
[[ -n "$RUN_DIR" ]] || { echo "请先在本sh顶部填写RUN_DIR。" >&2; exit 1; }
OUT="${REPORT_ROOT%/}/$(basename -- "$RUN_DIR")_$(date +%Y%m%d_%H%M%S)_$$"
"$PYTHON_BIN" tools/summarize_robot_rollout.py --run-dir "$RUN_DIR" --output-dir "$OUT"
printf '仅上传以下报告目录，不要上传原始rollout：%s\n' "$OUT"
