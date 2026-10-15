#!/usr/bin/env bash
# 第2对数值诊断：只读原始轨迹，不加载WAM，不控制机器人。
# 所有常用路径/参数只改下面；不修改已有筛选、回放脚本及其输出路径。

# -------------------- 修改这里 --------------------
PAIRS_JSON="ambiguity_outputs/history_ambiguity/20261014_191521_89946/pairs.json"
PAIR_RANKS=(2)                  # pairs.json中的rank，不是episode编号
CONFIG=""                      # 留空使用pairs.json中记录的config；挂载变化时填写
STATS=""                       # 留空使用pairs.json中记录的Fast-WAM stats.json
DATASET_DIR=""                 # 留空使用source_a/source_b中的数据根目录
DIAGNOSTIC_ROOT=""             # 留空：本轮结果目录/pair_diagnostics/新时间戳/
PYTHON_BIN="python"             # 已有WAM环境，也可填解释器绝对路径
AFTER_SECONDS=3                # 重点统计CURRENT之后3秒；另导出到记录末尾的全部数值
TAIL_SECONDS=2                 # 另外检查记录最后2秒，不借用CURRENT之前的行
TOLERANCE_S="0.002"             # 源时间戳校验容差；不是停止/静止阈值
BASE_MODE="velocity"           # 对应当前cmd_absolute_joint_vel：前三维是底盘速度
                               # 数据为底盘位置时改position；不确定填unknown
# 动作horizon/模型补齐维度从CONFIG读取。不会自动判定停止或任务完成。
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
command -v "$PYTHON_BIN" >/dev/null || { echo "找不到PYTHON_BIN，请先激活WAM环境。" >&2; exit 1; }
[[ -f tools/diagnose_ambiguity_pairs.py ]] || { echo "缺少诊断Python脚本，请更新仓库。" >&2; exit 1; }
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT="$("$PYTHON_BIN" - "$PAIRS_JSON" "$DIAGNOSTIC_ROOT" <<'PY'
import sys, time, uuid
from pathlib import Path
pairs = Path(sys.argv[1]).expanduser().resolve()
if not pairs.is_file():
    raise FileNotFoundError(f"找不到PAIRS_JSON：{pairs}")
root = Path(sys.argv[2]).expanduser().resolve() if sys.argv[2] else pairs.parent / "pair_diagnostics"
print(root / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]))
PY
)"
[[ ! -e "$OUT" ]] || { echo "诊断目录已存在，拒绝覆盖：$OUT" >&2; exit 1; }
ARGS=(--pairs-json "$PAIRS_JSON" --ranks "${PAIR_RANKS[@]}" --output-dir "$OUT"
      --after-seconds "$AFTER_SECONDS" --tail-seconds "$TAIL_SECONDS"
      --tolerance-s "$TOLERANCE_S" --base-mode "$BASE_MODE")
[[ -z "$CONFIG" ]] || ARGS+=(--config "$CONFIG")
[[ -z "$STATS" ]] || ARGS+=(--dataset-stats "$STATS")
[[ -z "$DATASET_DIR" ]] || ARGS+=(--dataset-dir "$DATASET_DIR")
LOG_FILE="$(mktemp)"
trap 'rm -f -- "$LOG_FILE"' EXIT
set +e
"$PYTHON_BIN" -u tools/diagnose_ambiguity_pairs.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUSES=("${PIPESTATUS[@]}")
set -e
if [[ -d "$OUT" ]]; then
    cp -- "$LOG_FILE" "$OUT/run.log"
    cp -- "$SCRIPT_PATH" "$OUT/launcher.sh"
fi
if [[ "${STATUSES[0]}" -ne 0 || "${STATUSES[1]}" -ne 0 ]]; then
    echo "诊断失败，请查看以上报错；未完整完成的目录不会有COMPLETE.json。" >&2
    exit 1
fi
printf '\n完成：%s\n先看 pair_0002/report.md；summary.json用于汇总分析。\n' "$OUT"
