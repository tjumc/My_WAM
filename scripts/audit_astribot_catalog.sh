#!/usr/bin/env bash
# 只读盘点catalog中的全部任务/目录。无模型，无GPU要求，不重训、不自动划分数据集。
# 原始数据路径仍由原catalog控制；这里只可选地覆盖本进程中的挂载根目录。
# ==================== 常用参数：只改这里 ====================
CATALOG="configs/data/astribot_dataset_catalog.sh"
TASKS=()                         # 空数组=全部任务；例如 TASKS=(dishwasher oven_put)
DATA_ROOT=""                     # 空=沿用catalog中的DATASET_ROOTPATH
BENCHMARK_ROOT=""                # 空=沿用catalog中的BENCHMARK_ROOTPATH
OUTPUT_ROOT="ambiguity_outputs/dataset_audit"  # 新增审计目录，不改旧实验输出路径
PYTHON_BIN="python"               # 现有WAM环境；需numpy/pyarrow，预览另需Pillow/PyAV
NUMERIC_SCAN=true                 # true:全量流式数值扫描；false:只看元信息
PREVIEW_EPISODES_PER_DIR=3         # 每个目录随机3条；0关闭图像预览
PREVIEW_FRAMES=5                  # 每条均匀取5个时刻（不是事件边界）
MAX_CAMERAS=3                     # 概览图最多几路相机；元信息仍报告全部相机
THUMB_WIDTH=240                   # 每个图像格的宽度，保留纵横比
BATCH_SIZE=4096                   # 数值Parquet读取批大小，不是训练batch
SEED=42
TIMESTAMP_TOLERANCE=0.002          # 时间对齐容差；不会自动放宽
LOW_CHANGE_EPSILON=0.0001         # 原始数值低变化提示，不是静止/成功阈值
MIN_FREE_GB=2                     # 剩余磁盘低于此值不再写预览图（单位GiB）
# ==========================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
command -v "$PYTHON_BIN" >/dev/null || { echo '请激活WAM环境或修改PYTHON_BIN' >&2; exit 1; }
[[ -f "$CATALOG" ]] || { echo "找不到catalog: $CATALOG" >&2; exit 1; }
[[ -f tools/audit_astribot_catalog.py ]] || { echo '缺少tools/audit_astribot_catalog.py，请更新仓库' >&2; exit 1; }
# No git pull/reset/add/push in this launcher.
OUT="${OUTPUT_ROOT%/}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
# Install the upload whitelist before creating any other run artifacts.
cat > "$OUT/.gitignore" <<'IGNORE'
*
!/.gitignore
!/report.md
!/directories.csv
!/tasks.csv
!/numeric_summary.csv
!/duplicate_candidates.csv
!/issues.csv
!/review.csv
!/manifest.json
!/COMPLETE.json
!/FAILED.json
IGNORE
cp -- "$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")" "$OUT/launcher.sh"
ARGS=(--catalog "$CATALOG" --output-dir "$OUT" --preview-episodes "$PREVIEW_EPISODES_PER_DIR"
      --preview-frames "$PREVIEW_FRAMES" --max-cameras "$MAX_CAMERAS" --thumb-width "$THUMB_WIDTH"
      --batch-size "$BATCH_SIZE" --seed "$SEED" --timestamp-tolerance "$TIMESTAMP_TOLERANCE"
      --low-change-epsilon "$LOW_CHANGE_EPSILON" --min-free-gb "$MIN_FREE_GB")
if ((${#TASKS[@]})); then ARGS+=(--tasks "${TASKS[@]}"); fi
[[ -z "$DATA_ROOT" ]] || ARGS+=(--data-root "$DATA_ROOT")
[[ -z "$BENCHMARK_ROOT" ]] || ARGS+=(--benchmark-root "$BENCHMARK_ROOT")
case "$NUMERIC_SCAN" in true) ;; false) ARGS+=(--no-numeric);; *) echo 'NUMERIC_SCAN必须为true/false'; exit 1;; esac
STATUS=0
"$PYTHON_BIN" -u tools/audit_astribot_catalog.py "${ARGS[@]}" 2>&1 | tee "$OUT/run.log" || STATUS=$?
if ((STATUS == 0)); then
    echo "盘点完成：$OUT"
elif ((STATUS == 2)) && [[ -f "$OUT/COMPLETE.json" ]]; then
    echo "已遍历清单，但存在缺失/异常/预览失败；先看 $OUT/issues.csv。未读数据不是空数据。"
else
    echo "盘点未完整结束：$OUT/run.log" >&2
fi
echo 'Git只提交小报告和摘要CSV；不要用git add -f上传_local/。'
exit "$STATUS"
