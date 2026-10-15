#!/usr/bin/env bash
# 真机客户端 + 可选闭环记录。会启动真实机器人，不是离线回放。
# 任务、模型和原执行参数仍来自已有 config.sh；本文件只增加记录设置。
# 先使用实验室现有安全流程验证，再运行；本记录器不提供硬件急停。

# -------------------- 只修改这里：记录设置 --------------------
RECORD_ENABLED=true
RECORD_ROOT="${HOME}/wam_rollouts"   # 客户端本机，默认在仓库外；不是远程推理服务器
RECORD_QUEUE_SIZE=8                 # 包含正在写的记录；满时丢记录，不等待磁盘
RECORD_MAX_GB=5                     # 每次运行上限（GiB）；到限关闭记录，不控制机器人
RECORD_MIN_FREE_GB=2                # 磁盘保留空间；不足则记录停用并标记缺失
RECORD_IMAGE_EVERY=1                # 每轮三相机；0=仅数值；N=每N轮保存图像
RECORD_CLOSE_TIMEOUT=3              # 退出后最多等待写盘秒数；超时不保证完整
PYTHON_BIN="python"                 # 机器人SDK所在环境，不是WAM推理服务器环境
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/config.sh"
command -v "$PYTHON_BIN" >/dev/null || { echo "找不到PYTHON_BIN" >&2; exit 1; }
[[ -f "$ASTRIBOT_SDK_ENV" ]] || { echo "找不到Astribot SDK环境文件" >&2; exit 1; }
source "$ASTRIBOT_SDK_ENV"
ARGS=(--server_host "$SERVER_HOST" --server_port "$SERVER_PORT"
      --execute_steps "$EXECUTE_STEPS" --control_hz "$CONTROL_HZ"
      --action_dt "$ACTION_DT" --boundary_blend_steps "$BOUNDARY_BLEND_STEPS"
      --max_runtime "$MAX_RUNTIME")
case "$RECORD_ENABLED" in
 true) ARGS+=(--record_dir "$RECORD_ROOT" --record_queue_size "$RECORD_QUEUE_SIZE"
               --record_max_gb "$RECORD_MAX_GB" --record_min_free_gb "$RECORD_MIN_FREE_GB"
               --record_image_every "$RECORD_IMAGE_EVERY" --record_close_timeout "$RECORD_CLOSE_TIMEOUT");;
 false) ;;
 *) echo "RECORD_ENABLED必须为true/false" >&2; exit 1;;
esac
printf '真机客户端：%s:%s；execute_steps=%s action_dt=%s control_hz=%s blend=%s\n' \
 "$SERVER_HOST" "$SERVER_PORT" "$EXECUTE_STEPS" "$ACTION_DT" "$CONTROL_HZ" "$BOUNDARY_BLEND_STEPS"
printf '记录启用=%s；本机目录=%s；原执行参数未更改。\n' "$RECORD_ENABLED" "$RECORD_ROOT"
exec "$PYTHON_BIN" -u experiments/astribot/real_robot_client.py "${ARGS[@]}"
