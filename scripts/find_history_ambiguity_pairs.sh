#!/usr/bin/env bash
# ============================================================
#      Fast-WAM 历史歧义候选筛选：只需修改下面的路径和参数
#
# 在已有 WAM Python 环境中运行：
#   bash scripts/find_history_ambiguity_pairs.sh
#
# 仅离线读取轨迹，不训练、不加载 WAM checkpoint、不控制机器人。
# 首次使用前先在仓库中 git pull --ff-only；本脚本不自动更新源码。
# ============================================================

# -------------------- 路径：优先检查这里 --------------------
# 以下数据路径来自之前记录；服务器挂载位置不同时请修改。
# DATASET 必须是一个含 meta/info.json 的 LeRobot 数据集目录。
DATASET="/data/share/1919650160032350208/foundation_model/datasets/astribot_lerobotv3/dishwasher_fx_20260312_compressed"

# 使用该 WAM 实验对应的 Fast-WAM 归一化文件。
# 不要直接替换成 LeRobot 的 meta/stats.json。
STATS="/efs/share/1919650160032350208/users/machong14/wam/runs/astribot_dishwasher_posttrain32/stats.json"
CONFIG="configs/train/astribot_posttrain32.yaml"

# 相对路径均以仓库根目录为基准；每次运行会创建独立子目录。
OUTPUT_ROOT="outputs/history_ambiguity"

# 默认使用当前激活环境的 python；也可填写解释器的绝对路径。
PYTHON_BIN="python"
VIDEO_BACKEND="pyav"

# -------------------- 采样参数：首次保持小规模 --------------------
MAX_SAMPLES=256                 # 最多读取的窗口起点数
SAMPLES_PER_EPISODE=64          # 每条轨迹最多采样多少个起点
SAMPLE_STRIDE=4                 # 起点候选间隔；不会改变窗口内部动作采样率
MIN_FRAME_GAP=150              # 同轨迹两起点最小帧距；30 Hz 时为 5 秒
SEED=42

# -------------------- 筛选与输出 --------------------
# 首次不按 proprio/action 过滤，先观察实际数值和图片。
# RMSE 位于数据集归一化空间，不是米、角度或成功概率。
MAX_PROPRIO_RMSE="inf"          # 越小越严格；inf 表示暂不过滤
MIN_ACTION_RMSE="0.0"           # 越大越严格；0.0 表示暂不过滤
CANDIDATE_K_PER_EPISODE=200
TOP_K=50                       # 最终保存多少组候选
SAVE_PAIR_IMAGES=20            # 保存前多少组对比图，0 表示不保存
FEATURE_H=24                   # pooled-pixel 特征尺寸
FEATURE_W=20

# ============================================================
#                    以下通常不需要修改
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "tools/find_history_ambiguity_pairs.py" ]]; then
    echo "错误：未找到 tools/find_history_ambiguity_pairs.py。" >&2
    echo "请将此 sh 文件放在 My_WAM/scripts/ 中，并先更新仓库。" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误：找不到 Python：$PYTHON_BIN；请激活 WAM 环境或修改 PYTHON_BIN。" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WAM_VIDEO_BACKEND="$VIDEO_BACKEND"

OUT="${OUTPUT_ROOT%/}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
cp -- "$SCRIPT_PATH" "$OUT/launcher.sh"
trap 'status=$?; printf "\n运行失败（退出码 %s）。日志：%s/run.log\n" "$status" "$OUT" >&2; exit "$status"' ERR

ARGS=(
    --config "$CONFIG"
    --dataset_stats "$STATS"
    --dataset_dir "$DATASET"
    --output_dir "$OUT"
    --max_samples "$MAX_SAMPLES"
    --samples_per_episode "$SAMPLES_PER_EPISODE"
    --sample_stride "$SAMPLE_STRIDE"
    --min_frame_gap "$MIN_FRAME_GAP"
    --seed "$SEED"
    --max_proprio_rmse "$MAX_PROPRIO_RMSE"
    --min_action_rmse "$MIN_ACTION_RMSE"
    --candidate_k_per_episode "$CANDIDATE_K_PER_EPISODE"
    --top_k "$TOP_K"
    --save_pair_images "$SAVE_PAIR_IMAGES"
    --feature_h "$FEATURE_H"
    --feature_w "$FEATURE_W"
)

# 在同一 Python 进程中进行预检、关闭底层随机换样本、运行原脚本。
# 修改只影响本进程，不修改仓库的 Python 文件。
"$PYTHON_BIN" -u - "${ARGS[@]}" <<'PY' 2>&1 | tee "$OUT/run.log"
import argparse
import json
import math
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
for name in ("config", "dataset_stats", "dataset_dir", "output_dir"):
    parser.add_argument(f"--{name}", required=True)
for name in ("max_samples", "samples_per_episode", "sample_stride", "min_frame_gap",
             "candidate_k_per_episode", "top_k", "save_pair_images", "feature_h", "feature_w"):
    parser.add_argument(f"--{name}", type=int, required=True)
parser.add_argument("--max_proprio_rmse", type=float, required=True)
parser.add_argument("--min_action_rmse", type=float, required=True)
args, _ = parser.parse_known_args()

for name in ("max_samples", "sample_stride", "min_frame_gap", "candidate_k_per_episode",
             "top_k", "feature_h", "feature_w"):
    if getattr(args, name) <= 0:
        raise ValueError(f"{name} 必须大于 0。")
if args.samples_per_episode < 2 or args.max_samples < 2:
    raise ValueError("MAX_SAMPLES 和 SAMPLES_PER_EPISODE 必须至少为 2。")
if not 0 <= args.save_pair_images <= args.top_k:
    raise ValueError("SAVE_PAIR_IMAGES 必须位于 0 到 TOP_K 之间。")
if math.isnan(args.max_proprio_rmse) or args.max_proprio_rmse < 0:
    raise ValueError("MAX_PROPRIO_RMSE 必须非负，允许 inf。")
if not math.isfinite(args.min_action_rmse) or args.min_action_rmse < 0:
    raise ValueError("MIN_ACTION_RMSE 必须是有限非负数。")

config_path = Path(args.config).expanduser().resolve()
stats_path = Path(args.dataset_stats).expanduser().resolve()
data_path = Path(args.dataset_dir).expanduser().resolve()
info_path = data_path / "meta/info.json"
for path in (config_path, stats_path, info_path):
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}\n请修改 sh 文件顶部的路径。")

print("Python:", sys.executable)
print("Config:", config_path)
print("Stats:", stats_path)
print("Dataset:", data_path)
info = json.loads(info_path.read_text(encoding="utf-8"))
print("FPS:", info.get("fps"), "Episodes:", info.get("total_episodes"))
features = info.get("features", {})
for key, value in features.items():
    if key in ("observation.state", "action") or key.startswith("observation.images"):
        print(" ", key, "shape =", value.get("shape"))

# 此启动器针对当前 Astribot raw-25D / model-32D 配置。
for key in ("observation.state", "action"):
    shape = features.get(key, {}).get("shape")
    if shape != [25]:
        raise ValueError(f"{key} 原始维度应为 [25]，实际为 {shape}。请勿混用 pose_6d 数据。")

from omegaconf import OmegaConf
from fastwam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()
cfg = OmegaConf.load(config_path)
train = cfg.data.train
if int(train.get("global_sample_stride", 1)) != 1:
    raise ValueError("当前候选脚本要求 global_sample_stride=1，避免窗口边界与帧距计算失配。")
if str(train.get("norm_stats_source", "meta")) != "pretrained":
    raise ValueError("请使用 norm_stats_source=pretrained 的配置，以使用指定 STATS。")
for kind in ("state", "action"):
    fields = train.shape_meta[kind]
    if len(fields) != 1 or int(fields[0].raw_shape) != 25:
        raise ValueError(f"当前启动器要求 shape_meta.{kind} 为单一 25D 字段。")

fps = float(info.get("fps", 0))
if not math.isfinite(fps) or fps <= 0:
    raise ValueError("meta/info.json 中的 fps 必须为正数。")
print(f"最小帧距：{args.min_frame_gap} 帧 / {args.min_frame_gap / fps:.2f} 秒")
if args.min_frame_gap < int(train.num_frames):
    print("注意：最小帧距小于训练窗口长度，部分候选窗口可能重叠。")

# RobotVideoDataset._get() 内部仍会调用 BaseLerobotDataset.__getitem__。
# 将尝试次数限制为 1，失败时抛错，不实际读取一个随机替代样本。
import fastwam.datasets.lerobot.base_lerobot_dataset as base
base.MAX_GETITEM_ATTEMPT = 1
print("预检通过；底层随机换样本重试已禁用。开始离线筛选。", flush=True)

sys.argv[0] = "tools/find_history_ambiguity_pairs.py"
runpy.run_path(sys.argv[0], run_name="__main__")

result = Path(args.output_dir) / "pairs.json"
payload = json.loads(result.read_text(encoding="utf-8"))
summary = payload.get("summary", {})
if int(summary.get("usable_samples", 0)) == 0:
    raise RuntimeError("未读取到有效样本；请先查看日志中的数据加载错误。")
print("有效样本数：", summary.get("usable_samples"))
print("读取错误数：", summary.get("skipped_error", 0))
print("保存候选数：", len(payload.get("pairs", [])))
if not payload.get("pairs"):
    print("没有候选通过筛选；不能据此断言任务不需要历史。请检查采样范围和阈值。")
print("注意：这里只筛选候选，未做 WAM 推理，也不自动判定历史歧义。")
PY

printf '\n完成。结果目录：%s\n' "$OUT"
printf '  pairs.csv       相似度 / proprio差异 / GT动作差异\n'
printf '  pairs.json      统计信息与候选来源\n'
printf '  pair_images/    左右对比图片\n'
printf '  run.log         完整运行日志\n'
printf '  launcher.sh     本次启动脚本副本（记录路径和参数）\n'
