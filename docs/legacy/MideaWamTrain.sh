# 必填，严格按照模板值
export NCCL_IB_HCA=mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

# 以下可选
export NCCL_P2P_LEVEL=NVL
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
# ====================================================
# 多机参数：从 Volcano 注入的变量换算
#   WORLD_SIZE    = 总 GPU 数（Volcano 注入，如 2机×16卡=32）
#   VC_TASK_INDEX = 当前节点编号（Volcano 注入，0=master, 1,2,...=worker）
#   /etc/volcano/worker.host = 所有节点 IP 列表，第一行为 master
# ====================================================
NPROC_PER_NODE=16                                                       # 每节点 GPU 数，按硬件修改
export NNODES=$(( ${WORLD_SIZE:-128} / NPROC_PER_NODE ))                 # 总节点数 = 总GPU / 每节点GPU
export NODE_RANK="${VC_TASK_INDEX:-0}"                                  # 当前节点编号，Volcano 注入
export MASTER_ADDR=$(awk 'NR==1{print $1}' /etc/volcano/worker.host)   # hostfile 第一行 IP 为 master
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "===== 关键环境变量确认 ====="
echo "NNODES=${NNODES}  NODE_RANK=${NODE_RANK}  MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "==========================================="

# ====================================================
# 诊断：打印平台实际暴露的 CUDA 设备数
# 若输出 < 16，说明容器资源限制导致只看到部分卡
# ====================================================
CUDA_VISIBLE_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "unknown")
echo "[诊断] torch.cuda.device_count() = ${CUDA_VISIBLE_COUNT}"
NPROC_PER_NODE=${CUDA_VISIBLE_COUNT}   # 自动按实际可见卡数启动进程
echo "[诊断] 实际使用 NPROC_PER_NODE = ${NPROC_PER_NODE}"

cd /data/share/1919650160032350208/wzy/FastWAM

# train.py 深度依赖 accelerate.Accelerator，必须通过 accelerate launch 启动
bash scripts/train_zero2.sh ${WORLD_SIZE} task=pick_place_1e-4