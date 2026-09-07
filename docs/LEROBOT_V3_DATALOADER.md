# LeRobot v3.0 数据格式支持说明

本文档说明如何在 FastWAM 中使用 LeRobot **v3.0** 格式的数据集进行训练。

---

## v3.0 与 v2.1 格式差异

| 项目 | v2.1 | v3.0 |
|------|------|------|
| 数据文件 | `data/chunk-XXX/episode_XXXXXX.parquet`（每集一个） | `data/chunk-XXX/file-XXX.parquet`（多集共享） |
| 视频文件 | `videos/chunk-XXX/{key}/episode_XXXXXX.mp4` | `videos/{key}/chunk-XXX/file-XXX.mp4` |
| episode 元数据 | `meta/episodes.jsonl` | `meta/episodes/chunk-XXX/file-XXX.parquet` |
| task 文件 | `meta/tasks.jsonl` | `meta/sub_tasks.jsonl` |
| episode 统计 | `meta/episodes_stats.jsonl` | 内嵌在 episode 元数据 parquet 中，无独立文件 |
| 全局统计 | `meta/stats.json` | `meta/stats.json`（同） |

---

## 目录结构（数据组织方式）

支持以下三种结构，会**自动识别**，无需手动配置：

### 方式 1：单个数据集目录
```
/data/my_dataset/
├── meta/
│   ├── info.json
│   ├── sub_tasks.jsonl       # v3.0
│   ├── stats.json
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── observation.images.cam_high/
        └── chunk-000/
            └── file-000.mp4
```

### 方式 2：根目录下含多个子数据集（自动扫描）
```
/data/Midea_DATa/
├── folding_napkin_olf_20250930_compressed/   # 含 meta/info.json，自动识别
│   └── ...
└── plastic_bag_50hz_c72_white_XXX/           # 含 meta/info.json，自动识别
    └── ...
```

### 方式 3：根目录下含 `dataloader.json`（推荐）
```
/data/Midea_DATa/
├── dataloader.json           # 指定子数据集路径列表

```

`dataloader.json` 格式：
```json
{
  "dataset_paths": [
    "/data/share/.../folding_napkin_olf_20250930_compressed",
    "/data/share/.../plastic_bag_50hz_c72_white_XXX"
  ],
  "description": "描述信息（可选）"
}
```

> **优先级**：`meta/info.json`（单集）> `dataloader.json` > 扫描子目录

---

## 训练前准备步骤
直接查看(MideaWamTrain.sh)

### 第一步：预计算文本嵌入

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/precompute_text_embeds.py task=pick_place_1e-4
```

结果缓存至 `data/text_embeds_cache/pick_place/`（由配置中 `text_embedding_cache_dir` 指定）。

### 第二步：预计算归一化统计（避免分布式训练超时）

```bash
python scripts/precompute_stats.py \
  task=pick_place_1e-4 \
  --output runs/pick_place_stats.json
```

> ⚠️ **为什么需要这一步？**
> 在多卡分布式训练时，stats 计算可能耗时超过 NCCL 通信超时阈值（默认 30 分钟），
> 导致 rank1 等待超时退出。预先计算并缓存可完全避免此问题。

### 第三步：启动训练

```bash
bash scripts/train_zero1.sh 2 task=pick_place_1e-4
```

---

## 配置文件说明

训练配置位于 `configs/data/`，以 `pick_place.yaml` 为例，关键字段：

```yaml
train:
  dataset_dirs:
    - /data/Midea_data           # 支持根目录（会自动展开子数据集）
  pretrained_norm_stats: null    # 填入预计算的 stats JSON 路径
  text_embedding_cache_dir: ./data/text_embeds_cache/pick_place
```

将 `dataset_dirs` 改为根目录路径后，代码会自动通过 `dataloader.json` 或扫描子目录展开所有数据集。

---

## 代码修改说明

为支持 v3.0 格式，以下文件做了适配：

| 文件 | 修改内容 |
|------|---------|
| `src/fastwam/datasets/lerobot/lerobot/datasets/utils.py` | `load_episodes()` 支持 `meta/episodes/**/*.parquet`；`load_tasks()` 支持 `sub_tasks.jsonl`；`load_episodes_stats()` v3.0 返回空字典；`get_episode_data_index()` 支持 `dataset_from_index`/`dataset_to_index` |
| `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py` | `LeRobotDatasetMetadata` 增加本地路径检测，跳过 HuggingFace Hub 验证；`get_data_file_path()`/`get_video_file_path()` 支持 v3.0 路径模板；`load_hf_dataset()` v3.0 按 episode index 过滤行；timestamp 列兼容 HF Column 和 list 两种格式 |
| `src/fastwam/datasets/lerobot/base_lerobot_dataset.py` | `_expand_dataset_dir()` 新增：优先读 `dataloader.json`，其次扫描子目录 |
| `scripts/precompute_text_embeds.py` | `_expand_dataset_dirs()` 支持根目录展开；`_read_unique_prompts()` 支持 `sub_tasks.jsonl` |
| `scripts/precompute_stats.py` | 新增脚本，用 Hydra 加载完整配置，单进程预计算归一化 stats |
