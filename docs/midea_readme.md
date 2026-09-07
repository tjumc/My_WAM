# Midea FastWAM 使用记录

## 1. 计算 Text Embedding

### Bash

推荐直接使用：

```bash
bash scripts/precompute_text.sh
```

### 参数配置

参数写在 `scripts/precompute_text.sh` 里：

- `--dataset-yaml`: 数据集路径列表 YAML。
- `--cache-dir`: text embedding 输出目录，需要和训练 YAML 里的 `data.train.text_embedding_cache_dir` 一致。
- `--context-len`: 文本 token 长度，需要和训练 YAML 里的 `data.train.context_len` 一致。
- `--batch-size`: 每张卡编码 text 的 batch size。
- `--skip-existing`: 已经存在的 cache 文件不重复计算。
- `--dry-run`: 只检查会读取到多少条 instruction，不加载模型。

cache 文件名由完整 prompt 的 SHA256 hash 决定，格式为：

```text
{sha256(prompt)}.t5_len{context_len}.wan22ti2v5b.pt
```

### 数据 YAML

Text embedding 只读取 YAML 里的 `dataset_dirs` 字段，推荐使用 list 写法：

```yaml
dataset_dirs: [
  "/data/path/dataset_a",
  "/data/path/dataset_b",
  "/data/path/dataset_c",
]
```

默认文件：

```bash
configs/data/post_train.yaml
```

每个路径可以是一个 LeRobot 数据集目录，目录下需要包含 `meta/info.json`。

## 2. 计算 Stats

### Bash

推荐直接使用：

```bash
bash scripts/precompute_stats.sh
```

### 参数配置

参数写在 `scripts/precompute_stats.sh` 里：

- `--dataset-yaml`: stats 配置和数据集路径列表 YAML。
- `--output-dir`: stats JSON 输出路径。
- `--skip-quantile`: 不计算 q01/q99，速度更快；当前使用 z-score 时推荐开启。
- `--profile`: 输出耗时统计。
- `--profile-interval`: profile 模式下每隔多少秒打印一次耗时摘要。
- `--num-workers`: 多线程读取/计算；如果没有收益就保持默认。

### 数据 YAML

和 text embedding 使用同一个数据 YAML。Stats 需要额外配置时间窗口、`shape_meta` 和 processor：

```yaml
num_frames: 33
obs_size: 33
action_size: 32
global_sample_stride: 1

shape_meta:
  images:
    - key: cam_high
      raw_shape: [3, 480, 640]
      shape: [3, 240, 320]
    - key: cam_left_wrist
      raw_shape: [3, 480, 640]
      shape: [3, 240, 320]
    - key: cam_right_wrist
      raw_shape: [3, 480, 640]
      shape: [3, 240, 320]
  action:
    - key: default
      raw_shape: 14
      shape: 14
  state:
    - key: default
      raw_shape: 14
      shape: 14

processor:
  _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
  shape_meta: ${shape_meta}
  num_obs_steps: ${num_frames}
  num_output_cameras: 3
  action_output_dim: 14
  proprio_output_dim: 14
  action_state_transforms: null
  use_stepwise_action_norm: false
  norm_default_mode: z-score
  norm_exception_mode: null
  action_state_merger:
    _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
  train_transforms:
    - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
    - _target_: torchvision.transforms.Resize
      size: [240, 320]
  val_transforms:
    - _target_: fastwam.datasets.lerobot.transforms.image.ToTensor
    - _target_: torchvision.transforms.Resize
      size: [240, 320]

dataset_dirs: [
  "/data/path/dataset_a",
  "/data/path/dataset_b",
  "/data/path/dataset_c",
]
```

默认文件：

```bash
configs/data/post_train.yaml
```

多个数据集会合并计算成一份 stats。所有数据集的 `fps` 和 state/action 结构需要一致。

## 3. 启动训练

### Bash

单机多卡：

```bash
bash scripts/train_ds_single.sh
```

多机多卡：

```bash
bash scripts/train_ds.sh
```

### 参数配置

参数写在 `scripts/train_ds_single.sh` 或 `scripts/train_ds.sh` 里。

脚本顶部配置：

- `CONFIG`: 训练 YAML，例如 `configs/train/robotwin.yaml` 或 `configs/train/post_train.yaml`。
- `ACCELERATE_CONFIG`: accelerate/deepspeed 配置。
- `TASK_NAME`: 当前任务名，用于默认输出目录。
- `OUTPUT_DIR`: 训练输出目录。
- `RESUME`: 可选恢复路径，默认空字符串表示从头训练。

主要训练参数直接写在 `accelerate launch` 命令后面：

- `--batch_size`: 单进程 batch size。
- `--num_workers`: 每个训练进程的 DataLoader workers。
- `--learning_rate`: 学习率。
- `--num_epochs`: 训练 epoch 数。
- `--mixed_precision`: 精度，当前为 `bf16`。
- `--save_every`: checkpoint 间隔。
- `--eval_every`: eval 间隔。
- `--resume "${RESUME}"`: 可选恢复路径。空字符串表示不恢复，`.pt` 表示只加载权重，`state/step_xxx` 表示完整续训。

`RESUME` 的取值含义：

| 目标 | `RESUME` 配置 | 效果 |
| --- | --- | --- |
| 从头训练 | `""` | 不加载任何 checkpoint |
| 只加载权重 | `runs/xxx/checkpoints/weights/step_005000.pt` | 只加载模型权重，不恢复 optimizer、scheduler、global step |
| 完整断点续训 | `runs/xxx/checkpoints/state/step_005000` | 恢复模型、optimizer、scheduler、global step 和 dataloader 进度 |

注意：`--resume` 是 `scripts/train.py` 的参数，需要保留在 `accelerate launch` 命令中 `scripts/train.py` 后面。

多机脚本里需要确认：

- `NNODES`: 总机器数。
- `GPUS_PER_NODE`: 单机 GPU 数。
- `MASTER_ADDR`: 默认从 `/etc/volcano/worker.host` 第一行读取。
- `NODE_RANK`: 默认从 `NODE_RANK` / `VC_TASK_INDEX` / `MACHINE_RANK` 读取。

### 训练 YAML

训练只读取 train YAML：

- `configs/train/pretrain.yaml`
- `configs/train/robotwin.yaml`
- `configs/train/post_train.yaml`

需要重点确认：

```yaml
data:
  train:
    dataset_dirs: [
      "/data/path/dataset_a",
      "/data/path/dataset_b",
    ]
    text_embedding_cache_dir: /path/to/text_embeds_cache
    context_len: 128
    pretrained_norm_stats: runs/robotwin_stats.json

  val:
    dataset_dirs: ${data.train.dataset_dirs}
    text_embedding_cache_dir: ${data.train.text_embedding_cache_dir}
    context_len: ${data.train.context_len}
    pretrained_norm_stats: ${data.train.pretrained_norm_stats}
```

`text_embedding_cache_dir` 和 `context_len` 需要和第一步 text embedding 保持一致。`pretrained_norm_stats` 需要指向第二步 stats 的输出文件。`val` 默认复用 `train` 的数据路径和缓存配置。
