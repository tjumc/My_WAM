# FastWAM readthrough notes

## Multi-node ZeRO-1 hang: Gloo loopback binding

Date: 2026-05-18

The multi-node `train_zero1_multi.sh` failure is happening before the first
training step. The traceback enters `RobotVideoDataset.__init__`, where
`PartialState().is_main_process` initializes distributed state. The important
log lines are:

```text
ProcessGroupGloo.cpp: Unable to resolve hostname ... Using the loopback address
Gloo connectFullMesh failed ... remote=[127.0.0.1]:15168$9
```

So NCCL is not the first blocker here. Gloo is choosing `127.0.0.1` for a
multi-node process group, then ranks on different machines try to connect to
each other's localhost addresses. That cannot work across nodes.

The launcher now sets a shared `NETWORK_IFNAME="eth0"` and exports both
`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` from it. If the cluster's
cross-node interface is not `eth0`, change `NETWORK_IFNAME` to the interface
that owns the reachable node IP, such as `bond0` or `ib0`.

On rerun, the first check is that the log no longer contains:

```text
Using the loopback address as fallback
remote=[127.0.0.1]
```

### Volcano wrapper note

An experimental `scripts/train_zero1ds_volcano.sh` wrapper was added while
checking the known working Hydra path. It keeps the original training call:

```bash
bash scripts/train_zero1ds.sh 8 task=pick_place_1e-4 resume=./checkpoints/step_055000.pt
```

and centralizes the Volcano-specific environment setup:

- reads `MASTER_ADDR` from `/etc/volcano/worker.host`
- maps `VC_TASK_INDEX` to `NODE_RANK`
- exports `NNODES`, `MASTER_ADDR`, and `MASTER_PORT`
- sets both `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`
- leaves killing old python processes off by default

The direct launcher below is the intended path for the no-Hydra work.

### Direct multi launcher update

The intended direct-entry launcher is `scripts/train_zero1_multi.sh`. It now
uses the same export style as the working Volcano script:

- `MASTER_ADDR` is read from `/etc/volcano/worker.host`
- `MASTER_PORT` defaults to `29604`
- `VC_TASK_INDEX`, `NODE_RANK`, or `SLURM_NODEID` is mapped to `MACHINE_RANK`
- `NCCL_SOCKET_IFNAME` defaults to `eth0,en0`
- `GLOO_SOCKET_IFNAME` defaults to `eth0` to avoid loopback binding
- `TORCH_NCCL_BLOCKING_WAIT`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`,
  `NCCL_IB_GID_INDEX`, `NCCL_TIMEOUT`, and `NCCL_SOCKET_TIMEOUT_MS` follow the
  known working script

This train launcher runs `scripts/train_direct.py` with
`configs/train/post_train.yaml`.

## RoboTwin text precompute: why 460516 prompts

Date: 2026-05-15

Question: running

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

shows a progress total around `460516`, but the expected RoboTwin scale is closer to `50 x 550` episodes if there is one instruction per episode.

### What the number means

The progress bar total is not the number of episodes. It is `len(prompts)` after `scripts/precompute_text_embeds.py` scans configured dataset metadata and de-duplicates prompt strings.

The relevant code path is:

1. Hydra loads `configs/train.yaml` plus `task=robotwin_uncond_3cam_384_1e-4`.
2. The task config overrides `/data: robotwin`.
3. `scripts/precompute_text_embeds.py` recursively finds all `dataset_dirs` under `cfg.data`.
4. `_read_unique_prompts()` expands those dirs into LeRobot dataset dirs.
5. For each dataset dir, it reads `meta/tasks.parquet` first; if absent, it falls back to `meta/tasks.jsonl`.
6. For `tasks.parquet`, the current script assumes the task text is stored in the parquet dataframe index.
7. It wraps each text with `DEFAULT_PROMPT`, hashes it, and stores one T5 cache file per unique prompt.

So a progress total of `460516` means the script believes there are `460516` unique prompt strings to encode after its parsing and de-duplication.

### Most likely causes

The most likely issue is that the RoboTwin `meta/tasks.parquet` is not a compact task table. It may contain frame-level, subtask-level, or otherwise per-sample task entries instead of one entry per task/episode.

Another likely issue is parquet schema mismatch. The current code expects LeRobot v3-style:

```text
index = task text
column = task_index
```

If the real task text is in a column such as `task`, but the dataframe index is a numeric `RangeIndex`, the script will accidentally treat `0, 1, 2, ...` as task text. In that case the count can become close to the number of rows in `tasks.parquet`, and the generated prompts would literally contain numeric strings.

The script also scans both `data.train.dataset_dirs` and `data.val.dataset_dirs`. In the current `configs/data/robotwin.yaml`, train and val point to the same root, but one path has a trailing slash and the other does not. This can duplicate the metadata scan count in logs. It should not double the progress total when task strings are identical, because prompts are de-duplicated, but it is still worth cleaning up.

### Why this matters for training

This is not only a precompute display issue. Training uses the cached text embedding looked up by the exact instruction string:

1. `LeRobotDataset.__getitem__()` maps `task_index` to `self.meta.tasks[task_idx]`.
2. `FastWAMProcessor.augment_instruction()` optionally splits bilingual `zh@en` text and returns the low-level instruction by default.
3. `RobotVideoDataset._get()` wraps it with `DEFAULT_PROMPT`.
4. `_get_cached_text_context()` hashes that prompt and loads the `.pt` cache.

So precompute must generate exactly the same prompt strings that training will request. If `tasks.parquet` is malformed or hugely expanded, the fix should align both precompute and training metadata, not just reduce the progress bar.

### Useful checks on the real dataset

The current shell could not access `/data/share/1919650160032350208/wzy/FastWAM/data/robotwin2.0/robotwin2.0`, so run these on the machine where the data is mounted:

```bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/data/share/1919650160032350208/wzy/FastWAM/data/robotwin2.0/robotwin2.0")
p = root / "meta" / "tasks.parquet"
t = pq.read_table(p)
print("rows:", t.num_rows)
print("columns:", t.column_names)
print("schema:", t.schema)
print("first rows:", t.slice(0, 5).to_pydict())

df = t.to_pandas()
print("index type:", type(df.index), "index name:", df.index.name)
print("index sample:", list(df.index[:10]))
print(df.head(10))
PY
```

Also check how many frame/data parquet rows point to task ids:

```bash
find /data/share/1919650160032350208/wzy/FastWAM/data/robotwin2.0/robotwin2.0 -path '*/data/*.parquet' | head
```

If `tasks.parquet` has `460516` rows, inspect whether `task_index` itself is unique per frame or whether task strings are stored in another column.

### Practical fixes

If this is intended to be unconditional training, use the same override for both precompute and training:

```bash
python scripts/precompute_text_embeds.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  override_instruction="robot manipulation task"
```

Then train with matching overrides:

```bash
python scripts/train.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  data.train.override_instruction="robot manipulation task" \
  data.val.override_instruction="robot manipulation task"
```

If training should be task-conditioned, fix the LeRobot metadata so `meta/tasks.parquet` is a compact mapping from task text to `task_index`, and the frame-level data parquet files reference those indices. For RoboTwin, depending on the intended semantics, that compact table should likely be at task-type scale or episode-scale, not frame-scale.

For script robustness, `_read_unique_prompts()` should validate the parquet schema before encoding. In particular, it should detect a numeric `RangeIndex` with many rows and refuse to treat it as task text unless a real task-text column or `__index_level_0__` exists.

## Training flow from scratch

This section follows the current RoboTwin task:

```bash
bash scripts/train_zero2.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

The selected task config is `configs/task/robotwin_uncond_3cam_384_1e-4.yaml`, which overrides:

```yaml
data: robotwin
model: fastwam
```

So the concrete stack is:

```text
scripts/train_zero2.sh
  -> accelerate launch
  -> scripts/train.py
  -> fastwam.runtime.run_training(cfg)
  -> instantiate(cfg.model): fastwam.runtime.create_fastwam(...)
  -> build_datasets(cfg.data): RobotVideoDataset(...)
  -> Wan22Trainer.train()
  -> model.training_loss(sample)
```

### 1. Launch layer

`scripts/train_zero2.sh` is only the launcher. It does not build the model or dataset itself.

It reads:

- first CLI arg: `NPROC_PER_NODE`, the number of local GPU processes;
- optional env vars: `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`;
- Hydra overrides after the first arg, for example `task=robotwin_uncond_3cam_384_1e-4`.

It derives:

- `TASK_BASENAME=robotwin_uncond_3cam_384_1e-4`;
- `RUN_ID`, synchronized by `torch.distributed.TCPStore` for multi-node runs;
- `output_dir=./runs/${TASK_BASENAME}/${RUN_ID}`;

Then it calls:

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "${EXTRA_ARGS[@]}"
```

`scripts/accelerate_configs/accelerate_zero2_ds.yaml` selects DeepSpeed. The actual ZeRO config is `scripts/ds_configs/ds_zero2_config.json`:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"}
  }
}
```

So ZeRO-2 shards optimizer states and gradients, with no CPU/NVMe offload.

### 2. Hydra config composition

`scripts/train.py` uses:

```python
@hydra.main(config_path="../configs", config_name="train")
```

The base `configs/train.yaml` has:

```yaml
defaults:
  - _self_
  - data: null
  - model: null
  - task: null
```

When passing `task=robotwin_uncond_3cam_384_1e-4`, Hydra loads `configs/task/robotwin_uncond_3cam_384_1e-4.yaml`, which contains:

```yaml
defaults:
  - override /data: robotwin
  - override /model: fastwam
  - _self_
```

Therefore the final resolved config is:

- general training options from `configs/train.yaml`;
- dataset options from `configs/data/robotwin.yaml`;
- model options from `configs/model/fastwam.yaml`;
- task-level overrides such as `batch_size`, `learning_rate`, `num_epochs`, `save_every`.

`run_training()` writes this resolved config to:

```text
${output_dir}/config.yaml
```

### 3. Runtime setup

`fastwam.runtime.run_training(cfg)` does the top-level orchestration:

1. Set logging.
2. Register `cfg.output_dir` as the working directory.
3. Save resolved `config.yaml`.
4. Resolve model device from `LOCAL_RANK`.
5. Convert `mixed_precision` to model dtype:
   - `bf16` -> `torch.bfloat16`;
   - `fp16` -> `torch.float16`;
   - `no` -> `torch.float32`.
6. Instantiate model by `instantiate(cfg.model, model_dtype=..., device=...)`.
7. Instantiate train and validation datasets by `build_datasets(cfg.data)`.
8. Create `Wan22Trainer`.
9. Call `trainer.train()`.

### 4. Model construction

`configs/model/fastwam.yaml` points to:

```yaml
_target_: fastwam.runtime.create_fastwam
```

`create_fastwam()` calls:

```python
FastWAM.from_wan22_pretrained(...)
```

The constructed model contains:

- `video_expert`: Wan2.2 video DiT, loaded from `Wan-AI/Wan2.2-TI2V-5B`;
- `action_expert`: `ActionDiT`, loaded from `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`;
- `mot`: mixture-of-transformers wrapper over video/action experts;
- `vae`: Wan2.2 VAE;
- `text_encoder/tokenizer`: not loaded for this config because `load_text_encoder: false`;
- `proprio_encoder`: `Linear(14, 4096)` because `proprio_dim=${data.train.processor.proprio_output_dim}=14`;
- separate flow-matching schedulers for video and action.

Important dimensions from `fastwam.yaml`:

```text
video expert:
  latent in/out dim: 48
  hidden dim: 3072
  text dim: 4096
  patch size: [1, 2, 2]
  layers: 30
  heads: 24
  head dim: 128
  video_attention_mask_mode: first_frame_causal
  action_conditioned: false

action expert:
  action dim: 14
  hidden dim: 1024
  text dim: 4096
  layers: 30
  heads: 24
  head dim: 128
```

Even though video/action hidden dims differ, their attention projection width is compatible:

```text
num_heads * attn_head_dim = 24 * 128 = 3072
```

That is why MoT can concatenate video and action Q/K/V along the sequence dimension.

### 5. Dataset construction

`build_datasets(cfg.data)` instantiates:

```yaml
fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
```

For RoboTwin, the key config values are:

```text
num_frames: 33
action_video_freq_ratio: 4
video_size: [384, 320]
concat_multi_camera: robotwin
context_len: 128
action/state dim: 14
num cameras: 3
normalization: z-score, use_stepwise_action_norm=False
```

Inside `RobotVideoDataset.__init__()`:

1. It creates `BaseLerobotDataset`.
2. `BaseLerobotDataset` expands `dataset_dirs`:
   - direct LeRobot dataset if `meta/info.json` exists;
   - otherwise `dataloader.json`;
   - otherwise subdirectories with `meta/info.json`.
3. It builds LeRobot metadata for every dataset.
4. It checks all datasets have the same FPS.
5. It splits train/val episodes by `val_set_proportion`.
6. It builds `MultiLeRobotDataset` with delta timestamps:
   - images: 33 timestamps;
   - state: 33 timestamps;
   - action: 32 timestamps.
7. It creates/loads normalization stats.
8. It attaches `FastWAMProcessor`.

If `pretrained_norm_stats` is set, stats are loaded from JSON. If it is null for train, the dataset iterates episodes to calculate stats and saves `dataset_stats.json` under the run directory.

### 6. Single-sample data path

When the trainer asks for one sample, the path is:

```text
DataLoader
  -> RobotVideoDataset.__getitem__
  -> RobotVideoDataset._get
  -> BaseLerobotDataset.__getitem__
  -> MultiLeRobotDataset.__getitem__
  -> FastWAMProcessor.preprocess
  -> RobotVideoDataset final video/text cache packaging
```

#### 6.1 LeRobot sample

`BaseLerobotDataset.__getitem__()` first pulls a temporal window from `MultiLeRobotDataset`.

Conceptually, before FastWAM processing, it has:

```text
images:
  cam_high:        [33, 3, H, W]
  cam_left_wrist:  [33, 3, H, W]
  cam_right_wrist: [33, 3, H, W]

state:
  default: [33, 14]

action:
  default: [32, 14]

task:
  string from task_index -> meta.tasks[task_index]
```

It also carries padding masks such as `action_is_pad`, `state_is_pad`, `image_is_pad`.

#### 6.2 Processor preprocessing

`FastWAMProcessor.preprocess()` does:

1. Instruction:
   - takes `data["task"]`;
   - if text is `zh@en`, selects English by default because `use_zh_instruction=False`;
   - drops high-level instruction by default because `drop_high_level_prob=1.0`;
   - returns the low-level instruction string.

2. Image transforms:
   - `ToTensor`;
   - resize to `[240, 320]`;
   - stack cameras.

3. Action/state transform:
   - optional action-state transforms are skipped because `action_state_transforms: null`;
   - z-score normalization is applied using dataset stats;
   - `ConcatLeftAlign` merges dict fields into dense tensors.

After processor:

```text
pixel_values: [3, 33, 3, 240, 320]
action:       [32, 14]
proprio:      [33, 14]
instruction:  str
```

#### 6.3 RobotVideoDataset video packaging

`RobotVideoDataset._get()` then samples video frames:

```python
video_sample_indices = range(0, 33, 4)
```

So 33 original frames become:

```text
T_video = 9
indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
```

For RoboTwin 3-camera layout:

```text
cam_high        -> resize to [256, 320]
cam_left_wrist  -> resize to [128, 160]
cam_right_wrist -> resize to [128, 160]
bottom = concat(left, right) -> [128, 320]
video = concat(top, bottom)  -> [384, 320]
```

Then it normalizes image values to `[-1, 1]` and permutes:

```text
video: [3, 9, 384, 320]
```

It also trims proprio to align with action:

```text
action:  [32, 14]
proprio: [32, 14]
```

It wraps the instruction:

```python
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
```

Then it loads the cached T5 embedding:

```text
context:      [128, 4096]
context_mask: [128]
```

The single training sample returned by the dataset is:

```text
video:          [3, 9, 384, 320]
action:         [32, 14]
proprio:        [32, 14]
prompt:         str
context:        [128, 4096]
context_mask:   [128]
image_is_pad:   [9]
action_is_pad:  [32]
proprio_is_pad: [32]
```

After DataLoader collation, the trainer sees:

```text
video:          [B, 3, 9, 384, 320]
action:         [B, 32, 14]
proprio:        [B, 32, 14]
context:        [B, 128, 4096]
context_mask:   [B, 128]
image_is_pad:   [B, 9]
action_is_pad:  [B, 32]
```

### 7. Trainer initialization

`Wan22Trainer.__init__()`:

1. Creates `Accelerator` with:
   - `gradient_accumulation_steps`;
   - `mixed_precision`;
   - `step_scheduler_with_optimizer=False`.
2. Checks dataset length consistency across ranks.
3. Freezes non-trainable modules:
   - `model.eval()`;
   - `model.requires_grad_(False)`;
   - `model.dit.train()` and `model.dit.requires_grad_(True)`;
   - `proprio_encoder.train()` and `requires_grad_(True)` if present.
4. Builds AdamW over:
   - `model.dit.parameters()`, where `model.dit` is the MoT wrapper;
   - `model.proprio_encoder.parameters()` if present.
5. Builds DataLoader with `ResumableEpochSampler`.
6. Estimates total optimizer steps.
7. Builds LR scheduler:
   - 5% warmup;
   - cosine decay by default.
8. Creates checkpoint/eval directories.
9. Calls `accelerator.prepare(model, optimizer, train_loader, scheduler)`.
10. Optionally resumes checkpoint.

Important: VAE and text encoder are frozen. The trainable part is MoT, which contains both the video expert and action expert transformer blocks, plus the proprio encoder.

### 8. One training step

`Wan22Trainer.train()` loops until `global_step == max_steps`.

For every micro-batch:

```python
with accelerator.accumulate(model):
    with accelerator.autocast():
        loss, loss_dict = model.training_loss(sample)
    accelerator.backward(loss)
```

When gradients are synchronized:

```python
clip_grad_norm_
optimizer.step()
scheduler.step()
optimizer.zero_grad()
global_step += 1
```

Then it may:

- log training loss every `log_every`;
- run evaluation every `eval_every`;
- save checkpoint every `save_every`;
- save a final checkpoint when `max_steps` is reached.

### 9. FastWAM.training_loss()

This is the core model computation.

#### 9.1 build_inputs()

Input batch:

```text
video:        [B, 3, 9, 384, 320]
action:       [B, 32, 14]
proprio:      [B, 32, 14]
context:      [B, 128, 4096]
context_mask: [B, 128]
```

The VAE encodes video without gradient:

```text
input_latents: [B, 48, 3, 48, 40]
```

Explanation:

- video frames: `9`;
- VAE temporal downsample keeps first frame and groups the remaining 8 frames by 4, so latent time is `1 + 8/4 = 3`;
- spatial downsample is effectively `/8`, so `384/8=48`, `320/8=40`;
- Wan latent channels are `48`.

Because `fuse_vae_embedding_in_latents=True`, the first latent frame is used as an image condition:

```text
first_frame_latents: [B, 48, 1, 48, 40]
```

The proprio encoder appends the first proprio state as one extra context token:

```text
proprio[:, 0, :] -> [B, 14]
Linear(14, 4096) -> [B, 1, 4096]
context           -> [B, 129, 4096]
context_mask      -> [B, 129]
```

Action remains:

```text
action: [B, 32, 14]
```

#### 9.2 Flow matching noise

Video and action have independent random timesteps:

```text
timestep_video:  [B]
timestep_action: [B]
```

The scheduler samples `sigma` through a shift transform and creates:

```text
noisy_video  = (1 - sigma_video)  * input_latents + sigma_video  * noise_video
target_video = noise_video - input_latents

noisy_action  = (1 - sigma_action) * action + sigma_action * noise_action
target_action = noise_action - action
```

Then the first video latent step is restored to the clean first-frame condition:

```python
latents[:, :, 0:1] = first_frame_latents
```

#### 9.3 Video pre-DiT

`video_expert.pre_dit()` patchifies the latent video.

Input:

```text
latents: [B, 48, 3, 48, 40]
```

With patch size `[1, 2, 2]`:

```text
latent temporal grid: 3
latent spatial grid: 48 x 40
patch grid: 3 x 24 x 20
tokens_per_frame = 24 * 20 = 480
video_seq_len = 3 * 480 = 1440
```

Output:

```text
video_tokens: [B, 1440, 3072]
video_context: [B, 129, 3072]
```

Since `action_conditioned: false`, the video branch does not cross-attend to action tokens in this config.

#### 9.4 Action pre-DiT

`action_expert.pre_dit()` embeds action tokens:

```text
noisy_action:  [B, 32, 14]
action_tokens: [B, 32, 1024]
action_context: [B, 129, 1024]
```

#### 9.5 MoT mixed attention

The MoT wrapper runs 30 layers. At each layer:

1. Video expert produces Q/K/V from `[B, 1440, 3072]`.
2. Action expert produces Q/K/V from `[B, 32, 1024]`.
3. Both are projected to attention width:

```text
24 heads * 128 head dim = 3072
```

4. Q/K/V are concatenated along sequence:

```text
joint sequence length = 1440 + 32 = 1472
attention mask: [1472, 1472]
```

The joint attention mask is:

```text
video query -> video keys only
action query -> action keys + first-frame video keys
video query -> action keys: disabled
action query -> future video keys: disabled
```

Then the mixed attention output is split back:

```text
video slice:  [B, 1440, 3072]
action slice: [B, 32, 3072]
```

Each expert applies its own output projection, cross-attention to text/proprio context, and MLP.

#### 9.6 Post-DiT and losses

Video post-DiT unpatchifies:

```text
pred_video: [B, 48, 3, 48, 40]
```

Action post-DiT projects:

```text
pred_action: [B, 32, 14]
```

Because the first latent frame is a clean condition, video loss drops the first latent time:

```text
pred_video loss region:   [B, 48, 2, 48, 40]
target_video loss region: [B, 48, 2, 48, 40]
```

`image_is_pad [B,9]` is compressed by VAE temporal factor 4:

```text
tail frames: [B, 8]
latent future mask: [B, 2]
```

Video loss is MSE over latent channels/spatial dims, masked by valid latent time steps, then weighted by the flow-matching timestep weight.

Action loss is:

```text
MSE(pred_action, target_action) -> [B, 32]
```

Then it masks padded action steps using `action_is_pad [B,32]`.

Total loss:

```text
loss_total = lambda_video * loss_video + lambda_action * loss_action
```

In `configs/model/fastwam.yaml`, only `lambda_action: 1.0` is explicitly set, so `lambda_video` defaults to `1.0` in `create_fastwam()`. This means the current config trains both video and action losses.

### 10. Evaluation flow

Every `eval_every` steps, `Wan22Trainer.evaluate()`:

1. Picks one random val sample per rank.
2. Computes `model.training_loss(sample)` as validation loss.
3. Uses the first video frame as input image.
4. Calls `model.infer(...)`.
5. Saves a stitched video:

```text
predicted rollout | VAE reconstruction | ground truth
```

6. Computes:
   - rollout vs ground-truth PSNR/SSIM;
   - rollout vs VAE decode PSNR/SSIM;
   - VAE decode vs ground-truth PSNR/SSIM;
   - action L1/L2 after denormalization when action is available.

### 11. Checkpoint flow

Every `save_every` steps:

1. Main process saves model weights:

```text
${output_dir}/checkpoints/weights/step_XXXXXX.pt
```

2. All ranks save Accelerate/DeepSpeed state:

```text
${output_dir}/checkpoints/state/step_XXXXXX/
```

3. The trainer also saves:

```text
trainer_state.json
```

containing:

```json
{
  "global_step": ...,
  "epoch": ...,
  "batch_in_epoch": ...
}
```

This is used to resume optimizer/scheduler/dataloader progress.

### 12. Important training implications

The current RoboTwin `fastwam` config is named `uncond`, but the loss still includes both video and action unless overridden. To train action-only, set:

```bash
model.loss.lambda_video=0.0
```

The video branch is not action-conditioned in `configs/model/fastwam.yaml`:

```yaml
video_dit_config:
  action_conditioned: false
```

So the video loss trains future video prediction from first frame plus text/proprio context, not from GT action.

The action branch sees:

```text
text/proprio context + noisy action tokens + first-frame video tokens
```

It does not attend to future GT video tokens because the MoT mask only allows action queries to attend to first-frame video keys.

The text encoder is not loaded during training. Every prompt that the dataset can emit must already exist in the text embedding cache.

`context_mask` is loaded from cache, padded context rows are zeroed, then the dataset sets `context_mask` to all ones. That means zero-padded T5 rows are still visible as valid cross-attention positions, matching the code comment's intended Wan2.2 behavior, but it is worth remembering when comparing to a strict padding-mask implementation.

## Normalization stats code locations

The normalization stats path starts in `RobotVideoDataset.__init__()`.

If `pretrained_norm_stats` is not provided and this is the training dataset, it calculates stats:

```python
dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
```

If `pretrained_norm_stats` is provided, it loads stats from JSON instead:

```python
dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
```

For validation/test datasets, `pretrained_norm_stats` is required because the code intentionally avoids calculating stats on validation/test splits.

The actual calculation is in `BaseLerobotDataset.get_dataset_stats(preprocessor)`. It iterates over every episode, loads raw action/state episode tensors, applies `preprocessor.action_state_transform(batch)`, then collects:

```text
state/action stepwise_min
state/action stepwise_max
state/action stepwise_q01
state/action stepwise_q99
state/action stepwise_mean
state/action stepwise_std
state/action global_min
state/action global_max
state/action global_q01
state/action global_q99
state/action global_mean
state/action global_std
```

The stats are then passed to `FastWAMProcessor.set_normalizer_from_stats()`, which constructs `LinearNormalizer`.

For the current RoboTwin config:

```yaml
use_stepwise_action_norm: false
norm_default_mode: z-score
```

So actions use `global_mean/global_std`, states also use `global_mean/global_std`, and `SingleFieldLinearNormalizer.forward()` applies:

```python
x = x * scale + offset
x = torch.clamp(x, -5.0, 5.0)
```

For z-score, this is equivalent to:

```text
x_norm = (x - global_mean) / (global_std + 1e-8)
x_norm = clamp(x_norm, -5, 5)
```

## Dataloader performance review

This review focuses on the current RoboTwin training path:

```text
DataLoader
  -> RobotVideoDataset.__getitem__
  -> RobotVideoDataset._get
  -> BaseLerobotDataset.__getitem__
  -> MultiLeRobotDataset.__getitem__
  -> LeRobotDataset.__getitem__
  -> FastWAMProcessor.preprocess
```

### Highest-impact issue: decoding 33 image frames but using 9

`RobotVideoDataset` is configured with:

```text
num_frames = 33
action_video_freq_ratio = 4
video_sample_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
```

The final model input uses only 9 video frames:

```text
video: [3, 9, 384, 320]
```

However, `BaseLerobotDataset` currently requests `obs_size=33` for images, state, and action/state-aligned observations. That means LeRobot decodes 33 frames per camera, and `FastWAMProcessor.preprocess()` resizes all 33 frames per camera. Only afterwards does `RobotVideoDataset._get()` select the 9 frames.

For RoboTwin 3-camera video, this means roughly:

```text
decoded/resized camera frames per sample: 3 cameras * 33 = 99
actually used camera frames per sample:   3 cameras * 9  = 27
```

This is likely the biggest dataloader-side waste.

A proper fix would separate image temporal sampling from action/state temporal sampling:

```text
image timestamps: 9 frames, stride 4
state timestamps: 33 frames
action timestamps: 32 frames
```

This requires refactoring because `FastWAMProcessor` currently assumes image length equals `num_obs_steps=33`. A clean design would add something like:

```text
num_image_steps = 9
image_sample_stride = 4
num_state_steps = 33
action_horizon = 32
```

Then `BaseLerobotDataset` should generate image delta timestamps using only the 9 required frame offsets.

### Text embedding cache is loaded from disk per sample

`RobotVideoDataset._get_cached_text_context()` calls `torch.load(cache_path)` every sample.

For task-conditioned data with repeated instructions, this is unnecessary repeated small-file I/O and repeated deserialization. For unconditional training, it can be especially wasteful because every sample may use the same prompt.

Recommended fix:

- move `os.makedirs(cache_dir)` to `__init__`;
- add a per-worker in-memory LRU cache keyed by hash/cache path;
- if `override_instruction` is used, preload that one context once.

Caution: the code mutates `context` after loading:

```python
context[~context_mask] = 0.0
context_mask = torch.ones_like(context_mask)
```

If contexts are cached in memory, return cloned tensors or store the already-zeroed context plus all-ones mask to avoid accidental shared mutation.

### Image resize pipeline does extra work

For RoboTwin, the current flow is:

1. `_get_image()` returns uint8 frames.
2. `FastWAMProcessor.preprocess()` applies:

```yaml
ToTensor
Resize [240, 320]
```

to all 33 frames per camera.

3. `RobotVideoDataset._get()` selects 9 frames.
4. For `concat_multi_camera="robotwin"`:

```text
top camera:   [240, 320] -> [256, 320]
wrist camera: [240, 320] -> [128, 160]
```

5. It concatenates to `[384, 320]`.
6. It calls final resize/crop to `[384, 320]`, which is usually a no-op in shape but still invokes transform code.

Optimization options:

- sample the 9 image frames before image transforms;
- build the RoboTwin mosaic directly from raw decoded frames;
- avoid resizing all cameras to `[240, 320]` before the RoboTwin-specific layout;
- skip final resize/crop when the mosaic is already exactly `[384, 320]`.

### DataLoader worker settings are conservative

`Wan22Trainer._build_loader()` currently uses:

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=worker_init_fn,
)
```

Potential knobs:

```python
persistent_workers = num_workers > 0
prefetch_factor = 2 or 4
drop_last = True for train, if exact epoch coverage is not required
```

`persistent_workers=True` avoids worker teardown/restart each epoch. `prefetch_factor` can help if video decode is the bottleneck. Too high can increase RAM pressure because each sample contains decoded multi-frame video.

Also consider setting worker CPU thread counts to avoid oversubscription:

```python
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

This matters because many DataLoader workers may each call torchvision resize routines.

### Multi-dataset indexing is linear in number of subdatasets

`MultiLeRobotDataset.__getitem__()` finds the underlying dataset by looping over `self._datasets` and subtracting lengths. If RoboTwin is expanded into many subdatasets, this becomes an `O(num_datasets)` operation per sample.

This is usually smaller than video decode cost, but easy to improve:

- precompute cumulative frame counts;
- use `bisect` to map global index to dataset index.

### Non-video data selection can be faster

`LeRobotDataset.__getitem__()` uses `hf_dataset.select(q_idx)` for non-video windows. This creates HuggingFace/Arrow selection objects per sample. The code already has comments about avoiding Arrow buffer accumulation.

For action/state columns, a faster path would be:

- load low-dimensional parquet columns into torch/numpy arrays once per worker or per dataset;
- gather windows with tensor indexing;
- keep video decoding separate.

This can reduce Python/HF overhead, especially when batch size is small and there are many workers.

### Normalization stats calculation can overload storage

`BaseLerobotDataset.get_dataset_stats()` uses `ThreadPoolExecutor()` and submits one future per episode immediately:

```python
futures = [executor.submit(process_episode, num) for num in range(episodes_num)]
```

For many episodes, this can create a large future list and make many concurrent parquet reads. On network storage, it can slow down or cause memory pressure.

Better:

- add configurable `stats_num_workers`;
- keep the number of in-flight futures bounded;
- use fewer workers on network filesystems;
- stream partial statistics instead of storing every per-episode tensor list when datasets get large.

### Suggested priority order

1. Reduce image decode/resize from 33 frames to 9 frames per sample.
2. Add in-memory LRU cache for text embedding `.pt` loads.
3. Enable `persistent_workers` and tune `prefetch_factor`.
4. Remove/no-op skip extra RoboTwin resize/crop work.
5. Use cumulative sizes + `bisect` in `MultiLeRobotDataset`.
6. Optimize non-video action/state column access.
7. Make stats computation bounded/configurable.

## Standalone text embedding precompute refactor

New script:

```text
scripts/precompute_text_embeds_direct.py
```

Goal: decouple text embedding precompute from Hydra task/data/model YAML nesting. The new script takes dataset paths and one output cache path directly from CLI.

Basic RoboTwin-style usage:

```bash
python scripts/precompute_text_embeds_direct.py \
  --dataset-dir /path/to/robotwin2.0/robotwin2.0 \
  --cache-dir /path/to/text_embeds_cache/robotwin
```

Multiple dataset roots are supported:

```bash
python scripts/precompute_text_embeds_direct.py \
  --dataset-dir /data/a /data/b \
  --cache-dir /cache/robotwin
```

It still supports `torchrun` sharding without Hydra:

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds_direct.py \
  --dataset-dir /path/to/dataset_root \
  --cache-dir /path/to/text_embeds_cache
```

Useful options:

```text
--context-len 128
--batch-size 16
--skip-existing
--dry-run
--override-instruction "robot manipulation task"
```

The model ids, tokenizer ids, prompt template, dtype, device selection, `enc_id`, and bilingual splitting policy are intentionally fixed to the repository defaults to keep the script small and hard to misconfigure.

Important behavior differences from `scripts/precompute_text_embeds.py`:

1. No Hydra import or config composition.
2. No dependency on `configs/task/*.yaml`, `configs/data/*.yaml`, or `configs/model/*.yaml`.
3. Existing cache files are overwritten by default, matching the original script. Pass `--skip-existing` to skip already generated files.
4. `--help` and `--dry-run` avoid importing torch/model code.
5. Debug prompt/hash prints were removed.
6. Parquet task text reading mirrors the training loader:
   - if `__index_level_0__` exists, set it as dataframe index;
   - otherwise use the dataframe index as task text.
7. If a parquet has a default numeric index plus a `task` column, the script warns but still mirrors training behavior instead of silently changing semantics.

The cache filename format remains compatible with training:

```text
{sha256(prompt)}.t5_len{context_len}.wan22ti2v5b.pt
```

This matches `RobotVideoDataset._get_cached_text_context()`.

## Direct training launcher refactor

Changed files:

```text
scripts/train_zero2.sh
scripts/train_direct.py
configs/train/robotwin.yaml
configs/train/post_train.yaml
```

The original launcher was backed up to:

```text
scripts/train_zero2.sh_bk
```

The new training path avoids Hydra CLI/task composition:

```text
scripts/train_zero2.sh
  -> accelerate launch with DeepSpeed ZeRO-2 config
  -> scripts/train_direct.py
  -> OmegaConf.load(single YAML)
  -> fastwam.runtime.run_training(cfg)
```

The launch parameters are now owned by the shell script itself:

```bash
bash scripts/train_zero2.sh
```

Current user-editable block in `scripts/train_zero2.sh`:

```text
CONFIG="configs/train/post_train.yaml"
NPROC_PER_NODE=8
LAUNCH_MODE="multi"
NUM_MACHINES=2
MACHINE_RANK=""
AUTO_DETECT_MULTI_NODE=1
HOSTFILE="/etc/volcano/worker.host"
MASTER_ADDR="${MASTER_ADDR:-}"
OUTPUT_DIR="./runs/pick_place_1e-4/direct"
```

The script is currently configured for 2-node x 8-GPU training. Keep the same command on every node:

```bash
bash scripts/train_zero2.sh
```

Expected launch summary:

```text
# node 0
nproc_per_node=8 num_machines=2 world_size=16
machine_rank=0 master=<first-host>:29500

# node 1
nproc_per_node=8 num_machines=2 world_size=16
machine_rank=1 master=<first-host>:29500
```

The script counts `/etc/volcano/worker.host`, uses the first host as `MASTER_ADDR`, and infers `MACHINE_RANK` from `VC_TASK_INDEX`, then `NODE_RANK`, then `SLURM_NODEID`. In multi-node mode, `MASTER_ADDR` is not allowed to fall back to `127.0.0.1`.

For single-node 8-GPU debug, edit:

```text
LAUNCH_MODE="single"
```

For manual multi-node training without scheduler env or hostfile, edit these lines in `scripts/train_zero2.sh` on each node:

```text
NUM_MACHINES=2
MACHINE_RANK=0        # node 0; change to 1 on node 1
MASTER_ADDR="10.0.0.1"  # reachable IP/hostname of node 0
```

`MACHINE_RANK` is the node index consumed by `accelerate --machine_rank`. It is not the GPU rank or process rank. GPU local ranks 0-7 are created by `NPROC_PER_NODE=8`.

The script auto-fills `MASTER_ADDR` from `/etc/volcano/worker.host` when present. This avoids calling it with `--machine_rank ${VC_TASK_INDEX}`, which can fail before the script starts when `VC_TASK_INDEX` is not defined in the parent shell.

Training overrides are written directly at the tail of the `accelerate launch` command:

```text
scripts/train_direct.py \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 16 \
  --num_workers 16 \
  --learning_rate 4.0e-4
```

The launcher intentionally rejects command-line arguments now, so the external command stays `bash scripts/train_zero2.sh`.

Notes:

1. The backend still uses Accelerate's DeepSpeed integration because `Wan22Trainer` is built around `Accelerator(...)`.
2. The user-facing launch path no longer uses `hydra.main`, `task=...`, or nested Hydra defaults.
3. The train YAML is intentionally complete: top-level train args, `data`, and `model` are in one file.

## Train post_train config

`configs/task/pick_place_1e-4.yaml` has been expanded into:

```text
configs/train/post_train.yaml
```

It combines:

```text
configs/train.yaml
configs/data/post_train.yaml
configs/model/fastwam.yaml
configs/task/pick_place_1e-4.yaml
```

The train config preserves the task overrides:

```text
batch_size: 16
num_workers: 16
learning_rate: 4.0e-4
num_epochs: 1
save_every: 10000
eval_every: 10001
model.mot_checkpoint_mixed_attn: false
```

`scripts/train_zero2.sh` now points to this train config by default:

```text
CONFIG="configs/train/post_train.yaml"
OUTPUT_DIR="./runs/pick_place_1e-4/direct"
```

## Single-node ZeRO-2 launcher

Added a single-node debugging launcher:

```text
scripts/train_zero2_single.sh
```

It uses the same direct pick_place config but fixes distributed launch parameters to one node:

```text
NPROC_PER_NODE=8
NUM_MACHINES=1
MACHINE_RANK=0
MASTER_ADDR="127.0.0.1"
WORLD_SIZE=8
```

Run it with:

```bash
bash scripts/train_zero2_single.sh
```

Expected launch summary:

```text
mode=single
nproc_per_node=8 num_machines=1 world_size=8
machine_rank=0 master=127.0.0.1:29500
```

This launcher avoids all `/etc/volcano/worker.host`, `VC_TASK_INDEX`, `NODE_RANK`, and cross-node NCCL variables, so it is useful for separating model/data/DeepSpeed issues from multi-node rendezvous issues.

## Single-node ZeRO-1 launcher

`scripts/train_zero1.sh` was rewritten as a direct single-node 8-GPU launcher, mirroring the current direct `train_zero2.sh` style but using ZeRO-1:

```text
ACCELERATE_CONFIG="scripts/accelerate_configs/accelerate_zero1_ds.yaml"
NPROC_PER_NODE=8
NUM_MACHINES=1
MACHINE_RANK=0
MASTER_ADDR="127.0.0.1"
WORLD_SIZE=8
```

The original Hydra-style launcher was backed up to:

```text
scripts/train_zero1.sh_bk
```

Run it with:

```bash
bash scripts/train_zero1.sh
```

Expected launch summary:

```text
mode=single zero_stage=1
nproc_per_node=8 num_machines=1 world_size=8
machine_rank=0 master=127.0.0.1:29500
```

## Multi-node ZeRO-1 launcher

Added a multi-node ZeRO-1 launcher:

```text
scripts/train_zero1_multi.sh
```

It mirrors the current direct `train_zero1.sh` training arguments and switches the launch parameters to multi-node:

```text
ACCELERATE_CONFIG="scripts/accelerate_configs/accelerate_zero1_ds.yaml"
NPROC_PER_NODE=8
NUM_MACHINES=2
MACHINE_RANK=""  # inferred from VC_TASK_INDEX, NODE_RANK, or SLURM_NODEID
HOSTFILE="/etc/volcano/worker.host"
WORLD_SIZE=NPROC_PER_NODE * NUM_MACHINES
```

Run the same command on every node:

```bash
bash scripts/train_zero1_multi.sh
```

Expected 2-node x 8-GPU summaries:

```text
# node 0
mode=multi zero_stage=1
nproc_per_node=8 num_machines=2 world_size=16
machine_rank=0 master=<first-host>:29500

# node 1
mode=multi zero_stage=1
nproc_per_node=8 num_machines=2 world_size=16
machine_rank=1 master=<first-host>:29500
```

The script refuses to run if `MASTER_ADDR` is empty or resolves to `127.0.0.1/localhost`, because that would make multi-node NCCL rendezvous hang.
