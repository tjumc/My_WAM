import sys
import os
import argparse
# ── EARLIEST POSSIBLE DEBUG ── guaranteed to run before any import or init ──
sys.stderr.write(
    f"[train.py LOADED] pid={os.getpid()} "
    f"LOCAL_RANK={os.environ.get('LOCAL_RANK','?')} "
    f"RANK={os.environ.get('RANK','?')} "
    f"script={__file__}\n"
)
sys.stderr.flush()

# 禁用 HuggingFace datasets 的磁盘缓存和内存缓存，防止 Arrow Table 随训练步数缓慢积累导致 OOM
import datasets as _hf_datasets
_hf_datasets.disable_caching()

from omegaconf import OmegaConf

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

sys.stderr.write("[train.py] fastwam imported OK — runtime.py is at: " + str(__import__('fastwam.runtime', fromlist=['']).__file__) + "\n")
sys.stderr.flush()

register_default_resolvers()


def _add_if_set(overrides: list[str], key: str, value):
    if value is not None:
        overrides.append(f"{key}={value}")


def _parse_direct_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Train FastWAM from one explicit YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to a complete training YAML config.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--lr_scheduler_type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_num_inference_steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_direct_args(argv)
    cfg = OmegaConf.load(args.config)

    overrides: list[str] = []
    _add_if_set(overrides, "output_dir", args.output_dir)
    _add_if_set(overrides, "batch_size", args.batch_size)
    _add_if_set(overrides, "num_workers", args.num_workers)
    _add_if_set(overrides, "learning_rate", args.learning_rate)
    _add_if_set(overrides, "weight_decay", args.weight_decay)
    _add_if_set(overrides, "num_epochs", args.num_epochs)
    _add_if_set(overrides, "max_steps", args.max_steps)
    _add_if_set(overrides, "gradient_accumulation_steps", args.gradient_accumulation_steps)
    _add_if_set(overrides, "mixed_precision", args.mixed_precision)
    _add_if_set(overrides, "lr_scheduler_type", args.lr_scheduler_type)
    _add_if_set(overrides, "seed", args.seed)
    _add_if_set(overrides, "max_grad_norm", args.max_grad_norm)
    _add_if_set(overrides, "log_every", args.log_every)
    _add_if_set(overrides, "save_every", args.save_every)
    _add_if_set(overrides, "eval_every", args.eval_every)
    _add_if_set(overrides, "eval_num_inference_steps", args.eval_num_inference_steps)
    _add_if_set(overrides, "resume", args.resume)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    sys.stderr.write(
        f"[train.py direct] LOCAL_RANK={os.environ.get('LOCAL_RANK','?')} "
        f"RANK={os.environ.get('RANK','?')} config={args.config} "
        f"output_dir={cfg.output_dir} "
        f"dataset_dirs={len(cfg.data.train.dataset_dirs)} -- calling run_training()\n"
    )
    sys.stderr.flush()
    run_training(cfg)


if __name__ == "__main__":
    main()
