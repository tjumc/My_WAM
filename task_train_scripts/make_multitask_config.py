import sys
from omegaconf import OmegaConf

src, dst = sys.argv[1], sys.argv[2]

cfg = OmegaConf.load(src)
cfg.data.train.override_instruction = None

if "val" in cfg.data:
    cfg.data.val.override_instruction = None

OmegaConf.save(cfg, dst, resolve=False)
