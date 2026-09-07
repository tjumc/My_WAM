"""
plot_loss.py  –  Parse robotw64.log and draw loss curves.

Usage:
    python plot_loss.py [logfile] [--out output.png] [--smooth N]

Defaults:
    logfile : robotw64.log (same directory as this script)
    --out   : loss_curves.png
    --smooth: 50   (moving-average window for train curves)
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("logfile", nargs="?",
                    default=str(Path(__file__).parent / "robotw64.log"))
parser.add_argument("--out", default=str(Path(__file__).parent / "loss_curves.png"))
parser.add_argument("--smooth", type=int, default=50,
                    help="Moving-average window for train curves (1 = raw)")
args = parser.parse_args()

log_path = Path(args.logfile)
print(f"[plot_loss] reading  : {log_path}")
print(f"[plot_loss] output   : {args.out}")
print(f"[plot_loss] smoothing: {args.smooth}")

# ─────────────────────────────────────────────────────────────────────────────
# Parse
# ─────────────────────────────────────────────────────────────────────────────
raw = log_path.read_text(encoding="utf-8", errors="replace")

# The log uses a two-column format: content wraps at ~50 chars.
# Strategy: collapse every run of whitespace → single space so that regex
# can find key=value pairs even across line boundaries.
flat = re.sub(r"\s+", " ", raw)

# ── Training records (trainer.py:715) ────────────────────────────────────────
# Pattern: epoch=X step=Y/Z  ...  loss=A  loss_action=B  loss_video=C  lr=D
RE_TRAIN = re.compile(
    r"epoch=(\d+)\s+step=(\d+)/\d+"          # epoch, step
    r".*?"
    r"loss=([\d.]+)\s+loss_action=([\d.]+)"   # loss, loss_action
    r".*?"
    r"loss_video=([\d.]+)"                    # loss_video
    r".*?"
    r"lr=([\d.eE+\-]+)"                       # lr
)

train = []
for m in RE_TRAIN.finditer(flat):
    train.append({
        "epoch":       int(m.group(1)),
        "step":        int(m.group(2)),
        "loss":        float(m.group(3)),
        "loss_action": float(m.group(4)),
        "loss_video":  float(m.group(5)),
        "lr":          float(m.group(6)),
    })

# ── Validation records (trainer.py:746) ──────────────────────────────────────
# Pattern: step=Y val_loss=A  infer_psnr=B  infer_ssim=C  action_l2=D  action_l1=E
RE_VAL = re.compile(
    r"step=(\d+)\s+val_loss=([\d.]+)"
    r".*?"
    r"infer_psnr=([\d.]+)\s+infer_ssim=([\d.]+)"
    r".*?"
    r"action_l2=([\d.]+)\s+action_l1=([\d.]+)"
)

val = []
for m in RE_VAL.finditer(flat):
    val.append({
        "step":       int(m.group(1)),
        "val_loss":   float(m.group(2)),
        "psnr":       float(m.group(3)),
        "ssim":       float(m.group(4)),
        "action_l2":  float(m.group(5)),
        "action_l1":  float(m.group(6)),
    })

print(f"[plot_loss] train records : {len(train)}")
print(f"[plot_loss] val   records : {len(val)}")

if not train and not val:
    print("[plot_loss] ERROR: no records parsed – check log format.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def moving_avg(arr, w):
    """Simple moving average (valid mode → shorter output) padded back."""
    if w <= 1 or len(arr) < w:
        return np.array(arr)
    kernel = np.ones(w) / w
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad front so length matches original
    pad = np.full(w - 1, smoothed[0])
    return np.concatenate([pad, smoothed])

def arr(records, key):
    return np.array([r[key] for r in records])

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "lines.linewidth":  1.3,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.size":        9,
}
plt.rcParams.update(STYLE)

has_train = len(train) > 0
has_val   = len(val)   > 0

# Layout: up to 3 rows × 2 cols
# Row 0: total loss (train) | action / video loss (train)
# Row 1: lr curve           | val loss
# Row 2: PSNR / SSIM        | action L1 / L2  (val only)
nrows = 1 + (1 if has_train else 0) + (1 if has_val else 0)
fig, axes = plt.subplots(3, 2, figsize=(14, 10), dpi=130)
fig.suptitle("Training Loss Curves", fontsize=13, fontweight="bold",
             color="#e6edf3", y=0.98)

def ax(r, c):
    return axes[r][c]

W = args.smooth    # smoothing window
ALPHA_RAW = 0.18   # alpha for raw (noisy) line

# ── Panel (0,0): Total train loss ────────────────────────────────────────────
if has_train:
    steps = arr(train, "step")
    loss  = arr(train, "loss")
    a = ax(0, 0)
    a.plot(steps, loss, color="#3d8bcd", alpha=ALPHA_RAW, linewidth=0.8)
    a.plot(steps, moving_avg(loss, W), color="#3d8bcd", label="loss (total)")
    a.set_title("Total Loss (train)")
    a.set_xlabel("step"); a.set_ylabel("loss")
    a.legend(); a.grid(True)

# ── Panel (0,1): action + video loss ─────────────────────────────────────────
if has_train:
    la = arr(train, "loss_action")
    lv = arr(train, "loss_video")
    a = ax(0, 1)
    a.plot(steps, la, color="#f0883e", alpha=ALPHA_RAW, linewidth=0.8)
    a.plot(steps, moving_avg(la, W), color="#f0883e", label="loss_action")
    a.plot(steps, lv, color="#56d364", alpha=ALPHA_RAW, linewidth=0.8)
    a.plot(steps, moving_avg(lv, W), color="#56d364", label="loss_video")
    a.set_title("Action & Video Loss (train)")
    a.set_xlabel("step"); a.set_ylabel("loss")
    a.legend(); a.grid(True)

# ── Panel (1,0): Learning rate ────────────────────────────────────────────────
if has_train:
    lr = arr(train, "lr")
    a = ax(1, 0)
    a.plot(steps, lr, color="#bc8cff")
    a.set_title("Learning Rate")
    a.set_xlabel("step"); a.set_ylabel("lr")
    a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    a.grid(True)

# ── Panel (1,1): Val total loss ───────────────────────────────────────────────
if has_val:
    vsteps   = arr(val, "step")
    val_loss = arr(val, "val_loss")
    a = ax(1, 1)
    a.plot(vsteps, val_loss, color="#f85149", marker="o", markersize=3,
           label="val_loss")
    a.set_title("Validation Loss")
    a.set_xlabel("step"); a.set_ylabel("val_loss")
    a.legend(); a.grid(True)

# ── Panel (2,0): PSNR & SSIM ─────────────────────────────────────────────────
if has_val:
    psnr = arr(val, "psnr")
    ssim = arr(val, "ssim")
    a = ax(2, 0)
    a2 = a.twinx()
    l1, = a.plot(vsteps, psnr, color="#58a6ff", marker="o", markersize=3,
                 label="PSNR (dB)")
    l2, = a2.plot(vsteps, ssim, color="#d2a8ff", marker="s", markersize=3,
                  linestyle="--", label="SSIM")
    a.set_title("Video Quality (PSNR / SSIM)")
    a.set_xlabel("step")
    a.set_ylabel("PSNR (dB)", color="#58a6ff")
    a2.set_ylabel("SSIM", color="#d2a8ff")
    a2.tick_params(axis="y", labelcolor="#d2a8ff")
    a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    lines = [l1, l2]
    a.legend(lines, [l.get_label() for l in lines], loc="lower right")
    a.grid(True)

# ── Panel (2,1): Action L1 & L2 ──────────────────────────────────────────────
if has_val:
    al1 = arr(val, "action_l1")
    al2 = arr(val, "action_l2")
    a = ax(2, 1)
    a.plot(vsteps, al1, color="#ffa657", marker="o", markersize=3, label="action_l1")
    a.plot(vsteps, al2, color="#ff7b72", marker="s", markersize=3,
           linestyle="--", label="action_l2")
    a.set_title("Action Error (val)")
    a.set_xlabel("step"); a.set_ylabel("error")
    a.legend(); a.grid(True)

# ── Hide unused panels ────────────────────────────────────────────────────────
for r in range(3):
    for c in range(2):
        axes[r][c].set_visible(False)

# Re-enable used panels
panels_used = []
if has_train:
    panels_used += [(0,0),(0,1),(1,0)]
if has_val:
    panels_used += [(1,1),(2,0),(2,1)]
for r, c in panels_used:
    axes[r][c].set_visible(True)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = args.out
plt.savefig(out_path, bbox_inches="tight")
print(f"[plot_loss] saved → {out_path}")
