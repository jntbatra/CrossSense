"""Generate README figures (honest, reproducible).

  python scripts/make_figures.py

Produces:
  assets/training_curve.png  -- synthetic smoke-test learning curve (NOT a
                                research result; synthetic data, for sanity only)
  assets/model_size.png      -- parameter-count comparison vs the VGG16 backbone
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import SyntheticPairs            # noqa: E402
from model import CrossModalAttentionFusion  # noqa: E402
from train import run_epoch                # noqa: E402

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)


def training_curve(epochs=20):
    ds = SyntheticPairs(seq_len=10)
    n_val = int(0.2 * len(ds))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    tdl, vdl = DataLoader(tr, batch_size=32, shuffle=True), DataLoader(va, batch_size=32)
    model = CrossModalAttentionFusion()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    hist = {"tr_acc": [], "va_acc": [], "tr_loss": [], "va_loss": []}
    for _ in range(epochs):
        trl, tra = run_epoch(model, tdl, "cpu", opt)
        val, vaa = run_epoch(model, vdl, "cpu")
        hist["tr_acc"].append(tra); hist["va_acc"].append(vaa)
        hist["tr_loss"].append(trl); hist["va_loss"].append(val)

    xs = range(1, epochs + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(xs, hist["tr_loss"], label="train"); a1.plot(xs, hist["va_loss"], label="val")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(xs, hist["tr_acc"], label="train"); a2.plot(xs, hist["va_acc"], label="val")
    a2.set_title("Pairing accuracy"); a2.set_xlabel("epoch"); a2.set_ylim(0, 1)
    a2.legend(); a2.grid(alpha=.3)
    fig.suptitle("Cross-modal attention — synthetic smoke test (sanity only, not a benchmark)")
    fig.tight_layout()
    out = os.path.join(ASSETS, "training_curve.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)


def model_size():
    params = sum(p.numel() for p in CrossModalAttentionFusion().parameters())
    # VGG16 convolutional backbone the baseline renders into (frozen, ImageNet).
    vgg16_backbone = 14_714_688
    names = ["VGG16 backbone\n(baseline, frozen)", "Cross-modal attention\n(this work, trainable)"]
    vals = [vgg16_backbone, params]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color=["#b0b0b0", "#3b7dd8"])
    ax.set_ylabel("parameters")
    ax.set_yscale("log")
    ax.set_title(f"Model size: {vgg16_backbone/params:.0f}x smaller than the VGG16 backbone")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    out = os.path.join(ASSETS, "model_size.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    training_curve()
    model_size()
