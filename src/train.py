"""Train the cross-modal attention pairing model.

Examples
--------
Smoke test on synthetic data (no dataset needed):
    python src/train.py --synthetic --epochs 2

Train on real CSVs:
    python src/train.py --data-dir sample-test-data --epochs 30
"""

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from data import SyntheticPairs, real_pairs
from model import CrossModalAttentionFusion


def run_epoch(model, loader, device, optim=None):
    train = optim is not None
    model.train(train)
    crit = torch.nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for insole, skel, label in loader:
        insole, skel, label = insole.to(device), skel.to(device), label.to(device)
        with torch.set_grad_enabled(train):
            logits = model(insole, skel)
            loss = crit(logits, label)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
        loss_sum += loss.item() * label.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += label.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default="sample-test-data")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="model/cross_attn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.synthetic:
        ds = SyntheticPairs(seq_len=args.seq_len)
        insole_dim, skel_dim = 8, 12
    else:
        ds = real_pairs(args.data_dir, args.seq_len)
        insole_dim, skel_dim = 8, 12
        print(f"real samples: {len(ds)}")

    n_val = max(1, int(0.2 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = CrossModalAttentionFusion(insole_dim=insole_dim, skel_dim=skel_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, device, optim)
        va_loss, va_acc = run_epoch(model, val_dl, device)
        print(f"epoch {ep:3d} | train {tr_loss:.4f}/{tr_acc:.3f} "
              f"| val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc >= best:
            best = va_acc
            import os
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "insole_dim": insole_dim, "skel_dim": skel_dim,
                        "seq_len": args.seq_len}, args.out)
    print(f"best val acc: {best:.3f} -> saved {args.out}")


if __name__ == "__main__":
    main()
