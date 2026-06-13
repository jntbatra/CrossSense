"""Subject-disjoint benchmark for cross-modal pairing.

Holds out whole subjects for the test set so the model is scored on people it
never saw in training -- the only honest way to measure person association (a
random split would leak per-subject signal and inflate the score).

    python src/evaluate.py --data-dir <full_dataset> --epochs 40 --test-frac 0.3

Reports Accuracy, macro-F1, and ROC-AUC. Needs >= 2 subjects in each split, so
the 2-subject sample data is NOT enough -- use the full released dataset.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PairDataset, load_people
from model import CrossModalAttentionFusion
from train import run_epoch


def split_subjects(people, test_frac, seed=0):
    names = sorted(people)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(names))
    n_test = max(1, int(round(test_frac * len(names))))
    n_test = min(n_test, len(names) - 1)  # keep at least one train subject
    test = set(perm[:n_test])
    train_people = {k: v for k, v in people.items() if k not in test}
    test_people = {k: v for k, v in people.items() if k in test}
    return train_people, test_people


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    ys, scores, preds = [], [], []
    for insole, skel, y in loader:
        logits = model(insole.to(device), skel.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1]  # P(UNPAIRED)
        ys.append(y.numpy())
        scores.append(prob.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return (np.concatenate(ys), np.concatenate(scores), np.concatenate(preds))


def macro_f1(y_true, y_pred):
    f1s = []
    for c in (0, 1):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def roc_auc(y_true, y_score):
    """Mann-Whitney U formulation; positive class = 1 (UNPAIRED)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    auc = (ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    people = load_people(args.data_dir, args.seq_len)
    print(f"subjects: {len(people)} ({', '.join(sorted(people))})")

    train_people, test_people = split_subjects(people, args.test_frac, args.seed)
    if len(train_people) < 2 or len(test_people) < 2:
        raise SystemExit(
            f"Subject-disjoint pairing needs >= 2 subjects per split "
            f"(got train={len(train_people)}, test={len(test_people)}). "
            f"The 2-subject sample data is not a valid benchmark -- "
            f"download the full dataset (see train-sets/).")

    train_ds = PairDataset(train_people, seed=args.seed)
    test_ds = PairDataset(test_people, seed=args.seed)
    print(f"train subjects: {sorted(train_people)} ({len(train_ds)} pairs) | "
          f"test subjects: {sorted(test_people)} ({len(test_ds)} pairs)")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    model = CrossModalAttentionFusion().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for ep in range(1, args.epochs + 1):
        loss, acc = run_epoch(model, train_dl, device, optim)
        if ep % 5 == 0 or ep == args.epochs:
            print(f"epoch {ep:3d} | train {loss:.4f}/{acc:.3f}")

    y, score, pred = collect(model, test_dl, device)
    print("\n=== held-out subject benchmark ===")
    print(f"Accuracy : {(pred == y).mean():.3f}")
    print(f"Macro-F1 : {macro_f1(y, pred):.3f}")
    print(f"ROC-AUC  : {roc_auc(y, score):.3f}")
    print(f"(n_test={len(y)}, positives/UNPAIRED={(y == 1).sum()})")


if __name__ == "__main__":
    main()
