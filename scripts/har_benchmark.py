"""Cross-modal pairing demo on the public UCI HAR dataset.

Tests the PressPose method's *generality*: can it decide whether an accelerometer
stream and a gyroscope stream came from the SAME person? 30 subjects, evaluated
subject-disjoint (test people never seen in training).

This is a public-dataset generalisation demo (accel<->gyro). It is NOT the
plantar-pressure / 2D-pose result -- report it labelled as "UCI HAR".

    python scripts/har_benchmark.py --zip /tmp/ppdata/har.zip --epochs 12
"""

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import PairDataset                       # noqa: E402
from model import CrossModalAttentionFusion        # noqa: E402
from train import run_epoch                        # noqa: E402
from evaluate import split_subjects, collect, macro_f1, roc_auc  # noqa: E402

ACC = ["total_acc_x", "total_acc_y", "total_acc_z",
       "body_acc_x", "body_acc_y", "body_acc_z"]      # modality A (6 ch)
GYRO = ["body_gyro_x", "body_gyro_y", "body_gyro_z"]  # modality B (3 ch)


def _read_signals(zf, root, split, names):
    out = []
    for n in names:
        path = f"{root}/{split}/Inertial Signals/{n}_{split}.txt"
        arr = np.loadtxt(io.TextIOWrapper(zf.open(path)))
        out.append(arr.astype(np.float32))          # (N, 128)
    return np.stack(out, axis=-1)                    # (N, 128, C)


def load_har(zip_path, seq_len=32):
    # Outer zip contains an inner "UCI HAR Dataset.zip".
    with zipfile.ZipFile(zip_path) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip"))
        zf = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    root = "UCI HAR Dataset"

    acc, gyro, subj = [], [], []
    for split in ("train", "test"):
        acc.append(_read_signals(zf, root, split, ACC))
        gyro.append(_read_signals(zf, root, split, GYRO))
        s = np.loadtxt(io.TextIOWrapper(zf.open(f"{root}/{split}/subject_{split}.txt")))
        subj.append(s.astype(int))
    acc = np.concatenate(acc); gyro = np.concatenate(gyro); subj = np.concatenate(subj)

    step = max(1, acc.shape[1] // seq_len)
    acc, gyro = acc[:, ::step][:, :seq_len], gyro[:, ::step][:, :seq_len]

    def standardise(x):
        mu = x.mean((0, 1), keepdims=True)
        sd = np.maximum(x.std((0, 1), keepdims=True), 1e-6)
        return ((x - mu) / sd).astype(np.float32)
    acc, gyro = standardise(acc), standardise(gyro)

    people = {}
    for i in range(len(subj)):
        people.setdefault(str(subj[i]), ([], []))
        people[str(subj[i])][0].append(acc[i])   # PAIRED: same window -> same person
        people[str(subj[i])][1].append(gyro[i])
    return people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="/tmp/ppdata/har.zip")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    people = load_har(args.zip, args.seq_len)
    print(f"UCI HAR: {len(people)} subjects, {sum(len(v[0]) for v in people.values())} windows")

    train_people, test_people = split_subjects(people, args.test_frac, args.seed)
    train_ds = PairDataset(train_people, seed=args.seed)
    test_ds = PairDataset(test_people, seed=args.seed)
    print(f"train: {len(train_people)} subj / {len(train_ds)} pairs | "
          f"test (held-out): {len(test_people)} subj / {len(test_ds)} pairs")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    model = CrossModalAttentionFusion(insole_dim=6, skel_dim=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for ep in range(1, args.epochs + 1):
        loss, acc = run_epoch(model, train_dl, device, opt)
        if ep % 3 == 0 or ep == args.epochs:
            print(f"epoch {ep:3d} | train {loss:.4f}/{acc:.3f}")

    y, score, pred = collect(model, test_dl, device)
    print("\n=== UCI HAR accel<->gyro pairing (subject-disjoint) ===")
    print(f"Accuracy : {(pred == y).mean():.3f}")
    print(f"Macro-F1 : {macro_f1(y, pred):.3f}")
    print(f"ROC-AUC  : {roc_auc(y, score):.3f}")
    print(f"(test subjects={len(test_people)}, n_test={len(y)})")


if __name__ == "__main__":
    main()
