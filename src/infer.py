"""Run pairing inference with a trained checkpoint.

    python src/infer.py --ckpt model/cross_attn.pt \
        --insole sample-test-data/smart-insole-A.csv \
        --openpose sample-test-data/open-pose-1.csv
"""

import argparse

import torch
import torch.nn.functional as F

from data import load_aligned, _normalise
from model import CrossModalAttentionFusion

LABELS = {0: "PAIRED", 1: "UNPAIRED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--insole", required=True)
    ap.add_argument("--openpose", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = CrossModalAttentionFusion(insole_dim=ckpt["insole_dim"],
                                      skel_dim=ckpt["skel_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ins, skel = load_aligned(args.insole, args.openpose, ckpt["seq_len"])
    if not ins:
        raise SystemExit("No overlapping seconds between the two files.")
    ins, skel = _normalise(ins), _normalise(skel)
    insole = torch.from_numpy(__import__("numpy").stack(ins)).float()
    skeleton = torch.from_numpy(__import__("numpy").stack(skel)).float()

    with torch.no_grad():
        probs = F.softmax(model(insole, skeleton), dim=1).mean(0)
    pred = int(probs.argmax())
    print(f"{LABELS[pred]}  (paired={probs[0]:.3f}, unpaired={probs[1]:.3f}, "
          f"n_windows={len(ins)})")


if __name__ == "__main__":
    main()
