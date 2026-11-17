"""Evaluate a checkpoint on the validation set: plain / hflip-TTA / vflip-TTA.

Reports the competition proxy (0.7 * mean per-class Dice + 0.3 * empty-row acc)
with the per-class thresholds from the checkpoint's sweep file, so we can decide
which prediction mode to use for the submission.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle import rle_decode_np  # noqa: E402
from infer import ARCHS  # noqa: E402
from train import CLASSES, CKPT, DATA, DEVICE, H, MEAN, STD, W  # noqa: E402


def load_val():
    train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    df = train_df.merge(split, on="image_id")
    va_ids = set(df.loc[df["split"] == "val", "image_id"])
    va = df[df["image_id"].isin(va_ids)].reset_index(drop=True)
    imgs, masks = [], []
    img_dir = os.path.join(DATA, "public", "train")
    for _, r in va.iterrows():
        img = cv2.imread(os.path.join(img_dir, r["image_id"] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((H, W), dtype=np.uint8)
        elif img.shape != (H, W):
            img = cv2.resize(img, (W, H))
        imgs.append(img)
        m = np.zeros((4, H, W), dtype=np.uint8)
        ci = CLASSES.index(r["defect_type"])
        m[ci] = rle_decode_np(r["mask_rle"], H, W)
        masks.append(m)
    x = np.stack(imgs).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x)[:, None], torch.from_numpy(np.stack(masks)).bool()


@torch.no_grad()
def predict_probs(model, x, tta=None):
    probs = torch.sigmoid(model(x))
    if tta == "hflip":
        probs = (probs + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
    elif tta == "vflip":
        probs = (probs + torch.sigmoid(model(x.flip(-2))).flip(-2)) / 2
    elif tta == "hvflip":
        probs = (probs + torch.sigmoid(model(x.flip(-1))).flip(-1)
                 + torch.sigmoid(model(x.flip(-2))).flip(-2)) / 3
    return probs


@torch.no_grad()
def eval_mode(probs, gt, thresholds):
    """Row-level proxy matching train.py: 0.7*mean(dice_c over GT-nonempty rows)
    + 0.3*(fraction of GT-empty rows predicted empty)."""
    dice_c = []
    gt_empty = ~gt.flatten(2).any(2)  # (N, 4)
    correct_empty = 0
    total_empty = int(gt_empty.sum())
    for c in range(4):
        b = probs[:, c] > thresholds[CLASSES[c]]
        g = gt[:, c]
        ne = gt_empty[:, c]
        if ne.any():
            inter = (b[ne] & g[ne]).sum(dim=(1, 2)).float()
            sums = (b[ne].sum(dim=(1, 2)) + g[ne].sum(dim=(1, 2))).float()
            dice_c.append(float((2 * inter / (sums + 1e-7)).mean()))
        else:
            dice_c.append(0.0)
        correct_empty += int((~b.flatten(1).any(1) & gt_empty[:, c]).sum())
    ea = correct_empty / max(total_empty, 1)
    return 0.7 * float(np.mean(dice_c)) + 0.3 * ea, dice_c, ea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT, "segformer_mit_b2_os_best.pt"))
    ap.add_argument("--sweep", default=os.path.join(CKPT, "segformer_mit_b2_os_sweep.json"))
    ap.add_argument("--arch", default="segformer")
    ap.add_argument("--encoder", default="mit_b2")
    args = ap.parse_args()

    with open(args.sweep) as f:
        sweep = json.load(f)
    thresholds = {c: sweep[c]["best_threshold"] for c in CLASSES}
    print("thresholds:", thresholds)

    x, gt = load_val()
    model = ARCHS[args.arch](encoder_name=args.encoder, encoder_weights=None,
                             in_channels=1, classes=4).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()

    batch = 10
    for mode in [None, "hflip", "vflip", "hvflip"]:
        probs = []
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(DEVICE)
            probs.append(predict_probs(model, xb, tta=mode).cpu())
        probs = torch.cat(probs)
        score, dice_c, ea = eval_mode(probs, gt, thresholds)
        print(f"mode={str(mode):6s} proxy={score:.4f} dice={[round(d, 3) for d in dice_c]} empty_acc={ea:.4f}")


if __name__ == "__main__":
    main()
