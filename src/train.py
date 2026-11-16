"""Train UNet(EfficientNet-B0) for 4-class defect segmentation at 800x128.

- BCE + Dice loss, class-weighted BCE for imbalance
- AMP on the local RTX 3050 (4GB)
- Per-epoch validation with the competition-metric proxy
- Per-class threshold sweep at the end; saves thresholds to JSON
"""
import json
import os
import sys
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle import rle_decode_np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
W, H = 800, 128
MEAN, STD = 0.3388, 0.2055
DEVICE = "cuda"
BATCH = 16
EPOCHS = 15
LR = 3e-4
NUM_WORKERS = 4

THRESHOLDS = np.arange(0.20, 0.91, 0.05, dtype=np.float32)


class DefectDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, augment: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.t = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=(MEAN,), std=(STD,), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        ) if augment else A.Compose(
            [A.Normalize(mean=(MEAN,), std=(STD,), max_pixel_value=1.0), ToTensorV2()]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row["image_id"] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((H, W), dtype=np.uint8)
        elif img.shape != (H, W):
            img = cv2.resize(img, (W, H))
        mask = np.zeros((H, W, 4), dtype=np.uint8)
        for ci, cls in enumerate(CLASSES):
            if row["defect_type"] == cls:
                mask[:, :, ci] = rle_decode_np(row["mask_rle"], H, W)
        out = self.t(image=img, mask=mask)
        return out["image"], out["mask"].permute(2, 0, 1)  # (1,H,W), (4,H,W)


def dice_soft_loss(pred, target, eps=1.0):
    """Soft Dice averaged over batch and classes, minimized (1 - dice)."""
    probs = torch.sigmoid(pred)
    inter = (probs * target).sum(dim=(0, 2, 3))
    sums = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (sums + eps)
    return 1 - dice.mean()


@torch.no_grad()
def validate(model, loader):
    """Per-class threshold sweep accumulating Dice over GT-nonempty rows and
    empty-row accuracy per class. Returns the best per-class thresholds and the
    combined proxy score at 0.5.

    proxy = 0.7 * mean_c(dice_c) + 0.3 * correctly_empty_all / total_empty_rows
    """
    model.eval()
    T = len(THRESHOLDS)
    tp = torch.zeros(T, 4, device=DEVICE)
    fp = torch.zeros(T, 4, device=DEVICE)
    fn = torch.zeros(T, 4, device=DEVICE)
    n_nonempty = torch.zeros(4, device=DEVICE)   # GT nonempty row counts
    n_empty = torch.zeros(4, device=DEVICE)      # GT empty row counts
    correct_empty = torch.zeros(T, 4, device=DEVICE)  # empty rows predicted empty
    dice50_num = torch.zeros(4, device=DEVICE)
    dice50_den = torch.zeros(4, device=DEVICE)
    for img, mask in loader:
        img = img.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True).bool()
        probs = torch.sigmoid(model(img))
        gte = mask.flatten(1).any(1)  # (B, 4) GT nonempty flags
        for ti, th in enumerate(THRESHOLDS):
            b = probs > th
            for c in range(4):
                ne = gte[:, c]
                if ne.any():
                    pb = b[:, c][ne]
                    gb = mask[:, c][ne]
                    tp[ti, c] += (pb & gb).sum().float()
                    fp[ti, c] += (pb & ~gb).sum().float()
                    fn[ti, c] += (~pb & gb).sum().float()
                correct_empty[ti, c] += ((~b[:, c].flatten(1).any(1)) & ~ne).sum().float()
                n_nonempty[c] += ne.float().sum()
                n_empty[c] += (~ne).float().sum()
        b5 = probs > 0.5
        for c in range(4):
            if gte[:, c].any():
                p = b5[:, c][gte[:, c]].float()
                g = mask[:, c][gte[:, c]].float()
                inter = (p * g).sum(dim=(1, 2))
                sums = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
                dice50_num[c] += (2 * inter).sum()
                dice50_den[c] += sums.sum()

    total_empty = n_empty.sum().clamp(min=1)
    sweep = {}
    best_proxy_sum = 0.0
    for ci, cls in enumerate(CLASSES):
        dice = (2 * tp[:, ci]) / (2 * tp[:, ci] + fp[:, ci] + fn[:, ci] + 1e-7)
        ea = correct_empty[:, ci] / total_empty  # class-c share of the empty term
        score_t = 0.7 * dice / 4 + 0.3 * ea
        best = int(torch.argmax(score_t))
        sweep[cls] = {
            "best_threshold": float(THRESHOLDS[best]),
            "best_dice": float(dice[best]),
            "dice50": float(dice50_num[ci] / (dice50_den[ci] + 1e-7)),
        }
        best_proxy_sum += float(score_t[best])
    sweep["best_proxy_total"] = best_proxy_sum
    sweep["proxy50"] = float(sum(
        0.7 * (dice50_num[i] / (dice50_den[i] + 1e-7)) / 4 for i in range(4)
    )) + float(sum(correct_empty[-1, i] for i in range(4)) / total_empty * 0.3)
    return sweep


def main():
    torch.backends.cudnn.benchmark = True
    train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    df = train_df.merge(split, on="image_id")
    tr = df[df["split"] == "train"]
    va_ids = set(df.loc[df["split"] == "val", "image_id"])
    va = df[df["image_id"].isin(va_ids)].reset_index(drop=True)
    tr = tr.reset_index(drop=True)
    img_dir = os.path.join(DATA, "public", "train")

    # class weights for BCE (presence frequency)
    freq = tr.groupby("defect_type").apply(
        lambda g: g["mask_rle"].fillna("").astype(str).str.strip().ne("").mean()
    )
    pos_weight = torch.tensor(
        [(1 - freq[c]) / max(freq[c], 1e-3) * 0.35 + 1.0 for c in CLASSES],
        dtype=torch.float32, device=DEVICE,
    )
    print("pos_weight:", pos_weight.tolist())

    train_ds = DefectDataset(tr, img_dir, augment=True)
    val_ds = DefectDataset(va, img_dir, augment=False)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=True, persistent_workers=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=True)

    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet",
                     in_channels=1, classes=4).to(DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    log_rows = []
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (img, mask) in enumerate(train_ld):
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(img)
                loss = bce(out, mask) + dice_soft_loss(out, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()
        sweep = validate(model, val_ld)
        elapsed = time.time() - t0
        score = sweep["proxy50"]
        log_rows.append({"epoch": ep, "loss": running / max(len(train_ld), 1),
                         "proxy50": score, "best_proxy": sweep["best_proxy_total"],
                         "dice50_a": sweep[CLASSES[0]]["dice50"], "dice50_b": sweep[CLASSES[1]]["dice50"],
                         "dice50_c": sweep[CLASSES[2]]["dice50"], "dice50_d": sweep[CLASSES[3]]["dice50"],
                         "sec": elapsed})
        print(f"ep {ep:02d} loss {running/len(train_ld):.4f} proxy50 {score:.4f} "
              f"best_proxy {sweep['best_proxy_total']:.4f} "
              f"dice50 {[round(sweep[c]['dice50'], 3) for c in CLASSES]} "
              f"({elapsed:.0f}s)")
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(CKPT, "unet_effb0_best.pt"))
            with open(os.path.join(CKPT, "unet_effb0_sweep.json"), "w") as f:
                json.dump(sweep, f, indent=2)
    pd.DataFrame(log_rows).to_csv(os.path.join(OUT, "train_log_unet_effb0.csv"), index=False)
    print("best proxy50:", best)


if __name__ == "__main__":
    main()
