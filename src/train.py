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
NUM_WORKERS = 0  # shared-memory worker transfer exhausts the Windows paging file on long runs (error 1455)
PIN_MEMORY = False  # Windows shared-memory paging-file limitation (error 1455)

THRESHOLDS = np.arange(0.20, 0.91, 0.05, dtype=np.float32)


class DefectDataset(Dataset):
    """Module-level so DataLoader workers on Windows (spawn) can pickle it."""

    def __init__(self, df: pd.DataFrame, img_dir: str, tf):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = tf

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
        out = self.tf(image=img, mask=mask)
        m = out["mask"]
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        if m.ndim == 3 and m.shape[-1] == 4:      # HWC -> CHW
            m = m.transpose(2, 0, 1)
        assert m.shape == (4, H, W), f"bad mask shape {m.shape}"
        return out["image"], torch.from_numpy(np.ascontiguousarray(m))


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
        gte = mask.flatten(2).any(2)  # (B, 4) per-class GT nonempty flags
        gte_f = gte.float()
        gsum = mask.sum(dim=(2, 3)).float()  # (B, 4) GT pixel counts
        n_nonempty += gte_f.sum(0)
        n_empty += (~gte).float().sum(0)
        for ti, th in enumerate(THRESHOLDS):
            b = probs > th
            inter = (b & mask).sum(dim=(2, 3)).float()  # (B, 4)
            psum = b.sum(dim=(2, 3)).float()  # (B, 4)
            # dice over GT-nonempty images only (GT-empty rows contribute 0)
            tp[ti] += (inter * gte_f).sum(0)
            fp[ti] += ((psum - inter) * gte_f).sum(0)
            fn[ti] += ((gsum - inter) * gte_f).sum(0)
            # GT-empty images whose class channel is predicted fully empty
            correct_empty[ti] += ((psum == 0) & ~gte).sum(0).float()
        b5 = probs > 0.5
        inter5 = (b5 & mask).sum(dim=(2, 3)).float()
        psum5 = b5.sum(dim=(2, 3)).float()
        dice50_num += (2 * inter5 * gte_f).sum(0)
        dice50_den += ((psum5 + gsum) * gte_f).sum(0)

    total_empty = n_empty.sum().clamp(min=1)
    idx50 = int(np.argmin(np.abs(THRESHOLDS - 0.5)))
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
    )) + float(sum(correct_empty[idx50, i] for i in range(4)) / total_empty * 0.3)
    return sweep


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None, help="encoder name; default depends on arch")
    ap.add_argument("--arch", default="segformer",
                    choices=["unet", "unetplusplus", "deeplabv3plus", "segformer", "upernet"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default=None, help="checkpoint/log tag; defaults to arch_encoder")
    ap.add_argument("--resume", action="store_true",
                    help="continue from {tag}_last.pt (model+optimizer+scheduler+scaler state)")
    ap.add_argument("--vflip", action="store_true", help="also use vertical flip augmentation")
    ap.add_argument("--rotate", action="store_true", help="add small ShiftScaleRotate augmentation")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.encoder is None:
        args.encoder = "mit_b2" if args.arch == "segformer" else "efficientnet-b0"
    tag = args.tag or f"{args.arch}_{args.encoder}"
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")  # Windows paging-file fix
    train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
    split = pd.read_csv(os.path.join(DATA, "split.csv"))
    df = train_df.merge(split, on="image_id")
    tr = df[df["split"] == "train"]
    va_ids = set(df.loc[df["split"] == "val", "image_id"])
    va = df[df["image_id"].isin(va_ids)].reset_index(drop=True)
    tr = tr.reset_index(drop=True)
    img_dir = os.path.join(DATA, "public", "train")

    # class weights for BCE (presence frequency)
    freq = tr.groupby("defect_type")["mask_rle"].apply(
        lambda s: s.fillna("").astype(str).str.strip().ne("").mean()
    )
    pos_weight = torch.tensor(
        [(1 - freq[c]) / max(freq[c], 1e-3) * 0.35 + 1.0 for c in CLASSES],
        dtype=torch.float32, device=DEVICE,
    ).view(1, 4, 1, 1)  # broadcast across (B, C, H, W)
    print("pos_weight:", pos_weight.flatten().tolist())

    augment = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(p=0.1),
    ]
    if args.vflip:
        augment.append(A.VerticalFlip(p=0.5))
    if args.rotate:
        augment.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                                          rotate_limit=5, border_mode=cv2.BORDER_REFLECT_101, p=0.3))
    aug_tf = A.Compose(augment + [
        A.Normalize(mean=(MEAN,), std=(STD,), max_pixel_value=1.0),
        ToTensorV2(),
    ])
    val_tf = A.Compose([A.Normalize(mean=(MEAN,), std=(STD,), max_pixel_value=1.0), ToTensorV2()])

    train_ds = DefectDataset(tr, img_dir, aug_tf)
    val_ds = DefectDataset(va, img_dir, val_tf)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0)

    arch = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3plus": smp.DeepLabV3Plus, "segformer": smp.Segformer,
            "upernet": smp.UPerNet}[args.arch]
    model = arch(encoder_name=args.encoder, encoder_weights="imagenet",
                 in_channels=1, classes=4).to(DEVICE)
    print(f"model: {args.arch}/{args.encoder}, params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    start_ep = 0
    best = -1.0
    last_path = os.path.join(CKPT, f"{tag}_last.pt")
    if args.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_ep = state["epoch"] + 1
        best = state["best"]
        print(f"resumed from {last_path}: epoch {start_ep}/{args.epochs}, best {best:.4f}")

    log_rows = []
    for ep in range(start_ep, args.epochs):
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
        score = sweep["best_proxy_total"]  # selection on tuned-threshold proxy (what inference deploys)
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
            torch.save(model.state_dict(), os.path.join(CKPT, f"{tag}_best.pt"))
            with open(os.path.join(CKPT, f"{tag}_sweep.json"), "w") as f:
                json.dump(sweep, f, indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best": best}, last_path)
    pd.DataFrame(log_rows).to_csv(os.path.join(OUT, f"train_log_{tag}.csv"), index=False)
    print("best proxy:", best)


if __name__ == "__main__":
    main()
