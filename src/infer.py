"""Generate test predictions -> submission CSV.

Uses the trained checkpoint + per-class thresholds from the validation sweep.
Writes submissions/submission_{method}.csv in the exact sample format:
columns ID, image_id, defect_type, mask_rle (empty rows as a single space,
quoted, matching sample_submission.csv).
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle import rle_encode  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
SUBS = os.path.join(ROOT, "submissions")
os.makedirs(SUBS, exist_ok=True)

CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
W, H = 800, 128
MEAN, STD = 0.3388, 0.2055
DEVICE = "cuda"


ARCHS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "segformer": smp.Segformer,
    "upernet": smp.UPerNet,
}


def load_model(ckpt_path: str, arch: str = "segformer", encoder: str = "mit_b2"):
    model = ARCHS[arch](encoder_name=encoder, encoder_weights=None,
                        in_channels=1, classes=4).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_paths, batch_size=16):
    probs_out = []
    for i in range(0, len(image_paths), batch_size):
        batch = []
        for p in image_paths[i:i + batch_size]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((H, W), dtype=np.uint8)
            elif img.shape != (H, W):
                img = cv2.resize(img, (W, H))
            img = (img.astype(np.float32) / 255.0 - MEAN) / STD
            batch.append(img)
        x = torch.from_numpy(np.stack(batch))[:, None].to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits)
        probs_out.append(probs.cpu())
    return torch.cat(probs_out)  # (N, 4, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT, "segformer_mit_b2_best.pt"))
    ap.add_argument("--sweep", default=os.path.join(CKPT, "segformer_mit_b2_sweep.json"))
    ap.add_argument("--arch", default="segformer")
    ap.add_argument("--encoder", default="mit_b2")
    ap.add_argument("--method", default="segformer_mit_b2")
    ap.add_argument("--thresholds", default=None,
                    help="JSON file overriding per-class thresholds")
    ap.add_argument("--tta_hflip", action="store_true")
    args = ap.parse_args()

    with open(args.sweep) as f:
        sweep = json.load(f)
    thresholds = {c: sweep[c]["best_threshold"] for c in CLASSES}
    if args.thresholds and os.path.exists(args.thresholds):
        with open(args.thresholds) as f:
            thresholds = {c: float(v) for c, v in json.load(f).items()}
    print("thresholds:", thresholds)

    sub = pd.read_csv(os.path.join(DATA, "public", "sample_submission.csv"))
    test_dir = os.path.join(DATA, "public", "test")
    unique_ids = sub["image_id"].drop_duplicates().tolist()
    paths = [os.path.join(test_dir, iid + ".jpg") for iid in unique_ids]

    model = load_model(args.ckpt, args.arch, args.encoder)
    probs = predict(model, paths)
    if args.tta_hflip:
        probs_f = []
        for i in range(0, len(paths), 32):
            batch = []
            for p in paths[i:i + 32]:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.zeros((H, W), dtype=np.uint8)
                elif img.shape != (H, W):
                    img = cv2.resize(img, (W, H))
                img = (img.astype(np.float32) / 255.0 - MEAN) / STD
                batch.append(img)
            x = torch.from_numpy(np.stack(batch))[:, None].to(DEVICE).flip(-1)
            logits = model(x)
            pr = torch.sigmoid(logits).flip(-1).cpu()
            probs_f.append(pr)
        probs = (probs + torch.cat(probs_f)) / 2

    probs_by_id = {iid: probs[i] for i, iid in enumerate(unique_ids)}

    rows = []
    for _, r in sub.iterrows():
        iid, cls = r["image_id"], r["defect_type"]
        ci = CLASSES.index(cls)
        mask = (probs_by_id[iid][ci].numpy() > thresholds[cls]).astype(np.uint8)
        rows.append(rle_encode(mask, order="col"))

    sub["mask_rle"] = [" " if s == "" else s for s in rows]
    out_path = os.path.join(SUBS, f"submission_{args.method}.csv")
    sub.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
    print("wrote", out_path, os.path.getsize(out_path), "bytes")


if __name__ == "__main__":
    main()
