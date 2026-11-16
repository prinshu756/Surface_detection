"""GPU-accelerated Phase-0 EDA for the SurfaceGuard AI competition.

Everything heavy (RLE decoding of ~45k masks, pixel statistics over thousands of
images) runs on the local CUDA GPU.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle import rle_decode_torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
report = {}


def chunked_mask_stats(rles, widths, heights, chunk=2048):
    """Decode all RLEs on GPU in chunks; return per-row stats as numpy arrays."""
    n = len(rles)
    nonempty = np.zeros(n, dtype=bool)
    areas = np.zeros(n, dtype=np.int64)
    t_x = np.zeros(n, dtype=np.int64)  # transitions along width axis
    t_y = np.zeros(n, dtype=np.int64)  # transitions along height axis
    bbox = np.full((n, 4), -1, dtype=np.int64)  # r0, r1, c0, c1 (inclusive)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        for j in range(sl.start, sl.stop):
            m = rle_decode_torch(rles.iat[j], int(widths.iat[j]), int(heights.iat[j]), device=DEV)
            f = m.float()
            nonempty[j] = bool(m.sum().item() > 0)
            if not nonempty[j]:
                continue
            areas[j] = int(m.sum().item())
            t_x[j] = int((m[:, 1:] != m[:, :-1]).sum().item())
            t_y[j] = int((m[1:, :] != m[:-1, :]).sum().item())
            rows_any = m.any(dim=1)
            cols_any = m.any(dim=0)
            r_idx = torch.nonzero(rows_any, as_tuple=False).flatten()
            c_idx = torch.nonzero(cols_any, as_tuple=False).flatten()
            bbox[j] = (int(r_idx[0]), int(r_idx[-1]), int(c_idx[0]), int(c_idx[-1]))
    return nonempty, areas, t_x, t_y, bbox


def main():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    pub_train = pd.read_csv(os.path.join(DATA, "public", "train.csv"))
    report["train_shape"] = list(train.shape)
    report["identical_train_csvs"] = bool(train.equals(pub_train))
    report["widths"] = sorted(train["width"].unique().tolist())
    report["heights"] = sorted(train["height"].unique().tolist())

    # image presence check
    train_dir = os.path.join(DATA, "public", "train")
    img_files = set(os.listdir(train_dir))
    ids = set(train["image_id"].unique())
    report["n_train_images"] = len(img_files)
    report["n_train_ids_in_csv"] = len(ids)
    missing = list(ids - {os.path.splitext(f)[0] for f in img_files})[:5]
    report["ids_missing_image"] = missing
    exts = {os.path.splitext(f)[1] for f in img_files}
    report["image_extensions"] = list(exts)

    # --- global mask stats (GPU) ---
    rles = train["mask_rle"].fillna("")
    nonempty, areas, t_x, t_y, bbox = chunked_mask_stats(rles, train["width"], train["height"])
    train["nonempty"] = nonempty
    train["area"] = areas
    train["t_x"] = t_x
    train["t_y"] = t_y
    for c in range(4):
        bbox_c = bbox[train["defect_type"].values == f"defect_{chr(97 + c)}"]
        bbox_c = bbox_c[bbox_c[:, 0] >= 0]
        if len(bbox_c):
            report[f"defect_{chr(97+c)}_bbox_rows_mean"] = float((bbox_c[:, 1] - bbox_c[:, 0] + 1).mean())
            report[f"defect_{chr(97+c)}_bbox_cols_mean"] = float((bbox_c[:, 3] - bbox_c[:, 2] + 1).mean())
    report["n_nonempty"] = int(nonempty.sum())
    report["p_nonempty"] = float(nonempty.mean())
    per_class = {}
    for cls in ["defect_a", "defect_b", "defect_c", "defect_d"]:
        sel = train["defect_type"] == cls
        ne = nonempty[sel.values]
        a = areas[sel.values]
        a = a[a > 0]
        per_class[cls] = {
            "n_rows": int(sel.sum()),
            "n_defect": int(ne.sum()),
            "frac_defect": float(ne.mean()),
            "area_mean": float(a.mean()) if len(a) else 0.0,
            "area_median": float(np.median(a)) if len(a) else 0.0,
            "area_p90": float(np.percentile(a, 90)) if len(a) else 0.0,
            "area_max": int(a.max()) if len(a) else 0,
            "t_x_over_area": float((t_x[sel.values][ne].sum() / max(a.sum(), 1))) if ne.sum() else 0.0,
            "t_y_over_area": float((t_y[sel.values][ne].sum() / max(a.sum(), 1))) if ne.sum() else 0.0,
        }
    report["per_class"] = per_class

    # images with multiple defect types
    g = train.groupby("image_id")["nonempty"].sum()
    report["images_with_multiple_defects"] = int((g > 1).sum())
    report["images_with_any_defect"] = int((g >= 1).sum())
    report["n_images"] = int(len(g))

    # --- sample submission filled masks analysis (GPU) ---
    sub = pd.read_csv(os.path.join(DATA, "public", "sample_submission.csv"))
    sub["filled"] = sub["mask_rle"].fillna("").astype(str).str.strip().ne("")
    report["sub_shape"] = list(sub.shape)
    report["sub_n_filled"] = int(sub["filled"].sum())
    report["sub_filled_per_class"] = {
        cls: int(sub.loc[(sub["defect_type"] == cls) & sub["filled"]].shape[0])
        for cls in ["defect_a", "defect_b", "defect_c", "defect_d"]
    }
    sub_ne, sub_areas, _, _, sub_bbox = chunked_mask_stats(
        sub["mask_rle"].fillna(""), pd.Series([800] * len(sub)), pd.Series([128] * len(sub))
    )
    sub_a = sub_areas[sub_areas > 0]
    report["sub_filled_area_mean"] = float(sub_a.mean()) if len(sub_a) else 0.0
    report["sub_filled_area_median"] = float(np.median(sub_a)) if len(sub_a) else 0.0

    # how the filled rows distribute over images: unique images touched
    report["sub_filled_unique_images"] = int(sub.loc[sub["filled"], "image_id"].nunique())

    # --- verify train images on GPU: mean/std over a sample ---
    ids_sample = train.loc[train["nonempty"], "image_id"].drop_duplicates().head(300).tolist()
    stack = np.empty((len(ids_sample), 128, 800, 3), dtype=np.float32)
    for k, iid in enumerate(ids_sample):
        img = cv2.imread(os.path.join(train_dir, iid + ".jpg"))
        if img is None:
            stack[k] = 0
            continue
        if img.shape != (128, 800, 3):
            img = cv2.resize(img, (800, 128))
        stack[k] = img.astype(np.float32) / 255.0
    t = torch.from_numpy(stack).permute(0, 3, 1, 2).to(DEV)
    report["img_mean_rgb"] = [round(float(v), 4) for v in t.mean(dim=(0, 2, 3)).tolist()]
    report["img_std_rgb"] = [round(float(v), 4) for v in t.std(dim=(0, 2, 3)).tolist()]
    del t, stack
    torch.cuda.empty_cache()

    # --- orientation sanity: overlay PNGs for 2 images (col vs row) ---
    multi = train[train["nonempty"]].groupby("image_id")["area"].sum().sort_values(ascending=False).head(2).index.tolist()
    for iid in multi:
        img = cv2.imread(os.path.join(train_dir, iid + ".jpg"))
        img = cv2.resize(img, (800, 128)) if img.shape[:2] != (128, 800) else img
        rows = train[(train["image_id"] == iid) & train["nonempty"]]
        canvas_col = img.copy()
        canvas_row = img.copy()
        for _, r in rows.iterrows():
            m_col = rle_decode_torch(r["mask_rle"], 800, 128, order="col", device="cuda").cpu().numpy()
            m_row = rle_decode_torch(r["mask_rle"], 800, 128, order="row", device="cuda").cpu().numpy()
            ci = ["defect_a", "defect_b", "defect_c", "defect_d"].index(r["defect_type"])
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)][ci]
            for m, canvas in ((m_col, canvas_col), (m_row, canvas_row)):
                canvas[m > 0] = (canvas[m > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT, f"overlay_col_{iid}.jpg"), canvas_col)
        cv2.imwrite(os.path.join(OUT, f"overlay_row_{iid}.jpg"), canvas_row)
        # transitions metric: fewer transitions => more coherent regions
        m_c = rle_decode_torch(rows.iloc[0]["mask_rle"], 800, 128, "col").float()
        m_r = rle_decode_torch(rows.iloc[0]["mask_rle"], 800, 128, "row").float()
        report[f"orient_{iid}"] = {
            "col_tx": int((m_c[:, 1:] != m_c[:, :-1]).sum().item()),
            "col_ty": int((m_c[1:, :] != m_c[:-1, :]).sum().item()),
            "row_tx": int((m_r[:, 1:] != m_r[:, :-1]).sum().item()),
            "row_ty": int((m_r[1:, :] != m_r[:-1, :]).sum().item()),
        }

    # --- metric model sanity: what would all-empty score? (rough local proxy) ---
    # local proxy: 0.7 * mean(Dice over GT-nonempty rows) + 0.3 * frac(GT-empty rows predicted empty)
    frac_defect_rows = float(nonempty.mean())
    report["all_empty_proxy_score"] = 0.7 * 0.0 + 0.3 * (1 - frac_defect_rows)

    with open(os.path.join(OUT, "eda_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
