"""Create a deterministic stratified train/val split.

Stratifies by the image's defect pattern (which of the 4 classes are present),
so rare patterns (e.g. images containing defect_b) appear in both splits.
Writes data/split.csv with columns: image_id, split.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SEED = 42


def main():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    train["has_mask"] = train["mask_rle"].fillna("").astype(str).str.strip().ne("")
    pattern = (
        train[train["has_mask"]]
        .groupby("image_id")["defect_type"]
        .apply(lambda s: "+".join(sorted(s)))
        .rename("pattern")
    )
    all_ids = train["image_id"].unique()
    pattern = pattern.reindex(all_ids, fill_value="none")  # include defect-free images
    ids = pattern.index.to_numpy()
    pat = pattern.values
    tr_idx, va_idx = train_test_split(
        np.arange(len(ids)), test_size=0.1, random_state=SEED, stratify=pat
    )
    df = pd.DataFrame({"image_id": ids, "pattern": pat})
    df["split"] = "train"
    df.loc[va_idx, "split"] = "val"
    df[["image_id", "split"]].to_csv(os.path.join(DATA, "split.csv"), index=False)
    # summary
    val_ids = set(df.loc[df["split"] == "val", "image_id"])
    val_rows = train[train["image_id"].isin(val_ids)]
    out = df["split"].value_counts().to_dict()
    per_class = val_rows[val_rows["has_mask"]]["defect_type"].value_counts().to_dict()
    print("split counts:", out)
    print("val defect rows per class:", per_class)


if __name__ == "__main__":
    main()
