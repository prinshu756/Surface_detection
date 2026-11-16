"""Validate a submission CSV before uploading to Kaggle.

Checks: exact row count/order vs sample_submission.csv, column names,
mask_rle is either empty-placeholder or valid even-token RLE, decodes to a
non-empty mask when filled, and no NaN/garbage tokens.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\MachineLearning\industrial_surface_defect\src")
from rle import rle_decode_np  # noqa: E402

ROOT = r"D:\MachineLearning\industrial_surface_defect"


def main(path: str):
    sub = pd.read_csv(path)
    sample = pd.read_csv(f"{ROOT}\\data\\public\\sample_submission.csv")
    errors = []

    # 1) identical keys in identical order
    if list(sub.columns) != list(sample.columns):
        errors.append(f"columns differ: {list(sub.columns)}")
    if len(sub) != len(sample):
        errors.append(f"row count {len(sub)} != {len(sample)}")
    elif not (sub["ID"].equals(sample["ID"]) and sub["image_id"].equals(sample["image_id"])
              and sub["defect_type"].equals(sample["defect_type"])):
        errors.append("ID/image_id/defect_type sequence differs from sample")

    # 2) mask_rle validity
    rle = sub["mask_rle"].fillna("").astype(str).str.strip()
    filled = rle.ne("")
    print(f"rows: {len(sub)}, filled: {int(filled.sum())}, empty: {int((~filled).sum())}")
    print("filled per class:", sub.loc[filled].groupby("defect_type").size().to_dict())

    bad_tokens = 0
    bad_decode = 0
    n_check = 0
    for s in rle[filled].sample(min(200, int(filled.sum())), random_state=0):
        toks = s.split()
        if len(toks) == 0 or len(toks) % 2 != 0:
            bad_tokens += 1
            continue
        try:
            vals = np.array(toks, dtype=np.int64)
        except ValueError:
            bad_tokens += 1
            continue
        if (vals <= 0).any() or (vals[1::2] <= 0).any():
            bad_tokens += 1
            continue
        m = rle_decode_np(s, 128, 800)
        n_check += 1
        if m.sum() == 0:
            bad_decode += 1
    if bad_tokens:
        errors.append(f"{bad_tokens} RLE strings with bad tokens")
    if bad_decode:
        errors.append(f"{bad_decode}/200 filled masks decode to empty")

    if errors:
        print("FAIL:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("PASS: submission format is valid")


if __name__ == "__main__":
    main(sys.argv[1])
