"""RLE encode/decode utilities for the industrial surface defect competition.

The competition masks are stored as run-length pairs: `start length start length ...`
on a flattened image grid. Two orderings are possible (column-major like Severstal,
or row-major). The decoder supports both; the EDA script determines which one the
competition uses by comparing compactness statistics on GPU.
"""
import numpy as np
import torch


def rle_decode_np(rle, height: int, width: int) -> np.ndarray:
    """Decode one RLE string (column-major, 1-based starts) into (H, W) uint8 mask."""
    H, W = height, width
    if rle is None:
        return np.zeros((H, W), dtype=np.uint8)
    s = str(rle).strip()
    if s == "" or s == "nan":
        return np.zeros((H, W), dtype=np.uint8)
    v = np.array(s.split(), dtype=np.int64)
    starts = v[0::2] - 1
    lens = v[1::2]
    pos = np.zeros(H * W, dtype=np.uint8)
    for st, ln in zip(starts, lens):
        pos[st:st + ln] = 1
    return pos.reshape((H, W), order="F")


def rle_decode_torch(rle: str, width: int, height: int, order: str = "col",
                     device: str = "cuda") -> torch.Tensor:
    """Decode one RLE string into a (H, W) uint8 mask tensor on `device`.

    order="col": flatten in column-major (Fortran) order, like Severstal.
    order="row": flatten in row-major (C) order.
    """
    H, W = height, width
    m = torch.zeros(H * W, dtype=torch.uint8, device=device)
    if rle is None:
        return m.view(H, W)
    rle = str(rle).strip()
    if rle == "" or rle == "nan":
        return m.view(H, W)
    v = torch.tensor([int(x) for x in rle.split()], dtype=torch.long, device=device)
    starts = v[0::2] - 1          # RLE positions are 1-based
    lengths = v[1::2]
    idx = starts[:, None] + torch.arange(int(lengths.max()), device=device)[None, :]
    keep = torch.arange(int(lengths.max()), device=device)[None, :] < lengths[:, None]
    flat_idx = idx[keep]
    m[flat_idx] = 1
    if order == "col":
        # position p in column-major of (W,H) -> row-major index: r = p % H, c = p // H
        rows = flat_idx % H
        cols = flat_idx // H
        m2 = torch.zeros(H * W, dtype=torch.uint8, device=device)
        m2[rows * W + cols] = 1
        return m2.view(H, W)
    return m.view(H, W)


def rle_encode(mask: np.ndarray, order: str = "col") -> str:
    """Encode a (H, W) binary mask (numpy) to RLE string (1-based start, length)."""
    if order == "col":
        flat = mask.flatten(order="F")
    else:
        flat = mask.flatten(order="C")
    flat = flat.astype(np.uint8)
    n = len(flat)
    if n == 0 or flat.sum() == 0:
        return ""
    change = np.nonzero(np.diff(flat))[0] + 1
    idx = np.concatenate(([0], change, [n]))
    runs = idx[1:] - idx[:-1]
    starts = idx[:-1] + 1  # 1-based
    if flat[0] == 1:
        # first run is foreground
        return " ".join(f"{s} {l}" for s, l in zip(starts[::2], runs[::2]))
    return " ".join(f"{s} {l}" for s, l in zip(starts[1::2], runs[1::2]))


def batch_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7,
               empty_as_one: bool = True) -> torch.Tensor:
    """Dice per element of a batch of (N, H, W) masks.

    empty_as_one=True mimics the common convention: when both pred and gt are
    empty, Dice counts as 1.0 for that row/class.
    """
    pred = pred.float()
    gt = gt.float()
    inter = (pred * gt).sum(dim=(1, 2))
    sums = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (sums + eps)
    both_empty = (sums == 0)
    if empty_as_one:
        dice = torch.where(both_empty, torch.ones_like(dice), dice)
    else:
        dice = torch.where(both_empty, torch.zeros_like(dice), dice)
    return dice
