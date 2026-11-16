"""Local proxy of the competition metric.

Official: Score = 0.70 * ForegroundDice + 0.20 * FalsePositiveScore + 0.10 * EmptyAccuracy

Exact definitions of the FP/Empty components are not published, so we use a
practical proxy:
    proxy = 0.7 * mean(Dice over GT-nonempty rows) + 0.3 * frac(GT-empty rows predicted empty)
(empty-vs-empty Dice counts as 1; GT-empty rows predicted nonempty count as 0
for the empty-accuracy term).
The optimization direction (high per-class thresholds, suppress FP) is identical.
"""
import torch


@torch.no_grad()
def competition_proxy_score(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """pred, gt: (N, H, W) binary tensors (0/1) for a single class or concatenated rows.

    Returns dict with fg_dice, empty_acc, proxy.
    """
    pred = pred.bool()
    gt = gt.bool()
    gt_nonempty = gt.flatten(1).any(1)
    pred_empty = ~pred.flatten(1).any(1)

    dice = torch.zeros(gt.shape[0], device=gt.device)
    if gt_nonempty.any():
        p = pred[gt_nonempty].float()
        g = gt[gt_nonempty].float()
        inter = (p * g).sum(dim=(1, 2))
        sums = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
        d = (2 * inter) / (sums + 1e-7)
        d[sums == 0] = 1.0  # both empty within the nonempty-GT subset (rare)
        dice[gt_nonempty] = d
    empty_acc = (pred_empty & ~gt_nonempty).float().sum() / (~gt_nonempty).float().sum().clamp(min=1)
    fg_dice = dice[gt_nonempty].mean() if gt_nonempty.any() else torch.tensor(0.0)
    proxy = 0.7 * fg_dice + 0.3 * empty_acc
    return {
        "fg_dice": float(fg_dice),
        "empty_acc": float(empty_acc),
        "proxy": float(proxy),
        "n_gt_nonempty": int(gt_nonempty.sum()),
        "n_gt_empty": int((~gt_nonempty).sum()),
    }
