# PROGRESS — SurfaceGuard AI (industrial-surface-defect-detection)

Target: **public LB ≥ 0.65**. Deadline: **2026-11-18 18:30 UTC**. Max 5 subs/day.

## Competition facts
- Metric: `0.70*ForegroundDice + 0.20*FalsePositiveScore + 0.10*EmptyAccuracy`
- RLE: column-major, 1-based starts. Images 800x128 grayscale JPG.
- Train: 11,393 images / 45,572 rows. Test: 3,000 images / 12,000 rows.
- Class presence (train rows): c 36.0%, d 15.0%, a 9.5%, b 2.7%. Only 15.8% of rows have defects.
- Defect shapes: b = thin vertical band (~12px), a = blob, c/d = wide bands.
- sample_submission.csv contains 1,146 pre-filled masks with *equal counts per class*
  (282/282/299/283) unlike the imbalanced train distribution -> synthetic demo/distractor
  masks, not test GT. Decision: ignore them, predict all rows ourselves.
- Baseline scores: all-empty ≈ 0.253 (proxy), sample-as-is = 0.273 (user's first sub).

## Setup
- Local: RTX 3050 4GB, torch 2.11+cu128, smp 0.6.0.
- Split: data/split.csv — 10,253 train / 1,140 val, stratified by defect pattern (seed 42).
- Local proxy metric: 0.7 * mean per-class Dice (GT-nonempty rows) + 0.3 * empty-row accuracy.

## Submissions log
| # | Method | File | CV proxy | Public LB | Notes |
|---|--------|------|----------|-----------|-------|
| 0 | sample-as-is (user) | submission.csv | — | 0.27337 | pre-filled sample masks ≈ noise |
