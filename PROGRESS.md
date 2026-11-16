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

## Hardware findings (benchmarked)
- fp16 autocast train step: 398 ms @ B=10 (6.8 min/epoch, 2.69 GiB peak) — fp32 @ B=10 is 14x SLOWER (5.6 s/step) → always autocast for training.
- fp32 eval forward: 0.15 s/batch @ B=10 → validation ~30 s for 1,140 images.
- First eval call pays a one-time ~1-2 min cuDNN benchmark search (eval-mode shapes).
- DataLoader: 18 ms/batch with 2 workers (Windows: file_system sharing, no pin_memory, module-level Dataset for spawn pickling).

## Model
- Primary: **SegFormer (smp.Segformer, mit_b2, 24.7M params)**, batch 10, 15 epochs, BCE(pos_weight)+Dice, cosine LR, HFlip/Brightness/GaussNoise aug.
- Per-class threshold sweep on val each epoch; checkpoint selected on tuned-threshold proxy.

## Phase map
| Phase | Files | Status |
|---|---|---|
| 0 — Recon/EDA | `src/eda_gpu.py`, `src/rle.py`, `src/metric.py`, `src/split.py`, `outputs/eda_report.json` | done |
| 1 — Baseline + first submission | `src/train.py`, `src/infer.py`, `src/check_submission.py` | running |
| 2 — LB-feedback iteration (thresholds/TTA/aug) | TBD from first LB score | next |
| 3 — Ensemble / final submissions | TBD | planned |

## Submissions log
| # | Method | File | CV proxy | Public LB | Notes |
|---|--------|------|----------|-----------|-------|
| 0 | sample-as-is (user) | submission.csv | — | 0.27337 | pre-filled sample masks ≈ noise |
