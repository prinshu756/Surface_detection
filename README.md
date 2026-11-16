# Surface_detection

Kaggle competition: [SurfaceGuard AI: Industrial Surface Anomaly Segmentation](https://www.kaggle.com/competitions/industrial-surface-defect-detection)

- 4-class defect segmentation (defect_a..d) on 800x128 industrial strip images
- Metric: `0.70*ForegroundDice + 0.20*FalsePositiveScore + 0.10*EmptyAccuracy`
- RLE masks, column-major order

## Layout
- `src/` — data utils, EDA (GPU), training, inference
- `data/` — competition data (gitignored)
- `checkpoints/` — model weights (gitignored)
- `submissions/` — generated submission CSVs (unique per method)
- `outputs/` — reports/logs (eda_report.json committed; images ignored)
- `PROGRESS.md` — experiment log and submission history

## Reproduce
```bash
python src/eda_gpu.py     # GPU EDA
python src/split.py       # stratified train/val split
python src/train.py       # UNet-EffB0 training (CUDA)
python src/infer.py --method unet_effb0
```
