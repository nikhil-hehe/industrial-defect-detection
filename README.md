# DefectLens — Unsupervised Industrial Defect Detection with Convolutional Autoencoders

Trains on **normal images only**, reconstructs inputs, and flags defects from reconstruction error.
Outputs an image-level anomaly score, a pixel-level heatmap/mask, evaluation metrics and a Streamlit demo.

## 1. Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # for an NVIDIA GPU, install torch from https://pytorch.org/get-started/locally/ first
```

## 2. Get the dataset (MVTec AD — start with `bottle`)
1. Open the official page: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
2. Fill in the short form. Download either **the whole dataset (~4.9 GB)** or just the **bottle** category.
3. Extract it into `data/raw/`:
```bash
python scripts/download_data.py --archive ~/Downloads/mvtec_anomaly_detection.tar.xz
# single-category archive:
python scripts/download_data.py --archive ~/Downloads/bottle.tar.xz --category-archive
```
Expected layout:
```
data/raw/mvtec_anomaly_detection/bottle/{train/good, test/{good,broken_large,...}, ground_truth/...}
```
Other datasets (VisA, MPDD, KolektorSDD2, BTAD ...) are listed in the project guide; to use one, arrange it in the same
`<category>/train/good`, `test/<type>`, `ground_truth/<type>/<name>_mask.png` layout and point `data_root` at it.
MVTec AD is **CC BY-NC-SA 4.0 (non-commercial)** — don't commit or redistribute it, and cite Bergmann et al., CVPR 2019.

## 3. Smoke test (optional, 1 minute, no dataset needed)
```bash
python scripts/make_dummy_data.py
python -m src.training.train --data_root data/raw/dummy --category toy --epochs 3
python -m src.evaluation.evaluate --category toy --data_root data/raw/dummy
```

## 4. Train
```bash
python -m src.training.train --category bottle --epochs 50
```
Defaults (`configs/default.yaml`): 256×256, batch 16, AdamW lr 1e-3, early stopping (patience 10), 10% of normal
training images held out for validation. The best checkpoint goes to `models/checkpoints/bottle_best.pt` together
with thresholds computed from validation-normal images. Useful flags:
`--batch_size 8` (out of memory) · `--latent_channels 32` (stronger bottleneck) · `--noise_std 0.05` (denoising AE).

Free GPU: Google Colab → Runtime → T4 GPU, clone/upload this folder, run the same commands.

## 5. Evaluate
```bash
python -m src.evaluation.evaluate --category bottle
```
Writes `outputs/bottle_evaluation.json` (image AUROC, precision/recall/F1, confusion matrix, pixel AUROC, IoU, Dice,
inference time) and plots in `outputs/plots/` (loss curve, ROC + confusion matrix, example grid).

## 6. Run the app
```bash
streamlit run app/streamlit_app.py
```

## 7. Other categories / comparisons
Repeat with `--category screw`, `leather`, `hazelnut`, ... Compare `latent_channels` 128 vs 32 vs 16 and
`noise_std` 0 vs 0.05, then (optionally) PatchCore/PaDiM via Anomalib using the same split.

## Notes
- Thresholds are chosen from held-out **normal validation** images only; the test set is used once for final metrics.
- If the model reconstructs defects too well, lower `latent_channels`; if normal images get flagged, raise `threshold_percentile`.
- Predictions are for education/research, not certified QC decisions.
