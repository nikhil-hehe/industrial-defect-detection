"""Evaluate on the official test split.

    python -m src.evaluation.evaluate --category bottle
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from src.data.dataset import TestDataset
from src.inference.predict import (anomaly_maps, heatmap_only, image_scores, overlay_heatmap)
from src.models.autoencoder import build_model
from src.utils.common import ensure_dir, get_device, load_config


def iou_dice(pred, gt, eps=1e-8):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + eps), 2 * inter / (pred.sum() + gt.sum() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--ckpt_dir", default="models/checkpoints")
    ap.add_argument("--n_examples", type=int, default=8)
    a = ap.parse_args()

    device = get_device()
    ck = torch.load(Path(a.ckpt_dir) / f"{a.category}_best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if a.data_root:
        cfg["data_root"] = a.data_root
    model = build_model(cfg); model.load_state_dict(ck["model"]); model.to(device).eval()
    img_thr, pix_thr = ck["image_threshold"], ck["pixel_threshold"]

    ds = TestDataset(Path(cfg["data_root"]) / cfg["category"], cfg["img_size"])
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    imgs, recs, maps, masks, labels, defects = [], [], [], [], [], []
    t_total, n = 0.0, 0
    with torch.no_grad():
        for x, m, y, _, d in dl:
            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = model(x)
            mp = anomaly_maps(x, r, cfg["smooth_sigma"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_total += time.perf_counter() - t0; n += x.size(0)
            imgs.append(x.cpu().numpy()); recs.append(r.cpu().numpy()); maps.append(mp)
            masks.append(m.numpy()[:, 0]); labels.extend(y.tolist()); defects.extend(d)
    imgs, recs = np.concatenate(imgs), np.concatenate(recs)
    maps, masks, labels = np.concatenate(maps), np.concatenate(masks), np.array(labels)

    scores = image_scores(maps, cfg["score_percentile"])
    pred = (scores > img_thr).astype(int)
    res = {"category": cfg["category"], "n_test": int(len(labels)),
           "n_normal": int((labels == 0).sum()), "n_defect": int((labels == 1).sum()),
           "image_threshold": img_thr, "pixel_threshold": pix_thr}
    res["image_auroc"] = float(roc_auc_score(labels, scores))
    res["precision"] = float(precision_score(labels, pred, zero_division=0))
    res["recall"] = float(recall_score(labels, pred, zero_division=0))
    res["f1"] = float(f1_score(labels, pred, zero_division=0))
    cm_ = confusion_matrix(labels, pred, labels=[0, 1])
    res["confusion_matrix"] = cm_.tolist()
    res["accuracy"] = float((pred == labels).mean())

    gt_pix = (masks > 0.5).reshape(-1)
    if gt_pix.any():
        res["pixel_auroc"] = float(roc_auc_score(gt_pix, maps.reshape(-1)))
        defect_idx = np.where(labels == 1)[0]
        ious, dices = [], []
        for i in defect_idx:
            iou, dice = iou_dice(maps[i] > pix_thr, masks[i] > 0.5)
            ious.append(iou); dices.append(dice)
        res["mean_iou"], res["mean_dice"] = float(np.mean(ious)), float(np.mean(dices))
    res["inference_ms_per_image"] = 1000 * t_total / max(n, 1)
    res["per_defect_type_mean_score"] = {d: float(scores[np.array(defects) == d].mean()) for d in sorted(set(defects))}

    out = ensure_dir(Path(cfg["out_dir"]))
    plots = ensure_dir(out / "plots")
    with open(out / f"{cfg['category']}_evaluation.json", "w") as f:
        json.dump(res, f, indent=2)

    # --- ROC + confusion matrix ---
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, lw=2, label=f"AUROC {res['image_auroc']:.3f}"); ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(); ax[0].set_title("Image-level ROC")
    ax[1].imshow(cm_, cmap="Purples")
    for (i, j), v in np.ndenumerate(cm_):
        ax[1].text(j, i, int(v), ha="center", va="center", fontsize=16)
    ax[1].set_xticks([0, 1], ["Normal", "Defect"]); ax[1].set_yticks([0, 1], ["Normal", "Defect"])
    ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True"); ax[1].set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(plots / f"{cfg['category']}_roc_confusion.png", dpi=150); plt.close(fig)

    # --- Example grid: input | reconstruction | heatmap | ground truth ---
    pick = list(np.where(labels == 1)[0][: a.n_examples // 2 + 1]) + list(np.where(labels == 0)[0][: a.n_examples // 2])
    fig, axes = plt.subplots(len(pick), 4, figsize=(10, 2.6 * len(pick)))
    axes = np.atleast_2d(axes)
    vmax = max(pix_thr * 2, float(maps.max()))
    for r, i in enumerate(pick):
        im = (imgs[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        rc = (recs[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        panels = [(im, "input"), (rc, "reconstruction"),
                  (overlay_heatmap(im, maps[i], vmax), f"score {scores[i]:.4f}"), (masks[i], f"GT: {defects[i]}")]
        for c, (p, t) in enumerate(panels):
            axes[r, c].imshow(p, cmap="gray"); axes[r, c].set_title(t, fontsize=9); axes[r, c].axis("off")
    fig.tight_layout(); fig.savefig(plots / f"{cfg['category']}_examples.png", dpi=130); plt.close(fig)

    print(json.dumps({k: v for k, v in res.items() if k != "per_defect_type_mean_score"}, indent=2))
    print(f"\nSaved: {out / (cfg['category'] + '_evaluation.json')} and plots in {plots}")


if __name__ == "__main__":
    main()
