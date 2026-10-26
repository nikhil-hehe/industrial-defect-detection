"""Train the autoencoder on NORMAL images only.

    python -m src.training.train --category bottle --epochs 50
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NormalDataset, split_normal_train
from src.inference.predict import anomaly_maps, image_scores
from src.models.autoencoder import build_model
from src.utils.common import ensure_dir, get_device, load_config, seed_everything


def run_epoch(model, loader, device, optimizer=None, scaler=None, noise_std=0.0, amp=False, desc=""):
    train = optimizer is not None
    model.train(train)
    crit = nn.MSELoss()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False)
    for x in bar:
        x = x.to(device, non_blocking=True)
        inp = x + noise_std * torch.randn_like(x) if (train and noise_std > 0) else x
        inp = inp.clamp(0, 1)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                recon = model(inp)
                loss = crit(recon, x)          # always reconstruct the CLEAN image
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        bar.set_postfix(loss=f"{loss.item():.5f}")
    return total / max(n, 1)


@torch.no_grad()
def validation_statistics(model, loader, device, cfg):
    """Scores + pixel errors on held-out NORMAL images -> thresholds (never uses test data)."""
    model.eval()
    scores, pix = [], []
    for x in loader:
        x = x.to(device)
        maps = anomaly_maps(x, model(x), cfg["smooth_sigma"])
        scores.extend(image_scores(maps, cfg["score_percentile"]).tolist())
        pix.append(maps.reshape(-1)[::7])          # subsample pixels to keep memory small
    scores = np.array(scores)
    pix = np.concatenate(pix)
    return {
        "image_threshold": float(np.percentile(scores, cfg["threshold_percentile"])),
        "pixel_threshold": float(np.percentile(pix, cfg["pixel_threshold_percentile"])),
        "val_score_mean": float(scores.mean()),
        "val_score_std": float(scores.std()),
        "val_score_max": float(scores.max()),
        "n_val": int(len(scores)),
    }


def plot_history(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["train_loss"], label="train", lw=2)
    ax.plot(hist["val_loss"], label="validation", lw=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend(); ax.set_title("Training history")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--category", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--latent_channels", type=int, default=None)
    ap.add_argument("--noise_std", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, category=a.category, data_root=a.data_root, epochs=a.epochs,
                      batch_size=a.batch_size, lr=a.lr, latent_channels=a.latent_channels,
                      noise_std=a.noise_std)

    seed_everything(cfg["seed"])
    device = get_device()
    cat_root = Path(cfg["data_root"]) / cfg["category"]
    train_paths, val_paths = split_normal_train(cat_root, cfg["val_split"], cfg["seed"])
    print(f"Device: {device} | category: {cfg['category']} | train {len(train_paths)} | val {len(val_paths)}")

    pin = device.type == "cuda"
    tr = DataLoader(NormalDataset(train_paths, cfg["img_size"], cfg["augment"]), cfg["batch_size"],
                    shuffle=True, num_workers=cfg["num_workers"], pin_memory=pin, drop_last=len(train_paths) > cfg["batch_size"])
    va = DataLoader(NormalDataset(val_paths, cfg["img_size"], False), cfg["batch_size"],
                    shuffle=False, num_workers=cfg["num_workers"], pin_memory=pin)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda") if (cfg["amp"] and device.type == "cuda") else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    ckpt_dir, out_dir = ensure_dir(cfg["ckpt_dir"]), ensure_dir(Path(cfg["out_dir"]) / "plots")
    best_path = ckpt_dir / f"{cfg['category']}_best.pt"
    hist = {"train_loss": [], "val_loss": [], "lr": []}
    best, bad, t0 = float("inf"), 0, time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        tl = run_epoch(model, tr, device, opt, scaler, cfg["noise_std"], cfg["amp"], f"epoch {epoch} train")
        vl = run_epoch(model, va, device, desc=f"epoch {epoch} val")
        sched.step(vl)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl); hist["lr"].append(opt.param_groups[0]["lr"])
        flag = ""
        if vl < best - 1e-7:
            best, bad, flag = vl, 0, "  * best"
            stats = validation_statistics(model, va, device, cfg)
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": vl,
                        "image_threshold": stats["image_threshold"],
                        "pixel_threshold": stats["pixel_threshold"], "val_stats": stats}, best_path)
        else:
            bad += 1
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  train {tl:.5f}  val {vl:.5f}  lr {opt.param_groups[0]['lr']:.1e}{flag}")
        if bad >= cfg["patience"]:
            print(f"Early stopping (no improvement for {cfg['patience']} epochs).")
            break

    hist["train_time_sec"] = time.time() - t0
    with open(out_dir / f"{cfg['category']}_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, out_dir / f"{cfg['category']}_loss.png")
    print(f"\nBest val loss {best:.5f}. Checkpoint: {best_path}")
    print(f"Next: python -m src.evaluation.evaluate --category {cfg['category']}")


if __name__ == "__main__":
    main()
