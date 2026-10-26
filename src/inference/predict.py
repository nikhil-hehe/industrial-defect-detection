"""Reconstruction, anomaly map, score, thresholding and heatmap overlay."""
import time

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.data.dataset import pil_to_tensor
from src.models.autoencoder import build_model
from src.utils.common import get_device


def anomaly_maps(x, recon, sigma=4.0) -> np.ndarray:
    """x, recon: (B,3,H,W) tensors in [0,1] -> (B,H,W) float32 smoothed error maps."""
    err = (x - recon).abs().mean(dim=1).cpu().numpy()
    if sigma and sigma > 0:
        err = np.stack([gaussian_filter(e, sigma=sigma) for e in err])
    return err.astype(np.float32)


def image_scores(maps: np.ndarray, percentile=99.0) -> np.ndarray:
    return np.array([np.percentile(m, percentile) for m in maps], dtype=np.float32)


def heatmap_only(amap: np.ndarray, vmax: float) -> np.ndarray:
    norm = np.clip(amap / max(vmax, 1e-8), 0, 1)
    return (cm.inferno(norm)[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(img_rgb: np.ndarray, amap: np.ndarray, vmax: float, alpha=0.55) -> np.ndarray:
    heat = heatmap_only(amap, vmax)
    return (img_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


class DefectDetector:
    def __init__(self, ckpt_path, device=None):
        self.device = device or get_device()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.size = self.cfg["img_size"]
        self.model = build_model(self.cfg)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.image_threshold = float(ck["image_threshold"])
        self.pixel_threshold = float(ck["pixel_threshold"])
        self.category = self.cfg["category"]
        self.val_stats = ck.get("val_stats", {})

    @torch.no_grad()
    def predict(self, pil_img: Image.Image, sensitivity: float = 1.0):
        """sensitivity > 1 = stricter (fewer alarms); < 1 = more sensitive."""
        x = pil_to_tensor(pil_img, self.size).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        recon = self.model(x)
        amap = anomaly_maps(x, recon, self.cfg["smooth_sigma"])[0]
        score = float(image_scores(amap[None], self.cfg["score_percentile"])[0])
        dt = time.perf_counter() - t0
        thr = self.image_threshold * sensitivity
        pthr = self.pixel_threshold * sensitivity
        inp = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rec = (recon[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return {
            "input": inp, "recon": rec, "amap": amap,
            "score": score, "threshold": thr, "pixel_threshold": pthr,
            "is_defect": score > thr, "mask": (amap > pthr),
            "vmax": max(pthr * 2.0, float(amap.max())),
            "time_ms": dt * 1000,
        }
