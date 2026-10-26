"""Create a tiny synthetic MVTec-style dataset to smoke-test the pipeline (NOT for real results).

    python scripts/make_dummy_data.py
    python -m src.training.train --data_root data/raw/dummy --category toy --epochs 3
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root = Path("data/raw/dummy/toy")
rng = np.random.default_rng(0)
S = 128


def base():
    yy, xx = np.mgrid[:S, :S]
    r = np.sqrt((xx - S / 2) ** 2 + (yy - S / 2) ** 2)
    img = np.stack([120 + 60 * np.cos(r / 9)] * 3, -1) + rng.normal(0, 4, (S, S, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


for split, n in [("train/good", 60), ("test/good", 10)]:
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        base().save(d / f"{i:03d}.png")

(root / "test/scratch").mkdir(parents=True, exist_ok=True)
(root / "ground_truth/scratch").mkdir(parents=True, exist_ok=True)
for i in range(10):
    im, mask = base(), Image.new("L", (S, S), 0)
    x0, y0 = rng.integers(20, 80, 2)
    for target in (ImageDraw.Draw(im), ImageDraw.Draw(mask)):
        target.line([x0, y0, x0 + 30, y0 + 20], fill=255, width=5)
    im.save(root / f"test/scratch/{i:03d}.png")
    mask.save(root / f"ground_truth/scratch/{i:03d}_mask.png")
print("Dummy data written to", root.parent)
