"""Helper for getting MVTec AD into data/raw/.

MVTec AD is CC BY-NC-SA 4.0 and its official download sits behind a short form, so the
reliable route is: download from the official page in a browser, then let this script extract it.

    python scripts/download_data.py --archive ~/Downloads/mvtec_anomaly_detection.tar.xz
    python scripts/download_data.py --archive ~/Downloads/bottle.tar.xz --category-archive
    python scripts/download_data.py --kaggle-slug <owner/dataset>      # needs kaggle CLI + API token
"""
import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

OFFICIAL = "https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads"
DEST = Path("data/raw")


def extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest} (this can take a few minutes)...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)
    print("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="Path to the downloaded .tar.xz / .zip")
    ap.add_argument("--category-archive", action="store_true",
                    help="Archive is a single category (e.g. bottle.tar.xz); extract into mvtec_anomaly_detection/")
    ap.add_argument("--kaggle-slug", help="Kaggle dataset slug of an MVTec AD mirror (verify its license/structure)")
    a = ap.parse_args()

    if a.kaggle_slug:
        DEST.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["kaggle", "datasets", "download", "-d", a.kaggle_slug, "-p", str(DEST), "--unzip"])
    elif a.archive:
        target = DEST / "mvtec_anomaly_detection" if a.category_archive else DEST
        extract(Path(a.archive).expanduser(), target)
    else:
        print(f"1) Open {OFFICIAL}\n2) Fill the form, download the full dataset (~4.9 GB) or just 'bottle'\n"
              f"3) Re-run: python scripts/download_data.py --archive <downloaded file>")
        sys.exit(0)

    root = DEST / "mvtec_anomaly_detection"
    if root.exists():
        cats = sorted(p.name for p in root.iterdir() if p.is_dir())
        print("Categories found:", ", ".join(cats))
    else:
        print("Extracted, but data/raw/mvtec_anomaly_detection/ not found - set data_root in configs/default.yaml.")


if __name__ == "__main__":
    main()
