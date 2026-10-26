"""Industrial Defect Inspector - Streamlit UI.   Run:  streamlit run app/streamlit_app.py"""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from src.data.dataset import list_images
from src.inference.predict import DefectDetector, heatmap_only, overlay_heatmap

st.set_page_config(page_title="DefectLens - Industrial Inspector", page_icon="🔍", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b1f4b 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, #2a1456 0%, transparent 55%), #0B0E1A; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
.hero { padding: 28px 32px; border-radius: 22px; margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(124,92,255,.22), rgba(0,212,255,.10));
        border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(10px); }
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em;
           background: linear-gradient(90deg,#fff,#b9a8ff 60%,#6fe3ff); -webkit-background-clip: text;
           -webkit-text-fill-color: transparent; }
.hero p { margin: 6px 0 0; color: #AEB4D6; font-size: 1.02rem; }
.pill { display:inline-block; padding: 3px 12px; margin: 10px 8px 0 0; border-radius: 999px; font-size:.78rem;
        color:#cfd3f5; background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.1); }
.card { padding: 14px 14px 10px; border-radius: 18px; background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08); }
.cap { text-align:center; color:#AEB4D6; font-size:.82rem; margin-top:6px; letter-spacing:.06em; text-transform:uppercase; }
.verdict { padding: 22px 26px; border-radius: 20px; display:flex; align-items:center; gap:18px; margin: 6px 0 16px; }
.verdict.ok  { background: linear-gradient(135deg, rgba(16,185,129,.22), rgba(16,185,129,.06)); border:1px solid rgba(16,185,129,.45); }
.verdict.bad { background: linear-gradient(135deg, rgba(244,63,94,.25), rgba(244,63,94,.06)); border:1px solid rgba(244,63,94,.55); }
.verdict .ic { font-size: 2.4rem; }
.verdict .t  { font-size: 1.55rem; font-weight: 800; letter-spacing:-.01em; }
.verdict .s  { color:#C5CAEA; font-size:.95rem; }
.gauge { position:relative; height:14px; border-radius:99px; margin: 14px 0 6px;
         background: linear-gradient(90deg,#10b981 0%,#f59e0b 50%,#f43f5e 100%); }
.gauge .mark { position:absolute; top:-6px; width:4px; height:26px; background:#fff; border-radius:3px; box-shadow:0 0 10px #fff; }
.gauge .thr  { position:absolute; top:-3px; width:2px; height:20px; background:rgba(255,255,255,.55); }
.glabels { display:flex; justify-content:space-between; color:#8f96c0; font-size:.75rem; }
.tile { padding:16px; border-radius:16px; background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08); text-align:center; }
.tile .v { font-size:1.7rem; font-weight:800; background: linear-gradient(90deg,#b9a8ff,#6fe3ff);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.tile .l { color:#9aa1c9; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }
.disc { color:#7f86b0; font-size:.8rem; padding-top:10px; }
section[data-testid="stSidebar"] { background: #0f1330; border-right: 1px solid rgba(255,255,255,.06); }
button[kind="primary"], .stDownloadButton button { border-radius: 12px !important; font-weight:600; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_detector(path: str) -> DefectDetector:
    return DefectDetector(path)


def tile(value, label):
    return f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div></div>'


def panel(img, caption):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown(f'<div class="cap">{caption}</div></div>', unsafe_allow_html=True)


def composite(res, overlay, mask_img):
    panels = [(res["input"], "INPUT"), (res["recon"], "RECONSTRUCTION"), (overlay, "ANOMALY HEATMAP"), (mask_img, "DEFECT MASK")]
    h, w = res["input"].shape[:2]
    bar = 34
    canvas = Image.new("RGB", (w * 4, h + bar + 40), (11, 14, 26))
    d = ImageDraw.Draw(canvas)
    for i, (arr, name) in enumerate(panels):
        canvas.paste(Image.fromarray(arr), (i * w, bar))
        d.text((i * w + 10, 10), name, fill=(190, 196, 235))
    verdict = "POTENTIAL DEFECT" if res["is_defect"] else "NORMAL"
    d.text((10, h + bar + 12), f"{verdict}  |  score {res['score']:.4f}  |  threshold {res['threshold']:.4f}  |  model prediction, not a certified QC decision",
           fill=(244, 63, 94) if res["is_defect"] else (16, 185, 129))
    buf = io.BytesIO(); canvas.save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control panel")
    ckpts = sorted((ROOT / "models" / "checkpoints").glob("*_best.pt"))
    if not ckpts:
        st.error("No trained model found.\n\nTrain one first:\n`python -m src.training.train --category bottle`")
        st.stop()
    names = [c.stem.replace("_best", "") for c in ckpts]
    choice = st.selectbox("Product category / model", names)
    det = load_detector(str(ckpts[names.index(choice)]))
    sens = st.slider("Decision strictness", 0.5, 2.0, 1.0, 0.05,
                     help="Multiplies the validation-selected thresholds. Higher = fewer alarms, lower = more sensitive.")
    alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.55, 0.05)
    st.divider()
    st.caption(f"Device: `{det.device}`  ·  Input: `{det.size}×{det.size}`")
    st.caption(f"Image threshold: `{det.image_threshold:.4f}`")
    st.caption(f"Pixel threshold: `{det.pixel_threshold:.4f}`")

st.markdown(f"""
<div class="hero">
  <h1>🔍 DefectLens</h1>
  <p>Unsupervised surface-defect detection &amp; localisation with a convolutional autoencoder.</p>
  <span class="pill">Trained on normal images only</span><span class="pill">Category: {det.category}</span>
  <span class="pill">Reconstruction-error anomaly maps</span>
</div>""", unsafe_allow_html=True)

tab_inspect, tab_metrics, tab_about = st.tabs(["🧪 Inspect", "📊 Model & metrics", "ℹ️ How it works"])

# ------------------------------------------------------------------ inspect
with tab_inspect:
    left, right = st.columns([1, 1])
    with left:
        up = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp", "tif"])
    sample_dir = Path(det.cfg["data_root"]) / det.category / "test"
    samples = list_images(sample_dir) if sample_dir.exists() else []
    with right:
        sample = st.selectbox("…or pick a test-set sample", ["—"] + [f"{p.parent.name}/{p.name}" for p in samples])

    pil = None
    if up is not None:
        pil = Image.open(up)
    elif sample != "—":
        pil = Image.open(sample_dir / sample)

    if pil is None:
        st.info("Upload an image or choose a sample to run an inspection.")
    else:
        res = det.predict(pil.convert("RGB"), sensitivity=sens)
        ratio = res["score"] / (res["threshold"] * 2)
        pos = float(np.clip(ratio, 0, 1)) * 100
        bad = res["is_defect"]
        st.markdown(f"""
        <div class="verdict {'bad' if bad else 'ok'}">
          <div class="ic">{'⚠️' if bad else '✅'}</div>
          <div><div class="t">{'Potential defect detected' if bad else 'Looks normal'}</div>
          <div class="s">Anomaly score <b>{res['score']:.4f}</b> vs threshold <b>{res['threshold']:.4f}</b>
          · inference {res['time_ms']:.0f} ms</div></div>
        </div>
        <div class="gauge"><div class="thr" style="left:50%"></div><div class="mark" style="left:calc({pos}% - 2px)"></div></div>
        <div class="glabels"><span>clean</span><span>threshold</span><span>anomalous</span></div>
        """, unsafe_allow_html=True)

        overlay = overlay_heatmap(res["input"], res["amap"], res["vmax"], alpha)
        mask_img = np.stack([(res["mask"] * 255).astype(np.uint8)] * 3, -1)
        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        with c1: panel(res["input"], "Input")
        with c2: panel(res["recon"], "Reconstruction")
        with c3: panel(overlay, "Anomaly heatmap")
        with c4: panel(mask_img, "Defect mask")

        area = 100 * res["mask"].mean()
        m1, m2, m3 = st.columns(3)
        m1.markdown(tile(f"{res['score']:.4f}", "anomaly score"), unsafe_allow_html=True)
        m2.markdown(tile(f"{area:.2f}%", "flagged area"), unsafe_allow_html=True)
        m3.markdown(tile(f"{res['amap'].max():.3f}", "peak error"), unsafe_allow_html=True)

        report = {"category": det.category, "score": res["score"], "threshold": res["threshold"],
                  "prediction": "potential_defect" if bad else "normal", "flagged_area_percent": area,
                  "strictness": sens, "inference_ms": res["time_ms"]}
        st.write("")
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇️ Result image (PNG)", composite(res, overlay, mask_img), "defect_result.png", "image/png")
        d2.download_button("⬇️ Report (JSON)", json.dumps(report, indent=2), "defect_report.json", "application/json")
        st.markdown('<div class="disc">This is a model prediction for educational/research use, not a certified industrial quality-control decision.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ metrics
with tab_metrics:
    out = ROOT / "outputs"
    ev = out / f"{choice}_evaluation.json"
    if ev.exists():
        m = json.loads(ev.read_text())
        cols = st.columns(6)
        items = [("image_auroc", "Image AUROC"), ("pixel_auroc", "Pixel AUROC"), ("f1", "F1"),
                 ("precision", "Precision"), ("recall", "Recall"), ("mean_iou", "Mean IoU")]
        for col, (k, lab) in zip(cols, items):
            if k in m:
                col.markdown(tile(f"{m[k]:.3f}", lab), unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        c1.markdown(tile(f"{m.get('mean_dice', 0):.3f}", "Mean Dice"), unsafe_allow_html=True)
        c2.markdown(tile(f"{m['inference_ms_per_image']:.1f} ms", "inference / image"), unsafe_allow_html=True)
    else:
        st.info(f"Run `python -m src.evaluation.evaluate --category {choice}` to see test metrics here.")
    st.write("")
    for fn, title in [(f"{choice}_loss.png", "Training history"), (f"{choice}_roc_confusion.png", "ROC & confusion matrix"),
                      (f"{choice}_examples.png", "Qualitative examples")]:
        p = out / "plots" / fn
        if p.exists():
            st.markdown(f"**{title}**"); st.image(str(p), use_container_width=True)

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Pipeline**
1. The autoencoder is trained **only on defect-free images**, so it learns what "normal" looks like.
2. At inference it reconstructs the input; defects are poorly reconstructed → large per-pixel error.
3. Error is averaged over colour channels, Gaussian-smoothed into an **anomaly map**; a high percentile becomes the **image score**.
4. Thresholds come from **held-out normal validation images** (never from the test set).

**Known limitations** – a plain autoencoder can reconstruct defects too well, or flag harmless lighting/texture changes.
Try a smaller `latent_channels`, denoising (`noise_std`), or compare with PatchCore/PaDiM (Anomalib).

**Dataset** – MVTec AD (CC BY-NC-SA 4.0, non-commercial). Cite: Bergmann et al., CVPR 2019.
""")
