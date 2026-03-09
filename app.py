# POLYP-CAD: Computer-Aided Detection System
# ResNet34 Attention U-Net with scSE Decoder  — DARK THEME

import streamlit as st
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import math
import datetime
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd

st.set_page_config(
    page_title="PolypCAD",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

T = dict(
    surface = "#07101c",
    card    = "#0d1a28",
    card2   = "#112030",
    rim     = "#1a2e42",
    rim2    = "#1f3a54",
    text    = "#ffffff",
    sub     = "#c8d8ec",
    muted   = "#6b8db5",
    dim     = "#2e4a66",
    green   = "#00e5a0",
    blue    = "#3399ff",
    amber   = "#f5a623",
    red     = "#ff3d5a",
)

st.markdown(f"""
<style>
:root {{
    --surface : {T['surface']};
    --card    : {T['card']};
    --card2   : {T['card2']};
    --rim     : {T['rim']};
    --rim2    : {T['rim2']};
    --text    : {T['text']};
    --sub     : {T['sub']};
    --muted   : {T['muted']};
    --dim     : {T['dim']};
    --green   : {T['green']};
    --blue    : {T['blue']};
    --amber   : {T['amber']};
    --red     : {T['red']};
    --ff-head : 'Barlow Condensed', 'Arial Black', sans-serif;
    --ff-body : 'Space Grotesk', 'Segoe UI', sans-serif;
    --ff-mono : 'JetBrains Mono', 'Consolas', monospace;
    --r       : 10px;
}}

/* ── BASE ────────────────────────────────────────────────── */
.stApp {{
    background: var(--surface) !important;
    font-family: var(--ff-body) !important;
    color: var(--text) !important;
}}
* {{ font-family: var(--ff-body); }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.6rem 2.4rem 3rem; max-width: 100%; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
button[data-testid="collapsedControl"] {{ display: none !important; }}

/* ── HEADER ──────────────────────────────────────────────── */
.app-header {{
    padding: 1.2rem 0 1.6rem;
    border-bottom: 1px solid var(--rim);
    margin-bottom: 2rem;
}}
.app-logo {{
    font-family: 'Barlow Condensed', 'Arial Black', sans-serif !important;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    line-height: 1;
    color: #ffffff;
}}
.app-logo span {{ color: var(--green); }}
.app-tagline {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem;
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}}

/* ── UPLOAD ZONE ─────────────────────────────────────────── */
div[data-testid="stFileUploader"] {{
    background: var(--card) !important;
    border: 2px dashed {T['rim2']} !important;
    border-radius: 16px !important;
    transition: all 0.25s !important;
    padding: 4rem 3rem !important;
    min-height: 260px !important;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: {T['green']}80 !important;
    background: #0f2035 !important;
}}
div[data-testid="stFileUploader"] > label {{
    font-family: 'Barlow Condensed', 'Arial Black', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 0.6rem !important;
    display: block !important;
    text-transform: uppercase !important;
}}
div[data-testid="stFileUploader"] p {{
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    line-height: 1.5 !important;
    margin: 0 0 0.3rem !important;
}}
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small {{
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem !important;
    font-weight: 600 !important;
}}
div[data-testid="stFileUploader"] button {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.8rem !important;
    border-radius: 8px !important;
    background: {T['green']} !important;
    color: #000000 !important;
    border: none !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    letter-spacing: 0.04em !important;
}}
div[data-testid="stFileUploader"] button:hover {{
    background: {T['green']}14 !important;
}}

/* ── RISK BANNER ─────────────────────────────────────────── */
.risk-banner {{
    display: flex;
    align-items: center;
    gap: 1.4rem;
    padding: 1.8rem 2rem;
    border-radius: 14px;
    border: 1px solid;
    margin: 1.4rem 0;
}}
.risk-none     {{ border-color:{T['blue']}60;   background:{T['blue']}14; }}
.risk-vlow     {{ border-color:{T['green']}60;  background:{T['green']}12; }}
.risk-low      {{ border-color:#b8e00060;      background:#b8e00012; }}
.risk-moderate {{ border-color:{T['amber']}60;  background:{T['amber']}12; }}
.risk-high     {{ border-color:{T['red']}60;    background:{T['red']}12; }}

.risk-icon-box {{
    width: 72px; height: 72px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.4rem; flex-shrink: 0;
}}
.risk-none     .risk-icon-box {{ background:{T['blue']}1a;  border:1px solid {T['blue']}50; }}
.risk-vlow     .risk-icon-box {{ background:{T['green']}1a; border:1px solid {T['green']}50; }}
.risk-low      .risk-icon-box {{ background:#b8e0001a;      border:1px solid #b8e00050; }}
.risk-moderate .risk-icon-box {{ background:{T['amber']}1a; border:1px solid {T['amber']}50; }}
.risk-high     .risk-icon-box {{ background:{T['red']}1a;   border:1px solid {T['red']}50; }}

.risk-title {{
    font-family: 'Barlow Condensed', 'Arial Black', sans-serif !important;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    line-height: 1;
}}
.risk-none     .risk-title {{ color:{T['blue']}; }}
.risk-vlow     .risk-title {{ color:{T['green']}; }}
.risk-low      .risk-title {{ color:#b8e000; }}
.risk-moderate .risk-title {{ color:{T['amber']}; }}
.risk-high     .risk-title {{ color:{T['red']}; }}

.risk-detail {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.2rem;
    color: #ffffff;
    margin-top: 0.4rem;
    font-weight: 600;
}}

/* ── RISK SCALE ──────────────────────────────────────────── */
.risk-scale {{
    background: var(--card);
    border: 1px solid var(--rim);
    border-radius: var(--r);
    padding: 1.4rem 1.8rem;
    margin-top: 1.2rem;
}}
.risk-scale-title {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem;
    font-weight: 600;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.1rem;
}}
.risk-scale-bar {{
    display: flex;
    height: 8px;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 0.5rem;
    gap: 2px;
}}
.rsb-none     {{ flex: 0 0 10%; background:{T['blue']}; border-radius:6px 0 0 6px; }}
.rsb-low      {{ flex: 0 0 22%; background:{T['green']}; }}
.rsb-moderate {{ flex: 0 0 28%; background:{T['amber']}; }}
.rsb-high     {{ flex: 1;       background:{T['red']};   border-radius:0 6px 6px 0; }}
.risk-scale-labels {{
    display: flex;
    gap: 2px;
    margin-top: 0.8rem;
}}
.rsl-item {{
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}}
.rsl-item.none     {{ flex: 0 0 10%; }}
.rsl-item.low      {{ flex: 0 0 22%; }}
.rsl-item.moderate {{ flex: 0 0 28%; }}
.rsl-item.high     {{ flex: 1; }}
.rsl-range {{
    font-family: 'Barlow Condensed', 'Arial Black', sans-serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1;
}}
.rsl-label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1.3;
}}
.rsl-item.none     .rsl-range, .rsl-item.none     .rsl-label {{ color:{T['blue']}; }}
.rsl-item.low      .rsl-range, .rsl-item.low      .rsl-label {{ color:{T['green']}; }}
.rsl-item.moderate .rsl-range, .rsl-item.moderate .rsl-label {{ color:{T['amber']}; }}
.rsl-item.high     .rsl-range, .rsl-item.high     .rsl-label {{ color:{T['red']}; }}
.rsl-active-marker {{
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-top: 0.3rem;
    box-shadow: 0 0 6px currentColor;
}}
.rsl-item.none     .rsl-active-marker {{ background:{T['blue']}; }}
.rsl-item.low      .rsl-active-marker {{ background:{T['green']}; }}
.rsl-item.moderate .rsl-active-marker {{ background:{T['amber']}; }}
.rsl-item.high     .rsl-active-marker {{ background:{T['red']}; }}

/* ── SECTION HEADER ──────────────────────────────────────── */
.sec-head {{
    display: flex; align-items: center; gap: 0.8rem;
    margin: 2.2rem 0 1.2rem;
}}
.sec-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    flex-shrink: 0;
}}
.sec-text {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem;
    font-weight: 800;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    white-space: nowrap;
}}
.sec-line {{ flex: 1; height: 1px; background: var(--rim); }}

/* ── METRIC CARD ─────────────────────────────────────────── */
.metric-card {{
    background: #0d1a28 !important;
    border: 1px solid #1a2e42 !important;
    border-radius: var(--r);
    padding: 1.4rem 1.5rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.metric-card:hover {{
    border-color: #1f3a54 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}}
.m-label {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
}}
.m-value {{
    font-family: 'Barlow Condensed', 'Arial Black', sans-serif !important;
    font-size: 3.0rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    line-height: 1;
    color: #ffffff;
}}
.m-unit {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem;
    color: #ffffff;
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.04em;
}}

/* ── POLYP SUB-CARD ──────────────────────────────────────── */
.polyp-sub-card {{
    background: var(--card2);
    border: 1px solid var(--rim2);
    border-radius: var(--r);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
}}
.polyp-sub-title {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--green);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
}}
.polyp-sub-grid {{
    display: flex;
    gap: 1.8rem;
    flex-wrap: wrap;
}}
.polyp-sub-val {{
    font-family: 'Barlow Condensed', 'Arial Black', sans-serif !important;
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}}
.polyp-sub-lbl {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem;
    color: #ffffff;
    font-weight: 600;
    margin-top: 0.2rem;
}}

/* ── MORPH CARD ──────────────────────────────────────────── */
.morph-card {{
    background: #0d1a28 !important;
    border: 1px solid #1a2e42 !important;
    border-radius: var(--r);
    padding: 1.4rem 1.5rem;
}}
.morph-label {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
}}
.morph-val {{
    font-family: 'Barlow Condensed', 'Arial Black', sans-serif !important;
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    line-height: 1;
    margin-bottom: 0.3rem;
}}
.morph-sublabel {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 0.5rem;
}}
.progress-track {{
    height: 6px; border-radius: 4px;
    background: var(--rim); margin: 0.8rem 0 0.5rem;
    overflow: hidden;
}}
.progress-fill {{ height: 100%; border-radius: 4px; }}

/* ── IMAGE PANEL ─────────────────────────────────────────── */
.img-panel {{
    background: var(--card);
    border: 1px solid var(--rim);
    border-radius: var(--r);
    overflow: hidden;
}}
.img-label {{
    padding: 0.75rem 1rem;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem;
    font-weight: 800;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--rim);
    background: var(--card2);
}}
.img-body {{ padding: 0.5rem; background: #040a10; }}

/* ── ASSESSMENT PANEL ────────────────────────────────────── */
.rp {{
    background: var(--card);
    border: 1px solid var(--rim);
    border-radius: var(--r);
    overflow: hidden;
}}
.rp-head {{
    padding: 0.75rem 1.4rem;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem;
    font-weight: 800;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--rim);
    background: var(--card2);
}}
.rtable {{ width: 100%; border-collapse: collapse; }}
.rtable tr {{ border-bottom: 1px solid var(--rim); }}
.rtable tr:last-child {{ border-bottom: none; }}
.rtable td {{ padding: 0.85rem 1.4rem; vertical-align: top; }}
.rtable td:first-child {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    width: 38%;
}}
.rtable td:last-child {{
    color: #ffffff;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem;
}}

/* ── WARN ────────────────────────────────────────────────── */
.warn-box {{
    display: flex; align-items: center; gap: 0.8rem;
    padding: 1rem 1.4rem;
    border-radius: 8px;
    border: 1px solid {T['amber']}55;
    background: {T['amber']}14;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.0rem;
    font-weight: 500;
    color: {T['amber']};
    margin-top: 0.8rem;
}}

/* ── DOWNLOAD BUTTON ─────────────────────────────────────── */
.stDownloadButton button {{
    background: transparent !important;
    color: {T['green']} !important;
    border: 1px solid {T['green']}70 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.6rem !important;
    margin-top: 1rem !important;
    transition: background 0.2s !important;
}}
.stDownloadButton button:hover {{
    background: {T['green']}14 !important;
}}

/* ── SPINNER / PROGRESS ──────────────────────────────────── */
.stSpinner > div {{ border-top-color: {T['green']} !important; }}
.stProgress > div > div {{ background: {T['green']} !important; }}

/* ── DATAFRAME ───────────────────────────────────────────── */
.stDataFrame {{ border-radius: var(--r) !important; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 256
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH    = "resnet34_attunet_model.pth"
THRESHOLD     = 0.50
MM_PER_PIXEL  = 0.117

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

MEAN_T = torch.tensor(IMAGENET_MEAN).view(1, 1, 3)
STD_T  = torch.tensor(IMAGENET_STD).view(1, 1, 3)


def denormalize(tensor):
    img = tensor.cpu().permute(1, 2, 0)
    img = img * STD_T + MEAN_T
    return (img.clamp(0, 1).numpy() * 255).astype(np.uint8)


def analyze_components(pred_mask, min_pixels=50):
    labeled, num_features = ndimage.label(pred_mask)
    components = []
    for i in range(1, num_features + 1):
        component = (labeled == i).astype(np.uint8)
        pixels = int(component.sum())
        if pixels < min_pixels:
            continue
        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox_w = int(cmax - cmin)
        bbox_h = int(rmax - rmin)
        comp_uint8 = (component * 255).astype(np.uint8)
        ctrs, _ = cv2.findContours(comp_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim_px = sum(cv2.arcLength(c, True) for c in ctrs)
        ca = pixels * (MM_PER_PIXEL ** 2)
        perim_mm = perim_px * MM_PER_PIXEL
        circ = (4 * math.pi * ca) / (perim_mm ** 2) if perim_mm > 0 else 0
        elong = round(max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + 1e-6), 2)
        components.append({
            "id"          : len(components) + 1,
            "pixels"      : pixels,
            "bbox"        : (int(cmin), int(rmin), int(cmax), int(rmax)),
            "bbox_h"      : bbox_h,
            "bbox_w"      : bbox_w,
            "centroid"    : (int((cmin + cmax) / 2), int((rmin + rmax) / 2)),
            "circularity" : round(circ, 4),
            "elongation"  : elong,
            "perimeter_mm": round(perim_mm, 2),
        })
    components.sort(key=lambda x: x["pixels"], reverse=True)
    return components, labeled


def get_polyp_risk(diameter_mm):
    """Return (risk_level, risk_class, risk_detail, paris_class) for a given diameter."""
    if diameter_mm < 5:
        return (
            "VERY LOW RISK", "vlow",
            "Diminutive polyp (< 5 mm). Extremely common; rarely contain advanced cancer or high-grade dysplasia.",
            "Ip / Is (Diminutive)"
        )
    elif diameter_mm < 10:
        return (
            "LOW RISK", "low",
            "Small polyp (5–9 mm). Low risk unless pathology shows advanced features such as villous growth.",
            "Ip / IIa (Small)"
        )
    elif diameter_mm < 20:
        return (
            "HIGH RISK", "moderate",
            "Large polyp (≥ 10 mm). Automatically classified as advanced adenoma — high-risk threshold reached.",
            "IIa / IIb (Large / Advanced)"
        )
    else:
        return (
            "VERY HIGH RISK", "high",
            "Giant polyp (≥ 20 mm). High likelihood of occult cancer; significantly harder to resect safely.",
            "IIa+IIb / III (Giant)"
        )


def calculate_metrics(pred_mask, mm_per_pixel):
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    polyp_pixels = int(pred_mask.sum())
    polyp_ratio  = polyp_pixels / total_pixels
    area_mm2     = polyp_pixels * (mm_per_pixel ** 2)
    diameter_mm  = 2 * math.sqrt(area_mm2 / math.pi) if area_mm2 > 0 else 0

    mask_uint8  = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_px = sum(cv2.arcLength(c, True) for c in contours)
    perimeter_mm = perimeter_px * mm_per_pixel
    circularity  = (4 * math.pi * area_mm2) / (perimeter_mm ** 2) if perimeter_mm > 0 else 0

    if contours:
        all_pts    = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        elongation = round(max(w, h) / (min(w, h) + 1e-6), 2)
    else:
        elongation = 0.0

    components, labeled_mask = analyze_components(pred_mask)
    num_polyps = len(components)
    for comp in components:
        ca = comp["pixels"] * (mm_per_pixel ** 2)
        comp["area_mm2"]    = round(ca, 2)
        comp["diameter_mm"] = round(2 * math.sqrt(ca / math.pi) if ca > 0 else 0, 2)
        # Per-polyp risk
        rl, rc, rd, pc = get_polyp_risk(comp["diameter_mm"])
        comp["risk_level"]  = rl
        comp["risk_class"]  = rc
        comp["risk_detail"] = rd
        comp["paris_class"] = pc

    largest_diameter = max((c["diameter_mm"] for c in components), default=diameter_mm)
    risk_diameter = largest_diameter if num_polyps > 0 else diameter_mm

    if polyp_pixels == 0:
        risk_level = "NO POLYP DETECTED"; risk_class = "none"; risk_icon = ""
        risk_detail = "No polyp region identified in this image."
        paris_class = "N/A"
    else:
        risk_level, risk_class, risk_detail, paris_class = get_polyp_risk(risk_diameter)
        risk_icon = ""

    min_circ = min((c["circularity"] for c in components), default=circularity)
    shape_warning = (
        "Irregular morphology detected — may indicate advanced or flat lesion."
        if min_circ < 0.4 and polyp_pixels > 0 else None
    )

    return {
        "polyp_pixels"    : polyp_pixels,
        "total_pixels"    : total_pixels,
        "polyp_ratio"     : polyp_ratio,
        "area_mm2"        : round(area_mm2, 3),
        "diameter_mm"     : round(diameter_mm, 3),
        "perimeter_mm"    : round(perimeter_mm, 3),
        "circularity"     : round(circularity, 4),
        "elongation"      : elongation,
        "num_polyps"      : num_polyps,
        "components"      : components,
        "labeled_mask"    : labeled_mask,
        "risk_level"      : risk_level,
        "risk_class"      : risk_class,
        "risk_detail"     : risk_detail,
        "paris_class"     : paris_class,
        "risk_icon"       : risk_icon,
        "shape_warning"   : shape_warning,
        "largest_diameter": round(risk_diameter, 3),
    }


def create_overlay(image_rgb, pred_mask, results):
    overlay    = image_rgb.copy()
    mask_color = np.zeros_like(overlay)
    risk_colors = {
        "none"    : (0, 180, 255),
        "vlow"    : (0, 229, 160),
        "low"     : (184, 224, 0),
        "moderate": (255, 170, 0),
        "high"    : (255, 61, 90),
    }
    color = risk_colors.get(results["risk_class"], (255, 255, 0))
    mask_color[pred_mask == 1] = color
    blended    = cv2.addWeighted(overlay, 0.6, mask_color, 0.4, 0)
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, 2)
    for comp in results["components"]:
        x1, y1, x2, y2 = comp["bbox"]
        # Use per-polyp color if available
        comp_color = risk_colors.get(comp.get("risk_class", results["risk_class"]), color)
        cv2.rectangle(blended, (x1, y1), (x2, y2), comp_color, 1)
        # ── FIX: Full "Polyp N" label instead of "PN" ──────────────────
        label = f"Polyp{comp['id']} {comp['diameter_mm']}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(blended, (x1, max(y1-th-6, 0)), (x1+tw+4, max(y1, th+6)), comp_color, -1)
        cv2.putText(blended, label, (x1+2, max(y1-4, th+2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return blended


def create_prob_heatmap(prob_map):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(T["card"])
    ax.set_facecolor(T["card"])
    im   = ax.imshow(prob_map, cmap="plasma", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=T["muted"])
    cbar.outline.set_edgecolor(T["rim"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=T["muted"], fontsize=7)
    ax.axis("off")
    ax.set_title("Confidence Map", fontsize=8, color=T["muted"], pad=6)
    plt.tight_layout(pad=0.5)
    return fig


@st.cache_resource
def load_model(path):
    model = smp.Unet(
        encoder_name           = "resnet34",
        encoder_weights        = None,
        decoder_attention_type = "scse",
        in_channels            = 3,
        classes                = 1,
        activation             = None,
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def predict(model, image_np):
    aug        = val_transform(image=image_np)
    img_tensor = aug["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output    = model(img_tensor)
        prob_map  = torch.sigmoid(output)[0, 0].cpu().numpy()
        pred_mask = (prob_map > THRESHOLD).astype(np.float32)
    img_display = denormalize(img_tensor[0])
    return pred_mask, prob_map, img_display


def _circ_label(circ):
    if circ > 0.8:   return "Regular — Likely benign"
    elif circ > 0.5: return "Moderate irregularity"
    elif circ > 0:   return "High irregularity — Advanced lesion"
    else:            return "No polyp detected"


def generate_report(filename, results, timestamp):
    """Full detailed report — per-polyp breakdown when multiple polyps exist."""
    lines = [
        "=" * 60,
        "  POLYP-CAD CLINICAL DETECTION REPORT",
        "=" * 60,
        f"  File          : {filename}",
        f"  Timestamp     : {timestamp}",
        f"  Threshold     : {THRESHOLD}",
        f"  Calibration   : {MM_PER_PIXEL} mm/pixel",
        "=" * 60,
        "  ASSESSMENT",
        "-" * 60,
        f"  Risk Level    : {results['risk_level']}",
        f"  Paris Class   : {results['paris_class']}",
        f"  Polyps Found  : {results['num_polyps']}",
        "=" * 60,
        "  SIZE ESTIMATION",
        "-" * 60,
    ]

    if results["num_polyps"] <= 1:
        lines += [
            f"  Area          : {results['area_mm2']} mm²",
            f"  Diameter      : {results['diameter_mm']} mm",
            f"  Perimeter     : {results['perimeter_mm']} mm",
        ]
    else:
        lines.append(f"  (Each polyp measured individually)")
        for comp in results["components"]:
            lines += [
                "",
                f"  Polyp {comp['id']}:",
                f"    Area        : {comp['area_mm2']} mm²",
                f"    Diameter    : {comp['diameter_mm']} mm",
                f"    Perimeter   : {comp['perimeter_mm']} mm",
                f"    Risk Level  : {comp['risk_level']}",
                f"    Paris Class : {comp['paris_class']}",
            ]

    lines += [
        "=" * 60,
        "  MORPHOLOGICAL ANALYSIS",
        "-" * 60,
    ]

    if results["num_polyps"] <= 1:
        circ  = results["circularity"]
        elong = results["elongation"]
        lines += [
            f"  Shape Regularity : {circ}  ({_circ_label(circ)})",
            f"  Elongation Ratio : {elong}  ({'Elongated shape' if elong > 2.0 else 'Normal aspect ratio'})",
            f"  Polyps Detected  : {results['num_polyps']}",
        ]
    else:
        lines.append(f"  Polyps Detected  : {results['num_polyps']}")
        for comp in results["components"]:
            circ  = comp["circularity"]
            elong = comp["elongation"]
            lines += [
                "",
                f"  Polyp {comp['id']}:",
                f"    Shape Regularity : {circ}  ({_circ_label(circ)})",
                f"    Elongation Ratio : {elong}  ({'Elongated shape' if elong > 2.0 else 'Normal aspect ratio'})",
            ]

    if results["shape_warning"]:
        lines += ["", f"  ⚠  {results['shape_warning']}"]

    lines += [
        "=" * 60,
        "  DISCLAIMER",
        "  For research and assistive use only.",
        "  All findings must be verified by a qualified",
        "  medical professional. Not for standalone diagnosis.",
        "=" * 60,
    ]
    return "\n".join(lines)


def generate_batch_report(all_names, all_results, timestamp):
    lines = [
        "=" * 60,
        "  POLYP-CAD BATCH CLINICAL REPORT",
        "=" * 60,
        f"  Generated     : {timestamp}",
        f"  Total Images  : {len(all_names)}",
        f"  Threshold     : {THRESHOLD}",
        f"  Calibration   : {MM_PER_PIXEL} mm/pixel",
        "=" * 60,
        "",
    ]
    for fname, res in zip(all_names, all_results):
        lines.append(generate_report(fname, res, timestamp))
        lines.append("")
    return "\n".join(lines)


def sec(label):
    st.markdown(f"""
    <div class="sec-head">
        <div class="sec-dot"></div>
        <div class="sec-text">{label}</div>
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_scale(active_class):
    """Render a visual size-to-risk scale with the active tier highlighted."""
    color_map = {
        "none":     "#3399ff",
        "vlow":     "#00e5a0",
        "low":      "#b8e000",
        "moderate": "#f5a623",
        "high":     "#ff3d5a",
    }
    tiers = [
        ("none",     "No Polyp", "—",           "10%"),
        ("vlow",     "< 5 mm",   "Very Low",    "20%"),
        ("low",      "5–9 mm",   "Low",         "18%"),
        ("moderate", "10–19 mm", "High",        "22%"),
        ("high",     "≥ 20 mm",  "Very High",   "30%"),
    ]

    bar_segs = ""
    for cls, _, _, width in tiers:
        c = color_map[cls]
        bar_segs += (
            f'<div style="flex:0 0 {width};height:8px;background:{c};'
            f'border-radius:{"6px 0 0 6px" if cls=="none" else ("0 6px 6px 0" if cls=="high" else "2px")};"></div>'
        )

    label_items = ""
    for cls, size_lbl, risk_lbl, width in tiers:
        c = color_map[cls]
        is_active = cls == active_class
        prefix = "&#10022; " if is_active else ""
        marker = (
            f'<div style="width:8px;height:8px;border-radius:50%;background:{c};'
            f'box-shadow:0 0 6px {c};margin-top:4px;"></div>'
            if is_active else ""
        )
        fw = "800" if is_active else "600"
        op = "1" if is_active else "0.55"
        label_items += (
            f'<div style="flex:0 0 {width};display:flex;flex-direction:column;gap:3px;opacity:{op};">'
            f'<div style="font-family:Barlow Condensed,Arial Black,sans-serif;font-size:1.05rem;'
            f'font-weight:{fw};color:{c};line-height:1;">{prefix}{size_lbl}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;font-weight:500;'
            f'color:{c};text-transform:uppercase;letter-spacing:0.06em;">{risk_lbl}</div>'
            f'{marker}</div>'
        )

    html = (
        '<div style="background:#0d1a28;border:1px solid #1a2e42;border-radius:10px;padding:1.4rem 1.8rem;">'
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;font-weight:600;'
        'color:#ffffff;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1.1rem;">'
        'Size &#8594; Risk Reference</div>'
        '<div style="display:flex;height:8px;border-radius:6px;overflow:hidden;gap:2px;margin-bottom:0.5rem;">'
        + bar_segs +
        '</div>'
        '<div style="display:flex;gap:2px;margin-top:0.8rem;">'
        + label_items +
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_polyp_risk_banner(comp, color_map_rc):
    """Render a risk banner for a single polyp component."""
    rc = comp["risk_class"]
    rc_color = color_map_rc.get(rc, "#fff")
    return (
        f'<div style="display:flex;align-items:center;gap:0px;border-radius:14px;'
        f'border:1px solid {rc_color}60;background:{rc_color}14;overflow:hidden;margin:0.8rem 0;">'
        f'<div style="width:8px;min-height:80px;background:{rc_color};flex-shrink:0;align-self:stretch;"></div>'
        f'<div style="padding:1.2rem 1.6rem;">'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;font-weight:700;'
        f'color:{rc_color};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">'
        f'Polyp {comp["id"]} — {comp["diameter_mm"]} mm</div>'
        f'<div style="font-family:Barlow Condensed,Arial Black,sans-serif;font-size:1.9rem;'
        f'font-weight:800;letter-spacing:0.04em;line-height:1;color:{rc_color};">{comp["risk_level"]}</div>'
        f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.0rem;color:#ffffff;'
        f'margin-top:0.4rem;font-weight:600;">{comp["risk_detail"]}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{rc_color};'
        f'margin-top:0.5rem;opacity:0.8;">Paris: {comp["paris_class"]}</div>'
        f'</div></div>'
    )


def render_result(image_np, filename, pred_mask, prob_map,
                  img_display, results, timestamp, idx=0):
    rc  = results["risk_class"]
    color_map_rc = {"none": T["blue"], "vlow": T["green"], "low": "#b8e000", "moderate": T["amber"], "high": T["red"]}
    col = color_map_rc.get(rc, "#fff")

    # ── 1. SEGMENTATION VISUALIZATION ────────────────────────────────────
    sec("Segmentation Visualization")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.markdown('<div class="img-panel"><div class="img-label">Original Image</div><div class="img-body">', unsafe_allow_html=True)
        st.image(image_np, width='stretch')
        st.markdown('</div></div>', unsafe_allow_html=True)
    with v2:
        st.markdown('<div class="img-panel"><div class="img-label">Predicted Mask</div><div class="img-body">', unsafe_allow_html=True)
        st.image((pred_mask * 255).astype(np.uint8), width='stretch')
        st.markdown('</div></div>', unsafe_allow_html=True)
    with v3:
        st.markdown('<div class="img-panel"><div class="img-label">Probability Heatmap</div><div class="img-body">', unsafe_allow_html=True)
        fig_heat = create_prob_heatmap(prob_map)
        st.pyplot(fig_heat, width='stretch')
        plt.close()
        st.markdown('</div></div>', unsafe_allow_html=True)
    with v4:
        st.markdown('<div class="img-panel"><div class="img-label">Annotated Overlay</div><div class="img-body">', unsafe_allow_html=True)
        overlay_img = create_overlay(img_display, pred_mask, results)
        st.image(overlay_img, width='stretch')
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── 2. SIZE ESTIMATION ────────────────────────────────────────────────
    sec("Size Estimation")

    if results["num_polyps"] <= 1:
        c1, c2, c3 = st.columns(3)
        for col_el, label, value, unit in [
            (c1, "Area",      results["area_mm2"],    "mm²"),
            (c2, "Diameter",  results["diameter_mm"], "mm"),
            (c3, "Perimeter", results["perimeter_mm"],"mm"),
        ]:
            with col_el:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="m-label">{label}</div>
                    <div class="m-value">{value}<span style="font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:400;color:#ffffff;margin-left:0.35rem;vertical-align:middle;font-opacity:0.7;">{unit}</span></div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                    color:{T['amber']};margin-bottom:0.9rem;letter-spacing:0.06em;">
            ⚠ &nbsp;{results['num_polyps']} polyps detected — each measured individually
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(min(results["num_polyps"], 3))
        for i, comp in enumerate(results["components"]):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="polyp-sub-card">
                    <div class="polyp-sub-title">Polyp {comp['id']}</div>
                    <div class="polyp-sub-grid">
                        <div>
                            <div class="polyp-sub-val">{comp['area_mm2']}</div>
                            <div class="polyp-sub-lbl">Area (mm²)</div>
                        </div>
                        <div>
                            <div class="polyp-sub-val">{comp['diameter_mm']}</div>
                            <div class="polyp-sub-lbl">Diameter (mm)</div>
                        </div>
                        <div>
                            <div class="polyp-sub-val">{comp['perimeter_mm']}</div>
                            <div class="polyp-sub-lbl">Perimeter (mm)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── 3. MORPHOLOGICAL ANALYSIS ─────────────────────────────────────────
    sec("Morphological Analysis")

    if results["num_polyps"] <= 1:
        m1, m2, m3 = st.columns(3)
        circ = results["circularity"]
        if circ > 0.8:   circ_lbl, circ_col = "Regular — Likely benign",             T["green"]
        elif circ > 0.5: circ_lbl, circ_col = "Moderate irregularity",               T["amber"]
        elif circ > 0:   circ_lbl, circ_col = "High irregularity — Advanced lesion", T["red"]
        else:            circ_lbl, circ_col = "No polyp detected",                   T["blue"]
        circ_pct = min(int(circ * 100), 100)
        with m1:
            st.markdown(f"""
            <div class="morph-card">
                <div class="morph-label">Shape Regularity</div>
                <div class="morph-val" style="color:#ffffff">{circ}</div>
                <div class="progress-track">
                    <div class="progress-fill" style="width:{circ_pct}%;background:#ffffff"></div>
                </div>
                <div class="morph-sublabel" style="color:#ffffff">{circ_lbl}</div>
            </div>
            """, unsafe_allow_html=True)
        elong     = results["elongation"]
        no_polyp  = results["num_polyps"] == 0
        elong_col = T["muted"] if no_polyp else (T["amber"] if elong > 2.0 else T["green"])
        elong_lbl = "No polyp detected" if no_polyp else ("Elongated shape" if elong > 2.0 else "Normal aspect ratio")
        with m2:
            st.markdown(f"""
            <div class="morph-card">
                <div class="morph-label">Elongation Ratio</div>
                <div class="morph-val" style="color:#ffffff">{elong}</div>
                <div class="morph-sublabel" style="color:#ffffff">{elong_lbl}</div>
            </div>
            """, unsafe_allow_html=True)
        n_polyps  = results["num_polyps"]
        count_col = T["blue"] if n_polyps == 0 else T["green"]
        count_lbl = "No lesion detected" if n_polyps == 0 else "Single lesion"
        with m3:
            st.markdown(f"""
            <div class="morph-card">
                <div class="morph-label">Polyps Detected</div>
                <div class="morph-val" style="color:#ffffff">{n_polyps}</div>
                <div class="morph-sublabel" style="color:#ffffff">{count_lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="morph-card" style="margin-bottom:1.2rem;">
            <div class="morph-label">Polyps Detected</div>
            <div class="morph-val" style="color:#ffffff">{results['num_polyps']}</div>
            <div class="morph-sublabel" style="color:#ffffff">Multiple lesions — morphology shown per polyp</div>
        </div>
        """, unsafe_allow_html=True)
        for comp in results["components"]:
            circ  = comp["circularity"]
            elong = comp["elongation"]
            if circ > 0.8:   circ_lbl, circ_col = "Regular — Likely benign",             T["green"]
            elif circ > 0.5: circ_lbl, circ_col = "Moderate irregularity",               T["amber"]
            elif circ > 0:   circ_lbl, circ_col = "High irregularity — Advanced lesion", T["red"]
            else:            circ_lbl, circ_col = "No polyp detected",                   T["blue"]
            circ_pct  = min(int(circ * 100), 100)
            elong_col = T["amber"] if elong > 2.0 else T["green"]
            elong_lbl = "Elongated shape" if elong > 2.0 else "Normal aspect ratio"
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                        color:{T['green']};margin:1rem 0 0.5rem;text-transform:uppercase;
                        letter-spacing:0.1em;">Polyp {comp['id']}</div>
            """, unsafe_allow_html=True)
            ma, mb = st.columns(2)
            with ma:
                st.markdown(f"""
                <div class="morph-card">
                    <div class="morph-label">Shape Regularity</div>
                    <div class="morph-val" style="color:#ffffff">{circ}</div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width:{circ_pct}%;background:#ffffff"></div>
                    </div>
                    <div class="morph-sublabel" style="color:#ffffff">{circ_lbl}</div>
                </div>
                """, unsafe_allow_html=True)
            with mb:
                st.markdown(f"""
                <div class="morph-card">
                    <div class="morph-label">Elongation Ratio</div>
                    <div class="morph-val" style="color:#ffffff">{elong}</div>
                    <div class="morph-sublabel" style="color:#ffffff">{elong_lbl}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── 4. RISK ANALYSIS ──────────────────────────────────────────────────
    sec("Risk Analysis")

    if results["num_polyps"] <= 1:
        # Single polyp — original layout (banner + scale side by side)
        left_col, right_col = st.columns([3, 2])
        with left_col:
            rc_color = color_map_rc.get(rc, "#fff")
            st.markdown(
                f'''<div style="display:flex;align-items:center;gap:0px;border-radius:14px;
                    border:1px solid {rc_color}60;background:{rc_color}14;overflow:hidden;margin:1.4rem 0;">
                    <div style="width:8px;min-height:100px;background:{rc_color};flex-shrink:0;align-self:stretch;"></div>
                    <div style="padding:1.8rem 2rem;">
                        <div style="font-family:Barlow Condensed,Arial Black,sans-serif;font-size:2.4rem;
                            font-weight:800;letter-spacing:0.04em;line-height:1;color:{rc_color};">{results["risk_level"]}</div>
                        <div style="font-family:Space Grotesk,sans-serif;font-size:1.2rem;color:#ffffff;
                            margin-top:0.5rem;font-weight:600;">{results["risk_detail"]}</div>
                    </div>
                </div>''',
                unsafe_allow_html=True
            )
            if results["shape_warning"]:
                st.markdown(f"""
                <div class="warn-box">⚠️ &nbsp;<strong>Morphology Alert:</strong>&nbsp; {results['shape_warning']}</div>
                """, unsafe_allow_html=True)
        with right_col:
            render_risk_scale(rc)

    else:
        # ── FIX: Multiple polyps — per-polyp risk banners + shared scale ──
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                    color:{T['amber']};margin-bottom:1rem;letter-spacing:0.06em;">
            ⚠ &nbsp;{results['num_polyps']} polyps detected — individual risk assessed per polyp diameter
        </div>
        """, unsafe_allow_html=True)

        banner_col, scale_col = st.columns([3, 2])
        with banner_col:
            for comp in results["components"]:
                st.markdown(render_polyp_risk_banner(comp, color_map_rc), unsafe_allow_html=True)
            if results["shape_warning"]:
                st.markdown(f"""
                <div class="warn-box">⚠️ &nbsp;<strong>Morphology Alert:</strong>&nbsp; {results['shape_warning']}</div>
                """, unsafe_allow_html=True)
        with scale_col:
            # Highlight the highest-risk polyp on the scale
            highest_rc = results["risk_class"]
            render_risk_scale(highest_rc)

    # ── 5. ASSESSMENT + DOWNLOAD ──────────────────────────────────────────
    sec("Assessment")

    if results["num_polyps"] <= 1:
        rows = "".join([
            f"<tr><td>File</td><td>{filename}</td></tr>",
            f"<tr><td>Generated</td><td>{timestamp}</td></tr>",
            f"<tr><td>Polyps Found</td><td>{results['num_polyps']}</td></tr>",
            f"<tr><td>Risk Level</td><td style='color:{col}'>{results['risk_level']}</td></tr>",
            f"<tr><td>Paris Class</td><td>{results['paris_class']}</td></tr>",
        ])
    else:
        per_polyp_rows = ""
        for comp in results["components"]:
            c_color = color_map_rc.get(comp["risk_class"], "#fff")
            per_polyp_rows += (
                f"<tr><td>Polyp {comp['id']} Risk</td>"
                f"<td style='color:{c_color}'>"
                f"{comp['risk_level']} ({comp['diameter_mm']} mm)</td></tr>"
            )
        rows = "".join([
            f"<tr><td>File</td><td>{filename}</td></tr>",
            f"<tr><td>Generated</td><td>{timestamp}</td></tr>",
            f"<tr><td>Polyps Found</td><td>{results['num_polyps']}</td></tr>",
            f"<tr><td>Overall Risk</td><td style='color:{col}'>{results['risk_level']} (largest polyp)</td></tr>",
            per_polyp_rows,
            f"<tr><td>Paris Class</td><td>{results['paris_class']}</td></tr>",
        ])

    st.markdown(f'<div class="rp"><div class="rp-head">Clinical Assessment</div>'
                f'<table class="rtable">{rows}</table></div>', unsafe_allow_html=True)

    report_txt = generate_report(filename, results, timestamp)
    st.download_button(
        label     = f"⬇  Download Clinical Report — {filename}",
        data      = report_txt,
        file_name = f"polypCAD_{filename}_{timestamp[:10]}.txt",
        mime      = "text/plain",
        key       = f"dl_{idx}",
    )


# ── HEADER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-logo">Polyp<span>CAD</span></div>
    <div class="app-tagline">Colonoscopy Polyp Detection System</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ── UPLOAD ────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload colonoscopy image(s) — PNG · JPG · JPEG",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# ── AUTO-RUN ON UPLOAD ────────────────────────────────────────────────────
if uploaded_files:
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results = []
    all_names   = []

    if len(uploaded_files) > 1:
        prog   = st.progress(0)
        status = st.empty()
        for i, f in enumerate(uploaded_files):
            status.markdown(
                f"<span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;"
                f"color:{T['green']};'>Processing {f.name}…</span>",
                unsafe_allow_html=True,
            )
            image_np = np.array(Image.open(f).convert("RGB"))
            pred_mask, prob_map, img_display = predict(model, image_np)
            results = calculate_metrics(pred_mask, MM_PER_PIXEL)
            all_results.append(results)
            all_names.append(f.name)
            prog.progress((i + 1) / len(uploaded_files))
        status.empty(); prog.empty()

        sec("Batch Analysis Summary")
        df = pd.DataFrame([{
            "File"           : fname,
            "Risk"           : res["risk_level"],
            "Polyps Found"   : res["num_polyps"],
            "Largest Ø (mm)" : res["largest_diameter"],
            "Paris Class"    : res["paris_class"],
        } for fname, res in zip(all_names, all_results)])
        st.dataframe(df, width='stretch', hide_index=True)

        batch_report = generate_batch_report(all_names, all_results, timestamp)
        st.download_button(
            label="⬇  Download Full Batch Report",
            data=batch_report,
            file_name=f"polypCAD_batch_{timestamp[:10]}.txt",
            mime="text/plain",
            key="dl_batch",
        )

        st.markdown("---")
        sec("Per-Image Detailed Results")
        for i, f in enumerate(uploaded_files):
            with st.expander(
                f"📋  {f.name}  —  {all_results[i]['risk_level']}  |  Largest Ø {all_results[i]['largest_diameter']} mm",
                expanded=(i == 0),
            ):
                image_np = np.array(Image.open(f).convert("RGB"))
                pred_mask, prob_map, img_display = predict(model, image_np)
                render_result(image_np, f.name, pred_mask, prob_map,
                              img_display, all_results[i], timestamp, idx=i)
    else:
        f        = uploaded_files[0]
        image_np = np.array(Image.open(f).convert("RGB"))
        with st.spinner("Running segmentation…"):
            pred_mask, prob_map, img_display = predict(model, image_np)
            results = calculate_metrics(pred_mask, MM_PER_PIXEL)
        render_result(image_np, f.name, pred_mask, prob_map,
                      img_display, results, timestamp, idx=0)