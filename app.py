# -*- coding: utf-8 -*-
"""
DermScan AI - Streamlit Web Interface
Final Year Project: Skin Disease Detection System using Deep Learning

Usage:
    streamlit run app.py
"""

import json
import base64
import html
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from tensorflow.keras.models import load_model

from preprocessing import (
    DISEASE_INFO,
    LESION_TYPE_DICT,
    hair_removal,
    preprocess_for_prediction,
)


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "skin_disease_model.h5"
BACKUP_MODEL_PATH = MODELS_DIR / "best_model.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
LOGO_PATH = ASSETS_DIR / "LogoGUI.jpg"
HERO_VIDEO_PATH = ASSETS_DIR / "mp_.mp4"
IMG_SIZE = (28, 28)

CONDITION_CARD_ORDER = [
    "Melanoma",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Melanocytic nevi",
    "Benign keratosis-like lesions",
    "Healthy skin",
    "Dermatofibroma",
    "Vascular lesions",
]

CONDITION_IMAGE_PATHS = {
    "Melanoma": ASSETS_DIR / "mel" / "ISIC_0032751.jpg",
    "Basal cell carcinoma": ASSETS_DIR / "bcc" / "ISIC_0031513.jpg",
    "Actinic keratoses": ASSETS_DIR / "akiec" / "ISIC_0025808.jpg",
    "Melanocytic nevi": ASSETS_DIR / "nv" / "ISIC_0028957.jpg",
    "Benign keratosis-like lesions": ASSETS_DIR / "bkl" / "ISIC_0027419.jpg",
    "Healthy skin": ASSETS_DIR / "normal skin" / "google 1.jpg",
    "Dermatofibroma": ASSETS_DIR / "df" / "ISIC_0028790.jpg",
    "Vascular lesions": ASSETS_DIR / "vasc" / "ISIC_0031901.jpg",
}

DISPLAY_CLASS_ORDER = [
    "Healthy skin",
    "Melanocytic nevi",
    "Melanoma",
    "Benign keratosis-like lesions",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Vascular lesions",
    "Dermatofibroma",
]

DEFAULT_CLASS_INDICES = {
    "Actinic keratoses": 0,
    "Basal cell carcinoma": 1,
    "Benign keratosis-like lesions": 2,
    "Dermatofibroma": 3,
    "Healthy skin": 4,
    "Vascular lesions": 5,
    "Melanocytic nevi": 6,
    "Melanoma": 7,
}


st.set_page_config(layout="wide", page_title="DermScan AI", page_icon="🩺")


def image_to_data_uri(path):
    """Return a base64 data URI for a local image path."""
    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{encoded}"


def video_to_data_uri(path):
    """Return a base64 data URI for a local MP4 video path."""
    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


# ============================================================================
# CUSTOM CSS
# ============================================================================
def load_custom_css():
    """Load the dark medical-AI landing dashboard styling."""
    st.markdown(
        """
        <style>
        html {
            scroll-behavior: smooth;
        }

        :root {
            --bg: #050B16;
            --bg-2: #071827;
            --card: rgba(15, 31, 52, 0.78);
            --card-strong: rgba(10, 25, 44, 0.92);
            --border: rgba(34, 211, 238, 0.18);
            --cyan: #22D3EE;
            --blue: #2563EB;
            --green: #22C55E;
            --orange: #F59E0B;
            --text: #F8FAFC;
            --muted: #94A3B8;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 20% 8%, rgba(34, 211, 238, 0.16), transparent 28rem),
                radial-gradient(circle at 84% 18%, rgba(37, 99, 235, 0.18), transparent 30rem),
                radial-gradient(circle at 50% 82%, rgba(34, 197, 94, 0.08), transparent 26rem),
                linear-gradient(135deg, #050B16 0%, #071827 48%, #020617 100%);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(34, 211, 238, 0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(34, 211, 238, 0.035) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent 78%);
            z-index: 0;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 6.5rem;
            padding-bottom: 3rem;
            position: relative;
            z-index: 1;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: inherit;
            letter-spacing: 0;
        }

        .section-anchor {
            height: 1px;
            scroll-margin-top: 6.5rem;
        }

        .top-nav {
            position: fixed;
            top: 0.85rem;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999;
            width: min(1120px, calc(100% - 2rem));
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.72rem 0.95rem;
            border: 1px solid rgba(34, 211, 238, 0.20);
            border-radius: 18px;
            background: rgba(5, 11, 22, 0.76);
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.34), 0 0 35px rgba(34, 211, 238, 0.08);
            backdrop-filter: blur(18px);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            text-decoration: none !important;
            min-width: fit-content;
        }

        .brand-mark {
            width: 2.35rem;
            height: 2.35rem;
            display: grid;
            place-items: center;
            overflow: hidden;
            border-radius: 0.72rem;
            color: #020617;
            background: #ffffff;
            box-shadow: 0 0 24px rgba(34, 211, 238, 0.30);
            font-weight: 900;
            flex: 0 0 2.35rem;
        }

        .brand-logo-img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .brand-text strong {
            display: block;
            color: var(--text);
            font-size: 0.98rem;
            line-height: 1.05;
        }

        .brand-text span {
            color: var(--muted);
            font-size: 0.66rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }

        .nav-links {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.2rem;
            flex-wrap: wrap;
        }

        .nav-links a {
            color: #CBD5E1 !important;
            text-decoration: none !important;
            font-size: 0.86rem;
            font-weight: 700;
            padding: 0.52rem 0.78rem;
            border-radius: 999px;
            transition: all 160ms ease;
        }

        .nav-links a:hover {
            color: var(--text) !important;
            background: rgba(34, 211, 238, 0.13);
            box-shadow: inset 0 0 0 1px rgba(34, 211, 238, 0.18);
        }

        .hero {
            min-height: calc(100vh - 8rem);
            display: grid;
            align-items: center;
            padding: 2.5rem 0 3rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(320px, 0.92fr);
            gap: 2rem;
            align-items: center;
        }

        .eyebrow {
            color: var(--cyan);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-weight: 800;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            margin: 0;
            color: #F8FAFC !important;
            font-size: clamp(3rem, 6vw, 5.6rem);
            line-height: 0.94;
            font-weight: 900;
        }

        .hero-title .brand-title {
            display: inline;
            color: #F8FAFC !important;
            text-shadow: none;
        }

        .hero-title span {
            display: block;
            color: var(--cyan);
            text-shadow: 0 0 32px rgba(34, 211, 238, 0.24);
        }

        .hero-subtitle {
            color: #DCEBFF;
            max-width: 660px;
            font-size: 1.22rem;
            line-height: 1.65;
            margin: 1.45rem 0 0.8rem;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 640px;
            font-size: 1rem;
            line-height: 1.75;
            margin-bottom: 1.6rem;
        }

        .cta-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin: 1.6rem 0 1.9rem;
        }

        .primary-cta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.9rem;
            padding: 0 1.25rem;
            color: #020617 !important;
            text-decoration: none !important;
            font-weight: 900;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--cyan), #7DD3FC);
            box-shadow: 0 0 30px rgba(34, 211, 238, 0.28);
        }

        .ghost-cta {
            color: #CBD5E1 !important;
            text-decoration: none !important;
            font-weight: 800;
        }

        .hero-visual {
            position: relative;
            min-height: 430px;
            border: 1px solid rgba(34, 211, 238, 0.20);
            border-radius: 26px;
            overflow: hidden;
            background: linear-gradient(145deg, rgba(15, 31, 52, 0.88), rgba(7, 24, 39, 0.56));
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.34), inset 0 0 70px rgba(34, 211, 238, 0.08);
            backdrop-filter: blur(18px);
        }

        .hero-video {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .hero-video-fallback {
            position: absolute;
            inset: 0;
            display: grid;
            place-items: center;
            color: var(--muted);
            padding: 2rem;
            text-align: center;
            background:
                radial-gradient(circle at 52% 42%, rgba(34, 211, 238, 0.24), transparent 10rem),
                linear-gradient(145deg, rgba(15, 31, 52, 0.88), rgba(7, 24, 39, 0.56));
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.5rem;
        }

        .metric-card,
        .glass-card,
        .process-card,
        .condition-card,
        .architecture-card {
            border: 1px solid var(--border);
            background: var(--card);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(16px);
        }

        .metric-card {
            border-radius: 18px;
            padding: 1rem;
            min-height: 96px;
        }

        .metric-card strong {
            display: block;
            color: var(--cyan);
            font-size: 1.42rem;
            line-height: 1.1;
        }

        .metric-card span {
            display: block;
            color: var(--muted);
            margin-top: 0.45rem;
            font-size: 0.82rem;
        }

        .section {
            padding: 4.4rem 0 2.2rem;
        }

        .section-heading {
            margin-bottom: 1.6rem;
        }

        .section-heading h2 {
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.05;
            margin: 0;
            font-weight: 900;
        }

        .section-heading p {
            color: var(--muted);
            max-width: 680px;
            margin: 0.8rem 0 0;
            line-height: 1.7;
        }

        .process-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
        }

        .process-card {
            border-radius: 20px;
            padding: 1.15rem;
            min-height: 190px;
        }

        .architecture-wrap {
            margin-top: 2.2rem;
        }

        .architecture-card {
            border-radius: 24px;
            padding: 1.25rem;
        }

        .architecture-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.2rem;
        }

        .architecture-header h3 {
            color: var(--text);
            font-size: 1.45rem;
            margin: 0;
        }

        .architecture-header p {
            color: var(--muted);
            line-height: 1.65;
            margin: 0.45rem 0 0;
            max-width: 680px;
        }

        .architecture-badge {
            color: #CFFAFE;
            background: rgba(34, 211, 238, 0.14);
            border: 1px solid rgba(34, 211, 238, 0.20);
            border-radius: 999px;
            padding: 0.5rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 900;
            white-space: nowrap;
        }

        .architecture-flow {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.8rem;
            align-items: stretch;
        }

        .architecture-node {
            position: relative;
            border: 1px solid rgba(34, 211, 238, 0.16);
            border-radius: 18px;
            padding: 0.95rem;
            min-height: 136px;
            background: rgba(2, 6, 23, 0.28);
        }

        .architecture-node:not(:last-child)::after {
            content: "";
            position: absolute;
            top: 50%;
            right: -0.72rem;
            width: 0.58rem;
            height: 2px;
            background: linear-gradient(90deg, var(--cyan), transparent);
            box-shadow: 0 0 12px rgba(34, 211, 238, 0.56);
        }

        .architecture-node span {
            display: block;
            color: var(--cyan);
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }

        .architecture-node strong {
            display: block;
            color: var(--text);
            font-size: 1rem;
            margin-bottom: 0.45rem;
        }

        .architecture-node p {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.55;
            margin: 0;
        }

        .architecture-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.95rem;
        }

        .architecture-stat {
            border-radius: 16px;
            border: 1px solid rgba(34, 211, 238, 0.13);
            background: rgba(2, 6, 23, 0.22);
            padding: 0.9rem;
        }

        .architecture-stat strong {
            display: block;
            color: var(--cyan);
            font-size: 1.1rem;
            margin-bottom: 0.25rem;
        }

        .architecture-stat span {
            color: var(--muted);
            font-size: 0.82rem;
        }

        .process-number {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.16em;
            margin-bottom: 1.1rem;
        }

        .process-card h3,
        .glass-card h3,
        .condition-card h3 {
            color: var(--text);
            font-size: 1.05rem;
            margin: 0 0 0.6rem;
        }

        .process-card p,
        .glass-card p,
        .condition-card p {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.65;
            margin: 0;
        }

        .glass-card {
            border-radius: 22px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .status-card {
            border: 1px solid rgba(34, 197, 94, 0.24);
            background: rgba(20, 83, 45, 0.18);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .status-card.offline {
            border-color: rgba(245, 158, 11, 0.28);
            background: rgba(120, 53, 15, 0.22);
        }

        .status-card strong {
            display: block;
            color: var(--green);
            margin-bottom: 0.35rem;
        }

        .status-card.offline strong {
            color: var(--orange);
        }

        .status-card span {
            color: #CBD5E1;
            font-size: 0.9rem;
        }

        .demo-copy {
            color: var(--muted);
            margin-bottom: 1rem;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed rgba(34, 211, 238, 0.32);
            background: rgba(2, 6, 23, 0.26);
            border-radius: 18px;
        }

        .stFileUploader label,
        .stFileUploader small {
            color: #CBD5E1 !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 3rem;
            color: #020617;
            font-weight: 900;
            border: 0;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--cyan), #7DD3FC);
            box-shadow: 0 0 28px rgba(34, 211, 238, 0.24);
        }

        .stButton > button:hover {
            color: #020617;
            border: 0;
            transform: translateY(-1px);
            box-shadow: 0 0 34px rgba(34, 211, 238, 0.34);
        }

        .stImage img {
            border-radius: 18px;
            border: 1px solid rgba(34, 211, 238, 0.16);
        }

        .result-card {
            border-radius: 24px;
            padding: 1.35rem;
            border: 1px solid rgba(34, 211, 238, 0.20);
            background: rgba(15, 31, 52, 0.80);
            box-shadow: 0 28px 80px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(16px);
        }

        .result-label {
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            font-weight: 900;
        }

        .predicted-condition {
            color: var(--text);
            font-size: clamp(1.8rem, 4vw, 3rem);
            line-height: 1.05;
            margin: 0.7rem 0;
            font-weight: 900;
        }

        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.5rem 0.75rem;
            font-weight: 900;
            margin-bottom: 1rem;
        }

        .confidence-badge.high {
            color: #DCFCE7;
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.26);
        }

        .confidence-badge.review {
            color: #FEF3C7;
            background: rgba(245, 158, 11, 0.18);
            border: 1px solid rgba(245, 158, 11, 0.28);
        }

        .probability-stack {
            display: grid;
            gap: 0.78rem;
            margin-top: 1rem;
        }

        .probability-row {
            display: grid;
            grid-template-columns: minmax(150px, 1fr) minmax(180px, 1.4fr) 4.4rem;
            gap: 0.8rem;
            align-items: center;
        }

        .probability-name {
            color: #E2E8F0;
            font-size: 0.9rem;
        }

        .probability-track {
            overflow: hidden;
            height: 0.72rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.14);
            border: 1px solid rgba(148, 163, 184, 0.10);
        }

        .probability-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--blue), var(--cyan));
            box-shadow: 0 0 18px rgba(34, 211, 238, 0.38);
        }

        .probability-value {
            color: var(--cyan);
            font-weight: 900;
            font-size: 0.9rem;
            text-align: right;
        }

        .note-card {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            background: rgba(2, 6, 23, 0.28);
            padding: 1rem;
            margin-top: 1rem;
        }

        .note-card strong {
            color: var(--text);
        }

        .note-card p {
            color: var(--muted);
            margin: 0.35rem 0 0;
            line-height: 1.65;
        }

        .disclaimer {
            border: 1px solid rgba(245, 158, 11, 0.24);
            border-radius: 18px;
            padding: 1rem;
            color: #FDE68A;
            background: rgba(120, 53, 15, 0.16);
            margin-top: 1rem;
            line-height: 1.6;
        }

        .condition-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.5rem;
        }

        .condition-card {
            display: flex;
            flex-direction: column;
            border-radius: 12px;
            padding: 1.45rem;
            min-height: 438px;
            background: rgba(15, 31, 52, 0.86);
            border: 1px solid rgba(34, 211, 238, 0.18);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(16px);
        }

        .condition-toolbar {
            display: grid;
            grid-template-columns: minmax(260px, 0.42fr) minmax(320px, 0.58fr);
            align-items: center;
            gap: 1.5rem;
            margin: 2.6rem 0 1.9rem;
        }

        .condition-search {
            width: 100%;
            min-height: 3.15rem;
            border-radius: 8px;
            border: 1px solid rgba(34, 211, 238, 0.24);
            background: rgba(2, 6, 23, 0.58);
            color: var(--text);
            font-size: 1rem;
            padding: 0 1rem;
            outline: none;
        }

        .condition-search::placeholder {
            color: rgba(248, 250, 252, 0.58);
        }

        .condition-filter {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.7rem;
            flex-wrap: wrap;
        }

        .condition-filter-label {
            color: #CBD5E1;
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .filter-chip {
            border: 1px solid rgba(34, 211, 238, 0.18);
            border-radius: 999px;
            background: rgba(15, 31, 52, 0.72);
            color: #CBD5E1;
            padding: 0.56rem 1.25rem;
            font-size: 0.82rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .filter-chip.active {
            border-color: var(--cyan);
            background: linear-gradient(135deg, var(--blue), var(--cyan));
            color: #020617;
            box-shadow: 0 0 22px rgba(34, 211, 238, 0.20);
        }

        .condition-image {
            position: relative;
            overflow: hidden;
            height: 158px;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            background: rgba(2, 6, 23, 0.36);
        }

        .condition-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .severity-pill {
            position: absolute;
            top: 0.75rem;
            right: 0.75rem;
            display: inline-flex;
            align-items: center;
            width: fit-content;
            border-radius: 999px;
            padding: 0.42rem 0.9rem;
            color: #CFFAFE;
            background: rgba(34, 211, 238, 0.14);
            border: 1px solid rgba(34, 211, 238, 0.18);
            font-size: 0.78rem;
            font-weight: 900;
        }

        .severity-pill.high {
            color: #FECACA;
            background: rgba(239, 68, 68, 0.14);
            border-color: rgba(239, 68, 68, 0.24);
        }

        .severity-pill.medium,
        .severity-pill.medium-high {
            color: #FEF3C7;
            background: rgba(245, 158, 11, 0.15);
            border-color: rgba(245, 158, 11, 0.24);
        }

        .severity-pill.none {
            color: #DCFCE7;
            background: rgba(34, 197, 94, 0.14);
            border-color: rgba(34, 197, 94, 0.24);
        }

        .condition-card h3 {
            min-height: 3.2rem;
            color: var(--text);
            font-size: 1.78rem;
            line-height: 1.1;
            margin: 0 0 0.75rem;
        }

        .condition-card p {
            color: #CBD5E1;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0;
        }

        .condition-details-link {
            margin-top: auto;
            padding-top: 1.5rem;
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .condition-empty {
            color: var(--muted);
            padding: 1rem 0;
        }

        .footer-note {
            color: var(--muted);
            text-align: center;
            padding: 2.5rem 0 1rem;
        }

        .footer-note p {
            color: var(--muted);
            margin: 0.45rem 0;
            font-size: 0.86rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            background: rgba(15, 31, 52, 0.78);
            color: var(--text);
            border: 1px solid var(--border);
        }

        div[data-testid="stSpinner"] > div {
            color: var(--cyan) !important;
        }

        @media (max-width: 920px) {
            .block-container {
                padding-top: 8.5rem;
            }

            .top-nav {
                align-items: flex-start;
                flex-direction: column;
            }

            .nav-links {
                justify-content: flex-start;
            }

            .hero-grid,
            .architecture-flow,
            .architecture-summary-grid,
            .process-grid,
            .condition-toolbar,
            .condition-grid,
            .metric-grid {
                grid-template-columns: 1fr;
            }

            .condition-filter {
                justify-content: flex-start;
            }

            .architecture-header {
                flex-direction: column;
            }

            .architecture-node:not(:last-child)::after {
                display: none;
            }

            .hero {
                min-height: auto;
            }

            .hero-visual {
                min-height: 320px;
            }

            .probability-row {
                grid-template-columns: 1fr;
                gap: 0.35rem;
            }

            .probability-value {
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_trained_model():
    """Load the trained model, cached for performance."""
    if MODEL_PATH.exists():
        return load_model(str(MODEL_PATH), compile=False)
    if BACKUP_MODEL_PATH.exists():
        return load_model(str(BACKUP_MODEL_PATH), compile=False)
    return None


@st.cache_data
def load_class_indices():
    """Load class indices mapping without changing the training label order."""
    if CLASS_INDICES_PATH.exists():
        with open(CLASS_INDICES_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_CLASS_INDICES


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_disease(model, image, class_indices):
    """
    Make prediction on uploaded image.

    Returns:
        dict with prediction results
    """
    processed = preprocess_for_prediction(image, target_size=IMG_SIZE)

    predictions = model.predict(processed, verbose=0)

    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = idx_to_class[predicted_idx]

    all_probs = {
        idx_to_class[i]: float(predictions[0][i])
        for i in range(len(predictions[0]))
        if i in idx_to_class
    }
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "all_probabilities": all_probs,
    }


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_navbar():
    """Render fixed top navigation with same-page anchor links."""
    logo_uri = image_to_data_uri(LOGO_PATH)
    logo_markup = (
        f'<img class="brand-logo-img" src="{logo_uri}" alt="DermScan AI logo">'
        if logo_uri
        else "D"
    )

    st.markdown(
        f"""
        <nav class="top-nav">
            <a class="brand" href="#home">
                <span class="brand-mark">{logo_markup}</span>
                <span class="brand-text">
                    <strong>DermScan AI</strong>
                    <span>Medical Vision System</span>
                </span>
            </a>
            <div class="nav-links">
                <a href="#home">Home</a>
                <a href="#system">System</a>
                <a href="#demo">Demo</a>
                <a href="#results">Results</a>
                <a href="#specs">Specs</a>
            </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(value, label):
    """Return one hero metric card."""
    return (
        f'<div class="metric-card">'
        f'<strong>{value}</strong>'
        f'<span>{label}</span>'
        f'</div>'
    )


def render_hero():
    """Render the landing hero section."""
    hero_video_uri = video_to_data_uri(HERO_VIDEO_PATH)
    hero_visual_markup = (
        f'<video class="hero-video" autoplay muted loop playsinline preload="auto" aria-label="DermScan AI animation">'
        f'<source src="{hero_video_uri}" type="video/mp4">'
        f'</video>'
        if hero_video_uri
        else '<div class="hero-video-fallback">Animation video mp_.mp4 was not found.</div>'
    )
    metric_cards = "".join(
        [
            render_metric_card("8", "Conditions"),
            render_metric_card("CNN", "Model"),
            render_metric_card("Image", "Upload"),
            render_metric_card("Score", "Confidence"),
        ]
    )

    hero_html = "\n".join(
        [
            '<div id="home" class="section-anchor"></div>',
            '<section class="hero">',
            '<div class="hero-grid">',
            '<div>',
            '<div class="eyebrow">Final Year Project - Dermatology AI</div>',
            '<h1 class="hero-title"><span class="brand-title">DermScan AI</span><span>Skin Disease Detection System</span></h1>',
            '<p class="hero-subtitle">AI-assisted skin lesion image classification using deep learning.</p>',
            '<p class="hero-copy">Upload a skin lesion image and receive AI-assisted prediction results with confidence score and probability breakdown.</p>',
            '<div class="cta-row">',
            '<a class="primary-cta" href="#demo">Try Prediction Demo</a>',
            '<a class="ghost-cta" href="#system">View Workflow</a>',
            '</div>',
            f'<div class="metric-grid">{metric_cards}</div>',
            '</div>',
            '<div class="hero-visual" aria-label="Dermatology AI scanning interface">',
            hero_visual_markup,
            '</div>',
            '</div>',
            '</section>',
        ]
    )

    st.markdown(hero_html, unsafe_allow_html=True)


def render_process_card(number, title, copy):
    """Return one process card."""
    return (
        f'<div class="process-card">'
        f'<div class="process-number">{number}</div>'
        f'<h3>{title}</h3>'
        f'<p>{copy}</p>'
        f'</div>'
    )


def render_cnn_architecture():
    """Return the CNN architecture panel for the System section."""
    nodes = [
        ("Input", "28 x 28 x 3 RGB", "Uploaded lesion image resized and normalized for inference."),
        ("Conv Block 1", "Conv2D 16 -> 32", "3x3 ReLU filters with same-padding feature extraction."),
        ("Pooling", "MaxPooling 2D", "Downsamples feature maps to reduce spatial complexity."),
        ("Conv Block 2", "Conv2D 32 -> 64", "Deeper convolution layers learn higher-level lesion patterns."),
        ("Classifier", "Dense 128 -> 64 -> 32 -> 8", "Flatten, dropout regularization, and softmax class output."),
    ]
    stats = [
        ("4", "Convolution layers"),
        ("2", "Max pooling stages"),
        ("0.4", "Dropout rate"),
        ("8", "Softmax outputs"),
    ]

    node_html = "".join(
        f'<div class="architecture-node">'
        f'<span>{label}</span>'
        f'<strong>{title}</strong>'
        f'<p>{copy}</p>'
        f'</div>'
        for label, title, copy in nodes
    )
    stat_html = "".join(
        f'<div class="architecture-stat">'
        f'<strong>{value}</strong>'
        f'<span>{label}</span>'
        f'</div>'
        for value, label in stats
    )

    return f"""
        <div class="architecture-wrap">
            <div class="architecture-card">
                <div class="architecture-header">
                    <div>
                        <div class="eyebrow">Neural Network</div>
                        <h3>CNN Architecture</h3>
                        <p>
                            The model follows the V2 training design: convolutional feature extraction,
                            pooling, dropout regularization, and dense classification for eight classes
                            (seven HAM10000 lesion types + healthy skin).
                        </p>
                    </div>
                    <div class="architecture-badge">Deep Learning Model</div>
                </div>
                <div class="architecture-flow">{node_html}</div>
                <div class="architecture-summary-grid">{stat_html}</div>
            </div>
        </div>
    """


def render_system_section():
    """Render the How It Works section."""
    cards = "".join(
        [
            render_process_card(
                "01",
                "Image Upload",
                "User uploads a skin lesion image through the Streamlit interface.",
            ),
            render_process_card(
                "02",
                "Preprocessing",
                "Image is resized, normalized, and prepared for model input.",
            ),
            render_process_card(
                "03",
                "CNN Prediction",
                "The trained deep learning model predicts the most likely skin condition.",
            ),
            render_process_card(
                "04",
                "Result Display",
                "The system displays predicted class, confidence score, and recommendation.",
            ),
        ]
    )
    architecture = render_cnn_architecture()

    st.markdown(
        f"""
        <div id="system" class="section-anchor"></div>
        <section class="section">
            <div class="section-heading">
                <div class="eyebrow">System Workflow</div>
                <h2>How It Works</h2>
                <p>
                    DermScan AI keeps the prediction flow simple for demonstration while preserving
                    the same trained CNN inference pipeline.
                </p>
            </div>
            <div class="process-grid">{cards}</div>
            {architecture}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(model_loaded):
    """Render the model readiness status."""
    if model_loaded:
        st.markdown(
            """
            <div class="status-card">
                <strong>Model loaded successfully</strong>
                <span>The CNN model is ready to analyze uploaded skin lesion images.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-card offline">
                <strong>Model not loaded</strong>
                <span>Please check the model file path before running prediction.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_demo_section(model, class_indices):
    """Render upload controls and run prediction button."""
    st.markdown(
        """
        <div id="demo" class="section-anchor"></div>
        <section class="section">
            <div class="section-heading">
                <div class="eyebrow">Interactive</div>
                <h2>Prediction Demo</h2>
                <p>
                    Upload a clear lesion image, preview the input, and run the trained model from
                    the same Streamlit page.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.18, 0.82], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Upload Skin Lesion Image</h3>
                <p class="demo-copy">Accepted formats are JPG, JPEG, and PNG. Use a focused, well-lit image for best results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the skin lesion for analysis.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded preview: {uploaded_file.name}", use_container_width=True)
        else:
            image = None
            st.info("Upload an image to enable the prediction demo.")

    with right_col:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Scan Readiness</h3>
                <p>Model status, supported format, and prediction control for the demo workflow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_status_card(model is not None)
        st.markdown(
            """
            <div class="glass-card">
                <h3>Supported Input</h3>
                <p>JPG, JPEG, or PNG image. The system applies the existing preprocessing pipeline before prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        button_disabled = model is None or uploaded_file is None
        run_prediction = st.button("Run Prediction", disabled=button_disabled, type="primary")

        if uploaded_file is not None and model is None:
            st.warning("Model file is missing, so prediction is currently disabled.")

    if run_prediction and image is not None and model is not None:
        image_np = np.array(image)
        processed_np = hair_removal(image_np)
        processed_image = Image.fromarray(processed_np)

        with st.spinner("Analyzing image with DermScan AI..."):
            result = predict_disease(model, image_np, class_indices)

        st.session_state.latest_result = result
        st.session_state.latest_image = image
        st.session_state.latest_processed_image = processed_image
        st.session_state.latest_file_name = uploaded_file.name
        st.session_state.scroll_to_results = True


def probability_bars(all_probs):
    """Return an HTML probability breakdown in the requested class display order."""
    ordered_names = [name for name in DISPLAY_CLASS_ORDER if name in all_probs]
    ordered_names.extend([name for name in all_probs if name not in ordered_names])

    rows = []
    for disease in ordered_names:
        prob = all_probs.get(disease, 0.0)
        percent = prob * 100
        rows.append(
            f'<div class="probability-row">'
            f'<div class="probability-name">{disease}</div>'
            f'<div class="probability-track">'
            f'<div class="probability-fill" style="width: {percent:.2f}%"></div>'
            f'</div>'
            f'<div class="probability-value">{percent:.1f}%</div>'
            f'</div>'
        )

    return f'<div class="probability-stack">{"".join(rows)}</div>'


def render_prediction_result(result, image, processed_image):
    """Render prediction result section."""
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    all_probs = result["all_probabilities"]

    disease_info = DISEASE_INFO.get(
        predicted_class,
        {
            "description": "No description available.",
            "severity": "Unknown",
            "recommendation": "Please consult a dermatologist for clinical review.",
        },
    )

    confidence_percent = confidence * 100
    confidence_class = "high" if confidence >= 0.70 else "review"
    confidence_text = "High confidence" if confidence >= 0.70 else "Needs review"

    st.markdown(
        """
        <div id="results" class="section-anchor"></div>
        <section class="section">
            <div class="section-heading">
                <div class="eyebrow">Model Output</div>
                <h2>Prediction Results</h2>
                <p>Review the uploaded image, predicted condition, model confidence, and probability breakdown.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    img_col, result_col = st.columns([0.9, 1.1], gap="large")

    with img_col:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Uploaded Image Preview</h3>
                <p>Original upload and existing hair-removal preprocessing visualization.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preview_col, processed_col = st.columns(2)
        with preview_col:
            st.image(image, caption="Original Image", use_container_width=True)
        with processed_col:
            st.image(processed_image, caption="Preprocessed", use_container_width=True)

    with result_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted condition</div>
                <div class="predicted-condition">{predicted_class}</div>
                <div class="confidence-badge {confidence_class}">
                    {confidence_text}: {confidence_percent:.1f}%
                </div>
                <div class="note-card">
                    <strong>Recommendation / note</strong>
                    <p>{disease_info["recommendation"]}</p>
                </div>
                <div class="note-card">
                    <strong>Condition context</strong>
                    <p>{disease_info["description"]}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="result-card" style="margin-top: 1.2rem;">
            <div class="result-label">Probability breakdown</div>
            {probability_bars(all_probs)}
            <div class="disclaimer">
                This AI prediction is intended to support clinical review and is not a final medical diagnosis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_section():
    """Render results anchor even before prediction, then show results if available."""
    if "latest_result" in st.session_state:
        render_prediction_result(
            st.session_state.latest_result,
            st.session_state.latest_image,
            st.session_state.latest_processed_image,
        )
        if st.session_state.pop("scroll_to_results", False):
            components.html(
                """
                <script>
                setTimeout(() => {
                    const target = window.parent.document.getElementById("results");
                    if (target) {
                        target.scrollIntoView({ behavior: "smooth", block: "start" });
                    } else {
                        window.parent.location.hash = "results";
                    }
                }, 150);
                </script>
                """,
                height=0,
            )
    else:
        st.markdown(
            """
            <div id="results" class="section-anchor"></div>
            <section class="section">
                <div class="section-heading">
                    <div class="eyebrow">Model Output</div>
                    <h2>Prediction Results</h2>
                    <p>Results will appear here after an image is uploaded and the prediction demo is run.</p>
                </div>
                <div class="glass-card">
                    <h3>No prediction yet</h3>
                    <p>Run the demo to view predicted condition, confidence score, probability breakdown, recommendation, and disclaimer.</p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )


def truncate_text(text, limit=92):
    """Keep condition summaries compact like the original library cards."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def escape_html(value):
    """Escape dynamic text before inserting into HTML."""
    return html.escape(str(value), quote=True)


def display_name(condition):
    """Use polished card titles while preserving model class names."""
    names = {
        "Melanocytic nevi": "Melanocytic Nevi",
        "Actinic keratoses": "Actinic Keratosis",
        "Benign keratosis-like lesions": "Benign Keratosis",
        "Basal cell carcinoma": "Basal Cell Carcinoma",
        "Vascular lesions": "Vascular Lesion",
        "Healthy skin": "Healthy Skin",
    }
    return names.get(condition, condition)


def severity_key(severity):
    """Convert severity labels into CSS-friendly values."""
    return str(severity).lower().replace(" ", "-")


def severity_bucket(severity):
    """Group detailed severities into the risk text used by the card design."""
    value = str(severity).lower()
    if "high" in value:
        return "High"
    if "medium" in value:
        return "Medium"
    if value == "none":
        return "None"
    return "Low"


def short_condition_description(condition, fallback):
    """Keep card copy close to the V3 card style."""
    custom = {
        "Melanocytic nevi": (
            "Common moles that are non-cancerous and typically harmless. "
            "Characterized by uniform color and regular borders."
        ),
        "Healthy skin": (
            "No visible lesion abnormality detected. Continue regular skin checks "
            "and sun protection."
        ),
    }
    return custom.get(condition, fallback)


def render_specs_section():
    """Render condition library cards in the bottom section."""
    cards = []
    for condition in CONDITION_CARD_ORDER:
        info = DISEASE_INFO.get(condition)
        if not info:
            continue

        severity = info.get("severity", "Unknown")
        risk_bucket = severity_bucket(severity)
        risk_text = "None" if risk_bucket == "None" else f"{risk_bucket} Risk"
        title = display_name(condition)
        description = short_condition_description(
            condition,
            info.get("description", "No description available."),
        )
        image_uri = image_to_data_uri(CONDITION_IMAGE_PATHS.get(condition, ASSETS_DIR / "__missing__.jpg"))
        cards.append(
            f'<article class="condition-card">'
            f'<div class="condition-image">'
            f'<img src="{image_uri}" alt="{escape_html(title)}">'
            f'<div class="severity-pill {severity_key(severity)}">{escape_html(risk_text)}</div>'
            f'</div>'
            f'<h3>{escape_html(title)}</h3>'
            f'<p>{escape_html(description)}</p>'
            f'<div class="condition-details-link">View Details -&gt;</div>'
            f'</article>'
        )

    card_markup = "".join(cards) or '<p class="condition-empty">No conditions available.</p>'

    st.markdown(
        f"""
        <div id="specs" class="section-anchor"></div>
        <section class="section">
            <div class="section-heading">
                <div class="eyebrow">Condition Library</div>
                <h2>Detectable Skin Conditions</h2>
                <p>The dashboard supports eight classes — seven lesion types plus healthy skin — and presents them as compact clinical summary cards.</p>
            </div>
            <div class="condition-toolbar">
                <input class="condition-search" placeholder="Search conditions..." aria-label="Search conditions" />
                <div class="condition-filter">
                    <span class="condition-filter-label">Filter by Severity:</span>
                    <span class="filter-chip active">All</span>
                    <span class="filter-chip">High</span>
                    <span class="filter-chip">Medium</span>
                    <span class="filter-chip">Low</span>
                </div>
            </div>
            <div class="condition-grid">{card_markup}</div>
            <div class="footer-note">
                <p>Skin Disease Detection System | Final Year Project</p>
                <p>Muhammad Haikhal Bin Omanudin Baki | 2026</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_no_model_details():
    """Render model path guidance when the trained model cannot be loaded."""
    st.markdown(
        f"""
        <div class="glass-card">
            <h3>Model File Required</h3>
            <p>
                Place the trained model at <code>{MODEL_PATH}</code> or <code>{BACKUP_MODEL_PATH}</code>.
                The interface remains available, but prediction is disabled until a model is found.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point."""
    load_custom_css()
    render_navbar()

    model = load_trained_model()
    class_indices = load_class_indices()

    render_hero()
    render_system_section()
    render_demo_section(model, class_indices)

    if model is None:
        render_no_model_details()

    render_results_section()
    render_specs_section()


if __name__ == "__main__":
    main()
