# -*- coding: utf-8 -*-
"""
DermScan AI V3 - one-page Stitch-inspired Streamlit interface.

Run:
    streamlit run app_V3.py
"""

import base64
import html
import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR if (APP_DIR / "preprocessing.py").exists() else APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import DISEASE_INFO, hair_removal, preprocess_for_prediction


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "skin_disease_model.h5"
BACKUP_MODEL_PATH = MODELS_DIR / "best_model.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "LogoGUI.jpg"
STYLE_PATH = PROJECT_ROOT / "styles" / "custom.css"
IMG_SIZE = (28, 28)
SCAN_VIDEO_PATH = ASSETS_DIR / "ScanSkin.mp4"
PAGE_ICON = Image.open(LOGO_PATH) if LOGO_PATH.exists() else "DS"

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

DISPLAY_CLASS_ORDER = [
    "Melanoma",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Melanocytic nevi",
    "Benign keratosis-like lesions",
    "Healthy skin",
    "Dermatofibroma",
    "Vascular lesions",
]

HERO_IMAGE = "https://lh3.googleusercontent.com/aida-public/AB6AXuAya-SqSXIoZEP3Po-NqaVc3BtS3Scp0Mw1IJAgC_AIuTrWozTuXv9ifKq77bZj8eFVBczel4esOAbwq4h3fuHPmdPQJ7kvrqeDZFDiUSUo5jY7Upt-9X67bofbqN8PoeUR5j4COkZnZrrqKGDBUMJv-Q5yuRi_tWTuPVVzIEb173InZMWqcjNybcjrRmfTbrWavPtBXma2rg7ccjFXYvVi0ybAdolxedtpIfIw8MV36lK4hTBgoOl7Q6z4otWqSQQHVloh63ef2sg"
RESULT_ORIGINAL = "https://lh3.googleusercontent.com/aida-public/AB6AXuDGWBbyb_PqjTo4eFpO3ImoucGWPuVPWsbq0D8Jpe0LMD_rTemTrHNgatYy-F8HhyjbW8tC9a3sO9fQ9rcj-SBci-OIWriLLHth2Tohj4JV4I7IBdeWn1WL2ktGxsZlw9qtdgrEVEcj6DYX-MeLWSHz2JI2ruwzjt_kIRBUyDSGzOgolSY4ksuCNGvWfRhg60bk6ku-AekKFIW4JzLrwFT2LHYk78d0trGFPV3MTKDdr1Pg9-JRLNIg8XSvtZE1u_n4yAk5tPgafUs"
RESULT_PROCESSED = "https://lh3.googleusercontent.com/aida-public/AB6AXuBTyIHP9NVPZlHRQ8c3SoehXd-du9QTpgtFHXiMS6MlyQpAuBzLjYyVAQsBS6FRHwcL_0Cv59UIRN8OjaOxL4lKsEtU9KUePRK_i8A5dpbtPLIgf4b8QHa_ZuezB2TfKUu3KKCjJN0ONeDL7KsF8LCqRNlYcIXz3JHHoOO29cQQ1YGR98HNhDnfYQItrm4ykD2p9iMN3tQMUHCWeYR-DYPrlzGgutS__U8o9OkkYBgqBs3efEpOUEpNW9dZ5pFuBWKCZYiruusnW3M"

CONDITION_IMAGES = {
    "Melanoma": "https://lh3.googleusercontent.com/aida-public/AB6AXuBOFzMFVW1UvdmUEPpRWCXvWKZ-lrQKOxo0U1AkmT1rAiQklu04eWLbddXLFUffUb0wzHeP4Iz9MSHd8qn_toEWWZQbD_ZHYd9Dlr3IuRC8ULmgyddEHGPAoH8Asthkxlzo6Jkl0-Qb22KliTdo0eeX2-0uqIltiKnNVy_75J3C2O9RtbKWKD5VgrAp_WVpjiO70CSuEU9i7l_IRnvYYtHQDyc_FZLi-VEqcu2gsZXZ8ATLvGcXTd2gSxyWck_yZMbheMKzQRNvxrc",
    "Basal cell carcinoma": "https://lh3.googleusercontent.com/aida-public/AB6AXuBHz6_ygTy6qvaW_d75TbyZRZs-gY30U65IW7pLudvspWgtWz85oZwetB322EE5AiHyAzaUh9luJcZf3yT8H4hgtBbsOhY_K4NsKJIMEobJWBjlrAbv486J6NI7bQVPFVAEy5vs-58OKU9znfzVl_DGwo0UZuUADa43nQy4a7DiTpmfNXMj4BeTfpC324bPRtKEKR8q6kos-maIpZniY6k5kZeU6WZZqJHxPPFV3Z8BWutifd4UwLxxacaRPvzb7sx-RsmXbsN22sM",
    "Actinic keratoses": "https://lh3.googleusercontent.com/aida-public/AB6AXuDYTgtSUPKKXhROBHfSJUCHLft9YFj86kS-ohDyX8R4bLyCaQHZJDqwraDAnwcd1Pu1RMcXtuQ9pDazUylzHm-KqH2uKQaRxS-KmSoCXpBjb-tJFA3IrA4qPKWTvzLxzDdfImDflXQDRTFg1VicDVI9zEI96fHd992Bls9t5kldg_fLLE6QxdXblKPhy9faRfqV7NF8O-bSr_055fvZTgqKOSpgFMh_wcemIVfE_aLcRUPTiI2imitKl6-4opAnm7lyxBcgUnUihlU",
    "Melanocytic nevi": "https://lh3.googleusercontent.com/aida-public/AB6AXuBhj0mij8OOu07L5JtAVtWqDuYvE2rpr-N6BnE1Ke-JNbZwVT_IlmE6_UR7cQfZKPxlkGm7-uGLfvFuADlGdM_KmabKBWHn4dgB9EHFvMHuyxSMWwn6fSxzV7smOj0hoJk6UKFrsdOoSM7UhpAC2PC8NUES_UE3Yc-omGxV9YJBj1XZrvmtkdk5Mep3pxE9kiQT2anCpqyU7HsNWH-qckLJ1YVwPEk7--1AdX3g5oa2EKSUsLee2XvMR3F6fE7jpBkSFlHpjE6tfYA",
    "Benign keratosis-like lesions": "https://lh3.googleusercontent.com/aida-public/AB6AXuB-0Wq5nq9Q0Qt83HzAJJoj0fFE7zOZvI1mH3LMT2Bo0KRM847Ed0sNVu1mfT44JgGJtCGUspL66bM75u4KjS9TcnrdlcSlBduWjgm25u5dPnEFiOsI4osnAyXvfyowx4n5UzU__z2ng4BdNYh-ZhIRnPLwVnzTypk72FARch7WHyyuc3y9nRd5VzaQM81WNPsQ_p7ZvZ4_y_7YLGn2vb4_3qS-gZE8VB4tUddmXxUE3p0ktgDu9dYROihS7-nAxPFvbM9pIeK1oQs",
    "Healthy skin": "https://lh3.googleusercontent.com/aida-public/AB6AXuDCow7Nk_AlK9cK0jCLm4fMpUerwajopBDZXviPoBZRskNKOAHbx7NlkZkAnY2CMdt7z8v1whZXdH1zQqLNWlUgcltMEih2CcPM4Dpl_HLEUYhsA9upGyZ_EI0g_AHLD252GUfPLyBxa4sC207EnfEtWkJPyS9oJQOjK5dj61XeJKjS-BoOoHBL8Nz0q-Ew5FAb5nElSxzHYhOMf2o5qJHH2veczA9mPlPtdd76LMhMYhUXy7zkqCcDm8EUHUjCK8Bl1nhUJ8uDrcU",
    "Dermatofibroma": "https://lh3.googleusercontent.com/aida-public/AB6AXuAGSvYQZzWjpVZWcVMpSsx7bBMKDXjmCtAhYDX3Trbnf8W_emk14UX_Ezy2nkwGpCTkE_vfJdMYpJh881ykTm8EzWHboqgNf8e7CAY6QUkCu6OZAKwYH4OCsXP2zMQGQIDgKKOmA6oXCFHiATBMcX95hV0E1FjflSgHcuTQ7sPq3-aBeQdCMq50CrcXEacBBfLTkptuSe7ctCF2l2Sid7Gg24lkl6cOZxcGLavwb7fGOtG0IgOnOkUGbm7jY1SOR6ZiTRFIpHpBSaQ",
    "Vascular lesions": "https://lh3.googleusercontent.com/aida-public/AB6AXuAXnAc8Q5Y839aYLAOWL-zYvfLEoHxUysBLu2yP2YCPSaOYmqqYRzHaJyRksEBtIQLdtTPjhYlM3VOcVwG-uoYww9WAknO0-hNit7rzT6rJzWuZMY6a-C50SxzOGJ6tQNY-vSns9gjw1FOfI45HePLKVzF4eol_RFSU0hC-zpY9PNGVUJ4mUK-BgKXqLB7cWCMv8SA32h0nz07sEHwm_G2egNn2LQ2jhawuB8PCl255PEXgYqieJXZUsMC4DfngFM7LtBOKUlZ4pJs",
}

CONDITION_IMAGE_PATHS = {
    "Melanoma": ASSETS_DIR / "mel" / "ISIC_0025081.jpg",
    "Basal cell carcinoma": ASSETS_DIR / "bcc" / "ISIC_0028155.jpg",
    "Actinic keratoses": ASSETS_DIR / "akiec" / "ISIC_0025808.jpg",
    "Melanocytic nevi": ASSETS_DIR / "nv" / "ISIC_0024975.jpg",
    "Benign keratosis-like lesions": ASSETS_DIR / "bkl" / "ISIC_0025030.jpg",
    "Healthy skin": ASSETS_DIR / "normal skin" / "google 1.jpg",
    "Dermatofibroma": ASSETS_DIR / "df" / "ISIC_0027008.jpg",
    "Vascular lesions": ASSETS_DIR / "vasc" / "ISIC_0029486.jpg",
}


st.set_page_config(
    page_title="DermScan AI - Skin Disease Detection System",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css():
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def apply_theme(theme: str = "light"):
    # Set a data-theme attribute on the top-level document element so CSS variables switch.
    components.html(
        f"""
        <script>
        try {{
            document.documentElement.setAttribute('data-theme', '{theme}');
        }} catch (e) {{}}
        </script>
        """,
        height=0,
        width=0,
    )


def esc(value):
    return html.escape(str(value), quote=True)


def array_to_data_uri(image_array):
    image = Image.fromarray(image_array.astype("uint8"))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def pil_to_data_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def file_to_data_uri(path):
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".")
    mime_by_suffix = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "mp4": "video/mp4",
    }
    mime = mime_by_suffix.get(suffix, f"application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def condition_image_src(condition):
    local_path = CONDITION_IMAGE_PATHS.get(condition)
    if local_path is not None:
        local_image = file_to_data_uri(local_path)
        if local_image:
            return local_image
    return CONDITION_IMAGES.get(condition, HERO_IMAGE)


def find_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH
    if BACKUP_MODEL_PATH.exists():
        return BACKUP_MODEL_PATH
    return None


@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_trained_model(model_path):
    from tensorflow.keras.models import load_model

    return load_model(model_path, compile=False)


@st.cache_data
def load_class_indices():
    if CLASS_INDICES_PATH.exists():
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return DEFAULT_CLASS_INDICES


def get_model():
    model_path = find_model_path()
    if model_path is None:
        return None, None, "No .h5 model found in the models folder."
    try:
        return load_trained_model(str(model_path)), model_path, ""
    except Exception as exc:
        return None, model_path, f"Model loading failed: {exc}"


def predict_disease(model, image, class_indices):
    processed = preprocess_for_prediction(image, target_size=IMG_SIZE)
    predictions = np.asarray(model.predict(processed, verbose=0))[0]
    idx_to_class = {int(value): key for key, value in class_indices.items()}
    predicted_idx = int(np.argmax(predictions))

    all_probs = {
        idx_to_class[index]: float(probability)
        for index, probability in enumerate(predictions)
        if index in idx_to_class
    }
    all_probs = dict(sorted(all_probs.items(), key=lambda item: item[1], reverse=True))

    return {
        "predicted_class": idx_to_class.get(predicted_idx, f"Class {predicted_idx}"),
        "confidence": float(predictions[predicted_idx]),
        "all_probabilities": all_probs,
    }


def init_state():
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("latest_original_uri", RESULT_ORIGINAL)
    st.session_state.setdefault("latest_processed_uri", RESULT_PROCESSED)
    st.session_state.setdefault("latest_time", "Oct 24, 2024 at 14:32 PST")
    st.session_state.setdefault("jump_to_results", False)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("theme", "light")


def jump_to_anchor(anchor_id):
    components.html(
        f"""
        <script>
        const anchorId = "{anchor_id}";

        function jump(retries) {{
            const target = window.parent.document.getElementById(anchorId);
            if (target) {{
                target.scrollIntoView({{ behavior: "smooth", block: "start" }});
                window.parent.history.replaceState(null, "", "#" + anchorId);
                return;
            }}
            if (retries > 0) {{
                window.setTimeout(() => jump(retries - 1), 100);
            }}
        }}

        window.setTimeout(() => jump(30), 100);
        </script>
        """,
        height=0,
        width=0,
    )


def nudge_scan_video_playback():
    components.html(
        """
        <script>
        (function () {
            function tryPlay() {
                const video = window.parent.document.getElementById("scan-video");
                if (!video) {
                    return false;
                }

                video.muted = true;
                video.loop = true;
                video.playsInline = true;
                video.setAttribute("playsinline", "playsinline");
                video.setAttribute("webkit-playsinline", "webkit-playsinline");

                const playPromise = video.play();
                if (playPromise && typeof playPromise.catch === "function") {
                    playPromise.catch(function () {});
                }
                return true;
            }

            if (tryPlay()) {
                return;
            }

            let attempts = 0;
            const timer = window.setInterval(function () {
                attempts += 1;
                if (tryPlay() || attempts > 20) {
                    window.clearInterval(timer);
                }
            }, 250);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def condition_info(condition):
    return DISEASE_INFO.get(
        condition,
        {
            "description": "Clinical information is not available for this class.",
            "severity": "Unknown",
            "recommendation": "Please consult a qualified medical professional.",
        },
    )


def severity_key(severity):
    value = str(severity).lower().replace(" ", "-")
    return "medium-high" if value == "medium-high" else value


def severity_bucket(severity):
    value = str(severity).lower()
    if "high" in value:
        return "High"
    if "medium" in value:
        return "Medium"
    if value == "none":
        return "None"
    return "Low"


def display_name(condition):
    names = {
        "Melanocytic nevi": "Melanocytic Nevi",
        "Actinic keratoses": "Actinic Keratosis",
        "Benign keratosis-like lesions": "Benign Keratosis",
        "Basal cell carcinoma": "Basal Cell Carcinoma",
        "Vascular lesions": "Vascular Lesion",
        "Healthy skin": "Healthy Skin",
    }
    return names.get(condition, condition)


def short_description(condition):
    custom = {
        "Melanocytic nevi": "Common moles that are non-cancerous and typically harmless. Characterized by uniform color and regular borders.",
        "Healthy skin": "No visible lesion abnormality detected. Continue regular skin checks and sun protection.",
    }
    return custom.get(condition, condition_info(condition).get("description", ""))


def render_topbar():
    logo_uri = file_to_data_uri(LOGO_PATH)
    logo_markup = f'<img src="{logo_uri}" alt="DermScan AI logo">' if logo_uri else '<span>DS</span>'
    st.markdown(
        f"""
        <header class="ds-topbar">
            <nav class="ds-nav">
                <a class="ds-brand-card" href="#home">
                    <span class="ds-brand-logo">{logo_markup}</span>
                    <span class="ds-brand-copy">
                        <span class="ds-brand-title">DermScan AI</span>
                        <span class="ds-brand-subtitle">Medical Vision System</span>
                    </span>
                </a>
                <div class="ds-links">
                    <a class="nav-link nav-home" href="#home">Home</a>
                    <a class="nav-link nav-system" href="#system">System</a>
                    <a class="nav-link nav-demo" href="#demo">Demo</a>
                    <a class="nav-link nav-results" href="#results">Results</a>
                    <a class="nav-link nav-library" href="#library">Library</a>
                </div>
                <div class="ds-actions">
                    <a class="ds-mobile-action" href="#demo" aria-label="Open scan section">
                        <span class="material-symbols-outlined">account_circle</span>
                    </a>
                    <a class="ds-cta" href="#demo">Get Started</a>
                    <button id="ds-theme-toggle" class="ds-theme-toggle" aria-label="Toggle theme">🌙</button>
                </div>
            </nav>
        </header>
        <nav class="ds-mobile-bottom-nav" aria-label="Primary mobile navigation">
            <a class="mobile-nav-link nav-home" href="#home">
                <span class="material-symbols-outlined">home</span>
                <span>Home</span>
            </a>
            <a class="mobile-nav-link nav-demo" href="#demo">
                <span class="material-symbols-outlined">photo_camera</span>
                <span>Scan</span>
            </a>
            <a class="mobile-nav-link nav-results" href="#results">
                <span class="material-symbols-outlined">analytics</span>
                <span>Results</span>
            </a>
            <a class="mobile-nav-link nav-library" href="#library">
                <span class="material-symbols-outlined">menu_book</span>
                <span>Library</span>
            </a>
        </nav>
        """,
        unsafe_allow_html=True,
    )
    # Inject client-side script to wire the theme toggle and persist choice to localStorage
    components.html(
        """
        <script>
        (function(){
            try{
                const parentDoc = window.parent.document;
                function setTheme(t){
                    parentDoc.documentElement.setAttribute('data-theme', t);
                    localStorage.setItem('dermscan-theme', t);
                    const btn = parentDoc.getElementById('ds-theme-toggle');
                    if(btn){
                        btn.classList.toggle('active', t === 'dark');
                        btn.textContent = t === 'dark' ? '☀' : '🌙';
                    }
                }

                const init = localStorage.getItem('dermscan-theme') || 'light';
                setTheme(init);

                const toggle = parentDoc.getElementById('ds-theme-toggle');
                if(toggle && !toggle._ds_init){
                    toggle._ds_init = true;
                    toggle.addEventListener('click', function(){
                        const current = localStorage.getItem('dermscan-theme') === 'dark' ? 'dark' : 'light';
                        const next = current === 'dark' ? 'light' : 'dark';
                        setTheme(next);
                    });
                }
            }catch(e){}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_home():
    scan_video_uri = file_to_data_uri(SCAN_VIDEO_PATH)
    scan_media_markup = (
        f'<video id="scan-video" class="scan-media" autoplay="autoplay" muted="muted" '
        f'loop="loop" playsinline="playsinline" webkit-playsinline="webkit-playsinline" '
        f'preload="metadata" controls="controls" controlsList="nodownload noplaybackrate">'
        f'<source src="{scan_video_uri}" type="video/mp4">'
        f'</video>'
        if scan_video_uri
        else f'<img class="scan-media" src="{HERO_IMAGE}" alt="Skin scan visualization">'
    )

    st.html(
        f"""
        <section id="home" class="ds-section">
            <div class="home-grid">
                <div>
                    <span class="ds-eyebrow">Clinical Decision Support</span>
                    <h1 class="ds-title">DermScan AI - <span>Skin Disease Detection System</span></h1>
                    <p class="ds-subtitle">
                        AI-assisted skin lesion detection for clinical demonstration. Leveraging advanced CNN architectures
                        to provide high-precision diagnostic support for dermatologists.
                    </p>
                    <div class="hero-actions">
                        <a class="primary-action" href="#demo">Start Prediction Demo</a>
                        <a class="secondary-action" href="#system">View Workflow</a>
                    </div>
                </div>
                <div class="scan-wrap">
                    <div class="scan-card">
                        {scan_media_markup}
                    </div>
                </div>
            </div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="icon-box">category</div>
                    <h3>8 Conditions</h3>
                    <p>Comprehensive classification of pigmented skin lesions.</p>
                </div>
                <div class="metric-card">
                    <div class="icon-box secondary">architecture</div>
                    <h3>CNN Model</h3>
                    <p>Custom multi-layer neural network optimized for clinical precision.</p>
                </div>
                <div class="metric-card">
                    <div class="icon-box">photo_size_select_small</div>
                    <h3>28x28 Input</h3>
                    <p>HAM10000 optimized grayscale and RGB processing pipeline.</p>
                </div>
                <div class="metric-card">
                    <div class="icon-box secondary">speed</div>
                    <h3>Real-time</h3>
                    <p>Inference latency under 150ms for instantaneous feedback.</p>
                </div>
            </div>
            <div class="mobile-metric-grid">
                <div class="mobile-metric-card">
                    <span class="material-symbols-outlined mobile-metric-icon">category</span>
                    <strong>8</strong>
                    <span>Conditions</span>
                </div>
                <div class="mobile-metric-card">
                    <span class="material-symbols-outlined mobile-metric-icon">neurology</span>
                    <strong>CNN</strong>
                    <span>Model</span>
                </div>
                <div class="mobile-metric-card">
                    <span class="material-symbols-outlined mobile-metric-icon">photo_size_select_small</span>
                    <strong>28x28</strong>
                    <span>Input</span>
                </div>
                <div class="mobile-metric-card">
                    <span class="material-symbols-outlined mobile-metric-icon">speed</span>
                    <strong>Real-time</strong>
                    <span>Analysis</span>
                </div>
            </div>
        </section>
        """
    )
    if scan_video_uri:
        nudge_scan_video_playback()


def render_system():
    st.markdown(
        """
        <section id="system" class="ds-band">
            <div class="ds-section workflow">
                <span class="ds-eyebrow">Methodology</span>
                <h2 class="ds-h2">Precision Diagnostic Workflow</h2>
                <div class="workflow-steps">
                    <div class="workflow-step">
                        <div class="step-number">01</div>
                        <h3>Data Acquisition</h3>
                        <p>Raw dermatoscopic imagery is captured and pre-processed for resolution standardization.</p>
                    </div>
                    <div class="workflow-step">
                        <div class="step-number">02</div>
                        <h3>CNN Analysis</h3>
                        <p>Multi-layered feature extraction identifies patterns invisible to the naked eye.</p>
                    </div>
                    <div class="workflow-step">
                        <div class="step-number active">03</div>
                        <h3>Class Output</h3>
                        <p>Weighted probability scores provided for 8 distinct diagnostic categories.</p>
                    </div>
                </div>
                <div class="architecture-panel">
                    <div class="architecture-head">
                        <div>
                            <span class="architecture-eyebrow">Neural Network</span>
                            <h3>CNN Architecture</h3>
                        </div>
                        <span class="architecture-badge">Deep Learning Model</span>
                    </div>
                    <p class="architecture-copy">
                        The model follows the V2 training design: convolutional feature extraction, pooling,
                        dropout regularization, and dense classification for eight classes
                        (seven HAM10000 lesion types + healthy skin).
                    </p>
                    <div class="architecture-flow">
                        <div class="architecture-node">
                            <span>Input</span>
                            <strong>28 x 28 x 3 RGB</strong>
                            <p>Uploaded lesion image resized and normalized for inference.</p>
                        </div>
                        <div class="architecture-node">
                            <span>Conv Block 1</span>
                            <strong>Conv2D 16 -> 32</strong>
                            <p>3x3 ReLU filters with same-padding feature extraction.</p>
                        </div>
                        <div class="architecture-node">
                            <span>Pooling</span>
                            <strong>MaxPooling 2D</strong>
                            <p>Downsamples feature maps to reduce spatial complexity.</p>
                        </div>
                        <div class="architecture-node">
                            <span>Conv Block 2</span>
                            <strong>Conv2D 32 -> 64</strong>
                            <p>Deeper convolution layers learn higher-level lesion patterns.</p>
                        </div>
                        <div class="architecture-node">
                            <span>Classifier</span>
                            <strong>Dense 128 -> 64 -> 32 -> 8</strong>
                            <p>Flatten, dropout regularization, and softmax class output.</p>
                        </div>
                    </div>
                    <div class="architecture-stats">
                        <div><strong>4</strong><span>Convolution layers</span></div>
                        <div><strong>2</strong><span>Max pooling stages</span></div>
                        <div><strong>0.4</strong><span>Dropout rate</span></div>
                        <div><strong>8</strong><span>Softmax outputs</span></div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_demo(model, model_path, model_error, class_indices):
    demo_light_bg = file_to_data_uri(ASSETS_DIR / "White Background.png")
    demo_dark_bg = file_to_data_uri(ASSETS_DIR / "Dark Background.png")
    st.markdown(
        f"""
        <style>
        :root {{
            --demo-bg-light: url("{demo_light_bg}");
            --demo-bg-dark: url("{demo_dark_bg}");
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section id="demo" class="ds-section demo-intro-section">
            <div class="section-header">
                <h2 class="ds-title" style="font-size: 50px;">Prediction Demo</h2>
                <p class="ds-subtitle">
                    Upload a high-resolution image of a skin lesion for immediate CNN-powered diagnostic analysis.
                    Our system performs multi-layered scanning for precise classification.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    image_np = None
    mobile_run_prediction = False
    desktop_run_prediction = False
    with st.container(border=False, key="demo_stage"):
        left, right = st.columns([0.58, 0.42], gap="large")

        with left:
            with st.container(border=False, key="demo_upload_card"):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="material-symbols-outlined">upload_file</span>
                        <h3>Upload Skin Lesion Image</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                upload_box = st.empty()
                uploaded = st.file_uploader(
                    "Drag and drop lesion image here",
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=False,
                    label_visibility="collapsed",
                )

                if uploaded is not None:
                    image = Image.open(uploaded).convert("RGB")
                    image_np = np.array(image)
                    upload_box.markdown(
                        f"""
                        <div class="upload-shell upload-shell-preview">
                            <img class="upload-preview-image" src="{pil_to_data_uri(image)}" alt="Selected lesion image">
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    upload_box.markdown(
                        """
                        <div class="upload-shell">
                            <div class="upload-placeholder">
                                <span class="material-symbols-outlined">add_a_photo</span>
                                <p>Drag and drop lesion image here</p>
                                <small>Maximum file size: 200MB</small>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if uploaded is not None:
                with st.container(border=False, key="demo_mobile_run"):
                    mobile_run_prediction = st.button(
                        "Run Prediction",
                        type="primary",
                        use_container_width=True,
                        disabled=model is None,
                        key="mobile_run_prediction",
                    )

        model_status = "Model Loaded" if model is not None else "Model Missing"
        model_version = model_path.name if model_path else "No model"
        model_class = "success" if model is not None else "neutral"
        with right:
            with st.container(border=False, key="status_card"):
                st.markdown(
                    f"""
                    <h3 class="status-heading">System Status</h3>
                    <div class="status-row {model_class}">
                        <span><span class="material-symbols-outlined">check_circle</span> <strong>{esc(model_status)}</strong></span>
                        <span>{esc(model_version)}</span>
                    </div>
                    <div class="status-row neutral">
                        <span><span class="material-symbols-outlined">info</span> <strong>Supported Input</strong></span>
                        <span>JPG, PNG, WEBP</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.container(border=False, key="execution_card"):
                st.markdown(
                    """
                    <h3 class="status-heading">Execution</h3>
                    <p class="ds-copy">
                        Once an image is uploaded, the existing preprocessing module removes hair artifacts,
                        normalizes the image to 28x28 RGB input, and sends it to the trained CNN model.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

                if model_error:
                    st.error(model_error)

                disabled = image_np is None or model is None
                desktop_run_prediction = st.button(
                    "Run Prediction",
                    type="primary",
                    use_container_width=True,
                    disabled=disabled,
                    key="desktop_run_prediction",
                )
                st.markdown(
                    '<p class="ds-copy" style="text-align:center;margin-top:16px;">Estimated processing time: &lt; 2.4s</p>',
                    unsafe_allow_html=True,
                )

    run_prediction = mobile_run_prediction or desktop_run_prediction
    if run_prediction and image_np is not None and model is not None:
        with st.spinner("Running hair removal, preprocessing, and CNN inference..."):
            cleaned = hair_removal(image_np)
            result = predict_disease(model, image_np, class_indices)
        st.session_state.latest_result = result
        st.session_state.latest_original_uri = pil_to_data_uri(Image.fromarray(image_np))
        st.session_state.latest_processed_uri = array_to_data_uri(cleaned)
        st.session_state.latest_time = datetime.now().strftime("%b %d, %Y at %H:%M")
        st.session_state.history.insert(
            0,
            {
                "filename": uploaded.name,
                "time": st.session_state.latest_time,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
            },
        )
        st.session_state.jump_to_results = True


def fallback_result():
    all_probs = {
        "Melanoma": 0.824,
        "Melanocytic nevi": 0.112,
        "Basal cell carcinoma": 0.031,
        "Actinic keratoses": 0.014,
        "Benign keratosis-like lesions": 0.009,
        "Dermatofibroma": 0.005,
        "Vascular lesions": 0.003,
        "Healthy skin": 0.002,
    }
    return {
        "predicted_class": "Melanoma",
        "confidence": 0.824,
        "all_probabilities": all_probs,
    }


def render_probability_rows(probabilities):
    rows = []
    for index, (condition, probability) in enumerate(probabilities.items()):
        percent = probability * 100
        fill_class = "hot" if index == 0 else "low" if percent < 5 else ""
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-meta">'
            f'<span>{esc(display_name(condition))}</span>'
            f'<strong style="color: {"var(--error)" if index == 0 else "var(--text-secondary)"}">{percent:.1f}%</strong>'
            f'</div>'
            f'<div class="prob-track"><div class="prob-fill {fill_class}" style="width: {max(percent, 1):.1f}%"></div></div>'
            f'</div>'
        )
    return "\n".join(rows)


def render_results():
    result = st.session_state.latest_result
    if result is None:
        st.html(
            """
            <section id="results" class="ds-section">
                <div class="section-header">
                    <h2 class="ds-title" style="font-size: 46px;">Prediction Results</h2>
                    <p class="ds-copy">No analysis completed yet.</p>
                </div>
                <div class="result-card no-results-card">
                    <span class="material-symbols-outlined">analytics</span>
                    <h3>No Prediction Yet</h3>
                    <p>Upload a skin lesion image in the Prediction Demo and run the CNN model to display results here.</p>
                </div>
            </section>
            """
        )
        return

    predicted = result["predicted_class"]
    confidence = result["confidence"] * 100
    is_healthy_skin = display_name(predicted).lower() == "healthy skin"
    health_class = "healthy" if is_healthy_skin else ""
    confidence_icon = "check_circle" if is_healthy_skin else "warning"
    clinical_icon = "health_and_safety" if is_healthy_skin else "clinical_notes"
    info = condition_info(predicted)
    recommendation = info.get("recommendation", "Please consult a dermatologist for proper clinical diagnosis.")
    description = info.get("description", "")

    probability_rows = render_probability_rows(result["all_probabilities"])
    st.html(
        f"""
        <section id="results" class="ds-section">
            <div class="section-header">
                <h2 class="ds-title" style="font-size: 46px;">Prediction Results</h2>
                <p class="ds-copy">Analysis completed: {esc(st.session_state.latest_time)}</p>
            </div>
            <div class="results-grid">
                <div class="results-left">
                    <div class="result-card">
                        <h3>Image Preprocessing</h3>
                        <div class="preprocess-strip">
                            <div>
                                <div class="preprocess-img"><img src="{st.session_state.latest_original_uri}" alt="Original lesion image"></div>
                                <div class="img-caption">Original Image</div>
                            </div>
                            <div>
                                <div class="preprocess-img"><img src="{st.session_state.latest_processed_uri}" alt="Preprocessed lesion image"></div>
                                <div class="img-caption" style="color: var(--primary);">Preprocessed</div>
                            </div>
                        </div>
                    </div>
                    <div class="result-card context-card">
                        <h3>Condition Context</h3>
                        <p>{esc(description)}</p>
                    </div>
                </div>
                <div>
                    <div class="result-card">
                        <div class="result-top">
                            <div>
                                <h3>Top Prediction</h3>
                                <div class="result-title">Predicted: {esc(display_name(predicted))}</div>
                            </div>
                            <div class="confidence-badge {health_class}">
                                <span class="material-symbols-outlined">{confidence_icon}</span>
                                <span>Confidence: {confidence:.1f}%</span>
                            </div>
                        </div>
                        <div class="clinical-note {health_class}">
                            <span class="material-symbols-outlined">{clinical_icon}</span>
                            <div>
                                <h4>Clinical Note</h4>
                                <p>{esc(recommendation)}</p>
                            </div>
                        </div>
                        <div class="feature-grid">
                            <div class="feature-box"><div class="feature-label">Preprocessing</div><div class="feature-value">Hair removal</div></div>
                            <div class="feature-box"><div class="feature-label">Input</div><div class="feature-value">28x28 RGB</div></div>
                            <div class="feature-box"><div class="feature-label">Classes</div><div class="feature-value">8 mapped</div></div>
                            <div class="feature-box"><div class="feature-label">Output</div><div class="feature-value">Probability</div></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="result-card probability-card">
                <h3 style="font-size: 13px;">Probability Breakdown</h3>
                <div style="font-size: 24px; font-weight: 700; margin-bottom: 4px;">AI Confidence Distribution Across All Classes</div>
                <div style="margin-top: 28px;">
                    {probability_rows}
                </div>
            </div>
            <p class="ds-copy" style="margin-top: 24px;">
                <strong style="color: var(--error);">Medical disclaimer:</strong>
                This AI prediction is intended to support clinical review and is not a final medical diagnosis.
            </p>
        </section>
        """
    )

    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Export Report", use_container_width=True):
            st.info("Export Report is display-only in this V3 demo.")
    with button_cols[1]:
        if st.button("Share Case", use_container_width=True):
            st.info("Share Case is display-only in this V3 demo.")


def render_library():
    st.markdown(
        """
        <section id="library" class="ds-section">
            <div class="section-header">
                <h2 class="ds-title" style="font-size: 58px;">Detectable Skin Conditions</h2>
                <p class="ds-subtitle" style="margin-top: 12px;">The dashboard supports eight classes — seven lesion types plus healthy skin — and presents them as compact clinical summary cards.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    toolbar_left, toolbar_right = st.columns([0.4, 0.6])
    with toolbar_left:
        query = st.text_input("Search conditions", placeholder="Search conditions...", label_visibility="collapsed")
    with toolbar_right:
        st.markdown(
            """
            <div class="severity-note">
                <span>Filter by Severity:</span>
                <span class="severity-chip active">All</span>
                <span class="severity-chip">High</span>
                <span class="severity-chip">Medium</span>
                <span class="severity-chip">Low</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    query = query.strip().lower()
    cards = []
    for condition in DISPLAY_CLASS_ORDER:
        info = condition_info(condition)
        title = display_name(condition)
        description = short_description(condition)
        if query and query not in title.lower() and query not in description.lower():
            continue
        severity = info.get("severity", "Unknown")
        bucket = severity_bucket(severity)
        risk_text = "None" if bucket == "None" else f"{bucket} Risk"
        cards.append(
            f'<div class="library-card">'
            f'<div class="library-image">'
            f'<img src="{condition_image_src(condition)}" alt="{esc(title)}">'
            f'<div class="risk-pill risk-{severity_key(severity)}">{esc(risk_text)}</div>'
            f'</div>'
            f'<h3>{esc(title)}</h3>'
            f'<p>{esc(description)}</p>'
            f'<div class="details-link">View Details <span class="material-symbols-outlined" style="font-size:18px;vertical-align:middle;">arrow_forward</span></div>'
            f'</div>'
        )

    card_markup = "".join(cards) or '<p class="ds-copy">No conditions match the current search.</p>'
    st.html(f'<section class="ds-section" style="padding-top: 26px;"><div class="library-grid">{card_markup}</div></section>')


def render_footer():
    st.markdown(
        """
        <footer class="ds-footer">
            <div>
                <strong>DermScan AI</strong>
                <p>
                    © 2026 DermScan AI. Clinical decision support tool. Not a replacement for professional medical advice.
                </p>
            </div>
            <div class="footer-links">
                <span>Privacy Policy</span>
                <span>Terms of Service</span>
                <span>Medical Disclaimer</span>
                <span>Contact Support</span>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_css()
    init_state()
    # Apply currently selected theme and render a compact toggle control
    apply_theme(st.session_state.get("theme", "light"))
    cols = st.columns([1, 0.08])
    with cols[1]:
        prefer_dark = st.session_state.get("theme", "light") == "dark"
        toggled = st.checkbox("🌙", value=prefer_dark, key="theme_toggle", help="Toggle dark mode")
        new_theme = "dark" if toggled else "light"
        if st.session_state.get("theme") != new_theme:
            st.session_state["theme"] = new_theme
            apply_theme(new_theme)

    class_indices = load_class_indices()
    model, model_path, model_error = get_model()

    render_topbar()
    render_home()
    render_system()
    render_demo(model, model_path, model_error, class_indices)
    render_results()
    if st.session_state.get("jump_to_results"):
        jump_to_anchor("results")
        st.session_state.jump_to_results = False
    render_library()
    render_footer()


if __name__ == "__main__":
    main()
