"""
Plant Disease Detector – Interactive Web App (v4)
Run:  python app.py
Then open  http://localhost:7860  in your browser.
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
IMG_SIZE             = 224
MODEL_PATH           = "models/plant_disease_model.keras"
BEST_MODEL           = "models/best_model.keras"
CLASS_MAP_PATH       = "models/class_names.json"
CONFIDENCE_THRESHOLD = 70.0

def load_resources():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else BEST_MODEL
    if not os.path.exists(path):
        raise FileNotFoundError("No trained model found. Run  python train.py  first.")
    model = tf.keras.models.load_model(path)
    with open(CLASS_MAP_PATH) as f:
        raw = json.load(f)
    return model, {int(k): v for k, v in raw.items()}

try:
    MODEL, CLASS_MAP = load_resources()
    MODEL_READY = True
    LOAD_ERROR  = ""
except Exception as e:
    MODEL_READY = False
    LOAD_ERROR  = str(e)

# ──────────────────────────────────────────────
# Disease database
# ──────────────────────────────────────────────
DISEASE_DB = {
    "healthy": {
        "display":   "Healthy Plant",
        "emoji":     "✅",
        "severity":  "None",
        "sev_bg":    "#e8f5e9", "sev_txt": "#1b5e20",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#66bb6a",
        "bar_color": "#4caf50",
        "remedy":    "Your plant looks healthy! Continue regular watering, ensure adequate sunlight, and inspect leaves weekly for early stress signs.",
        "learn_url": "https://extension.umn.edu/plant-health",
        "prev_url":  "https://www.rhs.org.uk/prevention-protection/plant-health",
    },
    "Early_blight": {
        "display":   "Early Blight",
        "emoji":     "🍂",
        "severity":  "Moderate",
        "sev_bg":    "#fff3e0", "sev_txt": "#bf360c",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ffa726",
        "bar_color": "#ff9800",
        "remedy":    "Remove and destroy infected leaves immediately. Apply copper-based or chlorothalonil fungicide every 7–10 days. Avoid wetting foliage when watering.",
        "learn_url": "https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
        "prev_url":  "https://www.gardeningknowhow.com/edible/vegetables/tomato/tomato-early-blight-prevention.htm",
    },
    "Late_blight": {
        "display":   "Late Blight",
        "emoji":     "⚠️",
        "severity":  "High",
        "sev_bg":    "#ffebee", "sev_txt": "#b71c1c",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ef5350",
        "bar_color": "#f44336",
        "remedy":    "Act immediately — Late Blight spreads rapidly. Remove and bag all infected material. Apply mancozeb or a systemic fungicide. Destroy severely infected plants.",
        "learn_url": "https://extension.umn.edu/disease-management/late-blight",
        "prev_url":  "https://www.rhs.org.uk/disease/potato-blight",
    },
    "Leaf_Mold": {
        "display":   "Leaf Mold",
        "emoji":     "🟤",
        "severity":  "Moderate",
        "sev_bg":    "#efebe9", "sev_txt": "#4e342e",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#a1887f",
        "bar_color": "#ff9800",
        "remedy":    "Increase air circulation by pruning dense foliage. Apply copper-based or mancozeb fungicide. Keep humidity below 85% in greenhouses.",
        "learn_url": "https://www.canr.msu.edu/news/tomato_leaf_mold_in_high_tunnels_and_greenhouses",
        "prev_url":  "https://extension.umn.edu/disease-management/leaf-mold-tomato",
    },
    "Septoria_leaf_spot": {
        "display":   "Septoria Leaf Spot",
        "emoji":     "🟠",
        "severity":  "Moderate",
        "sev_bg":    "#fff3e0", "sev_txt": "#bf360c",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ff7043",
        "bar_color": "#ff9800",
        "remedy":    "Remove lower infected leaves. Apply chlorothalonil or copper fungicide at first sign. Avoid overhead irrigation. Rotate crops next season.",
        "learn_url": "https://extension.umn.edu/disease-management/septoria-leaf-spot-tomato",
        "prev_url":  "https://www.almanac.com/pest/septoria-leaf-spot",
    },
    "Spider_mites": {
        "display":   "Spider Mites",
        "emoji":     "🕷️",
        "severity":  "Moderate",
        "sev_bg":    "#f3e5f5", "sev_txt": "#6a1b9a",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ab47bc",
        "bar_color": "#ff9800",
        "remedy":    "Spray plants with a strong water jet to dislodge mites. Apply neem oil or insecticidal soap every 5–7 days. Increase ambient humidity.",
        "learn_url": "https://extension.umn.edu/yard-and-garden-insects/spider-mites",
        "prev_url":  "https://www.almanac.com/pest/spider-mites",
    },
    "Target_Spot": {
        "display":   "Target Spot",
        "emoji":     "🎯",
        "severity":  "Moderate",
        "sev_bg":    "#fff3e0", "sev_txt": "#bf360c",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ffa726",
        "bar_color": "#ff9800",
        "remedy":    "Apply azoxystrobin or pyraclostrobin fungicide at first signs. Remove plant debris after harvest. Improve drainage and air circulation.",
        "learn_url": "https://www.plantix.net/en/library/plant-diseases/200044/target-spot-of-tomato",
        "prev_url":  "https://plantvillage.psu.edu/topics/tomato/infos",
    },
    "Mosaic_virus": {
        "display":   "Tomato Mosaic Virus",
        "emoji":     "🟡",
        "severity":  "High",
        "sev_bg":    "#ffebee", "sev_txt": "#b71c1c",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ef5350",
        "bar_color": "#f44336",
        "remedy":    "No chemical cure exists. Remove and destroy infected plants immediately. Disinfect tools with bleach. Control aphid and whitefly populations.",
        "learn_url": "https://extension.umn.edu/disease-management/tomato-mosaic-virus",
        "prev_url":  "https://www.rhs.org.uk/disease/tomato-mosaic-virus",
    },
    "Yellow_Leaf_Curl_Virus": {
        "display":   "Yellow Leaf Curl Virus",
        "emoji":     "🌀",
        "severity":  "High",
        "sev_bg":    "#ffebee", "sev_txt": "#b71c1c",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ef5350",
        "bar_color": "#f44336",
        "remedy":    "Remove infected plants promptly. Apply imidacloprid or spinosad to control whiteflies. Use reflective mulches to deter insects.",
        "learn_url": "https://plantvillage.psu.edu/topics/tomato/infos",
        "prev_url":  "https://www.plantix.net/en/library/plant-diseases/200028/tomato-yellow-leaf-curl-virus",
    },
    "Bacterial_spot": {
        "display":   "Bacterial Spot",
        "emoji":     "🦠",
        "severity":  "Moderate",
        "sev_bg":    "#fff3e0", "sev_txt": "#bf360c",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#ff7043",
        "bar_color": "#ff9800",
        "remedy":    "Apply copper-based bactericide preventively. Avoid working with wet plants. Remove heavily infected leaves. Use disease-free seed next season.",
        "learn_url": "https://extension.umn.edu/disease-management/bacterial-spot-pepper-and-tomato",
        "prev_url":  "https://www.almanac.com/pest/bacterial-spot",
    },
}

def get_info(class_name: str) -> dict:
    for key, info in DISEASE_DB.items():
        if key.lower() in class_name.lower():
            return info
    display = class_name.replace("___", " – ").replace("_", " ")
    return {
        "display":   display,  "emoji": "🔬",
        "severity":  "Unknown",
        "sev_bg":    "#eceff1", "sev_txt": "#263238",
        "conf_bg":   "#f1f8e9", "conf_txt": "#33691e",
        "border":    "#90a4ae", "bar_color": "#78909c",
        "remedy":    "Consult a local agricultural extension officer or plant pathologist for accurate diagnosis.",
        "learn_url": "https://plantvillage.psu.edu/",
        "prev_url":  "https://www.fao.org/plant-health-2020/en/",
    }

# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict(pil_image):
    if pil_image is None:
        return build_html("no_image"), ""
    if not MODEL_READY:
        return build_html("model_error", error=LOAD_ERROR), ""

    img       = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr       = np.array(img, dtype=np.float32) / 255.0
    preds     = MODEL.predict(np.expand_dims(arr, 0), verbose=0)[0]
    top_idx   = int(np.argmax(preds))
    top_conf  = float(preds[top_idx]) * 100
    top_class = CLASS_MAP[top_idx]

    if top_conf < CONFIDENCE_THRESHOLD:
        return build_html("low_confidence", conf=top_conf), ""

    return build_html("result", info=get_info(top_class), conf=top_conf, class_name=top_class), ""

# ──────────────────────────────────────────────
# HTML cards  — pure white backgrounds, dark text
# ──────────────────────────────────────────────
F = "font-family:'Poppins','Segoe UI',Arial,sans-serif;"

def build_html(mode, info=None, conf=0, class_name="", error=""):

    if mode == "no_image":
        return f"""
        <div class="pd-wrap" style="text-align:center;padding:52px 20px;background:#f9fdf5;
                    border-radius:14px;border:2px dashed #a5d6a7;">
            <div style="font-size:54px;margin-bottom:14px;">📷</div>
            <p style="font-size:16px;font-weight:700;color:#2e7d32;margin:0 0 6px;">
                Upload a leaf photo to begin
            </p>
            <p style="font-size:13px;color:#555;margin:0;">
                Supports JPG &amp; PNG · Bell Pepper, Tomato, Potato
            </p>
        </div>"""

    if mode == "model_error":
        return f"""
        <div class="pd-wrap" style="background:#fff8f8;border:1.5px solid #ef9a9a;border-radius:12px;
                    padding:22px;color:#b71c1c;">
            <p style="font-weight:700;font-size:15px;margin:0 0 10px;">⚠️ Model Not Loaded</p>
            <p style="color:#444;font-size:13px;margin:0 0 10px;">{error}</p>
            <p style="color:#555;font-size:13px;margin:0;">
                Run <code style="background:#fce4ec;padding:2px 8px;border-radius:4px;
                color:#c62828;">python train.py</code> first, then restart the app.
            </p>
        </div>"""

    if mode == "low_confidence":
        return f"""
        <div class="pd-wrap" style="background:#fffde7;border:2px solid #f9a825;border-radius:14px;padding:28px;">
            <div style="text-align:center;margin-bottom:18px;">
                <div style="font-size:50px;margin-bottom:10px;">🔁</div>
                <h2 style="color:#e65100;margin:0 0 8px;font-size:19px;font-weight:700;">
                    Photo Not Clear Enough
                </h2>
                <p style="color:#333;font-size:14px;line-height:1.6;margin:0;">
                    Model confidence was only
                    <strong style="color:#bf360c;">{conf:.1f}%</strong>
                    — needs at least <strong>70%</strong> for a reliable result.
                </p>
            </div>
            <div style="background:#fff9c4;border:1px solid #f9a825;border-radius:10px;padding:16px;">
                <p style="margin:0 0 10px;color:#e65100;font-weight:700;font-size:13px;
                          text-transform:uppercase;letter-spacing:0.5px;">
                    📸 Tips for a Better Photo
                </p>
                <ul style="margin:0;padding-left:20px;">
                    <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Use <strong style="color:#1b5e20 !important;font-weight:700;">one leaf only</strong> — no background clutter</li>
                    <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Shoot in <strong style="color:#1b5e20 !important;font-weight:700;">natural daylight</strong>, not under a bulb or flash</li>
                    <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Hold camera <strong style="color:#1b5e20 !important;font-weight:700;">20–30 cm</strong> above the leaf, keep it steady</li>
                    <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Ensure the <strong style="color:#1b5e20 !important;font-weight:700;">leaf fills most of the frame</strong></li>
                    <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Use a <strong style="color:#1b5e20 !important;font-weight:700;">plain white background</strong> behind the leaf</li>
                    <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Wipe your <strong style="color:#1b5e20 !important;font-weight:700;">camera lens</strong> before shooting</li>
                </ul>
            </div>
        </div>"""

    # ── Full result card ──
    return f"""
    <div class="pd-wrap" style="background:#ffffff;border:2px solid {info['border']};
                border-radius:16px;padding:24px;box-shadow:0 2px 10px rgba(0,0,0,0.07);">

        <!-- Title row -->
        <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:16px;">
            <span style="font-size:40px;line-height:1;flex-shrink:0;">{info['emoji']}</span>
            <div>
                <h2 style="margin:0 0 10px;font-size:21px;font-weight:800;color:#1a1a1a;">
                    {info['display']}
                </h2>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <span style="background:{info['sev_bg']};color:{info['sev_txt']};
                                 padding:4px 14px;border-radius:20px;font-size:12px;
                                 font-weight:700;border:1px solid {info['border']};">
                        Severity: {info['severity']}
                    </span>
                    <span style="background:{info['conf_bg']};color:{info['conf_txt']};
                                 padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;
                                 border:1px solid #a5d6a7;">
                        Confidence: {conf:.1f}%
                    </span>
                </div>
            </div>
        </div>

        <!-- Confidence bar -->
        <div style="background:#eeeeee;border-radius:8px;height:8px;margin-bottom:20px;overflow:hidden;">
            <div style="width:{conf:.0f}%;height:100%;background:{info['bar_color']};
                        border-radius:8px;"></div>
        </div>

        <!-- Recommendation -->
        <div style="background:#f1f8f1;border-left:4px solid #43a047;
                    border-radius:0 10px 10px 0;padding:14px 16px;margin-bottom:16px;">
            <p style="margin:0 0 6px;font-size:11px;font-weight:700;
                      color:#2e7d32 !important;letter-spacing:0.8px;
                      text-transform:uppercase;font-family:Poppins,Arial,sans-serif;">
                💊 Recommendation
            </p>
            <p style="margin:0;font-size:14px;color:#1a1a1a !important;font-weight:500;
                      line-height:1.8;font-family:Poppins,Arial,sans-serif;">
                {info['remedy']}
            </p>
        </div>

        <!-- What to do next -->
        <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:10px;
                    padding:14px 16px;margin-bottom:18px;">
            <p style="margin:0 0 8px;font-size:11px;font-weight:700;
                      color:#2e7d32 !important;letter-spacing:0.8px;
                      text-transform:uppercase;font-family:Poppins,Arial,sans-serif;">
                📋 What To Do Next
            </p>
            <ol style="margin:0;padding-left:18px;">
                <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Follow the recommendation above — act quickly for High severity.</li>
                <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Click <strong style="color:#1b5e20 !important;font-weight:700;">Learn More</strong> to understand this disease in depth.</li>
                <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Click <strong style="color:#1b5e20 !important;font-weight:700;">Prevention Guide</strong> to protect your healthy plants.</li>
                <li style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">When in doubt, consult a local agronomist with this diagnosis.</li>
            </ol>
        </div>

        <!-- Links -->
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <a href="{info['learn_url']}" target="_blank"
               style="flex:1;min-width:130px;display:block;background:#2e7d32;color:#ffffff;
                      text-decoration:none;padding:11px 14px;border-radius:10px;
                      text-align:center;font-size:13px;font-weight:700;">
                📖 Learn More
            </a>
            <a href="{info['prev_url']}" target="_blank"
               style="flex:1;min-width:130px;display:block;background:#f1f8e9;color:#33691e;
                      text-decoration:none;padding:11px 14px;border-radius:10px;
                      text-align:center;font-size:13px;font-weight:700;
                      border:1.5px solid #aed581;">
                🛡️ Prevention Guide
            </a>
        </div>
    </div>"""


# ──────────────────────────────────────────────
# CSS  — clean light theme, no dark backgrounds
# ──────────────────────────────────────────────
CUSTOM_CSS = """
body, .gradio-container {
    background: #f0f7f0 !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}
.app-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #43a047 100%);
    border-radius: 14px;
    padding: 26px 32px;
    margin-bottom: 20px;
    text-align: center;
}
.app-header h1 { font-size: 1.9rem; font-weight: 800; color: #ffffff; margin: 0 0 5px; }
.app-header p  { color: #c8e6c9; font-size: 13px; margin: 0; }

.left-panel, .right-panel {
    background: #ffffff;
    border: 1px solid #c8e6c9;
    border-radius: 14px;
    padding: 20px;
}
.right-panel { min-height: 340px; }

.ic-box {
    background: #f9fdf5;
    border: 1px solid #c5e1a5;
    border-radius: 10px;
    padding: 15px 18px;
    margin-bottom: 16px;
}
.ic-title {
    font-size: 12px; font-weight: 700; color: #2e7d32;
    text-transform: uppercase; letter-spacing: 0.6px; margin: 0 0 10px;
}
.ic-box ol { margin: 0; padding-left: 18px; color: #222; font-size: 13px; line-height: 2.1; }
.ic-box ol strong { color: #1b5e20; }

.diagnose-btn {
    width: 100%;
    background: #2e7d32 !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 13px !important;
    margin-top: 10px;
    transition: background 0.2s, box-shadow 0.2s;
}
.diagnose-btn:hover {
    background: #388e3c !important;
    box-shadow: 0 4px 14px rgba(46,125,50,0.35) !important;
}

.app-footer {
    text-align: center; color: #777; font-size: 12px;
    padding: 16px 0 6px; border-top: 1px solid #ddd; margin-top: 18px;
}
"""


# ──────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────
def build_ui():
    import gradio as gr

    MINIMAL_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Poppins:wght@400;500;600;700&display=swap');
    body, .gradio-container, .main, .wrap { background: #f4f9f4 !important; }
    """

    # Inline <style> tag — always wins over Gradio overrides
    FORCE_STYLE = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Poppins:wght@400;500;600;700&display=swap');
    .pd-wrap * {
        font-family: 'Poppins', 'Segoe UI', Arial, sans-serif !important;
        box-sizing: border-box;
    }
    .pd-wrap p, .pd-wrap li, .pd-wrap span, .pd-wrap a, .pd-wrap ol, .pd-wrap ul {
        color: #1a1a1a !important;
    }
    .pd-wrap strong { color: #1b5e20 !important; font-weight: 700 !important; }
    .pd-wrap h1, .pd-wrap h2 { color: #ffffff !important; }
    .pd-header h1  { font-family: 'Merriweather', Georgia, serif !important; color: #ffffff !important; }
    .pd-header p   { color: #c8e6c9 !important; }
    .pd-result-placeholder p.title { color: #2e7d32 !important; font-weight: 700 !important; }
    .pd-result-placeholder p.sub   { color: #777 !important; }
    .pd-ic-title { color: #e65100 !important; font-weight: 700 !important; }
    .pd-footer   { color: #777 !important; }
    </style>
    """

    with gr.Blocks(title="🌿 Plant Disease Detector", css=MINIMAL_CSS) as demo:

        # Force styles via inline <style> block
        gr.HTML(FORCE_STYLE)

        # ── HEADER ──
        gr.HTML("""
        <div class="pd-wrap pd-header" style="background:linear-gradient(135deg,#1a5c1e 0%,#2e7d32 50%,#4caf50 100%);
                    border-radius:18px;padding:32px 36px;margin-bottom:22px;
                    text-align:center;box-shadow:0 6px 24px rgba(46,125,50,0.25);">
            <div style="font-size:48px;margin-bottom:8px;">🌿</div>
            <h1 style="font-size:2rem;font-weight:900;margin:0 0 8px;
                       text-shadow:0 2px 8px rgba(0,0,0,0.2);">
                Plant Disease Detector
            </h1>
            <p style="font-size:14px;margin:0;font-weight:500;">
                🫑 Bell Pepper &nbsp;·&nbsp; 🍅 Tomato &nbsp;·&nbsp; 🥔 Potato
                &nbsp;|&nbsp; Powered by PlantVillage AI
            </p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── LEFT COLUMN ──
            with gr.Column(scale=1):

                gr.HTML("""
                <div style="background:linear-gradient(135deg,#fff9c4,#fff176);
                            border:2px solid #f9a825;border-radius:14px;
                            padding:18px 20px;margin-bottom:16px;
                            box-shadow:0 2px 10px rgba(249,168,37,0.2);">
                    <p style="font-family:Poppins,Arial,sans-serif;font-size:13px;font-weight:700;
                               color:#e65100 !important;text-transform:uppercase;
                               letter-spacing:0.8px;margin:0 0 12px;">
                        📷 &nbsp;How to Photograph Your Leaf
                    </p>
                    <ol style="margin:0;padding-left:20px;">
                        <li style="font-family:Poppins,Arial,sans-serif;font-size:13.5px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Pick <strong style="color:#1b5e20 !important;font-weight:700;">one single leaf</strong> — diseased or healthy.</li>
                        <li style="font-family:Poppins,Arial,sans-serif;font-size:13.5px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Lay it flat on a <strong style="color:#1b5e20 !important;font-weight:700;">plain white surface</strong>.</li>
                        <li style="font-family:Poppins,Arial,sans-serif;font-size:13.5px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Shoot in <strong style="color:#1b5e20 !important;font-weight:700;">natural daylight</strong> — avoid flash or bulb light.</li>
                        <li style="font-family:Poppins,Arial,sans-serif;font-size:13.5px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Hold camera <strong style="color:#1b5e20 !important;font-weight:700;">20–30 cm</strong> above the leaf, keep it steady.</li>
                        <li style="font-family:Poppins,Arial,sans-serif;font-size:13.5px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Ensure leaf is <strong style="color:#1b5e20 !important;font-weight:700;">sharp, centred and fills the frame</strong>.</li>
                        <li style="font-family:Poppins,Arial,sans-serif;font-size:13.5px;font-weight:500;color:#1a1a1a !important;line-height:2.3;margin-bottom:2px;">Wipe your <strong style="color:#1b5e20 !important;font-weight:700;">camera lens</strong> clean before shooting.</li>
                    </ol>
                </div>
                """)

                image_input = gr.Image(
                    type="pil",
                    label="📂  Upload or drag your leaf photo here",
                    height=290,
                )

                submit_btn = gr.Button("🔍  Diagnose Leaf", variant="primary")

            # ── RIGHT COLUMN ──
            with gr.Column(scale=1):
                result_html = gr.HTML("""
                <div class="pd-wrap" style="text-align:center;padding:64px 24px;
                            background:#ffffff;border-radius:16px;
                            border:2px dashed #a5d6a7;
                            box-shadow:0 2px 12px rgba(0,0,0,0.06);">
                    <div style="font-size:58px;margin-bottom:16px;">🌱</div>
                    <p class="title" style="font-size:16px;font-weight:700;
                                            color:#2e7d32 !important;margin:0 0 8px;">
                        Awaiting Diagnosis
                    </p>
                    <p class="sub" style="font-size:13px;color:#777 !important;
                                          margin:0;line-height:1.6;">
                        Upload a clear leaf photo on the left<br>
                        then click <strong style="color:#2e7d32 !important;">Diagnose Leaf</strong>
                    </p>
                </div>
                """)
                hidden_out = gr.Textbox(visible=False)

        # ── FOOTER ──
        gr.HTML("""
        <div class="pd-wrap pd-footer" style="text-align:center;font-size:12px;
                    padding:18px 0 8px;border-top:1px solid #dcedc8;margin-top:20px;">
            🌾 Trained on <strong style="color:#555 !important;">PlantVillage</strong> dataset &nbsp;·&nbsp;
            Bell Pepper, Tomato &amp; Potato &nbsp;·&nbsp;
            <em>Always confirm with a certified agronomist.</em>
        </div>
        """)

        submit_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[result_html, hidden_out],
        )

    return demo


if __name__ == "__main__":
    try:
        import gradio
    except ImportError:
        print("Gradio not installed. Run:  pip install gradio")
        exit(1)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # ── Paste your ngrok auth token here ──────
    NGROK_TOKEN = "PASTE_YOUR_TOKEN_HERE"
    # ──────────────────────────────────────────

    PORT = 7860

    # Start ngrok tunnel
    public_url = None
    try:
        from pyngrok import ngrok, conf
        if NGROK_TOKEN and NGROK_TOKEN != "PASTE_YOUR_TOKEN_HERE":
            conf.get_default().auth_token = NGROK_TOKEN
            # Kill all existing tunnels before starting a new one
            for t in ngrok.get_tunnels():
                ngrok.disconnect(t.public_url)
            tunnel     = ngrok.connect(PORT, "http")
            public_url = tunnel.public_url
            print("\n" + "═" * 55)
            print(f"  🌍  Public URL  →  {public_url}")
            print(f"  💻  Local  URL  →  http://localhost:{PORT}")
            print("═" * 55 + "\n")
        else:
            print("\n⚠️  No ngrok token set — running locally only.")
            print(f"   Local URL → http://localhost:{PORT}\n")
    except ImportError:
        print("\n⚠️  pyngrok not installed — running locally only.")
        print(f"   Local URL → http://localhost:{PORT}")
        print("   To get a public link: pip install pyngrok\n")

    demo = build_ui()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        quiet=True,
    )
