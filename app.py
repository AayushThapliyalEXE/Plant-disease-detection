"""
Plant Disease Detector – Interactive Web App (v3)
Powered by Gradio. Run:  python app.py
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
        "display":  "Healthy Plant",
        "emoji":    "✅",
        "severity": "None",
        "badge":    "#e8f5e9",
        "badge_txt":"#1b5e20",
        "border":   "#66bb6a",
        "remedy":   "Your plant looks healthy! Keep up regular watering, ensure adequate sunlight, and check leaves weekly for any early stress signs.",
        "learn_url":"https://extension.umn.edu/plant-health",
        "prev_url": "https://www.rhs.org.uk/prevention-protection/plant-health",
    },
    "Early_blight": {
        "display":  "Early Blight",
        "emoji":    "🍂",
        "severity": "Moderate",
        "badge":    "#fff3e0",
        "badge_txt":"#e65100",
        "border":   "#ffa726",
        "remedy":   "Remove and destroy infected leaves immediately. Apply copper-based or chlorothalonil fungicide every 7–10 days. Avoid wetting foliage when watering.",
        "learn_url":"https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
        "prev_url": "https://www.gardeningknowhow.com/edible/vegetables/tomato/tomato-early-blight-prevention.htm",
    },
    "Late_blight": {
        "display":  "Late Blight",
        "emoji":    "⚠️",
        "severity": "High",
        "badge":    "#ffebee",
        "badge_txt":"#b71c1c",
        "border":   "#ef5350",
        "remedy":   "Act immediately — Late Blight spreads rapidly. Remove and bag all infected material. Apply mancozeb or a systemic fungicide. Destroy severely infected plants.",
        "learn_url":"https://extension.umn.edu/disease-management/late-blight",
        "prev_url": "https://www.rhs.org.uk/disease/potato-blight",
    },
    "Leaf_Mold": {
        "display":  "Leaf Mold",
        "emoji":    "🟤",
        "severity": "Moderate",
        "badge":    "#efebe9",
        "badge_txt":"#4e342e",
        "border":   "#a1887f",
        "remedy":   "Increase air circulation by pruning dense foliage. Apply copper-based or mancozeb fungicide. Keep humidity below 85% in greenhouses.",
        "learn_url":"https://www.canr.msu.edu/news/tomato_leaf_mold_in_high_tunnels_and_greenhouses",
        "prev_url": "https://extension.umn.edu/disease-management/leaf-mold-tomato",
    },
    "Septoria_leaf_spot": {
        "display":  "Septoria Leaf Spot",
        "emoji":    "🟠",
        "severity": "Moderate",
        "badge":    "#fff3e0",
        "badge_txt":"#bf360c",
        "border":   "#ff7043",
        "remedy":   "Remove lower infected leaves. Apply chlorothalonil or copper fungicide at first sign. Avoid overhead irrigation. Rotate crops next season.",
        "learn_url":"https://extension.umn.edu/disease-management/septoria-leaf-spot-tomato",
        "prev_url": "https://www.almanac.com/pest/septoria-leaf-spot",
    },
    "Spider_mites": {
        "display":  "Spider Mites",
        "emoji":    "🕷️",
        "severity": "Moderate",
        "badge":    "#f3e5f5",
        "badge_txt":"#4a148c",
        "border":   "#ab47bc",
        "remedy":   "Spray plants with a strong water jet to dislodge mites. Apply neem oil or insecticidal soap every 5–7 days. Increase ambient humidity.",
        "learn_url":"https://extension.umn.edu/yard-and-garden-insects/spider-mites",
        "prev_url": "https://www.almanac.com/pest/spider-mites",
    },
    "Target_Spot": {
        "display":  "Target Spot",
        "emoji":    "🎯",
        "severity": "Moderate",
        "badge":    "#fff3e0",
        "badge_txt":"#e65100",
        "border":   "#ffa726",
        "remedy":   "Apply azoxystrobin or pyraclostrobin fungicide at first signs. Remove plant debris after harvest. Improve drainage and air circulation.",
        "learn_url":"https://www.plantix.net/en/library/plant-diseases/200044/target-spot-of-tomato",
        "prev_url": "https://plantvillage.psu.edu/topics/tomato/infos",
    },
    "Mosaic_virus": {
        "display":  "Tomato Mosaic Virus",
        "emoji":    "🟡",
        "severity": "High",
        "badge":    "#ffebee",
        "badge_txt":"#b71c1c",
        "border":   "#ef5350",
        "remedy":   "No chemical cure exists. Remove and destroy infected plants immediately. Disinfect tools with bleach solution. Control aphid and whitefly populations.",
        "learn_url":"https://extension.umn.edu/disease-management/tomato-mosaic-virus",
        "prev_url": "https://www.rhs.org.uk/disease/tomato-mosaic-virus",
    },
    "Yellow_Leaf_Curl_Virus": {
        "display":  "Yellow Leaf Curl Virus",
        "emoji":    "🌀",
        "severity": "High",
        "badge":    "#ffebee",
        "badge_txt":"#b71c1c",
        "border":   "#ef5350",
        "remedy":   "Remove infected plants promptly. Apply imidacloprid or spinosad to control whiteflies. Use reflective mulches to deter insects from spreading the virus.",
        "learn_url":"https://plantvillage.psu.edu/topics/tomato/infos",
        "prev_url": "https://www.plantix.net/en/library/plant-diseases/200028/tomato-yellow-leaf-curl-virus",
    },
    "Bacterial_spot": {
        "display":  "Bacterial Spot",
        "emoji":    "🦠",
        "severity": "Moderate",
        "badge":    "#fff3e0",
        "badge_txt":"#bf360c",
        "border":   "#ff7043",
        "remedy":   "Apply copper-based bactericide preventively. Avoid working with wet plants. Remove heavily infected leaves. Use disease-free seed next season.",
        "learn_url":"https://extension.umn.edu/disease-management/bacterial-spot-pepper-and-tomato",
        "prev_url": "https://www.almanac.com/pest/bacterial-spot",
    },
}

def get_info(class_name: str) -> dict:
    for key, info in DISEASE_DB.items():
        if key.lower() in class_name.lower():
            return info
    display = class_name.replace("___", " – ").replace("_", " ")
    return {
        "display":  display, "emoji": "🔬",
        "severity": "Unknown",
        "badge":    "#eceff1", "badge_txt": "#37474f", "border": "#90a4ae",
        "remedy":   "Consult a local agricultural extension officer or plant pathologist for accurate diagnosis.",
        "learn_url":"https://plantvillage.psu.edu/",
        "prev_url": "https://www.fao.org/plant-health-2020/en/",
    }

# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict(pil_image):
    if pil_image is None:
        return build_html("no_image"), ""
    if not MODEL_READY:
        return build_html("model_error", error=LOAD_ERROR), ""

    img  = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr  = np.array(img, dtype=np.float32) / 255.0
    x    = np.expand_dims(arr, 0)
    preds     = MODEL.predict(x, verbose=0)[0]
    top_idx   = int(np.argmax(preds))
    top_conf  = float(preds[top_idx]) * 100
    top_class = CLASS_MAP[top_idx]

    if top_conf < CONFIDENCE_THRESHOLD:
        return build_html("low_confidence", conf=top_conf), ""

    info = get_info(top_class)
    return build_html("result", info=info, conf=top_conf, class_name=top_class), ""

# ──────────────────────────────────────────────
# HTML result cards  (light theme, high contrast)
# ──────────────────────────────────────────────
def build_html(mode, info=None, conf=0, class_name="", error=""):

    BASE = "font-family:'Segoe UI',Arial,sans-serif;"

    if mode == "no_image":
        return f"""
        <div style="{BASE}text-align:center;padding:48px 20px;color:#555;background:#f9fdf9;
                    border-radius:14px;border:2px dashed #c8e6c9;">
            <div style="font-size:52px;margin-bottom:12px;">📷</div>
            <p style="font-size:16px;color:#2e7d32;font-weight:600;margin:0 0 6px;">
                Upload a leaf photo to begin
            </p>
            <p style="font-size:13px;color:#777;margin:0;">
                Supports JPG, PNG · Bell Pepper, Tomato, Potato
            </p>
        </div>"""

    if mode == "model_error":
        return f"""
        <div style="{BASE}background:#fff8f8;border:1px solid #ef9a9a;border-radius:12px;
                    padding:20px;color:#c62828;">
            <strong>⚠️ Model not loaded</strong><br><br>
            {error}<br><br>
            Run <code style="background:#fce4ec;padding:2px 6px;border-radius:4px;">
            python train.py</code> first, then restart the app.
        </div>"""

    if mode == "low_confidence":
        return f"""
        <div style="{BASE}background:#fffde7;border:2px solid #fbc02d;border-radius:14px;
                    padding:28px;text-align:center;">
            <div style="font-size:48px;margin-bottom:10px;">🔁</div>
            <h2 style="color:#f57f17;margin:0 0 8px;font-size:20px;">Photo Not Clear Enough</h2>
            <p style="color:#555;font-size:14px;line-height:1.6;margin:0 0 20px;">
                Confidence was only <strong style="color:#e65100;">{conf:.1f}%</strong>
                — the model needs at least 70% to give a reliable result.
            </p>
            <div style="background:#fff9c4;border:1px solid #f9a825;border-radius:10px;
                        padding:16px;text-align:left;">
                <p style="margin:0 0 10px;color:#f57f17;font-weight:700;font-size:14px;">
                    📸 Tips for a better photo
                </p>
                <ul style="margin:0;padding-left:20px;color:#444;font-size:13px;line-height:2.1;">
                    <li>Use <strong>one leaf only</strong> — remove background clutter</li>
                    <li>Shoot in <strong>natural daylight</strong>, not under a bulb or flash</li>
                    <li>Hold camera <strong>20–30 cm</strong> above the leaf, keep it steady</li>
                    <li>Ensure the <strong>leaf fills most of the frame</strong></li>
                    <li>Wipe your <strong>camera lens</strong> before shooting</li>
                    <li>Use a <strong>plain white or light background</strong> behind the leaf</li>
                </ul>
            </div>
        </div>"""

    # ── Full result ──
    bar_color = "#4caf50" if info["severity"] == "None" else \
                "#ff9800" if info["severity"] == "Moderate" else "#f44336"

    return f"""
    <div style="{BASE}background:#ffffff;border:2px solid {info['border']};
                border-radius:16px;padding:24px 26px;box-shadow:0 2px 12px rgba(0,0,0,0.08);">

        <!-- ── Title row ── -->
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
            <span style="font-size:42px;line-height:1;">{info['emoji']}</span>
            <div style="flex:1;">
                <h2 style="margin:0 0 8px;font-size:20px;color:#1a1a1a;font-weight:700;">
                    {info['display']}
                </h2>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <span style="background:{info['badge']};color:{info['badge_txt']};
                                 padding:3px 14px;border-radius:20px;font-size:12px;
                                 font-weight:700;letter-spacing:0.3px;">
                        Severity: {info['severity']}
                    </span>
                    <span style="background:#e8f5e9;color:#1b5e20;
                                 padding:3px 14px;border-radius:20px;font-size:12px;font-weight:700;">
                        Confidence: {conf:.1f}%
                    </span>
                </div>
            </div>
        </div>

        <!-- ── Confidence bar ── -->
        <div style="background:#f0f0f0;border-radius:8px;height:7px;margin-bottom:22px;overflow:hidden;">
            <div style="width:{conf:.0f}%;height:100%;background:{bar_color};
                        border-radius:8px;transition:width 0.4s ease;"></div>
        </div>

        <!-- ── Recommendation ── -->
        <div style="background:#f1f8f1;border-left:4px solid #43a047;
                    border-radius:0 10px 10px 0;padding:14px 16px;margin-bottom:18px;">
            <p style="margin:0 0 5px;font-size:11px;font-weight:700;color:#2e7d32;
                      letter-spacing:0.8px;text-transform:uppercase;">💊 Recommendation</p>
            <p style="margin:0;font-size:14px;color:#2d2d2d;line-height:1.7;">{info['remedy']}</p>
        </div>

        <!-- ── How to use ── -->
        <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:10px;
                    padding:14px 16px;margin-bottom:18px;">
            <p style="margin:0 0 8px;font-size:11px;font-weight:700;color:#388e3c;
                      letter-spacing:0.8px;text-transform:uppercase;">📋 What to do next</p>
            <ol style="margin:0;padding-left:18px;color:#333;font-size:13px;line-height:2.1;">
                <li>Follow the recommendation above — act quickly for High severity.</li>
                <li>Click <strong>Learn More</strong> to understand this disease in depth.</li>
                <li>Click <strong>Prevention Guide</strong> to protect your healthy plants.</li>
                <li>When in doubt, consult a local agronomist with this diagnosis.</li>
            </ol>
        </div>

        <!-- ── Links ── -->
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <a href="{info['learn_url']}" target="_blank"
               style="flex:1;min-width:140px;display:block;background:#2e7d32;color:#ffffff;
                      text-decoration:none;padding:11px 16px;border-radius:10px;
                      text-align:center;font-size:13px;font-weight:700;">
                📖 Learn More
            </a>
            <a href="{info['prev_url']}" target="_blank"
               style="flex:1;min-width:140px;display:block;background:#f1f8e9;color:#33691e;
                      text-decoration:none;padding:11px 16px;border-radius:10px;
                      text-align:center;font-size:13px;font-weight:700;
                      border:1.5px solid #aed581;">
                🛡️ Prevention Guide
            </a>
        </div>

    </div>"""


# ──────────────────────────────────────────────
# CSS  – clean light agriculture theme
# ──────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ── */
body, .gradio-container {
    background: #f2f7f2 !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px;
    letter-spacing: 0.5px;
}
.app-header p {
    color: #c8e6c9;
    font-size: 14px;
    margin: 0;
}

/* ── Left panel ── */
.left-panel {
    background: #ffffff;
    border: 1px solid #dcedc8;
    border-radius: 14px;
    padding: 20px;
}

/* ── Right panel ── */
.right-panel {
    background: #ffffff;
    border: 1px solid #dcedc8;
    border-radius: 14px;
    padding: 20px;
    min-height: 320px;
}

/* ── Instruction card ── */
.instruction-card {
    background: #f9fdf5;
    border: 1px solid #c5e1a5;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 16px;
}
.instruction-card .ic-title {
    font-size: 13px;
    font-weight: 700;
    color: #2e7d32;
    margin: 0 0 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.instruction-card ol {
    margin: 0;
    padding-left: 18px;
    color: #333;
    font-size: 13px;
    line-height: 2;
}
.instruction-card ol strong { color: #1b5e20; }

/* ── Diagnose button ── */
.diagnose-btn {
    width: 100%;
    background: #2e7d32 !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 13px !important;
    cursor: pointer;
    margin-top: 10px;
    transition: background 0.2s, transform 0.15s;
}
.diagnose-btn:hover {
    background: #388e3c !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(46,125,50,0.3) !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: #78909c;
    font-size: 12px;
    padding: 18px 0 8px;
    border-top: 1px solid #ddd;
    margin-top: 20px;
}

/* ── Gradio image uploader overrides ── */
.gradio-image { border-radius: 10px !important; border: 1.5px solid #c8e6c9 !important; }
label span { color: #333 !important; font-weight: 600 !important; font-size: 13px !important; }
"""

# ──────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────
def build_ui():
    import gradio as gr

    with gr.Blocks(css=CUSTOM_CSS, title="🌿 Plant Disease Detector") as demo:

        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🌿 Plant Disease Detector</h1>
            <p>AI-powered leaf diagnosis for Bell Pepper · Tomato · Potato &nbsp;|&nbsp; PlantVillage Dataset</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── LEFT: Instructions + Upload ──
            with gr.Column(scale=1, elem_classes=["left-panel"]):

                gr.HTML("""
                <div class="instruction-card">
                    <p class="ic-title">📷 How to photograph your leaf</p>
                    <ol>
                        <li>Pick <strong>one leaf</strong> — diseased or healthy.</li>
                        <li>Lay it flat on a <strong>plain white surface</strong>.</li>
                        <li>Shoot in <strong>natural daylight</strong> — no flash.</li>
                        <li>Hold camera <strong>20–30 cm</strong> above the leaf.</li>
                        <li>Keep the leaf <strong>sharp and centred</strong> in frame.</li>
                        <li>Wipe your <strong>camera lens</strong> before shooting.</li>
                    </ol>
                </div>
                """)

                image_input = gr.Image(
                    type="pil",
                    label="Upload or drag your leaf photo here",
                    height=280,
                )

                submit_btn = gr.Button(
                    "🔍  Diagnose Leaf",
                    elem_classes=["diagnose-btn"],
                )

            # ── RIGHT: Results ──
            with gr.Column(scale=1, elem_classes=["right-panel"]):
                result_html = gr.HTML("""
                <div style="text-align:center;padding:56px 20px;color:#888;
                            font-family:'Segoe UI',Arial,sans-serif;">
                    <div style="font-size:52px;margin-bottom:14px;">🌱</div>
                    <p style="font-size:15px;color:#444;font-weight:600;margin:0 0 6px;">
                        Results will appear here
                    </p>
                    <p style="font-size:13px;color:#888;margin:0;">
                        Upload a photo and click <strong>Diagnose Leaf</strong>
                    </p>
                </div>
                """)
                hidden_out = gr.Textbox(visible=False)

        # Footer
        gr.HTML("""
        <div class="app-footer">
            Trained on PlantVillage &nbsp;·&nbsp;
            Detects diseases in Bell Pepper, Tomato &amp; Potato &nbsp;·&nbsp;
            Results are indicative — always confirm with a certified agronomist.
        </div>
        """)

        submit_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[result_html, hidden_out],
        )

    return demo


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import gradio
    except ImportError:
        print("Gradio not installed. Run:  pip install gradio")
        exit(1)

    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7877)