"""
Plant Disease Detector – Prediction Script
Usage:
    python predict.py path/to/leaf_image.jpg
    python predict.py path/to/folder_of_images/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from PIL import Image
import tensorflow as tf

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_SIZE      = 224
MODEL_PATH    = "models/plant_disease_model.keras"
BEST_MODEL    = "models/best_model.keras"
CLASS_MAP_PATH = "models/class_names.json"

# Disease info dictionary (readable names + remedies)
DISEASE_INFO = {
    "healthy": {
        "display": "Healthy Plant 🌿",
        "severity": "None",
        "remedy": "Your plant looks healthy! Keep up good watering and sunlight routines.",
    },
    "Early_blight": {
        "display": "Early Blight",
        "severity": "Moderate",
        "remedy": "Remove infected leaves. Apply copper-based fungicide. Avoid overhead watering.",
    },
    "Late_blight": {
        "display": "Late Blight",
        "severity": "High",
        "remedy": "Remove and destroy infected plant parts. Apply chlorothalonil or mancozeb fungicide urgently.",
    },
    "Leaf_Mold": {
        "display": "Leaf Mold",
        "severity": "Moderate",
        "remedy": "Improve air circulation. Apply fungicide containing copper or mancozeb.",
    },
    "Septoria_leaf_spot": {
        "display": "Septoria Leaf Spot",
        "severity": "Moderate",
        "remedy": "Remove affected leaves. Apply fungicide early. Rotate crops next season.",
    },
    "Spider_mites": {
        "display": "Spider Mites",
        "severity": "Moderate",
        "remedy": "Spray with neem oil or insecticidal soap. Increase humidity around plants.",
    },
    "Target_Spot": {
        "display": "Target Spot",
        "severity": "Moderate",
        "remedy": "Apply azoxystrobin or pyraclostrobin fungicide. Remove debris after harvest.",
    },
    "Mosaic_virus": {
        "display": "Mosaic Virus",
        "severity": "High",
        "remedy": "No cure. Remove and destroy infected plants. Control aphid vectors.",
    },
    "Yellow_Leaf_Curl_Virus": {
        "display": "Yellow Leaf Curl Virus",
        "severity": "High",
        "remedy": "Remove infected plants. Control whitefly population with insecticides.",
    },
    "Bacterial_spot": {
        "display": "Bacterial Spot",
        "severity": "Moderate",
        "remedy": "Apply copper-based bactericide. Avoid working with wet plants.",
    },
    "Black_rot": {
        "display": "Black Rot",
        "severity": "High",
        "remedy": "Remove infected leaves/fruits. Apply copper fungicide. Improve drainage.",
    },
    "Esca": {
        "display": "Esca (Black Measles)",
        "severity": "High",
        "remedy": "No effective chemical cure. Prune infected wood. Protect wounds after pruning.",
    },
    "Haunglongbing": {
        "display": "Citrus Greening (HLB)",
        "severity": "Critical",
        "remedy": "No cure. Remove infected trees. Control psyllid insects aggressively.",
    },
    "Powdery_mildew": {
        "display": "Powdery Mildew",
        "severity": "Low–Moderate",
        "remedy": "Apply sulfur-based fungicide or neem oil. Improve air circulation.",
    },
    "Leaf_scorch": {
        "display": "Leaf Scorch",
        "severity": "Moderate",
        "remedy": "Ensure adequate watering. Mulch around plants. Avoid excess fertiliser.",
    },
    "Common_rust": {
        "display": "Common Rust",
        "severity": "Moderate",
        "remedy": "Apply fungicide containing propiconazole. Plant resistant varieties next season.",
    },
    "Northern_Leaf_Blight": {
        "display": "Northern Leaf Blight",
        "severity": "Moderate",
        "remedy": "Apply triazole or strobilurin fungicide. Use resistant corn hybrids.",
    },
    "Gray_leaf_spot": {
        "display": "Gray Leaf Spot",
        "severity": "Moderate",
        "remedy": "Apply strobilurin fungicide. Rotate crops. Till residue after harvest.",
    },
}


def get_disease_info(class_name: str) -> dict:
    """Return human-readable info for a predicted class."""
    for key, info in DISEASE_INFO.items():
        if key.lower() in class_name.lower():
            return info
    # Generic fallback
    return {
        "display": class_name.replace("_", " ").replace("__", " – "),
        "severity": "Unknown",
        "remedy": "Consult a local agricultural extension officer for advice.",
    }


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_model_and_classes():
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else BEST_MODEL
    if not os.path.exists(model_path):
        print("❌ No trained model found. Run  python train.py  first.")
        sys.exit(1)

    print(f"📦 Loading model from  {model_path} …")
    model = tf.keras.models.load_model(model_path)

    if not os.path.exists(CLASS_MAP_PATH):
        print("❌ class_names.json not found. Re-run training.")
        sys.exit(1)

    with open(CLASS_MAP_PATH) as f:
        class_map = json.load(f)           # {str(idx): "ClassName"}
    class_map = {int(k): v for k, v in class_map.items()}
    return model, class_map


# ──────────────────────────────────────────────
# PREPROCESS
# ──────────────────────────────────────────────
def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)   # (1, 224, 224, 3)


# ──────────────────────────────────────────────
# PREDICT SINGLE IMAGE
# ──────────────────────────────────────────────
def predict_image(model, class_map, image_path: str, top_k: int = 5):
    x      = preprocess(image_path)
    preds  = model.predict(x, verbose=0)[0]          # shape: (num_classes,)
    top_k  = min(top_k, len(preds))
    top_idx = np.argsort(preds)[::-1][:top_k]

    results = []
    for i, idx in enumerate(top_idx):
        class_name = class_map[idx]
        confidence = float(preds[idx]) * 100
        info       = get_disease_info(class_name)
        results.append({
            "rank":       i + 1,
            "class_name": class_name,
            "display":    info["display"],
            "confidence": confidence,
            "severity":   info["severity"],
            "remedy":     info["remedy"],
        })

    return results


# ──────────────────────────────────────────────
# VISUALISE RESULT
# ──────────────────────────────────────────────
def visualise(image_path: str, results: list, save_path: str = None):
    top = results[0]
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1b0d")

    # Left – image
    axes[0].imshow(img)
    axes[0].axis("off")
    conf_color = "#4caf50" if "healthy" in top["class_name"].lower() else "#ff6b35"
    axes[0].set_title(
        f"{top['display']}\nConfidence: {top['confidence']:.1f}%",
        color="white", fontsize=13, fontweight="bold", pad=10,
    )

    # Right – bar chart of top-5
    names   = [r["display"][:30] for r in results]
    confs   = [r["confidence"] for r in results]
    colors  = ["#4caf50" if "Healthy" in n else "#ff6b35" for n in names]
    bars    = axes[1].barh(names[::-1], confs[::-1], color=colors[::-1], edgecolor="none", height=0.6)
    axes[1].set_xlabel("Confidence (%)", color="white")
    axes[1].set_xlim(0, 100)
    axes[1].tick_params(colors="white")
    axes[1].set_facecolor("#162013")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#2e4a28")
    for bar, conf in zip(bars, confs[::-1]):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{conf:.1f}%", va="center", color="white", fontsize=9)
    axes[1].set_title("Top-5 Predictions", color="white", fontweight="bold")

    # Remedy box
    remedy_text = f"Severity: {top['severity']}\n\nRecommendation:\n{top['remedy']}"
    fig.text(0.5, 0.01, remedy_text, ha="center", va="bottom",
             color="#b5d6a7", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a2e17", edgecolor="#4caf50", alpha=0.8))

    plt.suptitle("🌿 Plant Disease Detection", color="#4caf50", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"   📊 Result image saved → {save_path}")
    else:
        plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close()


# ──────────────────────────────────────────────
# BATCH PREDICTION
# ──────────────────────────────────────────────
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def predict_folder(model, class_map, folder: str):
    paths = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in SUPPORTED]
    if not paths:
        print(f"No supported images found in {folder}")
        return

    print(f"\nRunning batch prediction on {len(paths)} images …\n")
    print(f"{'Image':<40} {'Prediction':<30} {'Confidence':>10}")
    print("-" * 82)

    all_results = []
    for path in sorted(paths):
        results = predict_image(model, class_map, path)
        top     = results[0]
        print(f"{os.path.basename(path):<40} {top['display'][:30]:<30} {top['confidence']:>9.1f}%")
        all_results.append({"image": path, "top": top})

    print("\nDone.")
    return all_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="🌿 Plant Disease Predictor")
    parser.add_argument("input", nargs="?",
                        help="Path to an image file or folder of images")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Show top-K predictions (default: 5)")
    parser.add_argument("--save",  type=str, default=None,
                        help="Save result visualisation to this path")
    args = parser.parse_args()

    model, class_map = load_model_and_classes()

    # Default demo image if none provided
    if not args.input:
        print("\n⚠️  No input provided.")
        print("Usage:  python predict.py path/to/leaf.jpg")
        print("        python predict.py path/to/folder/\n")
        sys.exit(0)

    inp = args.input

    if os.path.isdir(inp):
        predict_folder(model, class_map, inp)

    elif os.path.isfile(inp):
        print(f"\n🔍 Analysing: {inp}\n")
        results = predict_image(model, class_map, inp, top_k=args.top_k)

        print("━" * 55)
        print(f"  🌿 PLANT DISEASE DETECTION RESULT")
        print("━" * 55)
        for r in results:
            marker = "▶" if r["rank"] == 1 else " "
            print(f"  {marker} #{r['rank']}  {r['display']:<35} {r['confidence']:5.1f}%")
        print("━" * 55)
        top = results[0]
        print(f"\n  Severity     : {top['severity']}")
        print(f"  Remedy       : {top['remedy']}")
        print()

        visualise(inp, results, save_path=args.save)
    else:
        print(f"❌ Path not found: {inp}")
        sys.exit(1)


if __name__ == "__main__":
    main()
