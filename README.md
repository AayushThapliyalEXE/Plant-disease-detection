# 🌿 Plant Disease Detector

An end-to-end deep learning project that trains a **MobileNetV2** model on the
**PlantVillage** dataset and lets you predict plant diseases from new photos.

---

## 📁 Project Structure

```
plant_disease_detector/
├── download_dataset.py   # Step 1 – Download & organise dataset
├── train.py              # Step 2 – Train the model
├── predict.py            # Step 3 – Predict on new images (CLI)
├── app.py                # Step 4 – Interactive web app (optional)
├── requirements.txt
└── README.md
```

After running the scripts, these folders are created automatically:
```
dataset/
└── PlantVillage/
    ├── Tomato___Early_blight/
    ├── Tomato___healthy/
    └── … (38 folders total)
models/
├── best_model.keras
├── plant_disease_model.keras
└── class_names.json
logs/
├── training_log.csv
└── training_curves.png
```

---

## ⚡ Quick Start

### 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### 2 – Download the dataset
```bash
python download_dataset.py
```
This uses **TensorFlow Datasets** to download PlantVillage automatically (~1 GB).

> **Alternative (Kaggle):** If you prefer, download manually:
> ```bash
> pip install kaggle
> kaggle datasets download -d abdallahalidev/plantvillage-dataset
> unzip plantvillage-dataset.zip -d dataset/
> # Rename extracted folder to  dataset/PlantVillage/
> ```

### 3 – Train the model
```bash
python train.py
```
Training uses **two-phase transfer learning**:
- Phase 1 (~15 epochs): only the classification head is trained
- Phase 2 (~15 epochs): top 30 layers of MobileNetV2 are fine-tuned

Expected accuracy: **~95% on validation set** after fine-tuning.

### 4 – Predict on a new image
```bash
# Single image
python predict.py path/to/leaf.jpg

# Show top-3 predictions
python predict.py path/to/leaf.jpg --top_k 3

# Save result image
python predict.py path/to/leaf.jpg --save result.png

# Batch prediction on a folder
python predict.py path/to/folder_of_images/
```

### 5 – Launch the web app (optional)
```bash
python app.py
# Open  http://localhost:7860  in your browser
```

---

## 🌱 Supported Crops & Diseases (38 classes)

| Crop        | Diseases Detected |
|-------------|-------------------|
| Tomato      | Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Mosaic Virus, Yellow Curl Virus, Bacterial Spot, Healthy |
| Apple       | Apple Scab, Black Rot, Cedar Rust, Healthy |
| Corn (Maize)| Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy |
| Grape       | Black Rot, Esca, Leaf Blight, Healthy |
| Potato      | Early Blight, Late Blight, Healthy |
| Pepper      | Bacterial Spot, Healthy |
| Strawberry  | Leaf Scorch, Healthy |
| Orange      | Citrus Greening (HLB) |
| Peach       | Bacterial Spot, Healthy |
| + more      | Cherry, Blueberry, Raspberry, Soybean, Squash |

---

## 🧠 Model Architecture

```
Input (224×224×3)
    │
MobileNetV2 (ImageNet pre-trained, fine-tuned top 30 layers)
    │
GlobalAveragePooling2D
    │
BatchNormalization
    │
Dense(256, relu) → Dropout(0.5)
    │
Dense(128, relu) → Dropout(0.3)
    │
Dense(38, softmax)   ← output: probability per class
```

---

## 📊 Expected Results

| Metric          | Value       |
|-----------------|-------------|
| Val Accuracy    | ~95%        |
| Training Time   | ~30–60 min (GPU) / ~3–5 hr (CPU) |
| Model Size      | ~14 MB      |
| Inference Speed | <100 ms/image |

---

## 💡 Tips

- Use a **GPU** (NVIDIA + CUDA) for much faster training.
- The `best_model.keras` is saved automatically whenever validation accuracy improves.
- To re-train from scratch, delete the `models/` folder and re-run `train.py`.
- For best predictions, photograph the leaf against a plain background in good lighting.
# Plant-disease-detection
