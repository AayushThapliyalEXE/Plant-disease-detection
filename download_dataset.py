"""
Download PlantVillage dataset via TensorFlow Datasets
and organise it into  dataset/PlantVillage/<ClassName>/  folders.

PlantVillage contains 54,306 images across 38 plant-disease classes
(healthy + diseased variants of 14 crops).
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

──────────────────────────────────────────────
KAGGLE_INSTRUCTIONS = """
── Manual download via Kaggle ──────────────────────────────
1. Install kaggle:        pip install kaggle
2. Get your API token:    https://www.kaggle.com/settings → API → Create Token
   (places ~/.kaggle/kaggle.json automatically)
3. Download the dataset:
      kaggle datasets download -d abdallahalidev/plantvillage-dataset
      unzip plantvillage-dataset.zip -d dataset/
4. Rename the extracted folder so that the path becomes:
      dataset/PlantVillage/
   where each sub-folder is a disease class (e.g. Tomato___Early_blight)
5. Re-run  python train.py
─────────────────────────────────────────────────────────────
"""

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🌿 Plant Disease Detector – Dataset Setup")
    print("=" * 60)

    out_dir = "dataset/PlantVillageDataset/PlantVillage"

    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 5:
        print(f"\n✅ Dataset already present at '{out_dir}'. Nothing to do.")
        classes = os.listdir(out_dir)
        print(f"   Found {len(classes)} class folders.")
        return

    ok = download_via_tfds(out_dir)
    if not ok:
        print(KAGGLE_INSTRUCTIONS)

if __name__ == "__main__":
    main()
