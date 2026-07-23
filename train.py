"""
Plant Disease Detection - Training Script
Dataset: PlantVillage (via Kaggle or TensorFlow Datasets)
Model: MobileNetV2 (Transfer Learning)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
IMG_SIZE     = 224          # MobileNetV2 default
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
DATASET_DIR  = "dataset/PlantVillage"   # folder created by download_dataset.py
MODEL_DIR    = "models"
LOGS_DIR     = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)

# ──────────────────────────────────────────────
# DATA GENERATORS  (with augmentation)
# ──────────────────────────────────────────────
def build_generators(dataset_dir: str):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
    )
    val_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
    )

    train_ds = train_gen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )
    val_ds = val_gen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    return train_ds, val_ds


# ──────────────────────────────────────────────
# MODEL  (MobileNetV2 + custom head)
# ──────────────────────────────────────────────
def build_model(num_classes: int) -> Model:
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Freeze base initially
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_top_layers(model, base_model, num_layers: int = 30):
    """Fine-tune the top N layers of the base model."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=LR / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Fine-tuning: unfroze last {num_layers} layers of MobileNetV2.")


# ──────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────
def get_callbacks():
    return [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(os.path.join(LOGS_DIR, "training_log.csv")),
    ]


# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────
def plot_history(history, filename="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plant Disease Model – Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train Acc", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",   linewidth=2)
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, filename), dpi=150)
    print(f"✅ Training curves saved → {LOGS_DIR}/{filename}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🌿 Plant Disease Detector – Training")
    print("=" * 60)

    if not os.path.isdir(DATASET_DIR):
        print(f"\n❌  Dataset not found at '{DATASET_DIR}'")
        print("    Run  python download_dataset.py  first.\n")
        return

    # ── Data ──
    train_ds, val_ds = build_generators(DATASET_DIR)
    num_classes = len(train_ds.class_indices)
    print(f"\n📂 Classes found : {num_classes}")
    print(f"   Train samples : {train_ds.samples}")
    print(f"   Val   samples : {val_ds.samples}\n")

    # Save class mapping
    class_map = {v: k for k, v in train_ds.class_indices.items()}
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"✅ Class names saved → {MODEL_DIR}/class_names.json")

    # ── Phase 1: Train head only ──
    print("\n── Phase 1: Training head (base frozen) ──")
    model, base_model = build_model(num_classes)
    model.summary(line_length=80)

    history1 = model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=get_callbacks(),
    )

    # ── Phase 2: Fine-tune top layers ──
    print("\n── Phase 2: Fine-tuning top 30 layers ──")
    unfreeze_top_layers(model, base_model, num_layers=30)

    history2 = model.fit(
        train_ds,
        epochs=EPOCHS,
        initial_epoch=len(history1.history["loss"]),
        validation_data=val_ds,
        callbacks=get_callbacks(),
    )

    # Merge histories
    combined = {}
    for k in history1.history:
        combined[k] = history1.history[k] + history2.history[k]

    class FakeHistory:
        def __init__(self, h): self.history = h
    plot_history(FakeHistory(combined))

    # ── Evaluate ──
    print("\n── Final Evaluation on Validation Set ──")
    loss, acc = model.evaluate(val_ds, verbose=1)
    print(f"\n🏆 Val Accuracy : {acc * 100:.2f}%")
    print(f"   Val Loss     : {loss:.4f}")

    # Save final model
    final_path = os.path.join(MODEL_DIR, "plant_disease_model.keras")
    model.save(final_path)
    print(f"\n✅ Final model saved → {final_path}")
    print("\nDone! Run  python predict.py  to classify a new image.\n")


if __name__ == "__main__":
    main()
