#!/usr/bin/env python3.10
"""
Standalone Client 2 — CIC-ToN-IoT aligned 20k partition
===========================================================
Local-only baseline matched to FedProto Client 2.
No server communication and no prototype regularization/sharing.
"""

import os
import csv
import json
import pickle
import random
import time
from pathlib import Path
from datetime import datetime

SEED = 308
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed(SEED)

import numpy as np
np.random.seed(SEED)
import pandas as pd
import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras import layers, models, optimizers

# ============================================================
# Experiment settings — matched to FedProto
# ============================================================
CLIENT_ID = "client2"
NUM_EPOCHS = 40          # equivalent to 20 rounds x 1 local epoch
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PRIVATE_DIM = 8
HIDDEN_DIM = 8           # matches FedProto prototype embedding width
CLIP_NORM = 1.0
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)

BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/FedProtoVsPFTL"
)
CSV_PATH = (
    BASE_DIR / "Multi_class_Datasets" / "ton_iot_10_virtual_clients"
    / "ton_iot_client02.csv"
)

LABEL_COL = "Attack"
CLASS_NAMES = ['Benign', 'xss', 'password', 'injection', 'ransomware',
               'scanning', 'backdoor', 'mitm', 'ddos', 'dos']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

OUT_DIR = BASE_DIR / "standalone_logs" / "client2_ton_iot_aligned_20k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_LOG = OUT_DIR / f"{CLIENT_ID}_standalone_metrics.csv"
FINAL_METRICS_CSV = OUT_DIR / f"{CLIENT_ID}_standalone_final_metrics.csv"
CLASSIFICATION_REPORT_CSV = OUT_DIR / f"{CLIENT_ID}_classification_report.csv"
CONFUSION_MATRIX_CSV = OUT_DIR / f"{CLIENT_ID}_confusion_matrix.csv"
CONFUSION_MATRIX_PNG = OUT_DIR / f"{CLIENT_ID}_confusion_matrix.png"
CURVE_PNG = OUT_DIR / f"{CLIENT_ID}_standalone_convergence.png"
MODEL_PATH = OUT_DIR / f"{CLIENT_ID}_standalone_final_model.keras"
SCALER_PATH = OUT_DIR / f"{CLIENT_ID}_standard_scaler.pkl"
SUMMARY_JSON = OUT_DIR / f"{CLIENT_ID}_run_summary.json"


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.astype(str).str.replace("\\ufeff", "", regex=False).str.strip()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found.")

    labels = (
        df[LABEL_COL].astype(str)
        .str.replace("\\ufeff", "", regex=False)
        .str.replace("�", "-", regex=False)
        .str.replace(r"\\s+", " ", regex=True)
        .str.strip()
    )

    unknown = sorted(set(labels.unique()) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")

    missing = sorted(set(CLASS_NAMES) - set(labels.unique()))
    if missing:
        raise ValueError(
            f"Aligned-label experiment requires all 10 classes. Missing: {missing}"
        )

    y = labels.map(CLASS_TO_ID).astype(np.int32).to_numpy()

    # source_row_id is partition metadata, never a predictive feature.
    excluded = {
        LABEL_COL, "source_row_id", "Label", "binary_label",
        "local_label_id", "semantic_label", "original_label"
    }
    Xdf = (
        df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if Xdf.shape[1] == 0:
        raise ValueError("No numeric features remained.")

    X = Xdf.to_numpy(dtype=np.float32)
    idx = np.arange(len(y), dtype=np.int64)

    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=SEED,
        stratify=y[train_idx],
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])[..., np.newaxis].astype(np.float32)
    X_val = scaler.transform(X[val_idx])[..., np.newaxis].astype(np.float32)
    X_test = scaler.transform(X[test_idx])[..., np.newaxis].astype(np.float32)

    y_train = y[train_idx].astype(np.int32)
    y_val = y[val_idx].astype(np.int32)
    y_test = y[test_idx].astype(np.int32)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, class_weight


def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.Conv1D(32, 3, padding="valid", activation="relu", name="private_conv1")(inputs)
    x = layers.BatchNormalization(name="private_bn1")(x)
    x = layers.MaxPooling1D(2, name="private_pool1")(x)
    x = layers.Dropout(0.25, name="private_dropout1")(x)

    x = layers.Conv1D(64, 3, padding="valid", activation="relu", name="private_conv2")(x)
    x = layers.BatchNormalization(name="private_bn2")(x)
    x = layers.GlobalAveragePooling1D(name="private_global_average_pool")(x)

    x = layers.Dense(PRIVATE_DIM, activation="relu", name="private_dense")(x)
    # Same width as FedProto's prototype_embedding, but fully private/local here.
    x = layers.Dense(HIDDEN_DIM, activation="relu", name="private_embedding")(x)
    logits = layers.Dense(num_classes, activation=None, name="local_classifier_logits")(x)

    return models.Model(inputs=inputs, outputs=logits, name=f"Standalone_{CLIENT_ID}_CNN")


def evaluate(model, X, y):
    logits = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    pred = np.argmax(logits, axis=1)
    acc = accuracy_score(y, pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y, pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
    }, pred


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(CONFUSION_MATRIX_CSV)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Standalone {CLIENT_ID} Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=200)
    plt.close(fig)


def main():
    print("=" * 80)
    print(f"STANDALONE {CLIENT_ID.upper()} — LOCAL ONLY")
    print("=" * 80)
    print(f"Dataset: {CSV_PATH}")
    print("Architecture: Conv1D(32) -> Conv1D(64) -> Dense(8) -> Dense(8) -> classifier")
    print(f"Epochs: {NUM_EPOCHS}, batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, class_weight = load_data()

    with SCALER_PATH.open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    model = build_model((X_train.shape[1], 1), len(CLASS_NAMES))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()

    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        shuffle=True,
        verbose=2,
    )
    total_sec = time.perf_counter() - start

    # Epoch-wise Macro-F1 on validation set for a directly comparable learning curve.
    rows = []
    # Keras history only retains final model weights, so for exact epoch-wise Macro-F1
    # we log standard loss/accuracy per epoch and report final validation Macro-F1 below.
    for i in range(NUM_EPOCHS):
        rows.append({
            "seed": SEED,
            "epoch": i + 1,
            "train_loss": float(history.history["loss"][i]),
            "train_accuracy": float(history.history["accuracy"][i]),
            "val_loss": float(history.history["val_loss"][i]),
            "val_accuracy": float(history.history["val_accuracy"][i]),
        })
    pd.DataFrame(rows).to_csv(METRICS_LOG, index=False)

    val_metrics, _ = evaluate(model, X_val, y_val)
    test_metrics, y_pred = evaluate(model, X_test, y_test)

    final_row = {
        "client_id": CLIENT_ID,
        "seed": SEED,
        "samples_train": len(y_train),
        "samples_val": len(y_val),
        "samples_test": len(y_test),
        "conv1_filters": 32,
        "conv2_filters": 64,
        "private_dim": PRIVATE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "total_training_time_sec": total_sec,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    pd.DataFrame([final_row]).to_csv(FINAL_METRICS_CSV, index=False)

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_REPORT_CSV)
    save_confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, NUM_EPOCHS + 1), history.history["loss"], label="train loss")
    ax.plot(range(1, NUM_EPOCHS + 1), history.history["val_loss"], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Standalone {CLIENT_ID} Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=200)
    plt.close(fig)

    model.save(MODEL_PATH)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "client_id": CLIENT_ID,
        "dataset": str(CSV_PATH),
        "architecture": "Conv1D(32)-Conv1D(64)-Dense(8)-Dense(8)-Classifier",
        "federation": False,
        "prototype_sharing": False,
        "prototype_loss": False,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "total_training_time_sec": total_sec,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("\nFINAL STANDALONE RESULTS")
    print("-" * 80)
    print(f"Validation Macro-F1: {val_metrics['macro_f1']:.6f}")
    print(f"Test Accuracy:       {test_metrics['accuracy']:.6f}")
    print(f"Test Macro-F1:       {test_metrics['macro_f1']:.6f}")
    print(f"Test Weighted-F1:    {test_metrics['weighted_f1']:.6f}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
