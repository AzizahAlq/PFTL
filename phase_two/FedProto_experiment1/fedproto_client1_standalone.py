#!/usr/bin/env python3
# ============================================================
# Standalone Client 1 — CIC-ToN-IoT
# MATCHED TO FEDPROTO PIPELINE
# ============================================================
#
# Purpose:
#   Train a purely standalone Client 1 model using the same:
#   - mapped dataset
#   - train/validation/test split
#   - StandardScaler preprocessing
#   - CNN/private_dense/prototype_embedding architecture
#   - prototype embedding dimension = 8
#   - optimizer settings
#   - class weighting
#
# Differences from FedProto:
#   - NO server connection
#   - NO prototype exchange
#   - NO global prototypes
#   - NO prototype loss
#   - classification loss only
#
# The final 8-D sample-level representation is taken from
# "prototype_embedding", so this is the correct standalone
# reference for comparison with Client 1 FedProto.
# ============================================================

import os
import csv
import json
import time
import pickle
import random
from pathlib import Path
from datetime import datetime

SEED = 123

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
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from tensorflow.keras import layers, models, optimizers


# ============================================================
# Configuration — matched to FedProto Client 1
# ============================================================

CLIENT_ID = "client1"

EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

PRIVATE_DIM = 16
PROTO_DIM = 8
CLIP_NORM = 1.0

TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)

BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "M_PFTL_codes"
)

CSV_PATH = (
    BASE_DIR
    / "Multi_class_Datasets"
    / "D1_CIC-ToN-IoT_FedProto_mapped.csv"
)

GLOBAL_MAPPING_PATH = (
    BASE_DIR
    / "FedProto_Global_Mapping"
    / "client_local_to_global_ids.json"
)

LOCAL_LABEL_COL = "local_label_id"
SEMANTIC_LABEL_COL = "semantic_label"
ORIGINAL_LABEL_COL = "original_label"

OUT_DIR = (
    BASE_DIR
    / "standalone_outputs"
    / "client1_before_fedproto"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Output files
# ============================================================

EPOCH_LOG_CSV = OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_epoch_log_seed{SEED}.csv"
FINAL_METRICS_CSV = OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_final_metrics_seed{SEED}.csv"

CLASSIFICATION_REPORT_CSV = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_classification_report_seed{SEED}.csv"
)

CONFUSION_MATRIX_CSV = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_confusion_matrix_seed{SEED}.csv"
)

CONFUSION_MATRIX_NORMALIZED_CSV = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_confusion_matrix_normalized_seed{SEED}.csv"
)

CONFUSION_MATRIX_PNG = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_confusion_matrix_seed{SEED}.png"
)

CONFUSION_MATRIX_NORMALIZED_PNG = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_confusion_matrix_normalized_seed{SEED}.png"
)

MODEL_PATH = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_final_model_seed{SEED}.keras"
)

SCALER_PATH = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_standard_scaler_seed{SEED}.pkl"
)

EMBEDDINGS_CSV = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_embeddings_seed{SEED}.csv"
)

TSNE_COORDINATES_CSV = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_TSNE_coordinates_seed{SEED}.csv"
)

TSNE_PNG = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_embedding_TSNE_seed{SEED}.png"
)

UMAP_COORDINATES_CSV = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_UMAP_coordinates_seed{SEED}.csv"
)

UMAP_PNG = (
    OUT_DIR / f"{CLIENT_ID}_standalone_before_fedproto_embedding_UMAP_seed{SEED}.png"
)

VIS_MAX_SAMPLES = 5000
TSNE_SEED = 123
UMAP_SEED = 123


# ============================================================
# Utilities
# ============================================================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))

    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf1),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf1),
    }


# ============================================================
# Load Client 1 local-to-global semantic mapping
# ============================================================

def load_mapping():
    if not GLOBAL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Mapping file not found:\n{GLOBAL_MAPPING_PATH}"
        )

    with GLOBAL_MAPPING_PATH.open("r", encoding="utf-8") as f:
        all_mappings = json.load(f)

    if CLIENT_ID not in all_mappings:
        raise KeyError(
            f"{CLIENT_ID} is missing from:\n{GLOBAL_MAPPING_PATH}"
        )

    raw_mapping = all_mappings[CLIENT_ID].get("local_to_global")

    if not isinstance(raw_mapping, dict):
        raise KeyError(
            f"'local_to_global' is missing for {CLIENT_ID}."
        )

    local_to_global = {}
    local_to_semantic = {}

    for local_id_text, entry in raw_mapping.items():
        local_id = int(local_id_text)
        local_to_global[local_id] = int(entry["global_id"])
        local_to_semantic[local_id] = str(entry["semantic_label"]).strip()

    return local_to_global, local_to_semantic


# ============================================================
# Data — same pipeline as FedProto Client 1
# ============================================================

def load_data(local_to_semantic):
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Mapped Client 1 dataset not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH, low_memory=False)

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required = {LOCAL_LABEL_COL, SEMANTIC_LABEL_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Mapped dataset is missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=[LOCAL_LABEL_COL, SEMANTIC_LABEL_COL]
    ).copy()

    y = pd.to_numeric(
        df[LOCAL_LABEL_COL],
        errors="raise",
    ).astype(np.int32).to_numpy()

    semantic_values = (
        df[SEMANTIC_LABEL_COL]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    num_classes = len(local_to_semantic)

    expected_ids = set(range(num_classes))
    observed_ids = {int(v) for v in np.unique(y)}

    if observed_ids != expected_ids:
        raise ValueError(
            "Client 1 labels are not contiguous.\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Observed: {sorted(observed_ids)}"
        )

    for local_id in sorted(observed_ids):
        expected_semantic = local_to_semantic[local_id]
        observed_semantic = set(
            semantic_values[y == local_id]
        )

        if observed_semantic != {expected_semantic}:
            raise ValueError(
                f"Semantic mismatch for local ID {local_id}.\n"
                f"Expected: {expected_semantic}\n"
                f"Observed: {observed_semantic}"
            )

    excluded_columns = {
        LOCAL_LABEL_COL,
        SEMANTIC_LABEL_COL,
        ORIGINAL_LABEL_COL,
        "Label",
        "Attack",
        "binary_label",
    }

    feature_df = (
        df.drop(
            columns=[
                c for c in excluded_columns if c in df.columns
            ],
            errors="ignore",
        )
        .select_dtypes(include=[np.number])
        .copy()
    )

    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    if feature_df.shape[1] == 0:
        raise ValueError("No numeric features remained.")

    X = feature_df.to_numpy(dtype=np.float32)
    indices = np.arange(len(y), dtype=np.int64)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=SEED,
        stratify=y[train_idx],
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_val = X_val[..., np.newaxis].astype(np.float32)
    X_test = X_test[..., np.newaxis].astype(np.float32)

    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return (
        X,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        num_classes,
    )


# ============================================================
# Model — same representation path as FedProto
# ============================================================

def build_standalone_model(input_shape, num_classes):
    inputs = layers.Input(
        shape=input_shape,
        name="input",
    )

    x = layers.Conv1D(
        64,
        3,
        padding="valid",
        activation="relu",
        name="private_conv1",
    )(inputs)

    x = layers.BatchNormalization(
        name="private_bn1"
    )(x)

    x = layers.MaxPooling1D(
        2,
        name="private_pool1",
    )(x)

    x = layers.Dropout(
        0.25,
        name="private_dropout1",
    )(x)

    x = layers.Conv1D(
        128,
        3,
        padding="valid",
        activation="relu",
        name="private_conv2",
    )(x)

    x = layers.BatchNormalization(
        name="private_bn2"
    )(x)

    x = layers.MaxPooling1D(
        2,
        name="private_pool2",
    )(x)

    x = layers.Dropout(
        0.25,
        name="private_dropout2",
    )(x)

    x = layers.Conv1D(
        128,
        3,
        padding="valid",
        activation="relu",
        name="private_conv3",
    )(x)

    x = layers.BatchNormalization(
        name="private_bn3"
    )(x)

    x = layers.GlobalAveragePooling1D(
        name="private_global_average_pool"
    )(x)

    x = layers.Dense(
        PRIVATE_DIM,
        activation="relu",
        name="private_dense",
    )(x)

    embedding = layers.Dense(
        PROTO_DIM,
        activation="relu",
        name="prototype_embedding",
    )(x)

    logits = layers.Dense(
        num_classes,
        activation=None,
        name="local_classifier_logits",
    )(embedding)

    return models.Model(
        inputs=inputs,
        outputs={
            "logits": logits,
            "embedding": embedding,
        },
        name="Standalone_Client1_Match_FedProto",
    )


# ============================================================
# Standalone training
# ============================================================

def train_standalone(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weight_tensor,
):
    optimizer = optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIP_NORM,
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )

    if not EPOCH_LOG_CSV.exists():
        with EPOCH_LOG_CSV.open("w", newline="") as f:
            csv.writer(f).writerow([
                "seed",
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_accuracy",
                "validation_macro_precision",
                "validation_macro_recall",
                "validation_macro_f1",
                "validation_weighted_precision",
                "validation_weighted_recall",
                "validation_weighted_f1",
                "epoch_time_sec",
                "cumulative_time_sec",
                "timestamp",
            ])

    start_all = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()

        ds = (
            tf.data.Dataset
            .from_tensor_slices((X_train, y_train))
            .shuffle(
                len(y_train),
                seed=SEED + epoch,
                reshuffle_each_iteration=True,
            )
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        loss_sum = 0.0
        correct = 0
        processed = 0

        for x_batch, y_batch in ds:
            y_batch = tf.cast(y_batch, tf.int32)

            with tf.GradientTape() as tape:
                outputs = model(
                    x_batch,
                    training=True,
                )

                logits = outputs["logits"]

                per_sample_loss = loss_fn(
                    y_batch,
                    logits,
                )

                sample_weights = tf.gather(
                    class_weight_tensor,
                    y_batch,
                )

                loss = tf.reduce_mean(
                    per_sample_loss * sample_weights
                )

            gradients = tape.gradient(
                loss,
                model.trainable_variables,
            )

            pairs = [
                (g, v)
                for g, v in zip(
                    gradients,
                    model.trainable_variables,
                )
                if g is not None
            ]

            optimizer.apply_gradients(pairs)

            pred = tf.argmax(
                logits,
                axis=1,
                output_type=tf.int32,
            )

            batch_n = int(tf.shape(y_batch)[0].numpy())

            correct += int(
                tf.reduce_sum(
                    tf.cast(
                        tf.equal(pred, y_batch),
                        tf.int32,
                    )
                ).numpy()
            )

            processed += batch_n
            loss_sum += float(loss.numpy()) * batch_n

        train_loss = loss_sum / max(processed, 1)
        train_acc = correct / max(processed, 1)

        val_pred, _, _ = predict(
            model,
            X_val,
        )

        val_metrics = calculate_metrics(
            y_val,
            val_pred,
        )

        epoch_time = time.perf_counter() - epoch_start
        cumulative = time.perf_counter() - start_all

        with EPOCH_LOG_CSV.open("a", newline="") as f:
            csv.writer(f).writerow([
                SEED,
                epoch + 1,
                f"{train_loss:.8f}",
                f"{train_acc:.8f}",
                f"{val_metrics['accuracy']:.8f}",
                f"{val_metrics['macro_precision']:.8f}",
                f"{val_metrics['macro_recall']:.8f}",
                f"{val_metrics['macro_f1']:.8f}",
                f"{val_metrics['weighted_precision']:.8f}",
                f"{val_metrics['weighted_recall']:.8f}",
                f"{val_metrics['weighted_f1']:.8f}",
                f"{epoch_time:.4f}",
                f"{cumulative:.4f}",
                now_str(),
            ])

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| loss={train_loss:.6f} "
            f"| train_acc={train_acc:.6f} "
            f"| val_macro_f1={val_metrics['macro_f1']:.6f}"
        )

    return time.perf_counter() - start_all


# ============================================================
# Inference
# ============================================================

def predict(model, X):
    ds = (
        tf.data.Dataset
        .from_tensor_slices(X)
        .batch(BATCH_SIZE)
    )

    predictions = []
    probabilities = []
    embeddings = []

    for x_batch in ds:
        outputs = model(
            x_batch,
            training=False,
        )

        logits = outputs["logits"]
        batch_embeddings = outputs["embedding"]

        probs = tf.nn.softmax(
            logits,
            axis=1,
        )

        pred = tf.argmax(
            probs,
            axis=1,
            output_type=tf.int32,
        )

        predictions.append(pred.numpy())
        probabilities.append(probs.numpy())
        embeddings.append(batch_embeddings.numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(probabilities, axis=0),
        np.concatenate(embeddings, axis=0),
    )


# ============================================================
# Save representation outputs
# ============================================================

def save_embedding_outputs(
    model,
    X_test,
    y_test,
    local_to_semantic,
):
    X_vis = X_test
    y_vis = y_test

    if len(X_vis) > VIS_MAX_SAMPLES:
        all_idx = np.arange(len(y_vis))

        selected_idx, _ = train_test_split(
            all_idx,
            train_size=VIS_MAX_SAMPLES,
            random_state=SEED,
            stratify=y_vis,
        )

        X_vis = X_vis[selected_idx]
        y_vis = y_vis[selected_idx]

    _, _, embeddings = predict(
        model,
        X_vis,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2 or embeddings.shape[1] != PROTO_DIM:
        raise ValueError(
            f"Expected embedding shape (N, {PROTO_DIM}), "
            f"got {embeddings.shape}."
        )

    label_names = np.asarray(
        [
            str(
                local_to_semantic[int(label_id)]
            ).strip()
            for label_id in y_vis
        ],
        dtype=object,
    )

    # Raw 8-D sample embeddings
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[
            f"proto_dim_{i + 1}"
            for i in range(PROTO_DIM)
        ],
    )

    embedding_df["label_id"] = y_vis
    embedding_df["label_name"] = label_names

    embedding_df.to_csv(
        EMBEDDINGS_CSV,
        index=False,
    )

    class_ids = sorted(local_to_semantic)
    class_names = [
        str(local_to_semantic[i]).strip()
        for i in class_ids
    ]

    cmap = plt.get_cmap("tab20")
    denom = max(len(class_names) - 1, 1)

    class_colors = {
        class_name: cmap(i / denom)
        for i, class_name in enumerate(class_names)
    }

    # ---------------- t-SNE ----------------

    perplexity = (
        30
        if len(embeddings) > 31
        else max(5, min(30, len(embeddings) - 1))
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=TSNE_SEED,
    )

    tsne_2d = tsne.fit_transform(
        embeddings
    )

    pd.DataFrame({
        "tsne_1": tsne_2d[:, 0],
        "tsne_2": tsne_2d[:, 1],
        "label_id": y_vis,
        "label_name": label_names,
    }).to_csv(
        TSNE_COORDINATES_CSV,
        index=False,
    )

    plt.figure(figsize=(12, 9))

    for class_name in class_names:
        mask = label_names == class_name

        if not np.any(mask):
            continue

        plt.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            s=18,
            alpha=0.70,
            color=class_colors[class_name],
            label=class_name,
        )

    plt.title(
        "Standalone Before FedProto - t-SNE\n"
        f"Client 1 | {len(class_names)} Classes | "
        f"Train Seed={SEED}"
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        TSNE_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------- UMAP ----------------

    if UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=UMAP_SEED,
        )

        umap_2d = reducer.fit_transform(
            embeddings
        )

        pd.DataFrame({
            "umap_1": umap_2d[:, 0],
            "umap_2": umap_2d[:, 1],
            "label_id": y_vis,
            "label_name": label_names,
        }).to_csv(
            UMAP_COORDINATES_CSV,
            index=False,
        )

        plt.figure(figsize=(12, 9))

        for class_name in class_names:
            mask = label_names == class_name

            if not np.any(mask):
                continue

            plt.scatter(
                umap_2d[mask, 0],
                umap_2d[mask, 1],
                s=18,
                alpha=0.70,
                color=class_colors[class_name],
                label=class_name,
            )

        plt.title(
            "Standalone Before FedProto - UMAP\n"
            f"Client 1 | {len(class_names)} Classes | "
            f"Train Seed={SEED}"
        )

        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
            frameon=True,
        )

        plt.tight_layout()

        plt.savefig(
            UMAP_PNG,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# Main
# ============================================================

def main():
    local_to_global, local_to_semantic = load_mapping()

    (
        X_all,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        num_classes,
    ) = load_data(
        local_to_semantic
    )

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weight = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            classes,
            weights,
        )
    }

    class_weight_tensor = tf.constant(
        [
            class_weight[class_id]
            for class_id in range(num_classes)
        ],
        dtype=tf.float32,
    )

    model = build_standalone_model(
        (X_train.shape[1], 1),
        num_classes,
    )

    print("\n" + "=" * 80)
    print("STANDALONE CLIENT 1 MATCHED TO FEDPROTO")
    print("=" * 80)

    model.summary()

    print(f"\nDataset:       {CSV_PATH}")
    print(f"Rows:          {len(X_all):,}")
    print(f"Features:      {X_all.shape[1]}")
    print(f"Classes:       {num_classes}")
    print(f"Embedding dim: {PROTO_DIM}")
    print(
        f"Splits: train={len(y_train):,}, "
        f"val={len(y_val):,}, "
        f"test={len(y_test):,}"
    )

    print("\nClass mapping:")
    for local_id in sorted(local_to_semantic):
        print(
            f"  {local_id:2d} -> "
            f"{local_to_semantic[local_id]}"
        )

    total_train_time = train_standalone(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        class_weight_tensor,
    )

    y_pred, _, _ = predict(
        model,
        X_test,
    )

    final_metrics = calculate_metrics(
        y_test,
        y_pred,
    )

    print("\n" + "=" * 80)
    print("FINAL STANDALONE TEST RESULTS")
    print("=" * 80)

    for key, value in final_metrics.items():
        print(f"{key:<20}: {value:.6f}")

    pd.DataFrame([
        {
            "seed": SEED,
            **final_metrics,
            "total_train_time_sec": total_train_time,
        }
    ]).to_csv(
        FINAL_METRICS_CSV,
        index=False,
    )

    class_names = [
        local_to_semantic[i]
        for i in range(num_classes)
    ]

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report_dict
    ).transpose().to_csv(
        CLASSIFICATION_REPORT_CSV
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        CONFUSION_MATRIX_CSV
    )

    row_sums = cm.sum(
        axis=1,
        keepdims=True,
    )

    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=np.float64,
        ),
        where=row_sums != 0,
    )

    pd.DataFrame(
        cm_norm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        CONFUSION_MATRIX_NORMALIZED_CSV
    )

    # Confusion matrix images
    size = 12

    plt.figure(
        figsize=(size, size)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
        aspect="auto",
    )

    plt.title(
        "Standalone Before FedProto - Raw Confusion Matrix"
    )

    plt.colorbar()

    ticks = np.arange(num_classes)

    plt.xticks(
        ticks,
        class_names,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        ticks,
        class_names,
        fontsize=8,
    )

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(size, size)
    )

    plt.imshow(
        cm_norm,
        interpolation="nearest",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    plt.title(
        "Standalone Before FedProto - Row-Normalized Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        ticks,
        class_names,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        ticks,
        class_names,
        fontsize=8,
    )

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_NORMALIZED_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Save sample-level 8-D embeddings + t-SNE + UMAP
    save_embedding_outputs(
        model,
        X_test,
        y_test,
        local_to_semantic,
    )

    model.save(
        MODEL_PATH
    )

    with SCALER_PATH.open("wb") as f:
        pickle.dump(
            scaler,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nSaved outputs:")

    outputs = [
        EPOCH_LOG_CSV,
        FINAL_METRICS_CSV,
        CLASSIFICATION_REPORT_CSV,
        CONFUSION_MATRIX_CSV,
        CONFUSION_MATRIX_NORMALIZED_CSV,
        CONFUSION_MATRIX_PNG,
        CONFUSION_MATRIX_NORMALIZED_PNG,
        MODEL_PATH,
        SCALER_PATH,
        EMBEDDINGS_CSV,
        TSNE_COORDINATES_CSV,
        TSNE_PNG,
    ]

    if UMAP_AVAILABLE:
        outputs.extend([
            UMAP_COORDINATES_CSV,
            UMAP_PNG,
        ])

    for path in outputs:
        print(path)

    if not UMAP_AVAILABLE:
        print(
            "\nUMAP was skipped because umap-learn is not installed."
        )


if __name__ == "__main__":
    main()