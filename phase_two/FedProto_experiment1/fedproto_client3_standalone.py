#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""
Standalone-before-FedProto Client 3 — UNSW-NB15
=================================================

Fair baseline for FedProto Client 3.

Matched settings:
- Seed: 190
- 20 training epochs
- Batch size: 512
- Adam, learning rate 1e-3, clipnorm 1.0
- 70/15/15 split
- Balanced class weights
- Same mapped FedProto dataset
- Same CNN architecture:
    Conv1D(64) -> BN -> Pool
    Conv1D(128) -> BN -> Pool
    GlobalAveragePooling
    Dense(16)
    Dense(8) standalone embedding
    Local classifier

NO:
- federation
- gRPC
- server
- prototype communication
- global prototypes
- prototype loss

Outputs:
- Metrics log
- Final metrics
- Classification report
- Confusion matrices
- 8-D standalone embedding CSV
- t-SNE coordinates + PNG
- UMAP coordinates + PNG
"""

# ============================================================
# Imports
# ============================================================

import os
import csv
import json
import pickle
import random
from pathlib import Path
from datetime import datetime

# ============================================================
# Reproducibility
# ============================================================

SEED = 123

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import pandas as pd

# ============================================================
# Matplotlib
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ============================================================
# Scikit-learn
# ============================================================

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

# ============================================================
# UMAP
# ============================================================

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ============================================================
# Keras
# ============================================================

from tensorflow.keras import layers, models, optimizers


# ============================================================
# Experiment configuration
# ============================================================

CLIENT_ID = "client3"

EPOCHS = 20

BATCH_SIZE = 512

LEARNING_RATE = 0.001

CLIP_NORM = 1.0

PRIVATE_DIM = 16

EMBEDDING_DIM = 8


# ============================================================
# Dataset split
# ============================================================

TEST_SIZE = 0.15

VAL_SIZE_FROM_TRAIN = (
    0.15
    /
    (1.0 - TEST_SIZE)
)


# ============================================================
# Visualization
# ============================================================

VISUALIZE_EMBEDDING = True

VIS_MAX_SAMPLES = 5000

# Fixed visualization settings for fair comparison
TSNE_SEED = 123
UMAP_SEED = 123

CLASS_ORDER = [
    "DOS",
    "EXPLOITS",
    "FUZZERS",
    "GENERIC",
    "BENIGN",
    "RECONNAISSANCE",
]

CLASS_COLORS = {
    "DOS": "#bcbd22",
    "EXPLOITS": "#2ca02c",
    "FUZZERS": "#9467bd",
    "GENERIC": "#17becf",
    "BENIGN": "#1f77b4",
    "RECONNAISSANCE": "#e377c2",
}

def normalize_label_name(label):
    label = str(label).strip().upper()
    mapping = {
        "NORMAL": "BENIGN",
        "BENIGN": "BENIGN",
        "DOS": "DOS",
        "EXPLOITS": "EXPLOITS",
        "FUZZERS": "FUZZERS",
        "FUZZER": "FUZZERS",
        "GENERIC": "GENERIC",
        "RECONNAISSANCE": "RECONNAISSANCE",
        "RECON": "RECONNAISSANCE",
    }
    return mapping.get(label, label)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "M_PFTL_codes"
)


# IMPORTANT:
# Use SAME dataset as FedProto Client 3
CSV_PATH = (
    BASE_DIR
    / "Multi_class_Datasets"
    / "D3_UNSW-NB15_FedProto_mapped.csv"
)


GLOBAL_MAPPING_PATH = (
    BASE_DIR
    / "FedProto_Global_Mapping"
    / "client_local_to_global_ids.json"
)


LOCAL_LABEL_COL = "local_label_id"

SEMANTIC_LABEL_COL = "semantic_label"

ORIGINAL_LABEL_COL = "original_label"


# ============================================================
# Standalone output directory
# ============================================================

OUT_DIR = (
    BASE_DIR
    / "standalone_before_fedproto_logs"
    / "client3_unsw_nb15"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Output files
# ============================================================

METRICS_LOG = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_metrics_log.csv"
)


FINAL_METRICS_CSV = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_final_metrics.csv"
)


CLASSIFICATION_REPORT_CSV = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_classification_report.csv"
)


CONFUSION_MATRIX_CSV = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_confusion_matrix.csv"
)


CONFUSION_MATRIX_NORMALIZED_CSV = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_confusion_matrix_normalized.csv"
)


CONFUSION_MATRIX_PNG = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_confusion_matrix_raw.png"
)


CONFUSION_MATRIX_NORMALIZED_PNG = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_confusion_matrix_normalized.png"
)


MODEL_PATH = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_final_model.keras"
)


SCALER_PATH = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_standard_scaler.pkl"
)


# ============================================================
# Embedding files
# ============================================================

EMBEDDINGS_CSV = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_before_fedproto_embeddings_seed{SEED}.csv"
)


TSNE_COORDINATES_CSV = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_before_fedproto_TSNE_coordinates_seed{SEED}.csv"
)


TSNE_PNG = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_before_fedproto_embedding_TSNE_seed{SEED}.png"
)


UMAP_COORDINATES_CSV = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_before_fedproto_UMAP_coordinates_seed{SEED}.csv"
)


UMAP_PNG = (
    OUT_DIR
    / f"{CLIENT_ID}_standalone_before_fedproto_embedding_UMAP_seed{SEED}.png"
)


# ============================================================
# Utility
# ============================================================

def now_str():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    acc = float(
        accuracy_score(
            y_true,
            y_pred
        )
    )


    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        average="macro",

        zero_division=0
    )


    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        average="weighted",

        zero_division=0
    )


    return {

        "accuracy":
            acc,

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "weighted_precision":
            float(weighted_precision),

        "weighted_recall":
            float(weighted_recall),

        "weighted_f1":
            float(weighted_f1),
    }


# ============================================================
# Load Client 3 local/global semantic mapping
# ============================================================

def load_mapping():

    if not GLOBAL_MAPPING_PATH.exists():

        raise FileNotFoundError(

            f"Mapping file not found:\n"
            f"{GLOBAL_MAPPING_PATH}"
        )


    with GLOBAL_MAPPING_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        all_mappings = json.load(
            file
        )


    if CLIENT_ID not in all_mappings:

        raise KeyError(

            f"{CLIENT_ID} missing from "
            f"{GLOBAL_MAPPING_PATH}"
        )


    client_mapping = all_mappings[
        CLIENT_ID
    ]


    raw_mapping = client_mapping.get(
        "local_to_global"
    )


    if not isinstance(
        raw_mapping,
        dict
    ):

        raise KeyError(
            "'local_to_global' missing."
        )


    local_to_semantic = {}


    for local_id_text, entry in raw_mapping.items():

        local_id = int(
            local_id_text
        )

        local_to_semantic[
            local_id
        ] = str(
            entry["semantic_label"]
        )


    print(
        "\n=========================================="
    )

    print(
        "CLIENT 3 LABEL MAPPING"
    )

    print(
        "=========================================="
    )


    for local_id in sorted(
        local_to_semantic
    ):

        print(
            f"{local_id:2d} -> "
            f"{local_to_semantic[local_id]}"
        )


    return local_to_semantic


# ============================================================
# Load dataset
# ============================================================

def load_data(
    local_to_semantic
):

    if not CSV_PATH.exists():

        raise FileNotFoundError(

            f"Dataset not found:\n"
            f"{CSV_PATH}"
        )


    print(
        "\n=========================================="
    )

    print(
        "LOADING STANDALONE-BEFORE-FEDPROTO DATA"
    )

    print(
        "=========================================="
    )


    print(
        CSV_PATH
    )


    df = pd.read_csv(

        CSV_PATH,

        low_memory=False
    )


    df.columns = (

        df.columns

        .astype(str)

        .str.replace(
            "\ufeff",
            "",
            regex=False
        )

        .str.strip()
    )


    # ========================================================
    # Validate required columns
    # ========================================================

    required = {
        LOCAL_LABEL_COL,
        SEMANTIC_LABEL_COL
    }


    missing = (
        required
        -
        set(df.columns)
    )


    if missing:

        raise ValueError(

            f"Dataset missing columns: "
            f"{sorted(missing)}"
        )


    df = df.dropna(

        subset=[
            LOCAL_LABEL_COL,
            SEMANTIC_LABEL_COL
        ]

    ).copy()


    # ========================================================
    # Labels
    # ========================================================

    y = pd.to_numeric(

        df[
            LOCAL_LABEL_COL
        ],

        errors="raise"

    ).astype(
        np.int32
    ).to_numpy()


    semantic_values = (

        df[
            SEMANTIC_LABEL_COL
        ]

        .astype(str)

        .str.strip()

        .to_numpy()
    )


    num_classes = len(
        local_to_semantic
    )


    # ========================================================
    # Validate IDs
    # ========================================================

    expected_ids = set(
        range(
            num_classes
        )
    )


    observed_ids = {
        int(v)
        for v in np.unique(y)
    }


    if observed_ids != expected_ids:

        raise ValueError(

            "Labels are not contiguous.\n"

            f"Expected: {sorted(expected_ids)}\n"

            f"Observed: {sorted(observed_ids)}"
        )


    # ========================================================
    # Verify semantic mapping
    # ========================================================

    for local_id in sorted(
        observed_ids
    ):

        expected_semantic = (
            local_to_semantic[
                local_id
            ]
        )


        observed_semantic = set(

            semantic_values[
                y == local_id
            ]
        )


        if observed_semantic != {
            expected_semantic
        }:

            raise ValueError(

                f"Semantic mismatch "
                f"for local ID {local_id}.\n"

                f"Expected: "
                f"{expected_semantic}\n"

                f"Observed: "
                f"{observed_semantic}"
            )


    # ========================================================
    # Features
    # ========================================================

    excluded_columns = {

        LOCAL_LABEL_COL,

        SEMANTIC_LABEL_COL,

        ORIGINAL_LABEL_COL,

        "Label",

        "Attack",

        "binary_label",
    }


    feature_df = (

        df

        .drop(

            columns=[
                column
                for column
                in excluded_columns
                if column in df.columns
            ],

            errors="ignore"
        )

        .select_dtypes(
            include=[np.number]
        )

        .copy()
    )


    feature_df = (

        feature_df

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .fillna(
            0.0
        )
    )


    if feature_df.shape[1] == 0:

        raise ValueError(
            "No numeric features remained."
        )


    X = feature_df.to_numpy(
        dtype=np.float32
    )


    print(
        f"\nRows:     {len(X):,}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Classes:  {num_classes}"
    )


    # ========================================================
    # SAME 70 / 15 / 15 split as FedProto
    # ========================================================

    indices = np.arange(
        len(y),
        dtype=np.int64
    )


    (
        train_idx,
        test_idx
    ) = train_test_split(

        indices,

        test_size=TEST_SIZE,

        random_state=SEED,

        stratify=y
    )


    (
        train_idx,
        val_idx
    ) = train_test_split(

        train_idx,

        test_size=VAL_SIZE_FROM_TRAIN,

        random_state=SEED,

        stratify=y[
            train_idx
        ]
    )


    X_train = X[
        train_idx
    ]

    X_val = X[
        val_idx
    ]

    X_test = X[
        test_idx
    ]


    y_train = y[
        train_idx
    ]

    y_val = y[
        val_idx
    ]

    y_test = y[
        test_idx
    ]


    # ========================================================
    # Same StandardScaler
    # ========================================================

    scaler = StandardScaler()


    X_train = scaler.fit_transform(
        X_train
    )


    X_val = scaler.transform(
        X_val
    )


    X_test = scaler.transform(
        X_test
    )


    # Conv1D format
    X_train = X_train[
        ...,
        np.newaxis
    ].astype(
        np.float32
    )


    X_val = X_val[
        ...,
        np.newaxis
    ].astype(
        np.float32
    )


    X_test = X_test[
        ...,
        np.newaxis
    ].astype(
        np.float32
    )


    y_train = y_train.astype(
        np.int32
    )

    y_val = y_val.astype(
        np.int32
    )

    y_test = y_test.astype(
        np.int32
    )


    # ========================================================
    # Balanced class weights
    # ========================================================

    classes = np.unique(
        y_train
    )


    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y_train
    )


    class_weight = {

        int(class_id):
            float(weight)

        for class_id, weight
        in zip(
            classes,
            weights
        )
    }


    print(
        "\nSplits:"
    )

    print(
        f"Train      : "
        f"{len(y_train):,}"
    )

    print(
        f"Validation : "
        f"{len(y_val):,}"
    )

    print(
        f"Test       : "
        f"{len(y_test):,}"
    )


    print(
        "\nTraining class counts / weights:"
    )


    train_counts = (

        pd.Series(
            y_train
        )

        .value_counts()

        .sort_index()
    )


    for local_id in range(
        num_classes
    ):

        print(

            f"{local_id:2d} "

            f"{local_to_semantic[local_id]:<25} "

            f"count="
            f"{int(train_counts.get(local_id, 0)):<8d} "

            f"weight="
            f"{class_weight[local_id]:.6f}"
        )


    return (

        X_train,

        X_val,

        X_test,

        y_train,

        y_val,

        y_test,

        scaler,

        class_weight,

        num_classes,
    )


# ============================================================
# Standalone Model
#
# SAME feature extractor as FedProto.
# Only prototype communication/loss is removed.
# ============================================================

def build_standalone_model(
    input_shape,
    num_classes
):

    inputs = layers.Input(

        shape=input_shape,

        name="input"
    )


    # --------------------------------------------------------
    # Same private feature extractor
    # --------------------------------------------------------

    x = layers.Conv1D(

        64,

        3,

        padding="valid",

        activation="relu",

        name="private_conv1"

    )(inputs)


    x = layers.BatchNormalization(

        name="private_bn1"

    )(x)


    x = layers.MaxPooling1D(

        2,

        name="private_pool1"

    )(x)


    x = layers.Conv1D(

        128,

        3,

        padding="valid",

        activation="relu",

        name="private_conv2"

    )(x)


    x = layers.BatchNormalization(

        name="private_bn2"

    )(x)


    x = layers.MaxPooling1D(

        2,

        name="private_pool2"

    )(x)


    x = layers.GlobalAveragePooling1D(

        name="private_global_average_pool"

    )(x)


    x = layers.Dense(

        PRIVATE_DIM,

        activation="relu",

        name="private_dense"

    )(x)


    # --------------------------------------------------------
    # Same 8-D location as FedProto prototype_embedding
    #
    # But this is purely LOCAL / STANDALONE.
    # No prototype regularization.
    # --------------------------------------------------------

    embedding = layers.Dense(

        EMBEDDING_DIM,

        activation="relu",

        name="standalone_embedding"

    )(x)


    # --------------------------------------------------------
    # Local classifier
    # --------------------------------------------------------

    logits = layers.Dense(

        num_classes,

        activation=None,

        name="local_classifier_logits"

    )(embedding)


    model = models.Model(

        inputs=inputs,

        outputs=logits,

        name=(
            f"Standalone_Before_FedProto_"
            f"{CLIENT_ID}"
        )
    )


    model.compile(

        optimizer=optimizers.Adam(

            learning_rate=(
                LEARNING_RATE
            ),

            clipnorm=(
                CLIP_NORM
            )
        ),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),

        metrics=[
            "accuracy"
        ]
    )


    return model


# ============================================================
# Predict
# ============================================================

def predict(
    model,
    X
):

    logits = model.predict(

        X,

        batch_size=BATCH_SIZE,

        verbose=0
    )


    probabilities = tf.nn.softmax(

        logits,

        axis=1

    ).numpy()


    predictions = np.argmax(

        probabilities,

        axis=1

    ).astype(
        np.int32
    )


    return (
        predictions,
        probabilities
    )


# ============================================================
# Save confusion matrices
# ============================================================

def save_confusion_matrices(
    y_true,
    y_pred,
    class_names
):

    labels = np.arange(
        len(class_names)
    )


    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=labels
    )


    pd.DataFrame(

        matrix,

        index=class_names,

        columns=class_names

    ).to_csv(
        CONFUSION_MATRIX_CSV
    )


    row_sums = matrix.sum(

        axis=1,

        keepdims=True
    )


    normalized = np.divide(

        matrix,

        row_sums,

        out=np.zeros_like(
            matrix,
            dtype=np.float64
        ),

        where=row_sums != 0
    )


    pd.DataFrame(

        normalized,

        index=class_names,

        columns=class_names

    ).to_csv(
        CONFUSION_MATRIX_NORMALIZED_CSV
    )


    # Raw matrix
    plt.figure(
        figsize=(10, 8)
    )


    plt.imshow(

        matrix,

        interpolation="nearest",

        aspect="auto"
    )


    plt.title(
        "Standalone-before-FedProto Client 3\n"
        "Raw Confusion Matrix"
    )


    plt.colorbar()


    ticks = np.arange(
        len(class_names)
    )


    plt.xticks(

        ticks,

        class_names,

        rotation=45,

        ha="right"
    )


    plt.yticks(

        ticks,

        class_names
    )


    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )


    plt.tight_layout()


    plt.savefig(

        CONFUSION_MATRIX_PNG,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    # Normalized
    plt.figure(
        figsize=(10, 8)
    )


    plt.imshow(

        normalized,

        interpolation="nearest",

        aspect="auto",

        vmin=0.0,

        vmax=1.0
    )


    plt.title(
        "Standalone-before-FedProto Client 3\n"
        "Normalized Confusion Matrix"
    )


    plt.colorbar()


    plt.xticks(

        ticks,

        class_names,

        rotation=45,

        ha="right"
    )


    plt.yticks(

        ticks,

        class_names
    )


    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )


    plt.tight_layout()


    plt.savefig(

        CONFUSION_MATRIX_NORMALIZED_PNG,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# Save + visualize standalone embeddings
# ============================================================

def visualize_standalone_embedding(
    model,
    X_test,
    y_test,
    local_to_semantic,
    max_samples=VIS_MAX_SAMPLES
):

    print(
        "\n=========================================="
    )

    print(
        "STANDALONE-BEFORE-FEDPROTO "
        "EMBEDDING VISUALIZATION"
    )

    print(
        "=========================================="
    )


    X_vis = X_test

    y_vis = y_test


    print(
        f"Original test samples: "
        f"{len(X_vis)}"
    )


    # ========================================================
    # Same 5000 stratified samples as FedProto
    # ========================================================

    if len(X_vis) > max_samples:

        indices = np.arange(
            len(y_vis)
        )


        (
            selected_idx,
            _
        ) = train_test_split(

            indices,

            train_size=max_samples,

            random_state=SEED,

            stratify=y_vis
        )


        X_vis = X_vis[
            selected_idx
        ]


        y_vis = y_vis[
            selected_idx
        ]


    print(
        f"Visualization samples: "
        f"{len(X_vis)}"
    )


    # ========================================================
    # Intermediate model for 8-D embedding
    # ========================================================

    embedding_model = models.Model(

        inputs=model.input,

        outputs=model.get_layer(
            "standalone_embedding"
        ).output
    )


    embeddings = embedding_model.predict(

        X_vis,

        batch_size=1024,

        verbose=0
    )


    embeddings = np.asarray(

        embeddings,

        dtype=np.float32
    )


    print(
        "Standalone embedding shape:",
        embeddings.shape
    )


    if (
        embeddings.ndim != 2
        or
        embeddings.shape[1]
        != EMBEDDING_DIM
    ):

        raise ValueError(

            f"Expected embedding "
            f"(N, {EMBEDDING_DIM}), "

            f"got {embeddings.shape}"
        )


    # ========================================================
    # Semantic labels
    # ========================================================

    label_names = np.asarray(
        [
            normalize_label_name(
                local_to_semantic[int(label_id)]
            )
            for label_id in y_vis
        ],
        dtype=object
    )

    print("\nVisualization labels:")
    for name, count in zip(*np.unique(label_names, return_counts=True)):
        print(f"  {name:<18}: {count}")


    # ========================================================
    # Save original 8-D embedding
    # ========================================================

    embedding_df = pd.DataFrame(

        embeddings,

        columns=[
            f"standalone_dim_{i + 1}"

            for i
            in range(
                embeddings.shape[1]
            )
        ]
    )


    embedding_df[
        "label_id"
    ] = y_vis


    embedding_df[
        "label_name"
    ] = label_names


    embedding_df.to_csv(

        EMBEDDINGS_CSV,

        index=False
    )


    print(
        "\nStandalone 8-D embedding saved:"
    )

    print(
        EMBEDDINGS_CSV
    )


    # ========================================================
    # Plot setup
    # ========================================================

    # Fixed semantic colors/order; do not use automatic colormap.
# ========================================================
    # t-SNE
    # ========================================================

    print(
        "\nRunning t-SNE..."
    )


    if len(embeddings) > 31:

        perplexity = 30

    else:

        perplexity = max(

            5,

            min(
                30,
                len(embeddings) - 1
            )
        )


    tsne_model = TSNE(

        n_components=2,

        perplexity=perplexity,

        init="pca",

        learning_rate="auto",

        max_iter=1000,

        random_state=TSNE_SEED
    )


    tsne_2d = tsne_model.fit_transform(
        embeddings
    )


    pd.DataFrame({

        "tsne_1":
            tsne_2d[:, 0],

        "tsne_2":
            tsne_2d[:, 1],

        "label_id":
            y_vis,

        "label_name":
            label_names

    }).to_csv(

        TSNE_COORDINATES_CSV,

        index=False
    )


    plt.figure(
        figsize=(10, 8)
    )


    for class_name in CLASS_ORDER:

        mask = (label_names == class_name)

        if not np.any(mask):
            continue

        plt.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            s=18,
            alpha=0.70,
            color=CLASS_COLORS[class_name],
            label=class_name
        )


    plt.title(

        "Standalone-before-FedProto "
        "Embedding - t-SNE\n"

        f"{CLIENT_ID} | "
        f"{len(local_to_semantic)} Classes | "
        f"Train Seed={SEED} | t-SNE Seed={TSNE_SEED}"
    )


    plt.xlabel(
        "t-SNE Dimension 1"
    )


    plt.ylabel(
        "t-SNE Dimension 2"
    )


    plt.legend(

        bbox_to_anchor=(
            1.05,
            1
        ),

        loc="upper left",

        fontsize=8,

        frameon=True
    )


    plt.tight_layout()


    plt.savefig(

        TSNE_PNG,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    print(
        "t-SNE saved:"
    )

    print(
        TSNE_PNG
    )


    # ========================================================
    # UMAP
    # ========================================================

    if UMAP_AVAILABLE:

        print(
            "\nRunning UMAP..."
        )


        umap_model = umap.UMAP(

            n_components=2,

            n_neighbors=15,

            min_dist=0.1,

            metric="euclidean",

            random_state=UMAP_SEED
        )


        umap_2d = (
            umap_model.fit_transform(
                embeddings
            )
        )


        pd.DataFrame({

            "umap_1":
                umap_2d[:, 0],

            "umap_2":
                umap_2d[:, 1],

            "label_id":
                y_vis,

            "label_name":
                label_names

        }).to_csv(

            UMAP_COORDINATES_CSV,

            index=False
        )


        plt.figure(
            figsize=(10, 8)
        )


        for class_name in CLASS_ORDER:

            mask = (label_names == class_name)

            if not np.any(mask):
                continue

            plt.scatter(
                umap_2d[mask, 0],
                umap_2d[mask, 1],
                s=18,
                alpha=0.70,
                color=CLASS_COLORS[class_name],
                label=class_name
            )


        plt.title(

            "Standalone-before-FedProto "
            "Embedding - UMAP\n"

            f"{CLIENT_ID} | "
            f"{len(local_to_semantic)} Classes | "
            f"Train Seed={SEED} | UMAP Seed={UMAP_SEED}"
        )


        plt.xlabel(
            "UMAP Dimension 1"
        )


        plt.ylabel(
            "UMAP Dimension 2"
        )


        plt.legend(

            bbox_to_anchor=(
                1.05,
                1
            ),

            loc="upper left",

            fontsize=8,

            frameon=True
        )


        plt.tight_layout()


        plt.savefig(

            UMAP_PNG,

            dpi=300,

            bbox_inches="tight"
        )


        plt.close()


        print(
            "UMAP saved:"
        )

        print(
            UMAP_PNG
        )


    else:

        print(
            "\nUMAP not installed."
        )

        print(
            "Install with: "
            "pip install umap-learn"
        )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Mapping
    # ========================================================

    local_to_semantic = (
        load_mapping()
    )


    # ========================================================
    # Data
    # ========================================================

    (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        scaler,

        class_weight,

        num_classes

    ) = load_data(
        local_to_semantic
    )


    # ========================================================
    # Model
    # ========================================================

    model = build_standalone_model(

        input_shape=(
            X_train.shape[1],
            1
        ),

        num_classes=num_classes
    )


    print(
        "\n=========================================="
    )

    print(
        "STANDALONE-BEFORE-FEDPROTO CLIENT 3 MODEL"
    )

    print(
        "=========================================="
    )


    model.summary()


    # ========================================================
    # Metrics log header
    # ========================================================

    if not METRICS_LOG.exists():

        with METRICS_LOG.open(
            "w",
            newline=""
        ) as file:

            csv.writer(
                file
            ).writerow([
                "seed",
                "epoch",

                "train_loss",
                "train_accuracy",

                "validation_loss",
                "validation_accuracy",

                "validation_macro_precision",
                "validation_macro_recall",
                "validation_macro_f1",

                "timestamp",
            ])


    # ========================================================
    # Training
    #
    # 20 epochs = matched to 20 FedProto local rounds,
    # each with one local epoch.
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "TRAINING STANDALONE-BEFORE-FEDPROTO"
    )

    print(
        "=========================================="
    )


    for epoch in range(
        1,
        EPOCHS + 1
    ):


        print(
            f"\n===== Epoch "
            f"{epoch}/{EPOCHS} ====="
        )


        history = model.fit(

            X_train,

            y_train,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=1,

            batch_size=BATCH_SIZE,

            class_weight=class_weight,

            shuffle=True,

            verbose=1
        )


        # ====================================================
        # Validation Macro-F1
        # ====================================================

        y_val_pred, _ = predict(

            model,

            X_val
        )


        val_metrics = calculate_metrics(

            y_val,

            y_val_pred
        )


        train_loss = float(
            history.history[
                "loss"
            ][0]
        )


        train_accuracy = float(
            history.history[
                "accuracy"
            ][0]
        )


        val_loss = float(
            history.history[
                "val_loss"
            ][0]
        )


        val_accuracy = float(
            history.history[
                "val_accuracy"
            ][0]
        )


        print(

            f"Validation Macro-F1: "
            f"{val_metrics['macro_f1']:.6f}"
        )


        with METRICS_LOG.open(

            "a",

            newline=""

        ) as file:


            csv.writer(
                file
            ).writerow([

                SEED,

                epoch,

                f"{train_loss:.8f}",

                f"{train_accuracy:.8f}",

                f"{val_loss:.8f}",

                f"{val_accuracy:.8f}",

                f"{val_metrics['macro_precision']:.8f}",

                f"{val_metrics['macro_recall']:.8f}",

                f"{val_metrics['macro_f1']:.8f}",

                now_str(),
            ])


    # ========================================================
    # Final Test Evaluation
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=========================================="
    )


    y_test_pred, _ = predict(

        model,

        X_test
    )


    final_metrics = calculate_metrics(

        y_test,

        y_test_pred
    )


    print(
        f"Accuracy        : "
        f"{final_metrics['accuracy']:.6f}"
    )


    print(
        f"Macro-Precision : "
        f"{final_metrics['macro_precision']:.6f}"
    )


    print(
        f"Macro-Recall    : "
        f"{final_metrics['macro_recall']:.6f}"
    )


    print(
        f"Macro-F1        : "
        f"{final_metrics['macro_f1']:.6f}"
    )


    print(
        f"Weighted-F1     : "
        f"{final_metrics['weighted_f1']:.6f}"
    )


    # ========================================================
    # Save final metrics
    # ========================================================

    pd.DataFrame([
        {
            "seed":
                SEED,

            **final_metrics
        }

    ]).to_csv(

        FINAL_METRICS_CSV,

        index=False
    )


    # ========================================================
    # Class names
    # ========================================================

    class_names = [

        local_to_semantic[
            i
        ]

        for i
        in range(
            num_classes
        )
    ]


    # ========================================================
    # Classification report
    # ========================================================

    print(
        "\n===== Classification Report ====="
    )


    report = classification_report(

        y_test,

        y_test_pred,

        target_names=class_names,

        zero_division=0,

        output_dict=True
    )


    print(

        classification_report(

            y_test,

            y_test_pred,

            target_names=class_names,

            zero_division=0
        )
    )


    pd.DataFrame(

        report

    ).transpose().to_csv(

        CLASSIFICATION_REPORT_CSV
    )


    # ========================================================
    # Confusion matrices
    # ========================================================

    save_confusion_matrices(

        y_test,

        y_test_pred,

        class_names
    )


    # ========================================================
    # Save model
    # ========================================================

    model.save(
        MODEL_PATH
    )


    with SCALER_PATH.open(
        "wb"
    ) as file:

        pickle.dump(
            scaler,
            file
        )


    # ========================================================
    # Embedding visualization
    # ========================================================

    if VISUALIZE_EMBEDDING:

        visualize_standalone_embedding(

            model=model,

            X_test=X_test,

            y_test=y_test,

            local_to_semantic=local_to_semantic,

            max_samples=VIS_MAX_SAMPLES
        )


    # ========================================================
    # Final paths
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "STANDALONE-BEFORE-FEDPROTO COMPLETE"
    )

    print(
        "=========================================="
    )


    print(
        "\n8-D embeddings:"
    )

    print(
        EMBEDDINGS_CSV
    )


    print(
        "\nt-SNE:"
    )

    print(
        TSNE_PNG
    )


    if UMAP_AVAILABLE:

        print(
            "\nUMAP:"
        )

        print(
            UMAP_PNG
        )


    print(
        "\nFinal metrics:"
    )

    print(
        FINAL_METRICS_CSV
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()