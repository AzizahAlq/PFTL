#!/usr/bin/env python3.10

# ============================================================
# Standalone Client 3 - UNSW-NB15
# + Save 8-D shared embeddings
# + t-SNE visualization
# + UMAP visualization
# ============================================================


# ============================================================
# Imports
# ============================================================

import os
import csv
import random
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# TensorFlow
# ============================================================

import tensorflow as tf

tf.random.set_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ============================================================
# Scikit-learn
# ============================================================

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from sklearn.manifold import TSNE


# ============================================================
# Matplotlib
# ============================================================

import matplotlib

# Safe for NFS/Linux server without display
matplotlib.use("Agg")

import matplotlib.pyplot as plt


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

from tensorflow.keras import (
    layers,
    models,
    optimizers,
)


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/Multi_class_Datasets/"
    "D3_UNSW_NB15_ALL_MERGED_final.csv"
)

TARGET_COL = "attack_cat"


OUT_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/Standalone/"
    "standalone_logs/client3_unsw"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# Output paths
# ============================================================

METRICS_LOG = os.path.join(
    OUT_DIR,
    "standalone_metrics_log.csv"
)


EMBEDDING_CSV = os.path.join(
    OUT_DIR,
    f"client3_standalone_shared_embeddings_seed{SEED}.csv"
)


TSNE_COORDINATES_CSV = os.path.join(
    OUT_DIR,
    f"client3_standalone_TSNE_coordinates_seed{SEED}.csv"
)


TSNE_PNG = os.path.join(
    OUT_DIR,
    f"client3_standalone_shared_representation_TSNE_seed{SEED}.png"
)


UMAP_COORDINATES_CSV = os.path.join(
    OUT_DIR,
    f"client3_standalone_UMAP_coordinates_seed{SEED}.csv"
)


UMAP_PNG = os.path.join(
    OUT_DIR,
    f"client3_standalone_shared_representation_UMAP_seed{SEED}.png"
)


# ============================================================
# Training settings
# ============================================================

EPOCHS = 20

BATCH_SIZE = 512

LR = 1e-3


# ============================================================
# Split configuration
#
# 70% train
# 15% validation
# 15% test
# ============================================================

TEST_SIZE = 0.15

VAL_SIZE_FROM_TRAIN = (
    0.15
    /
    (1.0 - TEST_SIZE)
)


# ============================================================
# Visualization settings
# ============================================================

VIS_MAX_SAMPLES = 5000

# ============================================================
# Fixed labels / colors / legend order
# Matches the PFTL/FedProto comparison figures.
#
# IMPORTANT:
# - Training labels are NOT changed.
# - Only visualization/output labels are standardized.
# - If the dataset uses "Normal", it is displayed as "BENIGN".
# ============================================================

CLASS_ORDER = [
    "DOS",
    "EXPLOITS",
    "FUZZERS",
    "GENERIC",
    "BENIGN",
    "RECONNAISSANCE",
]

CLASS_COLORS = {
    "DOS": "#bcbd22",             # olive/yellow
    "EXPLOITS": "#2ca02c",        # green
    "FUZZERS": "#9467bd",         # purple
    "GENERIC": "#17becf",         # cyan
    "BENIGN": "#1f77b4",          # blue
    "RECONNAISSANCE": "#e377c2",  # pink
}


def normalize_label_name(label):
    """Standardize display names without changing training labels."""
    label = str(label).strip().upper()

    mapping = {
        "NORMAL": "BENIGN",
        "BENIGN": "BENIGN",
        "DOS": "DOS",
        "DO S": "DOS",
        "EXPLOITS": "EXPLOITS",
        "FUZZERS": "FUZZERS",
        "FUZZER": "FUZZERS",
        "GENERIC": "GENERIC",
        "RECONNAISSANCE": "RECONNAISSANCE",
        "RECON": "RECONNAISSANCE",
    }

    return mapping.get(label, label)



# ============================================================
# Build Standalone CNN
# ============================================================

def build_cnn(
    input_shape,
    num_classes
):

    inp = layers.Input(
        shape=input_shape
    )


    # --------------------------------------------------------
    # CNN backbone
    # --------------------------------------------------------

    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="valid",
        activation="relu"
    )(inp)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=2
    )(x)


    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="valid",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=2
    )(x)


    x = layers.GlobalAveragePooling1D()(x)


    # --------------------------------------------------------
    # Private representation
    # --------------------------------------------------------

    x = layers.Dense(
        16,
        activation="relu",
        name="private_dense"
    )(x)


    # --------------------------------------------------------
    # Shared-style 8-D representation
    #
    # This is what we save and visualize.
    # --------------------------------------------------------

    x = layers.Dense(
        8,
        activation="relu",
        name="shared_dense"
    )(x)


    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier_output"
    )(x)


    model = models.Model(
        inputs=inp,
        outputs=out,
        name="Standalone_Client3_UNSW"
    )


    model.compile(

        optimizer=optimizers.Adam(
            learning_rate=LR
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ],
    )


    return model


# ============================================================
# Save + Visualize Embeddings
# ============================================================

def save_and_visualize_test_embeddings(
    model,
    X_test,
    y_test,
    label_encoder,
    max_samples=VIS_MAX_SAMPLES
):

    print(
        "\n=========================================="
    )

    print(
        "STANDALONE EMBEDDING VISUALIZATION"
    )

    print(
        "=========================================="
    )


    # ========================================================
    # 1. Use held-out TEST set
    # ========================================================

    X_vis = X_test

    y_vis = y_test


    print(
        f"Original test samples: "
        f"{len(X_vis)}"
    )


    print(
        f"Number of classes: "
        f"{len(label_encoder.classes_)}"
    )


    print(
        "Class names:"
    )

    print(
        list(
            label_encoder.classes_
        )
    )


    # ========================================================
    # 2. Stratified subsampling
    #
    # Match PFTL/FedProto visualization size.
    # ========================================================

    if len(X_vis) > max_samples:

        print(
            f"\nSubsampling test set to "
            f"{max_samples} samples..."
        )


        all_indices = np.arange(
            len(y_vis)
        )


        (
            selected_indices,
            _
        ) = train_test_split(

            all_indices,

            train_size=max_samples,

            random_state=SEED,

            stratify=y_vis
        )


        X_vis = X_vis[
            selected_indices
        ]


        y_vis = y_vis[
            selected_indices
        ]


    print(
        f"Visualization samples: "
        f"{len(X_vis)}"
    )


    # ========================================================
    # 3. Extract 8-D shared_dense representation
    # ========================================================

    embedding_model = models.Model(

        inputs=model.input,

        outputs=(
            model
            .get_layer(
                "shared_dense"
            )
            .output
        )
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
        "\nStandalone embedding shape:",
        embeddings.shape
    )


    if (
        embeddings.ndim != 2
        or
        embeddings.shape[1] != 8
    ):

        raise ValueError(

            "Expected embedding shape "
            f"(N, 8), got {embeddings.shape}"
        )


    # ========================================================
    # 4. Labels
    # ========================================================

    original_label_names = (
        label_encoder
        .inverse_transform(
            y_vis
        )
    )

    # Standardize display/output names:
    # Normal -> BENIGN, and all labels are uppercase.
    label_names = np.asarray(
        [
            normalize_label_name(name)
            for name in original_label_names
        ],
        dtype=object
    )

    print(
        "\nLabels in visualization:"
    )
    unique_display, display_counts = np.unique(
        label_names,
        return_counts=True
    )
    for name, count in zip(unique_display, display_counts):
        print(f"  {name:<18} : {count}")

    # Check that every expected class is represented.
    missing_classes = [
        class_name
        for class_name in CLASS_ORDER
        if class_name not in set(unique_display)
    ]

    if missing_classes:
        print(
            "\nWARNING: Expected classes missing from visualization:",
            missing_classes
        )


    # ========================================================
    # 5. Save original 8-D embeddings
    # ========================================================

    embedding_df = pd.DataFrame(

        embeddings,

        columns=[
            f"shared_dim_{i + 1}"
            for i in range(
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

        EMBEDDING_CSV,

        index=False
    )


    print(
        "\nRaw 8-D embeddings saved:"
    )

    print(
        EMBEDDING_CSV
    )


    # ========================================================
    # Plot setup
    # ========================================================

    # Plotting uses fixed semantic label names, colors, and order.
    # This ensures Standalone, PFTL, and FedProto use the same legend.


    # ========================================================
    # 6. t-SNE
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "Running t-SNE..."
    )

    print(
        "=========================================="
    )


    # Perplexity must be < number of samples
    if len(embeddings) > 31:

        tsne_perplexity = 30

    else:

        tsne_perplexity = max(

            5,

            min(
                30,
                len(embeddings) - 1
            )
        )


    tsne_model = TSNE(

        n_components=2,

        perplexity=tsne_perplexity,

        init="pca",

        learning_rate="auto",

        max_iter=1000,

        random_state=SEED
    )


    tsne_2d = (
        tsne_model
        .fit_transform(
            embeddings
        )
    )


    # --------------------------------------------------------
    # Save t-SNE coordinates
    # --------------------------------------------------------

    tsne_df = pd.DataFrame(
        {

            "tsne_1":
                tsne_2d[:, 0],

            "tsne_2":
                tsne_2d[:, 1],

            "label_id":
                y_vis,

            "label_name":
                label_names,
        }
    )


    tsne_df.to_csv(

        TSNE_COORDINATES_CSV,

        index=False
    )


    # --------------------------------------------------------
    # Plot t-SNE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )


    for class_name in CLASS_ORDER:

        mask = (
            label_names == class_name
        )

        # Skip only if the class is genuinely absent.
        if not np.any(mask):
            continue

        plt.scatter(

            tsne_2d[
                mask,
                0
            ],

            tsne_2d[
                mask,
                1
            ],

            s=18,

            alpha=0.70,

            color=CLASS_COLORS[
                class_name
            ],

            label=class_name
        )


    plt.title(

        f"Standalone Shared Representation - t-SNE\n"
        f"client3 | "
        f"{len(label_encoder.classes_)} Classes | "
        f"Seed={SEED}"
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
        "\nt-SNE figure saved:"
    )

    print(
        TSNE_PNG
    )


    print(
        "\nt-SNE coordinates saved:"
    )

    print(
        TSNE_COORDINATES_CSV
    )


    # ========================================================
    # 7. UMAP
    # ========================================================

    if UMAP_AVAILABLE:

        print(
            "\n=========================================="
        )

        print(
            "Running UMAP..."
        )

        print(
            "=========================================="
        )


        umap_model = umap.UMAP(

            n_components=2,

            n_neighbors=15,

            min_dist=0.1,

            metric="euclidean",

            random_state=SEED
        )


        umap_2d = (
            umap_model
            .fit_transform(
                embeddings
            )
        )


        # ----------------------------------------------------
        # Save UMAP coordinates
        # ----------------------------------------------------

        umap_df = pd.DataFrame(
            {

                "umap_1":
                    umap_2d[:, 0],

                "umap_2":
                    umap_2d[:, 1],

                "label_id":
                    y_vis,

                "label_name":
                    label_names,
            }
        )


        umap_df.to_csv(

            UMAP_COORDINATES_CSV,

            index=False
        )


        # ----------------------------------------------------
        # Plot UMAP
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 8)
        )


        for class_name in CLASS_ORDER:

            mask = (
                label_names == class_name
            )

            # Skip only if the class is genuinely absent.
            if not np.any(mask):
                continue

            plt.scatter(

                umap_2d[
                    mask,
                    0
                ],

                umap_2d[
                    mask,
                    1
                ],

                s=18,

                alpha=0.70,

                color=CLASS_COLORS[
                    class_name
                ],

                label=class_name
            )


        plt.title(

            f"Standalone Shared Representation - UMAP\n"
            f"client3 | "
            f"{len(label_encoder.classes_)} Classes | "
            f"Seed={SEED}"
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
            "\nUMAP figure saved:"
        )

        print(
            UMAP_PNG
        )


        print(
            "\nUMAP coordinates saved:"
        )

        print(
            UMAP_COORDINATES_CSV
        )


    else:

        print(
            "\n=========================================="
        )

        print(
            "UMAP NOT INSTALLED"
        )

        print(
            "=========================================="
        )

        print(
            "Install with:"
        )

        print(
            "pip install umap-learn"
        )


    print(
        "\n=========================================="
    )

    print(
        "STANDALONE VISUALIZATION COMPLETE"
    )

    print(
        "=========================================="
    )


# ============================================================
# Main
# ============================================================

def main():


    # ========================================================
    # Load dataset
    # ========================================================

    print(
        "\n===== LOADING CLIENT 3 DATASET ====="
    )


    print(
        "Dataset:"
    )

    print(
        CSV_PATH
    )


    df = pd.read_csv(

        CSV_PATH,

        low_memory=False
    )


    # ========================================================
    # Clean column names
    # ========================================================

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


    if TARGET_COL not in df.columns:

        raise ValueError(

            f"Target column "
            f"'{TARGET_COL}' not found.\n"

            f"Available columns:\n"
            f"{list(df.columns)}"
        )


    # ========================================================
    # Clean infinities
    # ========================================================

    df = df.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan
    )


    # ========================================================
    # Labels
    # ========================================================

    y_raw = (

        df[
            TARGET_COL
        ]

        .astype(str)

        .str.strip()

        .values
    )


    # ========================================================
    # Features
    # ========================================================

    X_df = (

        df

        .drop(

            columns=[
                TARGET_COL,
                "id",
                "label"
            ],

            errors="ignore"
        )

        .select_dtypes(
            include=[
                np.number
            ]
        )

        .fillna(
            0.0
        )
    )


    X = (
        X_df
        .values
        .astype(
            np.float32
        )
    )


    # ========================================================
    # Label encoding
    # ========================================================

    le = LabelEncoder()


    y = (

        le

        .fit_transform(
            y_raw
        )

        .astype(
            int
        )
    )


    num_classes = int(

        len(
            le.classes_
        )
    )


    print(
        "\n===== DATASET SUMMARY ====="
    )


    print(
        f"Samples  : "
        f"{X.shape[0]}"
    )


    print(
        f"Features : "
        f"{X.shape[1]}"
    )


    print(
        f"Classes  : "
        f"{num_classes}"
    )


    print(
        "Class names:"
    )


    print(
        list(
            le.classes_
        )
    )


    # ========================================================
    # Train/Test split
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=SEED,

        stratify=y
    )


    # ========================================================
    # Train/Validation split
    # ========================================================

    (
        X_train,
        X_val,
        y_train,
        y_val
    ) = train_test_split(

        X_train,
        y_train,

        test_size=(
            VAL_SIZE_FROM_TRAIN
        ),

        random_state=SEED,

        stratify=y_train
    )


    print(
        "\n===== DATASET SPLITS ====="
    )


    print(
        "Train      :",
        len(
            y_train
        )
    )


    print(
        "Validation :",
        len(
            y_val
        )
    )


    print(
        "Test       :",
        len(
            y_test
        )
    )


    # ========================================================
    # StandardScaler
    # ========================================================

    scaler = (
        StandardScaler()
    )


    X_train = (
        scaler
        .fit_transform(
            X_train
        )
    )


    X_val = (
        scaler
        .transform(
            X_val
        )
    )


    X_test = (
        scaler
        .transform(
            X_test
        )
    )


    # ========================================================
    # Conv1D reshape
    # ========================================================

    X_train = (

        X_train[
            ...,
            np.newaxis
        ]

        .astype(
            np.float32
        )
    )


    X_val = (

        X_val[
            ...,
            np.newaxis
        ]

        .astype(
            np.float32
        )
    )


    X_test = (

        X_test[
            ...,
            np.newaxis
        ]

        .astype(
            np.float32
        )
    )


    # ========================================================
    # Build model
    # ========================================================

    model = build_cnn(

        input_shape=(
            X_train.shape[1],
            1
        ),

        num_classes=num_classes
    )


    print(
        "\n===== STANDALONE CLIENT 3 MODEL ====="
    )


    model.summary()


    # ========================================================
    # Metrics file
    # ========================================================

    if not os.path.exists(
        METRICS_LOG
    ):

        with open(
            METRICS_LOG,
            "w",
            newline=""
        ) as f:

            csv.writer(
                f
            ).writerow(
                [
                    "epoch",
                    "val_macro_f1",
                    "timestamp",
                ]
            )


    # ========================================================
    # Training
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "TRAINING STANDALONE CLIENT 3"
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


        model.fit(

            X_train,

            y_train,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=1,

            batch_size=BATCH_SIZE,

            verbose=1
        )


        # ====================================================
        # Validation Macro-F1
        # ====================================================

        val_probs = model.predict(

            X_val,

            batch_size=1024,

            verbose=0
        )


        y_val_pred = np.argmax(

            val_probs,

            axis=1
        )


        (
            _,
            _,
            val_macro_f1,
            _
        ) = (
            precision_recall_fscore_support(

                y_val,

                y_val_pred,

                average="macro",

                zero_division=0
            )
        )


        print(
            f"Validation Macro-F1: "
            f"{val_macro_f1:.6f}"
        )


        with open(

            METRICS_LOG,

            "a",

            newline=""
        ) as f:

            csv.writer(
                f
            ).writerow(
                [
                    epoch,

                    f"{val_macro_f1:.6f}",

                    datetime.now(),
                ]
            )


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


    test_probs = model.predict(

        X_test,

        batch_size=1024,

        verbose=0
    )


    y_pred = np.argmax(

        test_probs,

        axis=1
    )


    # ========================================================
    # Accuracy
    # ========================================================

    acc = accuracy_score(

        y_test,

        y_pred
    )


    # ========================================================
    # Macro metrics
    # ========================================================

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = (
        precision_recall_fscore_support(

            y_test,

            y_pred,

            average="macro",

            zero_division=0
        )
    )


    # ========================================================
    # Weighted metrics
    # ========================================================

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _
    ) = (
        precision_recall_fscore_support(

            y_test,

            y_pred,

            average="weighted",

            zero_division=0
        )
    )


    # ========================================================
    # Print results
    # ========================================================

    print(
        "\n===== STANDALONE RESULTS ====="
    )


    print(
        f"Accuracy        : "
        f"{acc:.6f}"
    )


    print(
        f"Macro-Precision : "
        f"{macro_precision:.6f}"
    )


    print(
        f"Macro-Recall    : "
        f"{macro_recall:.6f}"
    )


    print(
        f"Macro-F1        : "
        f"{macro_f1:.6f}"
    )


    print(
        "\n===== WEIGHTED METRICS ====="
    )


    print(
        f"Weighted-Precision : "
        f"{weighted_precision:.6f}"
    )


    print(
        f"Weighted-Recall    : "
        f"{weighted_recall:.6f}"
    )


    print(
        f"Weighted-F1        : "
        f"{weighted_f1:.6f}"
    )


    # ========================================================
    # Classification report
    # ========================================================

    print(
        "\n===== Classification Report ====="
    )


    print(

        classification_report(

            y_test,

            y_pred,

            target_names=(
                le.classes_
            ),

            zero_division=0
        )
    )


    # ========================================================
    # Confusion matrix
    # ========================================================

    print(
        "\n===== Confusion Matrix ====="
    )


    print(

        confusion_matrix(

            y_test,

            y_pred
        )
    )


    # ========================================================
    # Save + plot embeddings
    # ========================================================

    save_and_visualize_test_embeddings(

        model=model,

        X_test=X_test,

        y_test=y_test,

        label_encoder=le,

        max_samples=VIS_MAX_SAMPLES
    )


    # ========================================================
    # Saved outputs
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "SAVED OUTPUTS"
    )

    print(
        "=========================================="
    )


    print(
        "Metrics:"
    )

    print(
        METRICS_LOG
    )


    print(
        "\nRaw embeddings:"
    )

    print(
        EMBEDDING_CSV
    )


    print(
        "\nt-SNE coordinates:"
    )

    print(
        TSNE_COORDINATES_CSV
    )


    print(
        "\nt-SNE image:"
    )

    print(
        TSNE_PNG
    )


    if UMAP_AVAILABLE:

        print(
            "\nUMAP coordinates:"
        )

        print(
            UMAP_COORDINATES_CSV
        )


        print(
            "\nUMAP image:"
        )

        print(
            UMAP_PNG
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()