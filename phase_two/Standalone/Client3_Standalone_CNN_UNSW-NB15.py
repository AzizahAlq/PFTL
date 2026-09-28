#!/usr/bin/env python3.10
# ============================================================
# Standalone Client3 (UNSW-NB15)
# Same CNN/data split/training budget as PFTL, but NO federation/server communication
#
# Visualization changes:
#   1. Normal is displayed as BENIGN
#   2. Same fixed colors as FedProto
#   3. Training labels are NOT changed
#   4. Saved embedding/t-SNE/UMAP CSVs use normalized labels
#
# Fixed visualization colors:
#   BENIGN          -> blue
#   EXPLOITS        -> green
#   FUZZERS         -> purple
#   RECONNAISSANCE  -> pink
#   DOS             -> olive
#   GENERIC         -> cyan
# ============================================================

import os
import time
import csv
import pickle
import random

from datetime import datetime


# ============================================================
# Reproducibility
# ============================================================

SEED = 4096

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


# ============================================================
# Other imports
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

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
# TensorFlow Keras
# ============================================================

from tensorflow.keras import (
    layers,
    models,
    optimizers,
)


# ============================================================
# gRPC generated modules
# ============================================================



# ============================================================
# Standalone Settings
# ============================================================

SERVER_ADDRESS = os.environ.get(
    "SERVER_ADDRESS",
    "localhost:50051"
)

CLIENT_ID = os.environ.get(
    "CLIENT_ID",
    "client3"
)

NUM_EPOCHS = int(
    os.environ.get(
        "NUM_EPOCHS",
        "20"
    )
)


PRIVATE_DIM = 16
SHARED_DIM = 8


# ============================================================
# Data / Paths
# ============================================================

CSV_PATH = os.environ.get(
    "CSV_PATH",
    (
        "/nfs/aalqahtani/PFTL_Multi_class/"
        "M_PFTL_codes/Multi_class_Datasets/"
        "D3_UNSW_NB15_ALL_MERGED_final.csv"
    )
)


LABEL_COL = os.environ.get(
    "LABEL_COL",
    "attack_cat"
)


OUT_DIR = os.environ.get(
    "OUT_DIR",
    (
        "/nfs/aalqahtani/PFTL_Multi_class/"
        "standalone_logs/client3_unsw_v2"
    )
)


os.makedirs(
    OUT_DIR,
    exist_ok=True
)


COMM_LOG = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_comm_log.csv"
)


METRICS_LOG = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_metrics_log.csv"
)


CURVE_CSV = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_local_vs_global_curve.csv"
)


# ============================================================
# Visualization configuration
# ============================================================

VISUALIZE_SHARED = (
    os.environ.get(
        "VISUALIZE_SHARED",
        "1"
    )
    == "1"
)


VIS_MAX_SAMPLES = int(
    os.environ.get(
        "VIS_MAX_SAMPLES",
        "5000"
    )
)


# ============================================================
# SAME COLORS AS FEDPROTO
# ============================================================

CLASS_COLORS = {

    "BENIGN":
        "tab:blue",

    "EXPLOITS":
        "tab:green",

    "FUZZERS":
        "tab:purple",

    "RECONNAISSANCE":
        "tab:pink",

    "DOS":
        "tab:olive",

    "GENERIC":
        "tab:cyan",
}

# Exact legend/order used in the FedProto Client 3 figure
FEDPROTO_CLASS_ORDER = [
    "DOS",
    "EXPLOITS",
    "FUZZERS",
    "GENERIC",
    "BENIGN",
    "RECONNAISSANCE",
]


# ============================================================
# Label display mapping
#
# IMPORTANT:
# Training labels are NOT changed.
# Only labels saved/displayed in visualization are normalized.
# ============================================================

def normalize_label_name(
    label
):

    label_upper = (
        str(label)
        .strip()
        .upper()
    )


    mapping = {

        "NORMAL":
            "BENIGN",

        "BENIGN":
            "BENIGN",

        "EXPLOITS":
            "EXPLOITS",

        "FUZZERS":
            "FUZZERS",

        "FUZZER":
            "FUZZERS",

        "RECONNAISSANCE":
            "RECONNAISSANCE",

        "RECON":
            "RECONNAISSANCE",

        "DOS":
            "DOS",

        "GENERIC":
            "GENERIC",
    }


    return mapping.get(
        label_upper,
        label_upper
    )


# ============================================================
# Split style: 70 / 15 / 15
# ============================================================

TEST_SIZE = 0.15

VAL_SIZE_FROM_TRAIN = (
    0.17647058823529413
)


# ============================================================
# Train configuration
# ============================================================

EPOCHS_PER_ROUND = 1

BATCH_SIZE = 512

LR = 1e-3


# ============================================================
# Standalone configuration
# ============================================================
# No server, no shared-weight exchange, no gamma blending, and no safety gate.

# ============================================================
# Utils
# ============================================================

def now_str():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def clip(
    x,
    lo,
    hi
):

    return float(
        max(
            lo,
            min(
                hi,
                x
            )
        )
    )


def safe_unpickle_weights(
    b: bytes
):

    try:

        return (
            pickle.loads(b)
            if b
            else None
        )

    except Exception:

        return None


def is_valid_shared(
    weights_obj
) -> bool:

    return (
        isinstance(
            weights_obj,
            (list, tuple)
        )
        and
        len(weights_obj) == 2
    )


# ============================================================
# Convert encoded labels into normalized display labels
# ============================================================

def normalized_display_labels(
    label_encoder,
    y_values
):

    original_names = (
        label_encoder
        .inverse_transform(
            np.asarray(
                y_values,
                dtype=int
            )
        )
    )


    normalized_names = np.asarray(
        [
            normalize_label_name(
                name
            )
            for name
            in original_names
        ],
        dtype=object
    )


    return normalized_names


# ============================================================
# Prediction
# ============================================================

def predict_labels(
    model,
    X_batch,
    num_classes
):

    probs = model.predict(

        X_batch,

        batch_size=1024,

        verbose=0
    )


    if num_classes == 2:

        return (
            probs.reshape(-1)
            >= 0.5
        ).astype(
            int
        )


    return np.argmax(
        probs,
        axis=1
    )


# ============================================================
# Evaluation
# ============================================================

def eval_all_metrics(
    model,
    X_eval,
    y_eval,
    num_classes
):

    y_hat = predict_labels(

        model,
        X_eval,
        num_classes
    )


    acc = float(
        accuracy_score(
            y_eval,
            y_hat
        )
    )


    (
        mp,
        mr,
        mf1,
        _
    ) = precision_recall_fscore_support(

        y_eval,
        y_hat,

        average="macro",

        zero_division=0
    )


    (
        wp,
        wr,
        wf1,
        _
    ) = precision_recall_fscore_support(

        y_eval,
        y_hat,

        average="weighted",

        zero_division=0
    )


    return (

        acc,

        float(mp),

        float(mr),

        float(mf1),

        float(wp),

        float(wr),

        float(wf1),
    )


# ============================================================
# CSV headers
# ============================================================

def ensure_csv_headers():

    if not os.path.exists(
        COMM_LOG
    ):

        with open(

            COMM_LOG,

            "w",

            newline=""

        ) as f:

            csv.writer(
                f
            ).writerow(
                [
                    "seed",
                    "round",
                    "bytes_sent",
                    "rtt_sec",
                    "barrier_wait_sec",
                    "timestamp",
                ]
            )


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

                    "seed",

                    "round",

                    "stage",

                    "train_loss",

                    "train_acc",

                    "val_loss",

                    "val_acc",

                    "acc",

                    "macro_precision",

                    "macro_recall",

                    "macro_f1",

                    "weighted_precision",

                    "weighted_recall",

                    "weighted_f1",

                    "switch_on",

                    "eps_margin",

                    "gamma_local",

                    "gamma_global",

                    "delta_macro_f1_mixed_minus_local",

                    "sent_weights_type",

                    "local_train_time_sec",

                    "mix_time_sec",

                    "round_total_time_sec",

                    "timestamp",
                ]
            )


    if not os.path.exists(
        CURVE_CSV
    ):

        with open(

            CURVE_CSV,

            "w",

            newline=""

        ) as f:

            csv.writer(
                f
            ).writerow(
                [
                    "seed",
                    "round",
                    "local_macro_f1",
                    "global_macro_f1",
                    "switch_on",
                    "gamma_global",
                ]
            )


# ============================================================
# Metrics logging
# ============================================================

def log_stage_row(

    seed,

    round_id,

    stage,

    train_loss,

    train_acc,

    val_loss,

    val_acc,

    acc,

    mp,

    mr,

    mf1,

    wp,

    wr,

    wf1,

    switch_on,

    eps_margin,

    gamma_local,

    gamma_global,

    delta_f1,

    sent_type,

    local_train_time,

    mix_time,

    round_total_time,
):

    with open(

        METRICS_LOG,

        "a",

        newline=""

    ) as f:

        csv.writer(
            f
        ).writerow(
            [

                int(seed),

                int(round_id),

                str(stage),

                f"{float(train_loss):.6f}",

                f"{float(train_acc):.6f}",

                f"{float(val_loss):.6f}",

                f"{float(val_acc):.6f}",

                f"{float(acc):.6f}",

                f"{float(mp):.6f}",

                f"{float(mr):.6f}",

                f"{float(mf1):.6f}",

                f"{float(wp):.6f}",

                f"{float(wr):.6f}",

                f"{float(wf1):.6f}",

                int(switch_on),

                f"{float(eps_margin):.6f}",

                f"{float(gamma_local):.6f}",

                f"{float(gamma_global):.6f}",

                f"{float(delta_f1):+.6f}",

                str(sent_type),

                f"{float(local_train_time):.4f}",

                f"{float(mix_time):.4f}",

                f"{float(round_total_time):.4f}",

                now_str(),
            ]
        )


# ============================================================
# Model
# ============================================================

def build_cnn(
    input_shape,
    num_classes
):

    inp = layers.Input(
        shape=input_shape
    )


    x = layers.Conv1D(

        64,

        3,

        padding="valid",

        activation="relu"

    )(inp)


    x = layers.BatchNormalization()(x)


    x = layers.MaxPooling1D(
        2
    )(x)


    x = layers.Conv1D(

        128,

        3,

        padding="valid",

        activation="relu"

    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.MaxPooling1D(
        2
    )(x)


    x = layers.GlobalAveragePooling1D()(x)


    x = layers.Dense(

        PRIVATE_DIM,

        activation="relu",

        name="private_dense"

    )(x)


    x = layers.Dense(

        SHARED_DIM,

        activation="relu",

        name="shared_dense"

    )(x)


    out = layers.Dense(

        num_classes,

        activation="softmax",

        name="y_out"

    )(x)


    loss = (
        "sparse_categorical_crossentropy"
    )


    metrics = [

        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]


    model = models.Model(

        inp,

        out,

        name=f"Standalone_{CLIENT_ID}_CNN"
    )


    opt = optimizers.Adam(

        learning_rate=LR,

        clipnorm=1.0
    )


    model.compile(

        optimizer=opt,

        loss=loss,

        metrics=metrics
    )


    return model


# ============================================================
# Client
# ============================================================

class StandaloneClient3:

    def __init__(self):
        self._load_data()
        self.model = build_cnn(self.input_shape, self.num_classes)
        print("\n==== Standalone Client 3 Model Summary ====")
        self.model.summary()

    # ========================================================
    # Load dataset
    # ========================================================

    def _load_data(
        self
    ):

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


        if LABEL_COL not in df.columns:

            raise ValueError(

                f"Label column "
                f"'{LABEL_COL}' not found.\n"

                f"Available columns:\n"
                f"{list(df.columns)}"
            )


        df = df.replace(

            [
                np.inf,
                -np.inf
            ],

            np.nan
        )


        # ====================================================
        # IMPORTANT:
        # Keep original training labels unchanged.
        # ====================================================

        y_raw = (

            df[
                LABEL_COL
            ]

            .astype(str)

            .str.replace(
                "\ufeff",
                "",
                regex=False
            )

            .str.replace(
                "�",
                "-",
                regex=False
            )

            .str.replace(
                r"\s+",
                " ",
                regex=True
            )

            .str.strip()

            .values
        )


        X_df = (

            df

            .drop(

                columns=[
                    LABEL_COL,
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

            .copy()
        )


        X_df = (

            X_df

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(
                0.0
            )
        )


        X = X_df.values.astype(
            np.float32
        )


        # ====================================================
        # Original LabelEncoder remains unchanged
        # ====================================================

        self.le = LabelEncoder()


        y = (

            self.le

            .fit_transform(
                y_raw
            )

            .astype(
                int
            )
        )


        self.num_classes = int(
            len(
                self.le.classes_
            )
        )


        # ====================================================
        # 70 / 15 / 15 split
        # ====================================================

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


        # ====================================================
        # Scaling
        # ====================================================

        self.scaler = (
            StandardScaler()
        )


        X_train = (
            self.scaler
            .fit_transform(
                X_train
            )
        )


        X_val = (
            self.scaler
            .transform(
                X_val
            )
        )


        X_test = (
            self.scaler
            .transform(
                X_test
            )
        )


        self.X_train = (

            X_train[
                ...,
                np.newaxis
            ]

            .astype(
                np.float32
            )
        )


        self.X_val = (

            X_val[
                ...,
                np.newaxis
            ]

            .astype(
                np.float32
            )
        )


        self.X_test = (

            X_test[
                ...,
                np.newaxis
            ]

            .astype(
                np.float32
            )
        )


        self.y_train = y_train

        self.y_val = y_val

        self.y_test = y_test


        self.input_shape = (

            self.X_train.shape[1],

            1
        )


        # ====================================================
        # Balanced class weights
        # ====================================================

        classes_idx = np.unique(
            self.y_train
        )


        cw = compute_class_weight(

            class_weight="balanced",

            classes=classes_idx,

            y=self.y_train
        )


        self.class_weight = {

            int(i):
                float(w)

            for i, w
            in zip(
                classes_idx,
                cw
            )
        }


        print(
            "\n=== Standalone Client Readiness Summary ==="
        )


        print(
            f"Client : {CLIENT_ID}"
        )


        print(
            f"CSV    : {CSV_PATH}"
        )


        print(
            f"Samples: {X.shape[0]}"
        )


        print(
            f"Features: {X.shape[1]}"
        )


        print(
            f"Classes : {self.num_classes}"
        )


        print(
            "Original class names:"
        )


        print(
            list(
                self.le.classes_
            )
        )


        print(
            "Visualization class names:"
        )


        print(
            [
                normalize_label_name(
                    label
                )

                for label
                in self.le.classes_
            ]
        )


        print(

            "Splits:",

            len(
                self.X_train
            ),

            len(
                self.X_val
            ),

            len(
                self.X_test
            )
        )


    # ========================================================
    # Shared Representation Visualization
    # ========================================================

    def visualize_shared_representation(

        self,

        max_samples=VIS_MAX_SAMPLES
    ):

        print(
            "\n=========================================="
        )

        print(
            f"[{CLIENT_ID}] "
            f"STANDALONE INTERNAL-LAYER VISUALIZATION"
        )

        print(
            "=========================================="
        )


        # ====================================================
        # Extract learned 8-D shared representation
        # ====================================================

        embedding_model = tf.keras.Model(

            inputs=self.model.input,

            outputs=self.model.get_layer(
                "shared_dense"
            ).output
        )


        # ====================================================
        # Held-out test data only
        # ====================================================

        X_vis = self.X_test

        y_vis = self.y_test


        print(

            f"[{CLIENT_ID}] "
            f"Original test samples: "
            f"{len(X_vis)}"
        )


        print(

            f"[{CLIENT_ID}] "
            f"Number of classes: "
            f"{self.num_classes}"
        )


        print(

            f"[{CLIENT_ID}] "
            f"Original classes: "
            f"{list(self.le.classes_)}"
        )


        # ====================================================
        # Stratified subsampling
        # ====================================================

        if len(
            X_vis
        ) > max_samples:

            print(

                f"[{CLIENT_ID}] "
                f"Subsampling test data to "
                f"{max_samples} samples..."
            )


            (
                X_vis,
                _,
                y_vis,
                _

            ) = train_test_split(

                X_vis,

                y_vis,

                train_size=max_samples,

                random_state=SEED,

                stratify=y_vis
            )


        print(

            f"[{CLIENT_ID}] "
            f"Visualization samples: "
            f"{len(X_vis)}"
        )


        # ====================================================
        # Generate 8-D embeddings
        # ====================================================

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

            f"[{CLIENT_ID}] "
            f"Internal layer representation shape: "
            f"{embeddings.shape}"
        )


        # ====================================================
        # Normalized names
        #
        # Normal -> BENIGN
        # ====================================================

        display_labels = (
            normalized_display_labels(

                self.le,

                y_vis
            )
        )


        # ====================================================
        # Save raw 8-D embeddings
        # ====================================================

        embedding_csv = os.path.join(

            OUT_DIR,

            (
                f"{CLIENT_ID}_"
                f"shared_embeddings_"
                f"seed{SEED}.csv"
            )
        )


        emb_df = pd.DataFrame(

            embeddings,

            columns=[
                f"shared_dim_{i + 1}"

                for i
                in range(
                    embeddings.shape[1]
                )
            ]
        )


        emb_df[
            "label_id"
        ] = y_vis


        # NEW:
        # Save BENIGN rather than Normal
        emb_df[
            "label_name"
        ] = display_labels


        emb_df.to_csv(

            embedding_csv,

            index=False
        )


        print(

            f"[{CLIENT_ID}] "
            f"Raw embeddings saved: "
            f"{embedding_csv}"
        )


        unique_classes = np.unique(
            y_vis
        )


        # ====================================================
        # t-SNE
        # ====================================================

        print(

            f"\n[{CLIENT_ID}] "
            f"Running t-SNE..."
        )


        if len(
            embeddings
        ) > 31:

            tsne_perplexity = 30

        else:

            tsne_perplexity = max(

                5,

                min(
                    30,
                    len(
                        embeddings
                    )
                    -
                    1
                )
            )


        tsne_model = TSNE(

            n_components=2,

            perplexity=(
                tsne_perplexity
            ),

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


        # ====================================================
        # Save t-SNE coordinates
        # ====================================================

        tsne_csv = os.path.join(

            OUT_DIR,

            (
                f"{CLIENT_ID}_"
                f"shared_TSNE_coordinates_"
                f"seed{SEED}.csv"
            )
        )


        pd.DataFrame(
            {

                "tsne_1":
                    tsne_2d[:, 0],

                "tsne_2":
                    tsne_2d[:, 1],

                "label_id":
                    y_vis,

                "label_name":
                    display_labels,

            }

        ).to_csv(

            tsne_csv,

            index=False
        )


        # ====================================================
        # Plot t-SNE
        # ====================================================

        plt.figure(
            figsize=(10, 8)
        )


        # Plot in the exact same semantic-label order and colors as FedProto.
        # display_labels already maps UNSW-NB15 "Normal" -> "BENIGN".
        for class_name in FEDPROTO_CLASS_ORDER:

            mask = (display_labels == class_name)

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

            "Standalone Internal Layer - t-SNE\n"

            f"{CLIENT_ID} | "
            f"{self.num_classes} Classes | "
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


        tsne_path = os.path.join(

            OUT_DIR,

            (
                f"{CLIENT_ID}_"
                f"shared_representation_"
                f"TSNE_seed{SEED}.png"
            )
        )


        plt.savefig(

            tsne_path,

            dpi=300,

            bbox_inches="tight"
        )


        plt.close()


        print(

            f"[{CLIENT_ID}] "
            f"t-SNE figure saved: "
            f"{tsne_path}"
        )


        print(

            f"[{CLIENT_ID}] "
            f"t-SNE coordinates saved: "
            f"{tsne_csv}"
        )


        # ====================================================
        # UMAP
        # ====================================================

        if UMAP_AVAILABLE:

            print(

                f"\n[{CLIENT_ID}] "
                f"Running UMAP..."
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


            # ================================================
            # Save UMAP coordinates
            # ================================================

            umap_csv = os.path.join(

                OUT_DIR,

                (
                    f"{CLIENT_ID}_"
                    f"shared_UMAP_coordinates_"
                    f"seed{SEED}.csv"
                )
            )


            pd.DataFrame(
                {

                    "umap_1":
                        umap_2d[:, 0],

                    "umap_2":
                        umap_2d[:, 1],

                    "label_id":
                        y_vis,

                    "label_name":
                        display_labels,

                }

            ).to_csv(

                umap_csv,

                index=False
            )


            # ================================================
            # Plot UMAP
            # ================================================

            plt.figure(
                figsize=(10, 8)
            )


            # Plot in the exact same semantic-label order and colors as FedProto.
            # display_labels already maps UNSW-NB15 "Normal" -> "BENIGN".
            for class_name in FEDPROTO_CLASS_ORDER:

                mask = (display_labels == class_name)

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

                "Standalone Internal Layer - UMAP\n"

                f"{CLIENT_ID} | "
                f"{self.num_classes} Classes | "
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


            umap_path = os.path.join(

                OUT_DIR,

                (
                    f"{CLIENT_ID}_"
                    f"shared_representation_"
                    f"UMAP_seed{SEED}.png"
                )
            )


            plt.savefig(

                umap_path,

                dpi=300,

                bbox_inches="tight"
            )


            plt.close()


            print(

                f"[{CLIENT_ID}] "
                f"UMAP figure saved: "
                f"{umap_path}"
            )


            print(

                f"[{CLIENT_ID}] "
                f"UMAP coordinates saved: "
                f"{umap_csv}"
            )


        else:

            print(
                "\nWARNING: UMAP is not installed."
            )

            print(
                "Install it using: "
                "pip install umap-learn"
            )


        print(
            "\n=========================================="
        )


        print(

            f"[{CLIENT_ID}] "
            f"VISUALIZATION COMPLETE"
        )


        print(
            "=========================================="
        )


    # ========================================================
    # Train one epoch
    # ========================================================

    def train_one_epoch(
        self
    ):

        hist = self.model.fit(

            self.X_train,

            self.y_train,

            validation_data=(
                self.X_val,
                self.y_val
            ),

            epochs=(
                EPOCHS_PER_ROUND
            ),

            batch_size=(
                BATCH_SIZE
            ),

            class_weight=(
                self.class_weight
            ),

            verbose=1
        )


        train_loss = float(
            hist.history[
                "loss"
            ][0]
        )


        val_loss = float(
            hist.history[
                "val_loss"
            ][0]
        )


        train_acc = float(

            hist.history.get(
                "accuracy",
                [np.nan]
            )[0]
        )


        val_acc = float(

            hist.history.get(
                "val_accuracy",
                [np.nan]
            )[0]
        )


        return (

            train_loss,

            train_acc,

            val_loss,

            val_acc
        )


    # ========================================================
    # Standalone training
    # ========================================================

    def run(self):
        print(f"\n[{CLIENT_ID}] ===== STANDALONE TRAINING: {NUM_EPOCHS} EPOCHS =====")
        t0 = time.perf_counter()

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=self.class_weight,
            verbose=1,
        )

        total_train_time = time.perf_counter() - t0
        print(f"[{CLIENT_ID}] Standalone training completed in {total_train_time:.2f}s")

        # ====================================================
        # FINAL TEST
        # ====================================================

        print(

            f"\n[{CLIENT_ID}] "
            f"===== FINAL TEST EVALUATION ====="
        )


        y_pred = predict_labels(

            self.model,

            self.X_test,

            self.num_classes
        )


        acc = float(
            accuracy_score(
                self.y_test,
                y_pred
            )
        )


        (
            macro_p,
            macro_r,
            macro_f1,
            _

        ) = precision_recall_fscore_support(

            self.y_test,

            y_pred,

            average="macro",

            zero_division=0
        )


        (
            w_p,
            w_r,
            w_f1,
            _

        ) = precision_recall_fscore_support(

            self.y_test,

            y_pred,

            average="weighted",

            zero_division=0
        )


        print(
            "\n===== STANDALONE FINAL RESULTS ====="
        )


        print(
            f"Accuracy        : "
            f"{acc:.6f}"
        )


        print(
            f"Macro-Precision : "
            f"{macro_p:.6f}"
        )


        print(
            f"Macro-Recall    : "
            f"{macro_r:.6f}"
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
            f"{w_p:.6f}"
        )


        print(
            f"Weighted-Recall    : "
            f"{w_r:.6f}"
        )


        print(
            f"Weighted-F1        : "
            f"{w_f1:.6f}"
        )


        # ====================================================
        # Classification report
        #
        # Here we also display BENIGN rather than Normal.
        # This changes display only.
        # ====================================================

        display_class_names = [

            normalize_label_name(
                name
            )

            for name
            in self.le.classes_
        ]


        print(
            "\n==== Classification Report ===="
        )


        print(

            classification_report(

                self.y_test,

                y_pred,

                target_names=(
                    display_class_names
                ),

                zero_division=0
            )
        )


        print(
            "\n==== Confusion Matrix ===="
        )


        print(

            confusion_matrix(

                self.y_test,

                y_pred
            )
        )


        # ====================================================
        # FINAL shared representation visualization
        # ====================================================

        if VISUALIZE_SHARED:

            self.visualize_shared_representation(

                max_samples=(
                    VIS_MAX_SAMPLES
                )
            )


        else:

            print(
                "\nShared representation "
                "visualization disabled."
            )


        # ====================================================
        # Completed
        # ====================================================
        print("\n===== STANDALONE RUN COMPLETE =====")
        print("Visualization outputs:", OUT_DIR)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    StandaloneClient3().run()