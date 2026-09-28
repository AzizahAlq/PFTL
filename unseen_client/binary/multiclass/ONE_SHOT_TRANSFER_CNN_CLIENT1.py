#!/usr/bin/env python3.10
# ============================================================
# ONE_SHOT_TL_UNSW_NB15_MULTICLASS_FROM_SERVER_PKL_FIXED_CM.py
# ============================================================
# One-shot transfer learning for UNSW-NB15 multiclass:
#   - Loads a server shared-layer PKL snapshot
#   - Transfers only the "shared_dense" layer
#   - Blends local/global shared weights using gamma
#   - Trains locally
#   - Saves metrics, model, history, and a normalized
#     confusion matrix with bold decimal values in every cell
# ============================================================

import os
import pickle
import random
import re
import time
from datetime import datetime

# ============================================================
# Reproducibility
# ============================================================
SEED = 512

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ============================================================
# Imports
# ============================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, optimizers


# ============================================================
# Paths and experiment settings
# ============================================================
CSV_PATH = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "Multi_class_Datasets/UNSW_NB15_testing-set.csv"
)

LABEL_COL = "attack_cat"

SERVER_PKL = (
    "/nfs/aalqahtani/server_logs_pftl_shared/global_shared_round_019.pkl"
)

OUT_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "one_shot_unsw_nb15_multiclass_12"
)

os.makedirs(OUT_DIR, exist_ok=True)

# gamma = (local contribution, global contribution)
gamma = (0.9, 0.1)

EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

# 70% train, 15% validation, 15% test
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)

PRIVATE_DIM = 16
SHARED_DIM = 8


# ============================================================
# Helper functions
# ============================================================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text_series(series: pd.Series) -> np.ndarray:
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("�", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .values
    )


def infer_round_from_pkl(pkl_path: str, fallback: int = 0) -> int:
    match = re.search(
        r"global_shared_round_(\d+)\.pkl$",
        os.path.basename(pkl_path),
    )

    if match:
        return int(match.group(1))

    return int(fallback)


def blend(local_weights, global_weights, gamma_values=(0.5, 0.5)):
    gamma_local, gamma_global = gamma_values

    return [
        gamma_local * local_weight + gamma_global * global_weight
        for local_weight, global_weight in zip(
            local_weights,
            global_weights,
        )
    ]


def predict_labels_multiclass(
    model: tf.keras.Model,
    features: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    probabilities = model.predict(
        features,
        batch_size=batch_size,
        verbose=0,
    )

    return np.argmax(probabilities, axis=1)


# ============================================================
# Confusion matrix
# ============================================================
def plot_confusion_matrix_normalized(
    cm: np.ndarray,
    labels,
    outpath: str,
) -> None:
    """
    Saves a row-normalized confusion matrix with decimal values
    such as 0.20, 0.41, and 0.70 in every cell.
    """
    cm = np.asarray(cm, dtype=np.float64)

    row_sums = cm.sum(axis=1, keepdims=True)

    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )

    print("\n===== NORMALIZED CONFUSION MATRIX =====")
    print(np.round(cm_normalized, 2))

    number_of_classes = len(labels)

    # Large enough to prevent annotation overlap
    fig_width = max(12.0, number_of_classes * 1.65)
    fig_height = max(10.0, number_of_classes * 1.45)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        facecolor="white",
    )

    sns.heatmap(
        cm_normalized,
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        annot=False,
        linewidths=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        cbar_kws={
            "pad": 0.06,
            "shrink": 1.0,
        },
    )

    # Draw each value manually to guarantee it appears
    for row_index in range(number_of_classes):
        for column_index in range(number_of_classes):
            value = cm_normalized[row_index, column_index]

            text_color = "white" if value >= 0.50 else "black"

            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=22,
                fontweight="bold",
                clip_on=False,
            )

    ax.set_xlabel(
        "Predicted",
        fontsize=24,
        labelpad=18,
    )

    ax.set_ylabel(
        "Actual",
        fontsize=24,
        labelpad=18,
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=17,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=17,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )

    plt.close(fig)

    print("\nSaved normalized confusion matrix:")
    print(outpath)


# ============================================================
# Model
# ============================================================
def build_cnn_multiclass(
    input_shape,
    num_classes: int,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        64,
        3,
        padding="valid",
        activation="relu",
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(
        128,
        3,
        padding="valid",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(
        128,
        3,
        padding="valid",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        PRIVATE_DIM,
        activation="relu",
        name="private_dense",
    )(x)

    x = layers.Dense(
        SHARED_DIM,
        activation="relu",
        name="shared_dense",
    )(x)

    outputs = layers.Dense(
        int(num_classes),
        activation="softmax",
        name="y_out",
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="OneShot_UNSW_NB15_Multiclass",
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


# ============================================================
# Load server PKL
# ============================================================
def load_server_shared_pkl(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Missing server shared file: {pkl_path}"
        )

    with open(pkl_path, "rb") as file:
        obj = pickle.load(file)

    if not isinstance(obj, dict):
        raise ValueError(
            "Unexpected PKL format. Expected a dictionary."
        )

    if "global_shared" not in obj:
        raise ValueError(
            "Unexpected PKL format. Missing key 'global_shared'."
        )

    global_shared = obj["global_shared"]

    if not isinstance(global_shared, (list, tuple)):
        raise ValueError(
            "'global_shared' must be a list or tuple."
        )

    if len(global_shared) != 2:
        raise ValueError(
            "'global_shared' must contain [kernel, bias]."
        )

    return [
        np.asarray(weight, dtype=np.float32)
        for weight in global_shared
    ]


# ============================================================
# Main
# ============================================================
def main() -> None:
    round_id = infer_round_from_pkl(
        SERVER_PKL,
        fallback=0,
    )

    print("\n===== ONE-SHOT TL: UNSW-NB15 MULTICLASS =====")
    print("Seed       :", SEED)
    print("Server PKL :", SERVER_PKL)
    print("Round      :", round_id)
    print("Gamma      :", gamma)
    print("Output dir :", OUT_DIR)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH,
        low_memory=False,
    )

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    y_raw = clean_text_series(df[LABEL_COL])

    X_df = (
        df.drop(columns=[LABEL_COL], errors="ignore")
        .select_dtypes(include=[np.number])
        .copy()
    )

    if X_df.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns were found."
        )

    X_df = (
        X_df.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X = X_df.to_numpy(dtype=np.float32)

    print("Samples     :", X.shape[0])
    print("Features    :", X.shape[1])

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw).astype(np.int32)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    labels_all = np.arange(num_classes)

    print("Classes     :", num_classes)
    print("Class names :", class_names)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=SEED,
        stratify=y_train,
    )

    print("\nSplit sizes")
    print("Train:", len(y_train))
    print("Val  :", len(y_val))
    print("Test :", len(y_test))

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------
    classes_unique = np.unique(y_train)

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes_unique,
        y=y_train,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            classes_unique,
            weights,
        )
    }

    # --------------------------------------------------------
    # Load shared weights
    # --------------------------------------------------------
    global_shared = load_server_shared_pkl(
        SERVER_PKL
    )

    model = build_cnn_multiclass(
        input_shape=(X_train.shape[1], 1),
        num_classes=num_classes,
    )

    local_shared = (
        model.get_layer("shared_dense")
        .get_weights()
    )

    if len(local_shared) != len(global_shared):
        raise ValueError(
            "shared_dense weight-count mismatch: "
            f"local={len(local_shared)}, "
            f"global={len(global_shared)}"
        )

    for index, (
        local_weight,
        global_weight,
    ) in enumerate(zip(local_shared, global_shared)):
        if local_weight.shape != global_weight.shape:
            raise ValueError(
                f"shared_dense shape mismatch at index {index}: "
                f"local={local_weight.shape}, "
                f"global={global_weight.shape}. "
                f"Expected SHARED_DIM={SHARED_DIM}."
            )

    model.get_layer("shared_dense").set_weights(
        blend(
            local_shared,
            global_shared,
            gamma_values=gamma,
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    print("\n===== TRAINING =====")

    training_start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=1,
    )

    train_time = time.perf_counter() - training_start

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    y_hat = predict_labels_multiclass(
        model,
        X_test,
        batch_size=BATCH_SIZE,
    )

    cm = confusion_matrix(
        y_test,
        y_hat,
        labels=labels_all,
    )

    accuracy = accuracy_score(
        y_test,
        y_hat,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_test,
        y_hat,
        average="macro",
        zero_division=0,
        labels=labels_all,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_test,
        y_hat,
        average="weighted",
        zero_division=0,
        labels=labels_all,
    )

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Accuracy            : {accuracy:.6f}")
    print(f"Macro-Precision     : {macro_precision:.6f}")
    print(f"Macro-Recall        : {macro_recall:.6f}")
    print(f"Macro-F1            : {macro_f1:.6f}")
    print(f"Weighted-Precision  : {weighted_precision:.6f}")
    print(f"Weighted-Recall     : {weighted_recall:.6f}")
    print(f"Weighted-F1         : {weighted_f1:.6f}")
    print(f"Training time (sec) : {train_time:.2f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            y_hat,
            labels=labels_all,
            target_names=class_names,
            zero_division=0,
            digits=6,
        )
    )

    print("\n===== RAW CONFUSION MATRIX =====")
    print(cm)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    gamma_local_text = str(gamma[0]).replace(".", "p")
    gamma_global_text = str(gamma[1]).replace(".", "p")

    cm_png = os.path.join(
        OUT_DIR,
        (
            f"CM_WITH_VALUES_round{round_id:03d}"
            f"_seed{SEED}"
            f"_g{gamma_local_text}_{gamma_global_text}"
            f"_{timestamp_tag}.png"
        ),
    )

    plot_confusion_matrix_normalized(
        cm,
        class_names,
        cm_png,
    )

    metrics_csv = os.path.join(
        OUT_DIR,
        (
            f"metrics_one_shot_round{round_id:03d}"
            f"_seed{SEED}"
            f"_g{gamma_local_text}_{gamma_global_text}"
            f"_{timestamp_tag}.csv"
        ),
    )

    pd.DataFrame(
        [
            {
                "timestamp": now_str(),
                "seed": SEED,
                "round_id": round_id,
                "gamma_local": gamma[0],
                "gamma_global": gamma[1],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_classes": num_classes,
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
                "train_time_sec": float(train_time),
                "server_pkl_path": SERVER_PKL,
                "dataset_path": CSV_PATH,
            }
        ]
    ).to_csv(
        metrics_csv,
        index=False,
    )

    model_path = os.path.join(
        OUT_DIR,
        (
            f"one_shot_unsw_nb15_multiclass"
            f"_round{round_id:03d}"
            f"_seed{SEED}"
            f"_{timestamp_tag}.keras"
        ),
    )

    model.save(model_path)

    history_csv = os.path.join(
        OUT_DIR,
        (
            f"training_history_round{round_id:03d}"
            f"_seed{SEED}"
            f"_{timestamp_tag}.csv"
        ),
    )

    pd.DataFrame(history.history).to_csv(
        history_csv,
        index=False,
    )

    print("\n===== SAVED =====")
    print("Metrics CSV      :", metrics_csv)
    print("Normalized CM    :", cm_png)
    print("Training history :", history_csv)
    print("Model            :", model_path)


if __name__ == "__main__":
    main()