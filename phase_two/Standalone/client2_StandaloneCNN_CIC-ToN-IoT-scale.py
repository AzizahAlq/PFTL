#!/usr/bin/env python3.10
# ============================================================
# Standalone CNN — CIC-ToN-IoT Virtual Client 01
# Manual Class Weights + PTFL-Matched Configuration
# ============================================================
#
# Dataset:
#   ton_iot_client01.csv
#
# Target:
#   Attack
#
# Important:
#   The following columns are excluded from model features:
#   - Attack        : multiclass target
#   - Label         : binary target leakage
#   - source_row_id : audit/index column
#
# Manual class weights are based on this client distribution:
#
#   Benign        8525
#   backdoor       109
#   ddos            20
#   dos             15
#   injection     1082
#   mitm            51
#   password      1319
#   ransomware     479
#   scanning       143
#   xss           8257
#
# Balanced-weight formula:
#
#   weight(class) =
#       total samples / (number of classes × class samples)
#
# The ddos and dos weights are capped at 40 to reduce instability.
# Use the same manual weights and cap in PTFL for a fair comparison.
# ============================================================

import os
import time
import csv
import random
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# Reproducibility
# ============================================================
SEED = int(os.environ.get("SEED", "45"))

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ============================================================
# Imports
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras import layers, models, optimizers, callbacks


# ============================================================
# Paths and Settings
# ============================================================
CSV_PATH = os.environ.get(
    "CSV_PATH",
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "Multi_class_Datasets/ton_iot_10_virtual_clients/"
    "ton_iot_client01.csv",
)

LABEL_COL = os.environ.get("LABEL_COL", "Attack")

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "Multi_class_Datasets/ton_iot_10_virtual_clients/"
    "ton_iot_client01_logs",
)

os.makedirs(OUT_DIR, exist_ok=True)

EPOCH_LOG_CSV = os.path.join(
    OUT_DIR,
    "standalone_ton_iot_client01_epoch_log.csv",
)

RUN_SUMMARY_CSV = os.path.join(
    OUT_DIR,
    "standalone_ton_iot_client01_run_summary.csv",
)


# ============================================================
# Train / Validation / Test Split
# ============================================================
TEST_SIZE = float(
    os.environ.get("TEST_SIZE", "0.15")
)

VAL_SIZE_FROM_TRAIN = float(
    os.environ.get(
        "VAL_SIZE_FROM_TRAIN",
        str(0.15 / (1.0 - TEST_SIZE)),
    )
)


# ============================================================
# Hyperparameters
# ============================================================
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LR = float(os.environ.get("LR", "1e-3"))

PRIVATE_DIM = int(
    os.environ.get("PRIVATE_DIM", "16")
)

SHARED_DIM = int(
    os.environ.get("SHARED_DIM", "8")
)


# ============================================================
# Manual Class Weights
# ============================================================
#
# Raw balanced weights calculated from the complete
# 20,000-sample virtual-client distribution.
#
MANUAL_CLASS_WEIGHTS_BY_NAME = {
    "Benign": 0.234604,
    "backdoor": 18.348624,
    "ddos": 100.000000,
    "dos": 133.333333,
    "injection": 1.848429,
    "mitm": 39.215686,
    "password": 1.516300,
    "ransomware": 4.175365,
    "scanning": 13.986014,
    "xss": 0.242219,
}

# Cap large minority-class weights to improve training stability.
MAX_CLASS_WEIGHT = float(
    os.environ.get("MAX_CLASS_WEIGHT", "40.0")
)


# ============================================================
# Helpers
# ============================================================
def now_str():
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def ensure_csv_headers():
    if not os.path.exists(EPOCH_LOG_CSV):
        with open(
            EPOCH_LOG_CSV,
            "w",
            newline="",
        ) as file:
            csv.writer(file).writerow([
                "seed",
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_macro_precision",
                "val_macro_recall",
                "val_macro_f1",
                "val_weighted_precision",
                "val_weighted_recall",
                "val_weighted_f1",
                "epoch_time_sec",
                "total_time_sec",
                "timestamp",
            ])

    if not os.path.exists(RUN_SUMMARY_CSV):
        with open(
            RUN_SUMMARY_CSV,
            "w",
            newline="",
        ) as file:
            csv.writer(file).writerow([
                "timestamp",
                "seed",
                "samples_total",
                "features",
                "classes",
                "train_n",
                "val_n",
                "test_n",
                "test_accuracy",
                "test_macro_precision",
                "test_macro_recall",
                "test_macro_f1",
                "test_weighted_precision",
                "test_weighted_recall",
                "test_weighted_f1",
                "total_train_time_sec",
                "max_class_weight",
            ])


def clean_label_series(series):
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("�", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .values
    )


def predict_labels(
    model,
    X_batch,
    num_classes,
):
    probabilities = model.predict(
        X_batch,
        batch_size=1024,
        verbose=0,
    )

    if int(num_classes) == 2:
        probabilities = probabilities.reshape(-1)

        return (
            probabilities >= 0.5
        ).astype(int)

    return np.argmax(
        probabilities,
        axis=1,
    )


def build_manual_class_weight(
    label_encoder,
):
    """
    Convert manual name-based class weights to integer
    LabelEncoder indices safely.

    Example:
        0 -> Benign
        1 -> backdoor
        ...
    """
    encoded_class_names = list(
        label_encoder.classes_
    )

    missing_weight_names = [
        class_name
        for class_name in encoded_class_names
        if class_name
        not in MANUAL_CLASS_WEIGHTS_BY_NAME
    ]

    if missing_weight_names:
        raise ValueError(
            "Manual class weights are missing for these "
            f"dataset labels: {missing_weight_names}"
        )

    unexpected_weight_names = [
        class_name
        for class_name
        in MANUAL_CLASS_WEIGHTS_BY_NAME
        if class_name not in encoded_class_names
    ]

    if unexpected_weight_names:
        raise ValueError(
            "Manual class-weight dictionary contains labels "
            "not found in this client dataset: "
            f"{unexpected_weight_names}"
        )

    class_weight = {}

    for class_index, class_name in enumerate(
        encoded_class_names
    ):
        raw_weight = float(
            MANUAL_CLASS_WEIGHTS_BY_NAME[class_name]
        )

        capped_weight = min(
            raw_weight,
            MAX_CLASS_WEIGHT,
        )

        class_weight[int(class_index)] = float(
            capped_weight
        )

    return class_weight


# ============================================================
# Epoch Logger
# ============================================================
class EpochCSVLogger(callbacks.Callback):

    def __init__(
        self,
        X_val,
        y_val,
        num_classes,
    ):
        super().__init__()

        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = int(num_classes)

        self.total_start_time = None
        self.epoch_start_time = None

    def on_train_begin(
        self,
        logs=None,
    ):
        self.total_start_time = (
            time.perf_counter()
        )

    def on_epoch_begin(
        self,
        epoch,
        logs=None,
    ):
        self.epoch_start_time = (
            time.perf_counter()
        )

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):
        logs = logs or {}

        epoch_time = float(
            time.perf_counter()
            - self.epoch_start_time
        )

        total_time = float(
            time.perf_counter()
            - self.total_start_time
        )

        y_pred = predict_labels(
            self.model,
            self.X_val,
            self.num_classes,
        )

        macro_p, macro_r, macro_f1, _ = (
            precision_recall_fscore_support(
                self.y_val,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        weighted_p, weighted_r, weighted_f1, _ = (
            precision_recall_fscore_support(
                self.y_val,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        )

        train_acc = logs.get(
            "accuracy",
            logs.get(
                "binary_accuracy",
                np.nan,
            ),
        )

        val_acc = logs.get(
            "val_accuracy",
            logs.get(
                "val_binary_accuracy",
                np.nan,
            ),
        )

        with open(
            EPOCH_LOG_CSV,
            "a",
            newline="",
        ) as file:
            csv.writer(file).writerow([
                int(SEED),
                int(epoch + 1),

                f"{float(logs.get('loss', np.nan)):.6f}",
                f"{float(train_acc):.6f}",

                f"{float(logs.get('val_loss', np.nan)):.6f}",
                f"{float(val_acc):.6f}",

                f"{float(macro_p):.6f}",
                f"{float(macro_r):.6f}",
                f"{float(macro_f1):.6f}",

                f"{float(weighted_p):.6f}",
                f"{float(weighted_r):.6f}",
                f"{float(weighted_f1):.6f}",

                f"{epoch_time:.4f}",
                f"{total_time:.4f}",

                now_str(),
            ])


# ============================================================
# CNN Model — Matches PTFL Architecture
# ============================================================
def build_cnn(
    input_shape,
    num_classes,
):
    inputs = layers.Input(
        shape=input_shape
    )

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
        16,
        3,
        padding="valid",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(
        16,
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

    if int(num_classes) == 2:
        outputs = layers.Dense(
            1,
            activation="sigmoid",
            name="y_out",
        )(x)

        loss = "binary_crossentropy"

        metrics = [
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            )
        ]

    else:
        outputs = layers.Dense(
            int(num_classes),
            activation="softmax",
            name="y_out",
        )(x)

        loss = (
            "sparse_categorical_crossentropy"
        )

        metrics = [
            tf.keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]

    model = models.Model(
        inputs,
        outputs,
        name=(
            "Standalone_TON_IOT_Client01_"
            "CNN_Match_PTFL"
        ),
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LR,
            clipnorm=1.0,
        ),
        loss=loss,
        metrics=metrics,
    )

    return model


# ============================================================
# Main
# ============================================================
def main():
    ensure_csv_headers()

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{CSV_PATH}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
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
            f"Label column '{LABEL_COL}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Clean labels
    # --------------------------------------------------------
    y_raw = clean_label_series(
        df[LABEL_COL]
    )

    # --------------------------------------------------------
    # Remove target leakage and non-feature columns
    # --------------------------------------------------------
    drop_columns = [
        LABEL_COL,
        "Label",
        "source_row_id",
    ]

    X_df = (
        df.drop(
            columns=drop_columns,
            errors="ignore",
        )
        .select_dtypes(
            include=[np.number]
        )
        .copy()
    )

    X_df = (
        X_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    X = X_df.values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_raw
    ).astype(int)

    num_classes = int(
        len(label_encoder.classes_)
    )

    # --------------------------------------------------------
    # Verify expected labels
    # --------------------------------------------------------
    if num_classes != 10:
        raise ValueError(
            f"Expected 10 classes, but found "
            f"{num_classes}: "
            f"{list(label_encoder.classes_)}"
        )

    # --------------------------------------------------------
    # Stratified 70/15/15 split
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y,
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train,
            y_train,
            test_size=VAL_SIZE_FROM_TRAIN,
            random_state=SEED,
            stratify=y_train,
        )
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)

    X_train = X_train[
        ...,
        np.newaxis,
    ]

    X_val = X_val[
        ...,
        np.newaxis,
    ]

    X_test = X_test[
        ...,
        np.newaxis,
    ]

    # --------------------------------------------------------
    # Manual class weights
    # --------------------------------------------------------
    class_weight = (
        build_manual_class_weight(
            label_encoder
        )
    )

    print("\n" + "=" * 80)
    print("MANUAL CLASS WEIGHTS")
    print("=" * 80)

    for class_index, class_name in enumerate(
        label_encoder.classes_
    ):
        full_client_count = int(
            np.sum(y == class_index)
        )

        training_count = int(
            np.sum(y_train == class_index)
        )

        raw_weight = float(
            MANUAL_CLASS_WEIGHTS_BY_NAME[
                class_name
            ]
        )

        applied_weight = float(
            class_weight[class_index]
        )

        print(
            f"{class_index:2d} | "
            f"{class_name:12s} | "
            f"full={full_client_count:5d} | "
            f"train={training_count:5d} | "
            f"raw_weight={raw_weight:10.6f} | "
            f"applied_weight={applied_weight:10.6f}"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = build_cnn(
        input_shape=(
            X_train.shape[1],
            1,
        ),
        num_classes=num_classes,
    )

    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    model.summary()

    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    print(f"CSV path       : {CSV_PATH}")
    print(f"Samples        : {len(X):,}")
    print(f"Features       : {X.shape[1]}")
    print(f"Classes        : {num_classes}")
    print(
        f"Train/Val/Test : "
        f"{len(X_train):,} / "
        f"{len(X_val):,} / "
        f"{len(X_test):,}"
    )
    print(
        f"Label names    : "
        f"{list(label_encoder.classes_)}"
    )
    print(
        f"Dropped columns: "
        f"{drop_columns}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    training_start = (
        time.perf_counter()
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            EpochCSVLogger(
                X_val,
                y_val,
                num_classes,
            )
        ],
        verbose=1,
    )

    total_train_time = float(
        time.perf_counter()
        - training_start
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------
    y_pred = predict_labels(
        model,
        X_test,
        num_classes,
    )

    accuracy = float(
        accuracy_score(
            y_test,
            y_pred,
        )
    )

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print("\n" + "=" * 80)
    print("STANDALONE TEST RESULTS")
    print("=" * 80)

    print(
        f"Accuracy        : "
        f"{accuracy:.6f}"
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

    print("\n" + "=" * 80)
    print("WEIGHTED TEST METRICS")
    print("=" * 80)

    print(
        f"Weighted-Precision : "
        f"{weighted_p:.6f}"
    )

    print(
        f"Weighted-Recall    : "
        f"{weighted_r:.6f}"
    )

    print(
        f"Weighted-F1        : "
        f"{weighted_f1:.6f}"
    )

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(num_classes),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=np.arange(num_classes),
        )
    )

    # --------------------------------------------------------
    # Save run summary
    # --------------------------------------------------------
    with open(
        RUN_SUMMARY_CSV,
        "a",
        newline="",
    ) as file:
        csv.writer(file).writerow([
            now_str(),
            int(SEED),

            int(len(X)),
            int(X.shape[1]),
            int(num_classes),

            int(len(X_train)),
            int(len(X_val)),
            int(len(X_test)),

            f"{accuracy:.6f}",

            f"{float(macro_p):.6f}",
            f"{float(macro_r):.6f}",
            f"{float(macro_f1):.6f}",

            f"{float(weighted_p):.6f}",
            f"{float(weighted_r):.6f}",
            f"{float(weighted_f1):.6f}",

            f"{total_train_time:.4f}",
            f"{MAX_CLASS_WEIGHT:.6f}",
        ])

    print("\n" + "=" * 80)
    print("SAVED FILES")
    print("=" * 80)

    print(
        f"Epoch log : "
        f"{EPOCH_LOG_CSV}"
    )

    print(
        f"Run summary: "
        f"{RUN_SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()