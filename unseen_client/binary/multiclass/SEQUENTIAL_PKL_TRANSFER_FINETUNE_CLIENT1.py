#!/usr/bin/env python3.10

# ============================================================
# STANDALONE vs FROZEN vs FINETUNED
# SEQUENTIAL 20-ROUND PKL TRANSFER
# ============================================================
#
# Condition 1: Standalone
#   - No server PKL
#   - Train locally for 20 epochs
#
# Condition 2: Frozen
#   Round 001 PKL -> blend -> shared_dense frozen -> 1 epoch
#   Round 002 PKL -> blend -> shared_dense frozen -> 1 epoch
#   ...
#   Round 020 PKL -> blend -> shared_dense frozen -> 1 epoch
#
# Condition 3: FineTuned
#   Round 001 PKL -> blend -> shared_dense trainable -> 1 epoch
#   Round 002 PKL -> blend -> shared_dense trainable -> 1 epoch
#   ...
#   Round 020 PKL -> blend -> shared_dense trainable -> 1 epoch
#
# SAME:
#   - initial model
#   - train/val/test split
#   - architecture
#   - class weights
#   - optimizer settings
#   - seed
#   - number of local epochs
#
# Test set is evaluated only after training.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import time
import random
import pickle
from datetime import datetime

import numpy as np


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 512

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
# OTHER IMPORTS
# ============================================================

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.utils import class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import (
    layers,
    models,
    optimizers
)


# ============================================================
# PATHS
# ============================================================

CSV_PATH = (
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/Multi_class_Datasets/"
    "UNSW_NB15_testing-set.csv"
)

LABEL_COL = "attack_cat"


# ------------------------------------------------------------
# PKL directory
#
# Expected:
# global_shared_round_001.pkl
# ...
# global_shared_round_020.pkl
# ------------------------------------------------------------

SERVER_PKL_DIR = (
    "/nfs/aalqahtani/"
    "server_logs_pftl_shared"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUT_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "unseen_sequential_"
    "Standalone_Frozen_FineTuned"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

START_ROUND = 1
END_ROUND = 20

TOTAL_EPOCHS = (
    END_ROUND
    -
    START_ROUND
    +
    1
)


# ============================================================
# BLENDING
# ============================================================

GAMMA_LOCAL = 0.10
GAMMA_GLOBAL = 0.90


# ============================================================
# TRAINING
# ============================================================

BATCH = 1024

LR = 1e-4

PRIVATE_DIM = 16

SHARED_DIM = 8


# ============================================================
# UTILITIES
# ============================================================

def reset_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    tf.random.set_seed(SEED)


def now_str():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# PKL PATH
# ============================================================

def get_round_pkl(round_id):

    return os.path.join(
        SERVER_PKL_DIR,
        f"global_shared_round_{round_id:03d}.pkl"
    )


# ============================================================
# LOAD PKL
# ============================================================

def load_server_shared_pkl(
    pkl_path
):

    if not os.path.exists(
        pkl_path
    ):

        raise FileNotFoundError(
            f"\nMissing PKL:\n{pkl_path}"
        )


    with open(
        pkl_path,
        "rb"
    ) as f:

        obj = pickle.load(f)


    if (
        isinstance(obj, dict)
        and
        "global_shared" in obj
    ):

        shared = obj[
            "global_shared"
        ]


    elif (
        isinstance(obj, dict)
        and
        "shared_dense" in obj
    ):

        shared = obj[
            "shared_dense"
        ]


    elif isinstance(
        obj,
        (list, tuple)
    ):

        shared = obj


    else:

        raise ValueError(

            f"\nCannot find global_shared "
            f"in:\n{pkl_path}\n"

            f"Type: {type(obj)}\n"

            f"Keys: "
            f"{list(obj.keys()) if isinstance(obj, dict) else 'N/A'}"
        )


    shared = [

        np.asarray(w)

        for w in shared
    ]


    return shared


# ============================================================
# BLEND
# ============================================================

def blend_shared(
    local_shared,
    global_shared
):

    if (
        len(local_shared)
        !=
        len(global_shared)
    ):

        raise ValueError(

            "Shared parameter count mismatch: "

            f"local={len(local_shared)}, "
            f"global={len(global_shared)}"
        )


    blended = []


    for i, (
        local_w,
        global_w
    ) in enumerate(

        zip(
            local_shared,
            global_shared
        )
    ):

        if (
            local_w.shape
            !=
            global_w.shape
        ):

            raise ValueError(

                f"\nshared_dense shape mismatch "
                f"at index {i}:\n"

                f"local  = {local_w.shape}\n"

                f"global = {global_w.shape}\n"
            )


        new_w = (

            GAMMA_LOCAL
            *
            local_w

            +

            GAMMA_GLOBAL
            *
            global_w
        )


        blended.append(
            new_w
        )


    return blended


# ============================================================
# MODEL
# ============================================================

def build_cnn_multiclass(
    input_shape,
    num_classes
):

    inp = layers.Input(
        shape=input_shape,
        name="input"
    )


    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="valid"
    )(inp)


    x = layers.BatchNormalization()(x)


    x = layers.MaxPooling1D(
        pool_size=2
    )(x)


    x = layers.Dropout(
        0.25
    )(x)


    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="valid"
    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.MaxPooling1D(
        pool_size=2
    )(x)


    x = layers.Dropout(
        0.25
    )(x)


    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="valid"
    )(x)


    x = layers.BatchNormalization()(x)


    x = layers.GlobalAveragePooling1D()(x)


    # --------------------------------------------------------
    # PRIVATE
    # --------------------------------------------------------

    x = layers.Dense(
        PRIVATE_DIM,
        activation="relu",
        name="private_dense"
    )(x)


    # --------------------------------------------------------
    # SHARED
    # --------------------------------------------------------

    x = layers.Dense(
        SHARED_DIM,
        activation="relu",
        name="shared_dense"
    )(x)


    # --------------------------------------------------------
    # PRIVATE OUTPUT
    # --------------------------------------------------------

    out = layers.Dense(
        num_classes,
        activation="softmax",
        name="y_out"
    )(x)


    return models.Model(
        inputs=inp,
        outputs=out
    )


# ============================================================
# COMPILE
# ============================================================

def compile_model(
    model
):

    model.compile(

        optimizer=optimizers.Adam(

            learning_rate=LR,

            clipnorm=1.0
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[

            tf.keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )


# ============================================================
# PREDICT
# ============================================================

def predict_labels(
    model,
    X
):

    probs = model.predict(

        X,

        batch_size=BATCH,

        verbose=0
    )


    return np.argmax(
        probs,
        axis=1
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    labels_all
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        labels=labels_all,

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

        labels=labels_all,

        average="weighted",

        zero_division=0
    )


    return {

        "accuracy":
            float(accuracy),

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
            float(weighted_f1)
    }


# ============================================================
# CONFUSION MATRIX
#
# SAME BLUE STYLE AS YOUR IMAGE
# ============================================================

def save_cm(
    cm,
    class_names,
    save_path,
    normalized=False
):

    if normalized:

        row_sum = cm.sum(
            axis=1,
            keepdims=True
        )


        cm_plot = np.divide(

            cm.astype(float),

            row_sum,

            out=np.zeros_like(
                cm,
                dtype=float
            ),

            where=(
                row_sum != 0
            )
        )


        fmt = ".2f"

        vmin = 0.0
        vmax = 1.0


    else:

        cm_plot = cm

        fmt = "d"

        vmin = None
        vmax = None


    plt.figure(
        figsize=(11, 9)
    )


    ax = sns.heatmap(

        cm_plot,

        annot=True,

        fmt=fmt,

        cmap="Blues",

        vmin=vmin,

        vmax=vmax,

        xticklabels=class_names,

        yticklabels=class_names,

        cbar=True,

        annot_kws={
            "size": 18,
            "weight": "bold"
        }
    )


    # --------------------------------------------------------
    # BLACK / WHITE ANNOTATION
    # --------------------------------------------------------

    if normalized:

        threshold = 0.50

    else:

        threshold = (

            np.max(cm_plot)
            /
            2.0

            if np.max(cm_plot) > 0

            else 0
        )


    for text_obj, value in zip(

        ax.texts,

        cm_plot.flatten()
    ):

        if value > threshold:

            text_obj.set_color(
                "white"
            )

        else:

            text_obj.set_color(
                "black"
            )


    plt.xlabel(
        "Predicted",
        fontsize=18
    )


    plt.ylabel(
        "Actual",
        fontsize=18
    )


    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=14
    )


    plt.yticks(
        rotation=0,
        fontsize=14
    )


    cbar = (
        ax.collections[0]
        .colorbar
    )


    cbar.ax.tick_params(
        labelsize=13
    )


    plt.tight_layout()


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# SAVE CONDITION CONFUSION MATRICES
# ============================================================

def save_condition_cms(
    condition,
    y_true,
    y_pred,
    class_names,
    labels_all
):

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=labels_all
    )


    counts_path = os.path.join(

        OUT_DIR,

        f"CM_COUNTS_"
        f"{condition}_"
        f"seed{SEED}.png"
    )


    norm_path = os.path.join(

        OUT_DIR,

        f"CM_NORM_"
        f"{condition}_"
        f"seed{SEED}.png"
    )


    save_cm(

        cm,

        class_names,

        counts_path,

        normalized=False
    )


    save_cm(

        cm,

        class_names,

        norm_path,

        normalized=True
    )


    return (
        counts_path,
        norm_path
    )


# ============================================================
# PER-CLASS METRICS
# ============================================================

def add_per_class_metrics(
    row,
    y_true,
    y_pred,
    class_names,
    labels_all
):

    (
        precision,
        recall,
        f1,
        support

    ) = precision_recall_fscore_support(

        y_true,

        y_pred,

        labels=labels_all,

        average=None,

        zero_division=0
    )


    for i, class_name in enumerate(
        class_names
    ):

        safe_name = (

            str(class_name)

            .strip()

            .replace(
                " ",
                "_"
            )

            .replace(
                "/",
                "_"
            )
        )


        row[
            f"{safe_name}_precision"
        ] = float(
            precision[i]
        )


        row[
            f"{safe_name}_recall"
        ] = float(
            recall[i]
        )


        row[
            f"{safe_name}_f1"
        ] = float(
            f1[i]
        )


        row[
            f"{safe_name}_support"
        ] = int(
            support[i]
        )


    return row


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    name,
    metrics
):

    print(
        "\n"
        + "=" * 80
    )


    print(
        name
    )


    print(
        "=" * 80
    )


    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.6f}"
    )


    print(
        f"Macro-Precision : "
        f"{metrics['macro_precision']:.6f}"
    )


    print(
        f"Macro-Recall    : "
        f"{metrics['macro_recall']:.6f}"
    )


    print(
        f"Macro-F1        : "
        f"{metrics['macro_f1']:.6f}"
    )


    print(
        f"Weighted-Prec   : "
        f"{metrics['weighted_precision']:.6f}"
    )


    print(
        f"Weighted-Recall : "
        f"{metrics['weighted_recall']:.6f}"
    )


    print(
        f"Weighted-F1     : "
        f"{metrics['weighted_f1']:.6f}"
    )


# ============================================================
# STANDALONE
# ============================================================

def train_standalone(
    initial_weights,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    input_shape,
    num_classes
):

    print(
        "\n\n"
        + "#" * 90
    )


    print(
        "CONDITION 1: STANDALONE"
    )


    print(
        "#" * 90
    )


    reset_seed()


    model = build_cnn_multiclass(

        input_shape,

        num_classes
    )


    # --------------------------------------------------------
    # EXACT SAME INITIAL MODEL
    # --------------------------------------------------------

    model.set_weights(
        initial_weights
    )


    model.get_layer(
        "shared_dense"
    ).trainable = True


    compile_model(
        model
    )


    start = time.perf_counter()


    # --------------------------------------------------------
    # 20 LOCAL EPOCHS
    # --------------------------------------------------------

    model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=TOTAL_EPOCHS,

        batch_size=BATCH,

        class_weight=class_weights,

        verbose=1,

        shuffle=True
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    path = os.path.join(

        OUT_DIR,

        "Standalone_final.keras"
    )


    model.save(
        path
    )


    return (
        model,
        elapsed
    )


# ============================================================
# SEQUENTIAL TRANSFER CONDITION
#
# Used by BOTH:
#
# Frozen
# FineTuned
# ============================================================

def train_sequential_transfer(
    condition,
    shared_trainable,
    initial_weights,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    input_shape,
    num_classes,
    class_names,
    labels_all
):

    print(
        "\n\n"
        + "#" * 90
    )


    print(
        f"CONDITION: {condition}"
    )


    print(
        "shared_dense.trainable =",
        shared_trainable
    )


    print(
        "#" * 90
    )


    reset_seed()


    model = build_cnn_multiclass(

        input_shape,

        num_classes
    )


    # --------------------------------------------------------
    # EXACT SAME INITIAL MODEL
    # --------------------------------------------------------

    model.set_weights(
        initial_weights
    )


    # --------------------------------------------------------
    # SET TRAINABILITY BEFORE COMPILE
    # --------------------------------------------------------

    model.get_layer(
        "shared_dense"
    ).trainable = (
        shared_trainable
    )


    compile_model(
        model
    )


    round_rows = []


    start = time.perf_counter()


    # ========================================================
    # ROUNDS 1 -> 20
    # ========================================================

    for round_id in range(
        START_ROUND,
        END_ROUND + 1
    ):

        print(
            "\n"
            + "=" * 90
        )


        print(
            f"{condition} | "
            f"SERVER ROUND "
            f"{round_id:03d}"
        )


        print(
            "=" * 90
        )


        # ----------------------------------------------------
        # LOAD CURRENT SERVER ROUND
        # ----------------------------------------------------

        pkl_path = get_round_pkl(
            round_id
        )


        global_shared = (
            load_server_shared_pkl(
                pkl_path
            )
        )


        # ----------------------------------------------------
        # GET CURRENT LOCAL SHARED PARAMETERS
        #
        # FineTuned:
        # These include previous local updates.
        #
        # Frozen:
        # These contain the previously blended shared
        # parameters but no gradient updates to shared_dense.
        # ----------------------------------------------------

        current_local_shared = (

            model

            .get_layer(
                "shared_dense"
            )

            .get_weights()
        )


        # ----------------------------------------------------
        # BLEND
        # ----------------------------------------------------

        blended_shared = blend_shared(

            current_local_shared,

            global_shared
        )


        model.get_layer(
            "shared_dense"
        ).set_weights(
            blended_shared
        )


        print(
            "\nBlend:"
        )


        print(
            f"{GAMMA_LOCAL:.2f} "
            f"* current local"
        )


        print(
            f"+ {GAMMA_GLOBAL:.2f} "
            f"* global round "
            f"{round_id:03d}"
        )


        print(
            "shared_dense.trainable:",
            model.get_layer(
                "shared_dense"
            ).trainable
        )


        # ----------------------------------------------------
        # ONE LOCAL EPOCH
        # ----------------------------------------------------

        epoch_start = (
            time.perf_counter()
        )


        model.fit(

            X_train,
            y_train,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=1,

            batch_size=BATCH,

            class_weight=class_weights,

            verbose=1,

            shuffle=True
        )


        epoch_time = (

            time.perf_counter()

            -
            epoch_start
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        y_val_hat = predict_labels(

            model,

            X_val
        )


        val_metrics = calculate_metrics(

            y_val,

            y_val_hat,

            labels_all
        )


        print(
            f"\n{condition} "
            f"Round {round_id:03d}"
        )


        print(
            f"Validation Accuracy : "
            f"{val_metrics['accuracy']:.6f}"
        )


        print(
            f"Validation Macro-F1 : "
            f"{val_metrics['macro_f1']:.6f}"
        )


        print(
            f"Validation Recall   : "
            f"{val_metrics['macro_recall']:.6f}"
        )


        # ----------------------------------------------------
        # PER-CLASS VALIDATION
        # ----------------------------------------------------

        (
            class_precision,
            class_recall,
            class_f1,
            class_support

        ) = precision_recall_fscore_support(

            y_val,

            y_val_hat,

            labels=labels_all,

            average=None,

            zero_division=0
        )


        print(
            "\nPer-class validation:"
        )


        for i, class_name in enumerate(
            class_names
        ):

            print(

                f"{class_name:20s} | "

                f"P={class_precision[i]:.4f} | "

                f"R={class_recall[i]:.4f} | "

                f"F1={class_f1[i]:.4f}"
            )


        # ----------------------------------------------------
        # SAVE ROUND ROW
        # ----------------------------------------------------

        row = {

            "condition":
                condition,

            "seed":
                SEED,

            "server_round":
                round_id,

            "local_epoch":
                round_id,

            "gamma_local":
                GAMMA_LOCAL,

            "gamma_global":
                GAMMA_GLOBAL,

            "shared_trainable":
                shared_trainable,

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_macro_precision":
                val_metrics[
                    "macro_precision"
                ],

            "val_macro_recall":
                val_metrics[
                    "macro_recall"
                ],

            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],

            "val_weighted_precision":
                val_metrics[
                    "weighted_precision"
                ],

            "val_weighted_recall":
                val_metrics[
                    "weighted_recall"
                ],

            "val_weighted_f1":
                val_metrics[
                    "weighted_f1"
                ],

            "epoch_time_sec":
                epoch_time,

            "server_pkl":
                pkl_path
        }


        # ----------------------------------------------------
        # PER-CLASS VALIDATION COLUMNS
        # ----------------------------------------------------

        for i, class_name in enumerate(
            class_names
        ):

            safe_name = (

                str(class_name)

                .strip()

                .replace(
                    " ",
                    "_"
                )

                .replace(
                    "/",
                    "_"
                )
            )


            row[
                f"{safe_name}_precision"
            ] = float(
                class_precision[i]
            )


            row[
                f"{safe_name}_recall"
            ] = float(
                class_recall[i]
            )


            row[
                f"{safe_name}_f1"
            ] = float(
                class_f1[i]
            )


            row[
                f"{safe_name}_support"
            ] = int(
                class_support[i]
            )


        round_rows.append(
            row
        )


        # ----------------------------------------------------
        # SAVE ROUND MODEL
        # ----------------------------------------------------

        round_model_path = os.path.join(

            OUT_DIR,

            f"{condition}_"
            f"after_round_"
            f"{round_id:03d}.keras"
        )


        model.save(
            round_model_path
        )


        # ----------------------------------------------------
        # SAVE VALIDATION CM EACH ROUND
        #
        # Useful for your Fuzzers vs Reconnaissance analysis.
        # ----------------------------------------------------

        val_cm = confusion_matrix(

            y_val,

            y_val_hat,

            labels=labels_all
        )


        norm_path = os.path.join(

            OUT_DIR,

            f"CM_NORM_"
            f"{condition}_"
            f"round{round_id:03d}_"
            f"seed{SEED}.png"
        )


        save_cm(

            val_cm,

            class_names,

            norm_path,

            normalized=True
        )


        # ----------------------------------------------------
        # SAVE PROGRESS EVERY ROUND
        # ----------------------------------------------------

        progress_path = os.path.join(

            OUT_DIR,

            f"{condition}_"
            f"validation_by_round.csv"
        )


        pd.DataFrame(
            round_rows
        ).to_csv(

            progress_path,

            index=False
        )


    elapsed = (

        time.perf_counter()

        -
        start
    )


    # --------------------------------------------------------
    # SAVE FINAL ROUND MODEL
    # --------------------------------------------------------

    final_path = os.path.join(

        OUT_DIR,

        f"{condition}_final_round020.keras"
    )


    model.save(
        final_path
    )


    return (
        model,
        elapsed,
        round_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 90
    )


    print(
        "STANDALONE vs FROZEN vs FINETUNED"
    )


    print(
        "SEQUENTIAL PKL ROUNDS 001-020"
    )


    print(
        "=" * 90
    )


    print(
        "\nSeed:",
        SEED
    )


    print(
        "Gamma:",
        (
            GAMMA_LOCAL,
            GAMMA_GLOBAL
        )
    )


    print(
        "Epochs:",
        TOTAL_EPOCHS
    )


    print(
        "Batch:",
        BATCH
    )


    print(
        "LR:",
        LR
    )


    # ========================================================
    # CHECK ALL PKLs
    # ========================================================

    print(
        "\nChecking PKLs..."
    )


    missing = []


    for round_id in range(
        START_ROUND,
        END_ROUND + 1
    ):

        path = get_round_pkl(
            round_id
        )


        if not os.path.exists(
            path
        ):

            missing.append(
                path
            )


    if missing:

        print(
            "\nMissing PKLs:"
        )


        for path in missing:

            print(
                path
            )


        raise FileNotFoundError(
            "Missing server PKLs."
        )


    print(
        "All PKLs found."
    )


    # ========================================================
    # LOAD DATA ONCE
    # ========================================================

    print(
        "\nLoading:"
    )


    print(
        CSV_PATH
    )


    df = pd.read_csv(
        CSV_PATH
    )


    print(
        "Dataset shape:",
        df.shape
    )


    # ========================================================
    # CLEAN COLUMNS
    # ========================================================

    df.columns = (

        df.columns

        .astype(str)

        .str.strip()
    )


    if LABEL_COL not in df.columns:

        raise ValueError(

            f"{LABEL_COL} "
            "not found."
        )


    # ========================================================
    # CLEAN LABEL
    # ========================================================

    df[LABEL_COL] = (

        df[LABEL_COL]

        .astype(str)

        .str.strip()
    )


    # ========================================================
    # NUMERIC FEATURES ONLY
    # ========================================================

    feature_df = df.drop(
        columns=[
            LABEL_COL
        ]
    )


    numeric_cols = (

        feature_df

        .select_dtypes(
            include=[
                np.number
            ]
        )

        .columns

        .tolist()
    )


    print(
        "Numeric features:",
        len(numeric_cols)
    )


    X_df = feature_df[
        numeric_cols
    ].copy()


    X_df = X_df.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan
    )


    X_df = X_df.fillna(
        0.0
    )


    # ========================================================
    # LABEL ENCODER
    # ========================================================

    le = LabelEncoder()


    y = le.fit_transform(
        df[LABEL_COL]
    )


    class_names = (

        le.classes_

        .astype(str)

        .tolist()
    )


    num_classes = len(
        class_names
    )


    labels_all = np.arange(
        num_classes
    )


    print(
        "\nClasses:"
    )


    for i, name in enumerate(
        class_names
    ):

        print(
            i,
            name
        )


    # ========================================================
    # FEATURES
    # ========================================================

    X = X_df.to_numpy(
        dtype=np.float32
    )


    # ========================================================
    # SAME 70 / 15 / 15 SPLIT FOR ALL CONDITIONS
    # ========================================================

    X_train, X_temp, y_train, y_temp = (
        train_test_split(

            X,
            y,

            test_size=0.30,

            random_state=SEED,

            stratify=y
        )
    )


    X_val, X_test, y_val, y_test = (
        train_test_split(

            X_temp,
            y_temp,

            test_size=0.50,

            random_state=SEED,

            stratify=y_temp
        )
    )


    print(
        "\nTrain:",
        len(y_train)
    )


    print(
        "Validation:",
        len(y_val)
    )


    print(
        "Test:",
        len(y_test)
    )


    # ========================================================
    # SAME SCALER FOR ALL CONDITIONS
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


    X_train = np.expand_dims(
        X_train,
        axis=-1
    )


    X_val = np.expand_dims(
        X_val,
        axis=-1
    )


    X_test = np.expand_dims(
        X_test,
        axis=-1
    )


    input_shape = (

        X_train.shape[1],

        1
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    classes_unique = np.unique(
        y_train
    )


    weights = (
        class_weight
        .compute_class_weight(

            class_weight="balanced",

            classes=classes_unique,

            y=y_train
        )
    )


    cw = {

        int(c):
            float(w)

        for c, w in zip(
            classes_unique,
            weights
        )
    }


    print(
        "\nClass weights:"
    )


    print(
        cw
    )


    # ========================================================
    # CREATE MASTER INITIAL MODEL
    #
    # All 3 conditions receive EXACT SAME initial parameters.
    # ========================================================

    reset_seed()


    master_model = build_cnn_multiclass(

        input_shape,

        num_classes
    )


    initial_weights = [

        w.copy()

        for w in master_model.get_weights()
    ]


    print(
        "\nMaster initial model created."
    )


    print(
        "All three conditions will start "
        "from these exact same weights."
    )


    # ========================================================
    # CHECK SERVER SHAPE BEFORE LONG EXPERIMENT
    # ========================================================

    first_global = load_server_shared_pkl(

        get_round_pkl(
            START_ROUND
        )
    )


    master_shared = (

        master_model

        .get_layer(
            "shared_dense"
        )

        .get_weights()
    )


    print(
        "\nLocal shared shapes:"
    )


    print(
        [
            w.shape
            for w in master_shared
        ]
    )


    print(
        "Server shared shapes:"
    )


    print(
        [
            w.shape
            for w in first_global
        ]
    )


    # Trigger shape validation now.
    _ = blend_shared(

        master_shared,

        first_global
    )


    del master_model


    # ========================================================
    # 1. STANDALONE
    # ========================================================

    standalone_model, standalone_time = (
        train_standalone(

            initial_weights,

            X_train,
            y_train,

            X_val,
            y_val,

            cw,

            input_shape,

            num_classes
        )
    )


    # ========================================================
    # STANDALONE TEST
    # ========================================================

    y_test_standalone = predict_labels(

        standalone_model,

        X_test
    )


    standalone_metrics = calculate_metrics(

        y_test,

        y_test_standalone,

        labels_all
    )


    print_results(

        "FINAL TEST — STANDALONE",

        standalone_metrics
    )


    save_condition_cms(

        "Standalone",

        y_test,

        y_test_standalone,

        class_names,

        labels_all
    )


    # ========================================================
    # 2. FROZEN
    # ========================================================

    frozen_model, frozen_time, frozen_rounds = (
        train_sequential_transfer(

            condition="Frozen",

            shared_trainable=False,

            initial_weights=initial_weights,

            X_train=X_train,

            y_train=y_train,

            X_val=X_val,

            y_val=y_val,

            class_weights=cw,

            input_shape=input_shape,

            num_classes=num_classes,

            class_names=class_names,

            labels_all=labels_all
        )
    )


    # ========================================================
    # FROZEN TEST
    # ========================================================

    y_test_frozen = predict_labels(

        frozen_model,

        X_test
    )


    frozen_metrics = calculate_metrics(

        y_test,

        y_test_frozen,

        labels_all
    )


    print_results(

        "FINAL TEST — FROZEN",

        frozen_metrics
    )


    save_condition_cms(

        "Frozen",

        y_test,

        y_test_frozen,

        class_names,

        labels_all
    )


    # ========================================================
    # 3. FINETUNED
    # ========================================================

    finetuned_model, finetuned_time, finetuned_rounds = (
        train_sequential_transfer(

            condition="FineTuned",

            shared_trainable=True,

            initial_weights=initial_weights,

            X_train=X_train,

            y_train=y_train,

            X_val=X_val,

            y_val=y_val,

            class_weights=cw,

            input_shape=input_shape,

            num_classes=num_classes,

            class_names=class_names,

            labels_all=labels_all
        )
    )


    # ========================================================
    # FINETUNED TEST
    # ========================================================

    y_test_finetuned = predict_labels(

        finetuned_model,

        X_test
    )


    finetuned_metrics = calculate_metrics(

        y_test,

        y_test_finetuned,

        labels_all
    )


    print_results(

        "FINAL TEST — FINETUNED",

        finetuned_metrics
    )


    save_condition_cms(

        "FineTuned",

        y_test,

        y_test_finetuned,

        class_names,

        labels_all
    )


    # ========================================================
    # FINAL COMPARISON TABLE
    # ========================================================

    final_rows = []


    conditions = [

        (
            "Standalone",
            standalone_metrics,
            standalone_time,
            y_test_standalone,
            False,
            True
        ),

        (
            "Frozen",
            frozen_metrics,
            frozen_time,
            y_test_frozen,
            True,
            False
        ),

        (
            "FineTuned",
            finetuned_metrics,
            finetuned_time,
            y_test_finetuned,
            True,
            True
        )
    ]


    for (
        condition,
        metrics,
        elapsed,
        y_pred,
        transfer_used,
        shared_trainable
    ) in conditions:

        row = {

            "timestamp":
                now_str(),

            "condition":
                condition,

            "seed":
                SEED,

            "transfer_used":
                transfer_used,

            "sequential_rounds":
                (
                    "001-020"
                    if transfer_used
                    else "None"
                ),

            "gamma_local":
                (
                    GAMMA_LOCAL
                    if transfer_used
                    else np.nan
                ),

            "gamma_global":
                (
                    GAMMA_GLOBAL
                    if transfer_used
                    else np.nan
                ),

            "shared_dense_trainable":
                shared_trainable,

            "local_epochs":
                TOTAL_EPOCHS,

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "macro_precision":
                metrics[
                    "macro_precision"
                ],

            "macro_recall":
                metrics[
                    "macro_recall"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "weighted_precision":
                metrics[
                    "weighted_precision"
                ],

            "weighted_recall":
                metrics[
                    "weighted_recall"
                ],

            "weighted_f1":
                metrics[
                    "weighted_f1"
                ],

            "training_time_sec":
                elapsed
        }


        row = add_per_class_metrics(

            row,

            y_test,

            y_pred,

            class_names,

            labels_all
        )


        final_rows.append(
            row
        )


    # ========================================================
    # SAVE FINAL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        final_rows
    )


    comparison_csv = os.path.join(

        OUT_DIR,

        "FINAL_TEST_"
        "Standalone_vs_Frozen_vs_FineTuned.csv"
    )


    comparison_df.to_csv(

        comparison_csv,

        index=False
    )


    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    print(
        "\n\n"
        + "=" * 100
    )


    print(
        "FINAL COMPARISON"
    )


    print(
        "=" * 100
    )


    print(

        comparison_df[

            [
                "condition",

                "accuracy",

                "macro_precision",

                "macro_recall",

                "macro_f1",

                "weighted_f1"
            ]

        ].to_string(
            index=False
        )
    )


    print(
        "\nSaved comparison:"
    )


    print(
        comparison_csv
    )


    # ========================================================
    # CLASSIFICATION REPORTS
    # ========================================================

    for (
        condition,
        y_pred
    ) in [

        (
            "Standalone",
            y_test_standalone
        ),

        (
            "Frozen",
            y_test_frozen
        ),

        (
            "FineTuned",
            y_test_finetuned
        )
    ]:

        print(
            "\n"
            + "=" * 90
        )


        print(
            f"{condition} "
            f"CLASSIFICATION REPORT"
        )


        print(
            "=" * 90
        )


        print(

            classification_report(

                y_test,

                y_pred,

                labels=labels_all,

                target_names=class_names,

                zero_division=0
            )
        )


    print(
        "\n"
        + "=" * 90
    )


    print(
        "DONE"
    )


    print(
        "=" * 90
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()