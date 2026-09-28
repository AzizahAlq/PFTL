#!/usr/bin/env python3.10
# ============================================================
# Client 6: CIC-IoT IDAD-2024 PFTL Virtual Client
#
# Original Client-6 model:
#   Conv1D(64)  -> BatchNorm -> Pool -> Dropout
#   Conv1D(128) -> BatchNorm -> Pool -> Dropout
#   Conv1D(128) -> BatchNorm
#   GlobalAveragePooling
#   Dense(16) private
#   Dense(8) shared
#
# Client-specific preprocessing:
#   1. Replace +/- infinity with NaN
#   2. Fit median imputer on local training data
#   3. Fit StandardScaler on local training data
#
# Scalability settings:
#   NUM_VIRTUAL_CLIENTS = 1  -> 6 total federation clients
#   NUM_VIRTUAL_CLIENTS = 5  -> 30 total federation clients
#   NUM_VIRTUAL_CLIENTS = 9  -> 54 total federation clients
#   NUM_VIRTUAL_CLIENTS = 15 -> 90 total federation clients
#   NUM_VIRTUAL_CLIENTS = 20 -> 120 total federation clients
#
# Run examples:
#   python3.10 client6_IDAD_2024_PFTL_virtual.py 0
#   python3.10 client6_IDAD_2024_PFTL_virtual.py 1
# ============================================================

import os
import sys
import time
import csv
import pickle
import random
from datetime import datetime

# ============================================================
# Client identity
# ============================================================

BASE_CLIENT_ID = "client6_idad_2024"
ORIGINAL_CLIENT_NUMBER = 6
NUM_ORIGINAL_CLIENTS = 6


# ============================================================
# Scalability configuration
# ============================================================

# Change only this value for each scalability experiment.
NUM_VIRTUAL_CLIENTS = 20

TOTAL_EXPERIMENT_CLIENTS = (
    NUM_ORIGINAL_CLIENTS * NUM_VIRTUAL_CLIENTS
)

NONIID_ALPHA = 0.3
MIN_VIRTUAL_CLIENT_SAMPLES = 20
MAX_DIRICHLET_ATTEMPTS = 2000

# The command-line argument identifies the virtual-client process.
if len(sys.argv) > 1:
    VIRTUAL_CLIENT_INDEX = int(sys.argv[1])
else:
    VIRTUAL_CLIENT_INDEX = 0

if not 0 <= VIRTUAL_CLIENT_INDEX < NUM_VIRTUAL_CLIENTS:
    raise ValueError(
        f"Invalid VIRTUAL_CLIENT_INDEX={VIRTUAL_CLIENT_INDEX}. "
        f"Expected a value from 0 to {NUM_VIRTUAL_CLIENTS - 1}."
    )

CLIENT_ID = (
    f"{BASE_CLIENT_ID}_v{VIRTUAL_CLIENT_INDEX + 1}"
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 45

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
# Libraries
# ============================================================

import pandas as pd
import grpc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras import layers, models, optimizers

import myproto_pb2
import myproto_pb2_grpc


# ============================================================
# PFTL configuration
# ============================================================

SERVER_ADDRESS = "localhost:50051"

NUM_ROUNDS = 20
EPOCHS_PER_ROUND = 1

BATCH_SIZE = 256
PREDICT_BATCH_SIZE = 1024

LEARNING_RATE = 1e-3

PRIVATE_DIM = 16
SHARED_DIM = 8


# ============================================================
# Client 6 dataset configuration
# ============================================================

CSV_PATH = (
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/Multi_class_Datasets/"
    "D6_CIC_IOT_IDAD_2024_Balanced_Final.csv"
)

LABEL_COL = "Label"

TEST_SIZE = 0.15
VAL_SIZE_FROM_REMAINING = 0.15 / (1.0 - TEST_SIZE)


# ============================================================
# Safety gate and adaptive gamma
# ============================================================

ADAPT_GAMMA = True

EPS = 0.001

GAMMA_GLOBAL_INIT = 0.50
ETA = 0.05
TAU = 0.05

GAMMA_MIN = 0.10
GAMMA_MAX = 0.90


# ============================================================
# Networking and strict barrier
# ============================================================

MAX_GRPC_MSG = 50 * 1024 * 1024

WAIT_SERVER_MAX_SEC = 600
WAIT_SERVER_POLL_SEC = 2.0

BARRIER_TIMEOUT_SEC = 1800
BARRIER_POLL_SEC = 1.0

RPC_TIMEOUT_SEC = 1800


# ============================================================
# Output paths
# ============================================================

ROOT_OUTPUT_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/scalability_logs"
)

EXPERIMENT_NAME = (
    f"clients_{TOTAL_EXPERIMENT_CLIENTS}"
    f"_alpha_{NONIID_ALPHA}"
    f"_seed_{SEED}"
)

OUT_DIR = os.path.join(
    ROOT_OUTPUT_DIR,
    EXPERIMENT_NAME,
    BASE_CLIENT_ID,
    CLIENT_ID,
)

os.makedirs(OUT_DIR, exist_ok=True)

COMM_LOG = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_communication.csv",
)

METRICS_LOG = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_metrics.csv",
)

CURVE_LOG = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_convergence_curve.csv",
)

SCALABILITY_LOG = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_scalability.csv",
)

PARTITION_LOG = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_partition.csv",
)


# ============================================================
# General utilities
# ============================================================

def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clip(value, minimum, maximum):
    return float(
        max(minimum, min(maximum, value))
    )


def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as file:
            csv.writer(file).writerow(header)


def safe_unpickle(payload):
    if not payload:
        return None

    try:
        return pickle.loads(payload)

    except Exception as exc:
        print(
            f"[{CLIENT_ID}] Failed to deserialize "
            f"shared weights: {exc}"
        )

        return None


def is_valid_shared(weights):
    if not isinstance(weights, (list, tuple)):
        return False

    if len(weights) != 2:
        return False

    try:
        for weight in weights:
            array = np.asarray(weight)

            if np.isnan(array).any():
                return False

            if np.isinf(array).any():
                return False

    except Exception:
        return False

    return True


# ============================================================
# Log initialization
# ============================================================

def initialize_logs():
    ensure_csv(
        COMM_LOG,
        [
            "seed",
            "round",
            "client_id",
            "base_client_id",
            "virtual_client_index",
            "virtual_clients_in_group",
            "total_experiment_clients",
            "train_samples",
            "server_round_before_send",
            "server_round_after_barrier",
            "bytes_uploaded",
            "final_model_bytes_downloaded",
            "total_poll_bytes_downloaded",
            "total_application_bytes",
            "serialization_sec",
            "upload_rpc_sec",
            "barrier_wait_sec",
            "poll_rpc_sec",
            "poll_sleep_sec",
            "deserialization_sec",
            "poll_count",
            "send_attempts",
            "barrier_timeout",
            "timestamp",
        ],
    )

    ensure_csv(
        METRICS_LOG,
        [
            "seed",
            "round",
            "stage",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "switch_on",
            "eps_margin",
            "sent_weights_type",
            "gamma_local",
            "gamma_global",
            "candidate_delta_macro_f1",
            "timestamp",
        ],
    )

    ensure_csv(
        CURVE_LOG,
        [
            "seed",
            "round",
            "local_macro_f1",
            "mixed_candidate_macro_f1",
            "post_aggregation_global_macro_f1",
            "switch_on",
            "gamma_global",
            "elapsed_experiment_sec",
            "timestamp",
        ],
    )

    ensure_csv(
        SCALABILITY_LOG,
        [
            "seed",
            "round",
            "client_id",
            "base_client_id",
            "virtual_client_index",
            "virtual_clients_in_group",
            "total_experiment_clients",
            "train_samples",
            "validation_samples",
            "test_samples",
            "alignment_pull_sec",
            "local_training_sec",
            "local_validation_eval_sec",
            "global_pull_and_mix_sec",
            "mixed_candidate_eval_sec",
            "safety_gate_total_sec",
            "serialization_sec",
            "upload_rpc_sec",
            "barrier_wait_sec",
            "poll_rpc_sec",
            "poll_sleep_sec",
            "deserialization_sec",
            "global_validation_eval_sec",
            "round_total_sec",
            "client_cpu_sec",
            "non_training_overhead_sec",
            "non_training_overhead_percent",
            "safety_gate_overhead_percent",
            "bytes_uploaded",
            "final_model_bytes_downloaded",
            "total_poll_bytes_downloaded",
            "total_application_bytes",
            "poll_count",
            "send_attempts",
            "barrier_timeout",
            "timestamp",
        ],
    )

    ensure_csv(
        PARTITION_LOG,
        [
            "client_id",
            "base_client_id",
            "virtual_client_index",
            "num_virtual_clients",
            "total_experiment_clients",
            "seed",
            "noniid_alpha",
            "class_id",
            "class_name",
            "sample_count",
            "timestamp",
        ],
    )


# ============================================================
# Multiclass Dirichlet partition
# ============================================================

def dirichlet_multiclass_partition(
    labels,
    num_clients,
    alpha,
    seed,
    min_client_samples,
    max_attempts,
):
    labels = np.asarray(labels)
    total_samples = len(labels)

    if num_clients == 1:
        all_indices = np.arange(total_samples)

        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)

        return [all_indices.tolist()]

    if total_samples < num_clients:
        raise ValueError(
            f"Training pool contains {total_samples} samples, "
            f"but {num_clients} virtual clients were requested."
        )

    maximum_feasible_minimum = max(
        1,
        total_samples // num_clients,
    )

    required_minimum = min(
        int(min_client_samples),
        int(maximum_feasible_minimum),
    )

    class_ids = np.unique(labels)

    # Every process uses the same seed and reconstructs the
    # same complete partition map.
    rng = np.random.default_rng(seed)

    for attempt in range(1, max_attempts + 1):
        client_indices = [
            [] for _ in range(num_clients)
        ]

        for class_id in class_ids:
            class_indices = np.where(
                labels == class_id
            )[0].copy()

            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.full(
                    num_clients,
                    float(alpha),
                )
            )

            expected_counts = (
                proportions * len(class_indices)
            )

            integer_counts = np.floor(
                expected_counts
            ).astype(int)

            remainder = (
                len(class_indices)
                - int(integer_counts.sum())
            )

            if remainder > 0:
                fractional_parts = (
                    expected_counts
                    - integer_counts
                )

                priority_order = np.argsort(
                    fractional_parts
                )[::-1]

                for client_index in priority_order[:remainder]:
                    integer_counts[client_index] += 1

            start_position = 0

            for client_index, count in enumerate(
                integer_counts
            ):
                end_position = (
                    start_position + int(count)
                )

                if end_position > start_position:
                    selected_indices = class_indices[
                        start_position:end_position
                    ]

                    client_indices[
                        client_index
                    ].extend(
                        selected_indices.tolist()
                    )

                start_position = end_position

        client_sizes = np.asarray(
            [
                len(indices)
                for indices in client_indices
            ],
            dtype=int,
        )

        if client_sizes.min() >= required_minimum:
            for indices in client_indices:
                rng.shuffle(indices)

            flattened_indices = [
                sample_index
                for indices in client_indices
                for sample_index in indices
            ]

            if len(flattened_indices) != total_samples:
                raise RuntimeError(
                    "Not all IDAD-2024 training samples "
                    "were assigned."
                )

            if len(set(flattened_indices)) != total_samples:
                raise RuntimeError(
                    "Duplicate IDAD-2024 training samples "
                    "were assigned."
                )

            print(
                f"[{CLIENT_ID}] Dirichlet split accepted "
                f"after {attempt} attempt(s)."
            )

            print(
                f"[{CLIENT_ID}] Partition sizes: "
                f"min={client_sizes.min()}, "
                f"mean={client_sizes.mean():.2f}, "
                f"max={client_sizes.max()}"
            )

            return client_indices

    raise RuntimeError(
        f"Unable to generate a valid Dirichlet split after "
        f"{max_attempts} attempts. Increase NONIID_ALPHA, "
        f"reduce NUM_VIRTUAL_CLIENTS, or reduce "
        f"MIN_VIRTUAL_CLIENT_SAMPLES."
    )


# ============================================================
# Prediction and metrics
# ============================================================

def predict_labels(
    model,
    features,
    num_classes,
):
    probabilities = model.predict(
        features,
        batch_size=PREDICT_BATCH_SIZE,
        verbose=0,
    )

    if num_classes == 2:
        return (
            probabilities.reshape(-1) >= 0.5
        ).astype(int)

    return np.argmax(
        probabilities,
        axis=1,
    )


def evaluate_model(
    model,
    features,
    labels,
    num_classes,
):
    predictions = predict_labels(
        model,
        features,
        num_classes,
    )

    accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "predictions": predictions,
    }


def update_gamma_global(
    current_gamma_global,
    local_macro_f1,
    mixed_macro_f1,
):
    delta = float(
        mixed_macro_f1
        - local_macro_f1
    )

    step = float(ETA) * float(
        np.tanh(
            delta / max(float(TAU), 1e-8)
        )
    )

    new_gamma_global = clip(
        float(current_gamma_global) + step,
        GAMMA_MIN,
        GAMMA_MAX,
    )

    new_gamma_local = (
        1.0 - new_gamma_global
    )

    return (
        new_gamma_local,
        new_gamma_global,
        delta,
    )


# ============================================================
# Client 6 CNN model
# ============================================================

def build_client6_cnn(
    input_shape,
    num_classes,
):
    model_input = layers.Input(
        shape=input_shape
    )

    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="valid",
        activation="relu",
    )(model_input)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="valid",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
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

    if num_classes == 2:
        model_output = layers.Dense(
            1,
            activation="sigmoid",
            name="y_out",
        )(x)

        loss_function = (
            "binary_crossentropy"
        )

        metric_list = [
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            )
        ]

    else:
        model_output = layers.Dense(
            num_classes,
            activation="softmax",
            name="y_out",
        )(x)

        loss_function = (
            "sparse_categorical_crossentropy"
        )

        metric_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]

    model = models.Model(
        model_input,
        model_output,
        name=f"PFTL_{CLIENT_ID}_IDAD_CNN",
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=loss_function,
        metrics=metric_list,
    )

    return model


# ============================================================
# Client 6 implementation
# ============================================================

class Client6IDAD2024VirtualPFTL:
    def __init__(self):
        initialize_logs()

        self.current_round = 0

        self.gamma_global = float(
            GAMMA_GLOBAL_INIT
        )

        self.gamma_local = (
            1.0 - self.gamma_global
        )

        self.experiment_start = (
            time.perf_counter()
        )

        self.channel = grpc.insecure_channel(
            SERVER_ADDRESS,
            options=[
                (
                    "grpc.max_send_message_length",
                    MAX_GRPC_MSG,
                ),
                (
                    "grpc.max_receive_message_length",
                    MAX_GRPC_MSG,
                ),
            ],
        )

        self.stub = (
            myproto_pb2_grpc.AggregatorStub(
                self.channel
            )
        )

        self.load_data()

        self.model = build_client6_cnn(
            self.input_shape,
            self.num_classes,
        )

        print(
            "\n==== Client 6 IDAD-2024 Model Summary ===="
        )

        self.model.summary()

        shared_shapes = [
            tuple(weight.shape)
            for weight in self.get_shared()
        ]

        expected_shapes = [
            (PRIVATE_DIM, SHARED_DIM),
            (SHARED_DIM,),
        ]

        print(
            f"[{CLIENT_ID}] Shared-layer shapes: "
            f"{shared_shapes}"
        )

        if shared_shapes != expected_shapes:
            raise ValueError(
                f"Unexpected shared-layer shapes. "
                f"Expected {expected_shapes}, "
                f"received {shared_shapes}."
            )

        self.wait_for_server()

        initial_state = self.pull_global()

        self.current_round = int(
            initial_state["round"]
        )

        if is_valid_shared(
            initial_state["weights"]
        ):
            loaded = self.set_shared_if_valid(
                initial_state["weights"]
            )

            print(
                f"[{CLIENT_ID}] Initial global shared layer "
                f"loaded at server round "
                f"{self.current_round}; "
                f"set_shared={loaded}."
            )

        else:
            print(
                f"[{CLIENT_ID}] No initial global shared "
                f"layer. Starting local-first training at "
                f"server round {self.current_round}."
            )

    # ========================================================
    # Data loading and virtual partitioning
    # ========================================================

    def load_data(self):
        print(
            f"\n[{CLIENT_ID}] Loading CIC-IoT IDAD-2024..."
        )

        dataframe = pd.read_csv(
            CSV_PATH,
            low_memory=False,
        )

        dataframe.columns = (
            dataframe.columns
            .astype(str)
            .str.replace(
                "\ufeff",
                "",
                regex=False,
            )
            .str.strip()
        )

        if LABEL_COL not in dataframe.columns:
            raise ValueError(
                f"Label column '{LABEL_COL}' was not found. "
                f"Available columns:\n"
                f"{list(dataframe.columns)}"
            )

        raw_labels = (
            dataframe[LABEL_COL]
            .astype(str)
            .str.replace(
                "\ufeff",
                "",
                regex=False,
            )
            .str.replace(
                "�",
                "-",
                regex=False,
            )
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
            .str.strip()
            .values
        )

        feature_dataframe = (
            dataframe
            .drop(
                columns=[LABEL_COL],
                errors="ignore",
            )
            .select_dtypes(
                include=[np.number]
            )
            .copy()
        )

        # Preserve NaN for median imputation.
        feature_dataframe = (
            feature_dataframe.replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        all_features = (
            feature_dataframe
            .values
            .astype(np.float32)
        )

        self.label_encoder = LabelEncoder()

        all_labels = (
            self.label_encoder
            .fit_transform(raw_labels)
            .astype(np.int32)
        )

        self.label_classes = (
            self.label_encoder.classes_
        )

        self.num_classes = int(
            len(self.label_classes)
        )

        print(
            "\n=== Client 6 IDAD-2024 Dataset Summary ==="
        )

        print(f"Dataset  : {CSV_PATH}")
        print(f"Samples  : {len(all_labels)}")
        print(f"Features : {all_features.shape[1]}")
        print(f"Classes  : {self.num_classes}")

        print("\nTop 15 label counts:")

        print(
            pd.Series(raw_labels)
            .value_counts()
            .head(15)
        )

        # ----------------------------------------------------
        # Common test partition
        # ----------------------------------------------------
        try:
            (
                training_validation_features,
                test_features,
                training_validation_labels,
                test_labels,
            ) = train_test_split(
                all_features,
                all_labels,
                test_size=TEST_SIZE,
                random_state=SEED,
                stratify=all_labels,
            )

        except ValueError as exc:
            print(
                f"[{CLIENT_ID}] Stratified test split failed: "
                f"{exc}. Using a non-stratified split."
            )

            (
                training_validation_features,
                test_features,
                training_validation_labels,
                test_labels,
            ) = train_test_split(
                all_features,
                all_labels,
                test_size=TEST_SIZE,
                random_state=SEED,
                stratify=None,
            )

        # ----------------------------------------------------
        # Common validation partition
        # ----------------------------------------------------
        try:
            (
                training_pool_features,
                validation_features,
                training_pool_labels,
                validation_labels,
            ) = train_test_split(
                training_validation_features,
                training_validation_labels,
                test_size=VAL_SIZE_FROM_REMAINING,
                random_state=SEED,
                stratify=training_validation_labels,
            )

        except ValueError as exc:
            print(
                f"[{CLIENT_ID}] Stratified validation split "
                f"failed: {exc}. Using a non-stratified split."
            )

            (
                training_pool_features,
                validation_features,
                training_pool_labels,
                validation_labels,
            ) = train_test_split(
                training_validation_features,
                training_validation_labels,
                test_size=VAL_SIZE_FROM_REMAINING,
                random_state=SEED,
                stratify=None,
            )

        # ----------------------------------------------------
        # Dirichlet partition only the training pool
        # ----------------------------------------------------
        partitions = dirichlet_multiclass_partition(
            labels=training_pool_labels,
            num_clients=NUM_VIRTUAL_CLIENTS,
            alpha=NONIID_ALPHA,
            seed=SEED,
            min_client_samples=(
                MIN_VIRTUAL_CLIENT_SAMPLES
            ),
            max_attempts=(
                MAX_DIRICHLET_ATTEMPTS
            ),
        )

        local_indices = np.asarray(
            partitions[
                VIRTUAL_CLIENT_INDEX
            ],
            dtype=int,
        )

        local_training_features = (
            training_pool_features[
                local_indices
            ]
        )

        local_training_labels = (
            training_pool_labels[
                local_indices
            ]
        )

        if len(local_training_labels) == 0:
            raise RuntimeError(
                f"{CLIENT_ID} received an empty "
                f"IDAD-2024 training partition."
            )

        # ----------------------------------------------------
        # Client-local median imputation
        # ----------------------------------------------------
        self.imputer = SimpleImputer(
            strategy="median"
        )

        local_training_features = (
            self.imputer.fit_transform(
                local_training_features
            )
        )

        validation_features = (
            self.imputer.transform(
                validation_features
            )
        )

        test_features = (
            self.imputer.transform(
                test_features
            )
        )

        # ----------------------------------------------------
        # Client-local standardization
        # ----------------------------------------------------
        self.scaler = StandardScaler()

        local_training_features = (
            self.scaler.fit_transform(
                local_training_features
            )
        )

        validation_features = (
            self.scaler.transform(
                validation_features
            )
        )

        test_features = (
            self.scaler.transform(
                test_features
            )
        )

        self.X_train = (
            local_training_features
            .astype(np.float32)[..., np.newaxis]
        )

        self.X_val = (
            validation_features
            .astype(np.float32)[..., np.newaxis]
        )

        self.X_test = (
            test_features
            .astype(np.float32)[..., np.newaxis]
        )

        self.y_train = (
            local_training_labels
            .astype(np.int32)
        )

        self.y_val = (
            validation_labels
            .astype(np.int32)
        )

        self.y_test = (
            test_labels
            .astype(np.int32)
        )

        self.input_shape = (
            self.X_train.shape[1],
            1,
        )

        # ----------------------------------------------------
        # Partition distribution
        # ----------------------------------------------------
        local_class_ids, local_class_counts = np.unique(
            self.y_train,
            return_counts=True,
        )

        local_distribution = {}

        with open(
            PARTITION_LOG,
            "a",
            newline="",
        ) as file:
            writer = csv.writer(file)

            for class_id, count in zip(
                local_class_ids,
                local_class_counts,
            ):
                class_name = (
                    self.label_encoder.classes_[
                        int(class_id)
                    ]
                )

                local_distribution[
                    class_name
                ] = int(count)

                writer.writerow([
                    CLIENT_ID,
                    BASE_CLIENT_ID,
                    int(VIRTUAL_CLIENT_INDEX),
                    int(NUM_VIRTUAL_CLIENTS),
                    int(TOTAL_EXPERIMENT_CLIENTS),
                    int(SEED),
                    f"{NONIID_ALPHA:.6f}",
                    int(class_id),
                    class_name,
                    int(count),
                    now_string(),
                ])

        print(
            "\n=== Client 6 Virtual Partition Summary ==="
        )

        print(f"Client ID                : {CLIENT_ID}")
        print(f"Original client          : {BASE_CLIENT_ID}")

        print(
            f"Virtual client           : "
            f"{VIRTUAL_CLIENT_INDEX + 1}/"
            f"{NUM_VIRTUAL_CLIENTS}"
        )

        print(
            f"Total federation clients : "
            f"{TOTAL_EXPERIMENT_CLIENTS}"
        )

        print(f"Dirichlet alpha          : {NONIID_ALPHA}")

        print(
            f"Complete training pool   : "
            f"{len(training_pool_labels)}"
        )

        print(
            f"Local training samples   : "
            f"{len(self.y_train)}"
        )

        print(
            f"Common validation samples: "
            f"{len(self.y_val)}"
        )

        print(
            f"Common test samples      : "
            f"{len(self.y_test)}"
        )

        print(
            f"Local classes present    : "
            f"{len(local_class_ids)}/"
            f"{self.num_classes}"
        )

        print(
            f"Local class distribution : "
            f"{local_distribution}"
        )

        print("Imputation               : local median")
        print("Standardization          : local StandardScaler")
        print(f"Output directory         : {OUT_DIR}")

    # ========================================================
    # Shared-layer helpers
    # ========================================================

    def get_shared(self):
        return (
            self.model
            .get_layer("shared_dense")
            .get_weights()
        )

    def set_shared_if_valid(
        self,
        weights,
    ):
        if not is_valid_shared(weights):
            return False

        try:
            self.model.get_layer(
                "shared_dense"
            ).set_weights(list(weights))

            return True

        except Exception as exc:
            print(
                f"[{CLIENT_ID}] Failed to set shared "
                f"weights: {exc}"
            )

            return False

    def blend(
        self,
        local_weights,
        global_weights,
    ):
        return [
            (
                self.gamma_local * local_weight
                + self.gamma_global * global_weight
            )
            for local_weight, global_weight
            in zip(local_weights, global_weights)
        ]

    # ========================================================
    # Pull global shared layer
    # ========================================================

    def pull_global(self):
        rpc_start = time.perf_counter()

        response = self.stub.GetSharedWeights(
            myproto_pb2.EmptyRequest(),
            metadata=[
                ("client_id", CLIENT_ID)
            ],
            timeout=RPC_TIMEOUT_SEC,
        )

        rpc_sec = (
            time.perf_counter()
            - rpc_start
        )

        payload = (
            response.weights
            if response.weights
            else b""
        )

        bytes_received = int(
            len(payload)
        )

        deserialization_start = (
            time.perf_counter()
        )

        weights = (
            safe_unpickle(payload)
            if payload
            else None
        )

        deserialization_sec = (
            time.perf_counter()
            - deserialization_start
        )

        return {
            "weights": weights,
            "round": int(response.round),
            "bytes_received": bytes_received,
            "rpc_sec": float(rpc_sec),
            "deserialization_sec": float(
                deserialization_sec
            ),
        }

    # ========================================================
    # Upload shared update
    # ========================================================

    def send_shared_update(
        self,
        shared_weights,
        num_samples,
    ):
        serialization_start = (
            time.perf_counter()
        )

        payload = pickle.dumps(
            shared_weights,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        serialization_sec = (
            time.perf_counter()
            - serialization_start
        )

        request = myproto_pb2.SharedUpdate(
            weights=payload,
            round=int(self.current_round),
            num_samples=int(num_samples),
        )

        upload_start = time.perf_counter()

        acknowledgement = (
            self.stub.SendSharedUpdate(
                request,
                metadata=[
                    ("client_id", CLIENT_ID)
                ],
                timeout=RPC_TIMEOUT_SEC,
            )
        )

        upload_rpc_sec = (
            time.perf_counter()
            - upload_start
        )

        return {
            "ok": bool(
                getattr(
                    acknowledgement,
                    "ok",
                    False,
                )
            ),
            "bytes_uploaded": int(
                len(payload)
            ),
            "serialization_sec": float(
                serialization_sec
            ),
            "upload_rpc_sec": float(
                upload_rpc_sec
            ),
        }

    def send_update_with_retry(
        self,
        shared_weights,
        num_samples,
    ):
        total_bytes_uploaded = 0
        total_serialization_sec = 0.0
        total_upload_rpc_sec = 0.0

        for attempt in range(2):
            result = self.send_shared_update(
                shared_weights,
                num_samples,
            )

            total_bytes_uploaded += int(
                result["bytes_uploaded"]
            )

            total_serialization_sec += float(
                result["serialization_sec"]
            )

            total_upload_rpc_sec += float(
                result["upload_rpc_sec"]
            )

            if result["ok"]:
                return {
                    "ok": True,
                    "bytes_uploaded": int(
                        total_bytes_uploaded
                    ),
                    "serialization_sec": float(
                        total_serialization_sec
                    ),
                    "upload_rpc_sec": float(
                        total_upload_rpc_sec
                    ),
                    "attempts": attempt + 1,
                }

            print(
                f"[{CLIENT_ID}] Upload attempt "
                f"{attempt + 1} was rejected. "
                f"Resynchronizing with the server."
            )

            server_state = self.pull_global()

            self.current_round = int(
                server_state["round"]
            )

            if is_valid_shared(
                server_state["weights"]
            ):
                self.set_shared_if_valid(
                    server_state["weights"]
                )

        return {
            "ok": False,
            "bytes_uploaded": int(
                total_bytes_uploaded
            ),
            "serialization_sec": float(
                total_serialization_sec
            ),
            "upload_rpc_sec": float(
                total_upload_rpc_sec
            ),
            "attempts": 2,
        }

    # ========================================================
    # Server availability
    # ========================================================

    def wait_for_server(self):
        start = time.perf_counter()

        print(
            f"[{CLIENT_ID}] Waiting for aggregator "
            f"at {SERVER_ADDRESS}..."
        )

        while True:
            try:
                server_state = self.pull_global()

                print(
                    f"[{CLIENT_ID}] Aggregator reachable. "
                    f"Server round="
                    f"{server_state['round']}."
                )

                return

            except Exception as exc:
                elapsed = (
                    time.perf_counter()
                    - start
                )

                if elapsed > WAIT_SERVER_MAX_SEC:
                    raise RuntimeError(
                        f"[{CLIENT_ID}] Aggregator was not "
                        f"reachable after "
                        f"{WAIT_SERVER_MAX_SEC}s: {exc}"
                    ) from exc

                time.sleep(
                    WAIT_SERVER_POLL_SEC
                )

    def align_to_server(self):
        server_state = self.pull_global()

        self.current_round = int(
            server_state["round"]
        )

        if is_valid_shared(
            server_state["weights"]
        ):
            self.set_shared_if_valid(
                server_state["weights"]
            )

        return server_state

    # ========================================================
    # Strict barrier
    # ========================================================

    def wait_for_barrier(
        self,
        before_round,
    ):
        target_round = (
            int(before_round) + 1
        )

        barrier_start = (
            time.perf_counter()
        )

        poll_count = 0
        poll_rpc_sec = 0.0
        poll_sleep_sec = 0.0
        deserialization_sec = 0.0

        total_poll_bytes = 0
        final_model_bytes = 0

        latest_weights = None
        latest_round = int(before_round)

        while True:
            server_state = self.pull_global()

            poll_count += 1

            poll_rpc_sec += float(
                server_state["rpc_sec"]
            )

            deserialization_sec += float(
                server_state[
                    "deserialization_sec"
                ]
            )

            total_poll_bytes += int(
                server_state[
                    "bytes_received"
                ]
            )

            final_model_bytes = int(
                server_state[
                    "bytes_received"
                ]
            )

            latest_weights = (
                server_state["weights"]
            )

            latest_round = int(
                server_state["round"]
            )

            elapsed = (
                time.perf_counter()
                - barrier_start
            )

            if latest_round >= target_round:
                if is_valid_shared(
                    latest_weights
                ):
                    self.set_shared_if_valid(
                        latest_weights
                    )

                return {
                    "weights": latest_weights,
                    "round": latest_round,
                    "barrier_wait_sec": float(
                        elapsed
                    ),
                    "poll_count": int(
                        poll_count
                    ),
                    "poll_rpc_sec": float(
                        poll_rpc_sec
                    ),
                    "poll_sleep_sec": float(
                        poll_sleep_sec
                    ),
                    "deserialization_sec": float(
                        deserialization_sec
                    ),
                    "final_model_bytes": int(
                        final_model_bytes
                    ),
                    "total_poll_bytes": int(
                        total_poll_bytes
                    ),
                    "timed_out": 0,
                }

            if elapsed > BARRIER_TIMEOUT_SEC:
                return {
                    "weights": latest_weights,
                    "round": latest_round,
                    "barrier_wait_sec": float(
                        elapsed
                    ),
                    "poll_count": int(
                        poll_count
                    ),
                    "poll_rpc_sec": float(
                        poll_rpc_sec
                    ),
                    "poll_sleep_sec": float(
                        poll_sleep_sec
                    ),
                    "deserialization_sec": float(
                        deserialization_sec
                    ),
                    "final_model_bytes": int(
                        final_model_bytes
                    ),
                    "total_poll_bytes": int(
                        total_poll_bytes
                    ),
                    "timed_out": 1,
                }

            sleep_start = time.perf_counter()

            time.sleep(BARRIER_POLL_SEC)

            poll_sleep_sec += (
                time.perf_counter()
                - sleep_start
            )

    # ========================================================
    # Training
    # ========================================================

    def train_one_epoch(self):
        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(
                self.X_val,
                self.y_val,
            ),
            epochs=EPOCHS_PER_ROUND,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        return {
            "train_loss": float(
                history.history["loss"][0]
            ),
            "train_accuracy": float(
                history.history.get(
                    "accuracy",
                    [np.nan],
                )[0]
            ),
            "validation_loss": float(
                history.history["val_loss"][0]
            ),
            "validation_accuracy": float(
                history.history.get(
                    "val_accuracy",
                    [np.nan],
                )[0]
            ),
        }

    # ========================================================
    # Metrics logging
    # ========================================================

    def log_metric_stage(
        self,
        round_id,
        stage,
        metrics,
        training_history,
        switch_on,
        sent_type,
        delta_f1,
    ):
        with open(
            METRICS_LOG,
            "a",
            newline="",
        ) as file:
            csv.writer(file).writerow([
                int(SEED),
                int(round_id),
                stage,
                f"{metrics['accuracy']:.6f}",
                f"{metrics['macro_precision']:.6f}",
                f"{metrics['macro_recall']:.6f}",
                f"{metrics['macro_f1']:.6f}",
                f"{metrics['weighted_precision']:.6f}",
                f"{metrics['weighted_recall']:.6f}",
                f"{metrics['weighted_f1']:.6f}",
                f"{training_history['train_loss']:.6f}",
                f"{training_history['train_accuracy']:.6f}",
                f"{training_history['validation_loss']:.6f}",
                f"{training_history['validation_accuracy']:.6f}",
                int(switch_on),
                f"{EPS:.6f}",
                sent_type,
                f"{self.gamma_local:.6f}",
                f"{self.gamma_global:.6f}",
                f"{delta_f1:+.6f}",
                now_string(),
            ])

    # ========================================================
    # Main training loop
    # ========================================================

    def run(self):
        for local_iteration in range(
            NUM_ROUNDS
        ):
            round_id = local_iteration + 1

            round_wall_start = (
                time.perf_counter()
            )

            round_cpu_start = (
                time.process_time()
            )

            # ------------------------------------------------
            # Align to current server round
            # ------------------------------------------------
            alignment_start = (
                time.perf_counter()
            )

            self.align_to_server()

            alignment_pull_sec = (
                time.perf_counter()
                - alignment_start
            )

            print(
                f"\n[{CLIENT_ID}] "
                f"===== Round {round_id}/"
                f"{NUM_ROUNDS} | "
                f"server_round="
                f"{self.current_round} ====="
            )

            # ------------------------------------------------
            # Local training
            # ------------------------------------------------
            training_start = (
                time.perf_counter()
            )

            training_history = (
                self.train_one_epoch()
            )

            local_training_sec = (
                time.perf_counter()
                - training_start
            )

            local_weights = self.get_shared()

            # ------------------------------------------------
            # Evaluate local model
            # ------------------------------------------------
            local_eval_start = (
                time.perf_counter()
            )

            local_metrics = evaluate_model(
                self.model,
                self.X_val,
                self.y_val,
                self.num_classes,
            )

            local_validation_eval_sec = (
                time.perf_counter()
                - local_eval_start
            )

            # ------------------------------------------------
            # Pull and mix current global candidate
            # ------------------------------------------------
            mix_start = time.perf_counter()

            candidate_state = self.pull_global()

            global_weights = (
                candidate_state["weights"]
            )

            if not is_valid_shared(
                global_weights
            ):
                global_weights = (
                    local_weights
                )

            mixed_weights = self.blend(
                local_weights,
                global_weights,
            )

            self.set_shared_if_valid(
                mixed_weights
            )

            global_pull_and_mix_sec = (
                time.perf_counter()
                - mix_start
            )

            # ------------------------------------------------
            # Evaluate mixed candidate
            # ------------------------------------------------
            mixed_eval_start = (
                time.perf_counter()
            )

            mixed_metrics = evaluate_model(
                self.model,
                self.X_val,
                self.y_val,
                self.num_classes,
            )

            mixed_candidate_eval_sec = (
                time.perf_counter()
                - mixed_eval_start
            )

            safety_gate_total_sec = (
                local_validation_eval_sec
                + global_pull_and_mix_sec
                + mixed_candidate_eval_sec
            )

            # ------------------------------------------------
            # Macro-F1 safety gate
            # ------------------------------------------------
            switch_on = int(
                mixed_metrics["macro_f1"]
                >= (
                    local_metrics["macro_f1"]
                    + EPS
                )
            )

            if switch_on:
                sent_type = "mixed"
                weights_to_send = (
                    mixed_weights
                )

            else:
                sent_type = "local"
                weights_to_send = (
                    local_weights
                )

                self.set_shared_if_valid(
                    local_weights
                )

            # ------------------------------------------------
            # Adaptive gamma
            # ------------------------------------------------
            if ADAPT_GAMMA:
                (
                    self.gamma_local,
                    self.gamma_global,
                    delta_f1,
                ) = update_gamma_global(
                    self.gamma_global,
                    local_metrics["macro_f1"],
                    mixed_metrics["macro_f1"],
                )

            else:
                delta_f1 = float(
                    mixed_metrics["macro_f1"]
                    - local_metrics["macro_f1"]
                )

            print(
                f"[{CLIENT_ID}] "
                f"Macro-F1 Local="
                f"{local_metrics['macro_f1']:.6f} | "
                f"Mixed="
                f"{mixed_metrics['macro_f1']:.6f} | "
                f"Switch="
                f"{'ON' if switch_on else 'OFF'} | "
                f"Send={sent_type} | "
                f"gamma_global="
                f"{self.gamma_global:.4f} | "
                f"delta={delta_f1:+.6f}"
            )

            # ------------------------------------------------
            # Upload selected weights
            # ------------------------------------------------
            server_round_before_send = int(
                self.current_round
            )

            send_stats = (
                self.send_update_with_retry(
                    weights_to_send,
                    len(self.y_train),
                )
            )

            # Retry can update current_round.
            server_round_before_send = int(
                self.current_round
            )

            if not send_stats["ok"]:
                print(
                    f"[{CLIENT_ID}] Warning: update "
                    f"was rejected after retry."
                )

            # ------------------------------------------------
            # Strict synchronization barrier
            # ------------------------------------------------
            barrier_stats = (
                self.wait_for_barrier(
                    server_round_before_send
                )
            )

            self.current_round = int(
                barrier_stats["round"]
            )

            if barrier_stats["timed_out"]:
                print(
                    f"[{CLIENT_ID}] Barrier timeout. "
                    f"Server round="
                    f"{self.current_round}; expected >= "
                    f"{server_round_before_send + 1}."
                )

            else:
                print(
                    f"[{CLIENT_ID}] Barrier passed: "
                    f"{server_round_before_send} -> "
                    f"{self.current_round} | "
                    f"wait="
                    f"{barrier_stats['barrier_wait_sec']:.3f}s"
                )

            # ------------------------------------------------
            # Evaluate actual post-aggregation global model
            # ------------------------------------------------
            global_eval_start = (
                time.perf_counter()
            )

            global_metrics = evaluate_model(
                self.model,
                self.X_val,
                self.y_val,
                self.num_classes,
            )

            global_validation_eval_sec = (
                time.perf_counter()
                - global_eval_start
            )

            round_total_sec = (
                time.perf_counter()
                - round_wall_start
            )

            client_cpu_sec = (
                time.process_time()
                - round_cpu_start
            )

            non_training_overhead_sec = max(
                0.0,
                round_total_sec
                - local_training_sec,
            )

            if round_total_sec > 0:
                non_training_overhead_percent = (
                    100.0
                    * non_training_overhead_sec
                    / round_total_sec
                )

                safety_gate_overhead_percent = (
                    100.0
                    * safety_gate_total_sec
                    / round_total_sec
                )

            else:
                non_training_overhead_percent = 0.0
                safety_gate_overhead_percent = 0.0

            total_application_bytes = (
                int(send_stats["bytes_uploaded"])
                + int(
                    barrier_stats[
                        "total_poll_bytes"
                    ]
                )
            )

            # ------------------------------------------------
            # Communication log
            # ------------------------------------------------
            with open(
                COMM_LOG,
                "a",
                newline="",
            ) as file:
                csv.writer(file).writerow([
                    int(SEED),
                    int(round_id),
                    CLIENT_ID,
                    BASE_CLIENT_ID,
                    int(VIRTUAL_CLIENT_INDEX),
                    int(NUM_VIRTUAL_CLIENTS),
                    int(TOTAL_EXPERIMENT_CLIENTS),
                    int(len(self.y_train)),
                    int(server_round_before_send),
                    int(self.current_round),
                    int(send_stats["bytes_uploaded"]),
                    int(
                        barrier_stats[
                            "final_model_bytes"
                        ]
                    ),
                    int(
                        barrier_stats[
                            "total_poll_bytes"
                        ]
                    ),
                    int(total_application_bytes),
                    f"{send_stats['serialization_sec']:.6f}",
                    f"{send_stats['upload_rpc_sec']:.6f}",
                    f"{barrier_stats['barrier_wait_sec']:.6f}",
                    f"{barrier_stats['poll_rpc_sec']:.6f}",
                    f"{barrier_stats['poll_sleep_sec']:.6f}",
                    f"{barrier_stats['deserialization_sec']:.6f}",
                    int(barrier_stats["poll_count"]),
                    int(send_stats["attempts"]),
                    int(barrier_stats["timed_out"]),
                    now_string(),
                ])

            # ------------------------------------------------
            # Scalability log
            # ------------------------------------------------
            with open(
                SCALABILITY_LOG,
                "a",
                newline="",
            ) as file:
                csv.writer(file).writerow([
                    int(SEED),
                    int(round_id),
                    CLIENT_ID,
                    BASE_CLIENT_ID,
                    int(VIRTUAL_CLIENT_INDEX),
                    int(NUM_VIRTUAL_CLIENTS),
                    int(TOTAL_EXPERIMENT_CLIENTS),
                    int(len(self.y_train)),
                    int(len(self.y_val)),
                    int(len(self.y_test)),
                    f"{alignment_pull_sec:.6f}",
                    f"{local_training_sec:.6f}",
                    f"{local_validation_eval_sec:.6f}",
                    f"{global_pull_and_mix_sec:.6f}",
                    f"{mixed_candidate_eval_sec:.6f}",
                    f"{safety_gate_total_sec:.6f}",
                    f"{send_stats['serialization_sec']:.6f}",
                    f"{send_stats['upload_rpc_sec']:.6f}",
                    f"{barrier_stats['barrier_wait_sec']:.6f}",
                    f"{barrier_stats['poll_rpc_sec']:.6f}",
                    f"{barrier_stats['poll_sleep_sec']:.6f}",
                    f"{barrier_stats['deserialization_sec']:.6f}",
                    f"{global_validation_eval_sec:.6f}",
                    f"{round_total_sec:.6f}",
                    f"{client_cpu_sec:.6f}",
                    f"{non_training_overhead_sec:.6f}",
                    f"{non_training_overhead_percent:.6f}",
                    f"{safety_gate_overhead_percent:.6f}",
                    int(send_stats["bytes_uploaded"]),
                    int(
                        barrier_stats[
                            "final_model_bytes"
                        ]
                    ),
                    int(
                        barrier_stats[
                            "total_poll_bytes"
                        ]
                    ),
                    int(total_application_bytes),
                    int(barrier_stats["poll_count"]),
                    int(send_stats["attempts"]),
                    int(barrier_stats["timed_out"]),
                    now_string(),
                ])

            # ------------------------------------------------
            # Metrics logs
            # ------------------------------------------------
            self.log_metric_stage(
                round_id,
                "LOCAL",
                local_metrics,
                training_history,
                switch_on,
                sent_type,
                delta_f1,
            )

            self.log_metric_stage(
                round_id,
                "MIXED_CANDIDATE",
                mixed_metrics,
                training_history,
                switch_on,
                sent_type,
                delta_f1,
            )

            self.log_metric_stage(
                round_id,
                "GLOBAL",
                global_metrics,
                training_history,
                switch_on,
                sent_type,
                delta_f1,
            )

            # ------------------------------------------------
            # Convergence curve
            # ------------------------------------------------
            elapsed_experiment_sec = (
                time.perf_counter()
                - self.experiment_start
            )

            with open(
                CURVE_LOG,
                "a",
                newline="",
            ) as file:
                csv.writer(file).writerow([
                    int(SEED),
                    int(round_id),
                    f"{local_metrics['macro_f1']:.16f}",
                    f"{mixed_metrics['macro_f1']:.16f}",
                    f"{global_metrics['macro_f1']:.16f}",
                    int(switch_on),
                    f"{self.gamma_global:.16f}",
                    f"{elapsed_experiment_sec:.6f}",
                    now_string(),
                ])

            print(
                f"[{CLIENT_ID}] Round summary | "
                f"global Macro-F1="
                f"{global_metrics['macro_f1']:.6f} | "
                f"train={local_training_sec:.3f}s | "
                f"barrier="
                f"{barrier_stats['barrier_wait_sec']:.3f}s | "
                f"round={round_total_sec:.3f}s | "
                f"upload="
                f"{send_stats['bytes_uploaded']} B | "
                f"download="
                f"{barrier_stats['final_model_bytes']} B"
            )

        self.final_test_evaluation()

    # ========================================================
    # Final test evaluation
    # ========================================================

    def final_test_evaluation(self):
        print(
            f"\n[{CLIENT_ID}] "
            f"===== FINAL TEST EVALUATION ====="
        )

        test_metrics = evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
            self.num_classes,
        )

        print(
            f"Accuracy            : "
            f"{test_metrics['accuracy']:.6f}"
        )

        print(
            f"Macro-Precision     : "
            f"{test_metrics['macro_precision']:.6f}"
        )

        print(
            f"Macro-Recall        : "
            f"{test_metrics['macro_recall']:.6f}"
        )

        print(
            f"Macro-F1            : "
            f"{test_metrics['macro_f1']:.6f}"
        )

        print(
            f"Weighted-Precision  : "
            f"{test_metrics['weighted_precision']:.6f}"
        )

        print(
            f"Weighted-Recall     : "
            f"{test_metrics['weighted_recall']:.6f}"
        )

        print(
            f"Weighted-F1         : "
            f"{test_metrics['weighted_f1']:.6f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                self.y_test,
                test_metrics["predictions"],
                labels=np.arange(
                    self.num_classes
                ),
                target_names=(
                    self.label_classes
                ),
                zero_division=0,
            )
        )

        print("Confusion Matrix:")

        print(
            confusion_matrix(
                self.y_test,
                test_metrics["predictions"],
                labels=np.arange(
                    self.num_classes
                ),
            )
        )

        print("\nSaved files:")
        print(f"Communication : {COMM_LOG}")
        print(f"Metrics       : {METRICS_LOG}")
        print(f"Convergence   : {CURVE_LOG}")
        print(f"Scalability   : {SCALABILITY_LOG}")
        print(f"Partition     : {PARTITION_LOG}")

        print(f"\n[{CLIENT_ID}] DONE")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    Client6IDAD2024VirtualPFTL().run()