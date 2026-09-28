#!/usr/bin/env python3.10
# ============================================================
# Client 1: CIC-ToN-IoT Multiclass PFTL
# CNN + Adaptive Gamma + Safety Gate + Strict Barrier
#
# Scalability support:
#   NUM_VIRTUAL_CLIENTS = 1  -> 6 total federation clients
#   NUM_VIRTUAL_CLIENTS = 5  -> 30 total federation clients
#   NUM_VIRTUAL_CLIENTS = 9  -> 54 total federation clients
#   NUM_VIRTUAL_CLIENTS = 15 -> 90 total federation clients
#   NUM_VIRTUAL_CLIENTS = 20 -> 120 total federation clients
#
# Virtual-client index:
#   python3.10 this_script.py 0
#   python3.10 this_script.py 1
#   ...
#
# Only the first command-line argument identifies the process.
# All experiment parameters are defined inside this file.
# ============================================================

import os
import sys
import time
import csv
import pickle
import random
from datetime import datetime

# ============================================================
# Fixed experiment configuration
# ============================================================

# -------------------------
# Original client identity
# -------------------------
BASE_CLIENT_ID = "client1_ton_iot"
ORIGINAL_CLIENT_NUMBER = 1
NUM_ORIGINAL_CLIENTS = 6

# -------------------------
# Scalability configuration
# -------------------------
#
# Change only this parameter for each scalability experiment:
#
# 1  virtual client per dataset  = 6 total clients
# 5  virtual clients per dataset = 30 total clients
# 9  virtual clients per dataset = 54 total clients
# 15 virtual clients per dataset = 90 total clients
# 20 virtual clients per dataset = 120 total clients
#
NUM_VIRTUAL_CLIENTS = 20

TOTAL_EXPERIMENT_CLIENTS = (
    NUM_ORIGINAL_CLIENTS * NUM_VIRTUAL_CLIENTS
)

NONIID_ALPHA = 0.3
MIN_VIRTUAL_CLIENT_SAMPLES = 20
MAX_DIRICHLET_ATTEMPTS = 2000

# The process index is the only value supplied when launching.
if len(sys.argv) > 1:
    VIRTUAL_CLIENT_INDEX = int(sys.argv[1])
else:
    VIRTUAL_CLIENT_INDEX = 0

if not 0 <= VIRTUAL_CLIENT_INDEX < NUM_VIRTUAL_CLIENTS:
    raise ValueError(
        f"VIRTUAL_CLIENT_INDEX={VIRTUAL_CLIENT_INDEX} is invalid. "
        f"It must be between 0 and {NUM_VIRTUAL_CLIENTS - 1}."
    )

CLIENT_ID = (
    f"{BASE_CLIENT_ID}_v{VIRTUAL_CLIENT_INDEX + 1}"
)

# -------------------------
# Reproducibility
# -------------------------
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

import pandas as pd
import grpc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras import layers, models, optimizers

import myproto_pb2
import myproto_pb2_grpc

# -------------------------
# Federation settings
# -------------------------
SERVER_ADDRESS = "localhost:50051"

NUM_ROUNDS = 20
EPOCHS_PER_ROUND = 1
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

PRIVATE_DIM = 16
SHARED_DIM = 8

# -------------------------
# Dataset
# -------------------------
CSV_PATH = (
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/Multi_class_Datasets/"
    "D1_CIC-ToN-IoT_new.csv"
)

LABEL_COL = "Attack"

TEST_SIZE = 0.15
VAL_SIZE_FROM_REMAINING = 0.17647058823529413

# -------------------------
# Safety gate
# -------------------------
ADAPT_GAMMA = True
EPS = 0.001

GAMMA_GLOBAL_INIT = 0.50
ETA = 0.05
TAU = 0.05
GAMMA_MIN = 0.0
GAMMA_MAX = 1.0

# -------------------------
# Networking
# -------------------------
MAX_GRPC_MSG = 50 * 1024 * 1024

WAIT_FOR_SERVER_MAX_SEC = 600
WAIT_FOR_SERVER_SLEEP_SEC = 2.0

BARRIER_TIMEOUT_SEC = 900
BARRIER_POLL_SEC = 1.0

RPC_TIMEOUT_SEC = 900

# -------------------------
# Output paths
# -------------------------
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
    return float(max(minimum, min(maximum, value)))


def safe_unpickle(payload):
    if not payload:
        return None

    try:
        return pickle.loads(payload)
    except Exception as exc:
        print(
            f"[{CLIENT_ID}] Warning: unable to deserialize "
            f"weights: {exc}"
        )
        return None


def is_valid_shared(weights):
    if not isinstance(weights, (list, tuple)):
        return False

    if len(weights) != 2:
        return False

    try:
        if any(np.isnan(array).any() for array in weights):
            return False
    except Exception:
        return False

    return True


def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as file:
            csv.writer(file).writerow(header)


# ============================================================
# Logging setup
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
            "bytes_uploaded",
            "final_model_bytes_downloaded",
            "total_poll_bytes_downloaded",
            "total_application_bytes",
            "serialization_sec",
            "upload_rpc_sec",
            "barrier_wait_sec",
            "poll_rpc_sec",
            "poll_sleep_sec",
            "poll_count",
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
            "global_macro_f1",
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
            "total_experiment_clients",
            "train_samples",
            "validation_samples",
            "test_samples",
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
# Dirichlet multiclass partition
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
    number_of_samples = len(labels)

    if num_clients == 1:
        all_indices = np.arange(number_of_samples)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        return [all_indices.tolist()]

    if number_of_samples < num_clients:
        raise ValueError(
            f"The training pool has {number_of_samples} samples, "
            f"which is smaller than {num_clients} virtual clients."
        )

    feasible_minimum = max(
        1,
        number_of_samples // num_clients,
    )

    min_client_samples = min(
        min_client_samples,
        feasible_minimum,
    )

    class_ids = np.unique(labels)

    # A deterministic RNG is recreated using the same seed in every
    # virtual process. Therefore, all virtual processes reconstruct
    # exactly the same global partition map and select their own part.
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
                np.full(num_clients, float(alpha))
            )

            expected_counts = (
                proportions * len(class_indices)
            )

            class_counts = np.floor(
                expected_counts
            ).astype(int)

            remainder = (
                len(class_indices)
                - int(class_counts.sum())
            )

            if remainder > 0:
                fractional_parts = (
                    expected_counts - class_counts
                )

                priority_order = np.argsort(
                    fractional_parts
                )[::-1]

                for client_index in priority_order[:remainder]:
                    class_counts[client_index] += 1

            start_index = 0

            for client_index, sample_count in enumerate(
                class_counts
            ):
                end_index = (
                    start_index + int(sample_count)
                )

                if end_index > start_index:
                    client_indices[client_index].extend(
                        class_indices[
                            start_index:end_index
                        ].tolist()
                    )

                start_index = end_index

        client_sizes = np.asarray(
            [
                len(indices)
                for indices in client_indices
            ],
            dtype=int,
        )

        if client_sizes.min() >= min_client_samples:
            for indices in client_indices:
                rng.shuffle(indices)

            flattened_indices = [
                index
                for indices in client_indices
                for index in indices
            ]

            if len(flattened_indices) != number_of_samples:
                raise RuntimeError(
                    "Not all training samples were assigned."
                )

            if len(set(flattened_indices)) != number_of_samples:
                raise RuntimeError(
                    "Duplicate training indices were assigned."
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
        f"Unable to create a valid partition after "
        f"{max_attempts} attempts. Increase NONIID_ALPHA "
        f"or reduce MIN_VIRTUAL_CLIENT_SAMPLES."
    )


# ============================================================
# Metrics
# ============================================================

def predict_labels(model, features, num_classes):
    probabilities = model.predict(
        features,
        batch_size=BATCH_SIZE,
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
        accuracy_score(labels, predictions)
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "predictions": predictions,
    }


def update_gamma_global(
    current_gamma_global,
    local_macro_f1,
    mixed_macro_f1,
):
    delta = float(
        mixed_macro_f1 - local_macro_f1
    )

    denominator = max(float(TAU), 1e-8)

    step = float(ETA) * float(
        np.tanh(delta / denominator)
    )

    new_gamma_global = clip(
        float(current_gamma_global) + step,
        GAMMA_MIN,
        GAMMA_MAX,
    )

    new_gamma_local = 1.0 - new_gamma_global

    return (
        new_gamma_local,
        new_gamma_global,
        delta,
    )


# ============================================================
# Client-1 CNN model
# ============================================================

def build_client1_cnn(input_shape, num_classes):
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

        loss_function = "binary_crossentropy"

        model_metrics = [
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

        model_metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]

    model = models.Model(
        model_input,
        model_output,
        name=f"PFTL_{CLIENT_ID}_CNN",
    )

    optimizer = optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=model_metrics,
    )

    return model


# ============================================================
# PFTL client
# ============================================================

class Client1CICToNIoTVirtualPFTL:
    def __init__(self):
        initialize_logs()

        self.current_round = 0

        self.gamma_global = float(
            GAMMA_GLOBAL_INIT
        )

        self.gamma_local = (
            1.0 - self.gamma_global
        )

        self.experiment_start = time.perf_counter()

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

        self.model = build_client1_cnn(
            self.input_shape,
            self.num_classes,
        )

        print("\n==== Client 1 Model Summary ====")
        self.model.summary()

        shared_shapes = [
            weight.shape
            for weight in self.get_shared()
        ]

        print(
            f"[{CLIENT_ID}] Shared-layer shapes: "
            f"{shared_shapes}"
        )

        if shared_shapes != [(PRIVATE_DIM, SHARED_DIM), (SHARED_DIM,)]:
            raise ValueError(
                f"Unexpected shared-layer shapes: "
                f"{shared_shapes}"
            )

        self.wait_for_aggregator()

        initial_pull = self.pull_global_shared()

        self.current_round = int(
            initial_pull["round"]
        )

        if is_valid_shared(
            initial_pull["weights"]
        ):
            self.set_shared(
                initial_pull["weights"]
            )

            print(
                f"[{CLIENT_ID}] Initial global weights "
                f"loaded at server round "
                f"{self.current_round}."
            )
        else:
            print(
                f"[{CLIENT_ID}] No initial global weights. "
                f"Starting from the local initialization."
            )

    # ========================================================
    # Data
    # ========================================================

    def load_data(self):
        print(
            f"\n[{CLIENT_ID}] Loading CIC-ToN-IoT..."
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
                f"Label column '{LABEL_COL}' not found. "
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

        feature_dataframe = (
            feature_dataframe
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
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

        self.num_classes = int(
            len(self.label_encoder.classes_)
        )

        # ----------------------------------------------------
        # Fixed common validation and test sets
        # ----------------------------------------------------
        try:
            (
                training_pool_features,
                test_features,
                training_pool_labels,
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
                f"[{CLIENT_ID}] Stratified test split "
                f"failed: {exc}"
            )

            (
                training_pool_features,
                test_features,
                training_pool_labels,
                test_labels,
            ) = train_test_split(
                all_features,
                all_labels,
                test_size=TEST_SIZE,
                random_state=SEED,
                stratify=None,
            )

        try:
            (
                training_pool_features,
                validation_features,
                training_pool_labels,
                validation_labels,
            ) = train_test_split(
                training_pool_features,
                training_pool_labels,
                test_size=VAL_SIZE_FROM_REMAINING,
                random_state=SEED,
                stratify=training_pool_labels,
            )
        except ValueError as exc:
            print(
                f"[{CLIENT_ID}] Stratified validation "
                f"split failed: {exc}"
            )

            (
                training_pool_features,
                validation_features,
                training_pool_labels,
                validation_labels,
            ) = train_test_split(
                training_pool_features,
                training_pool_labels,
                test_size=VAL_SIZE_FROM_REMAINING,
                random_state=SEED,
                stratify=None,
            )

        # ----------------------------------------------------
        # Partition only the fixed training pool
        # ----------------------------------------------------
        virtual_partitions = (
            dirichlet_multiclass_partition(
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
        )

        local_indices = np.asarray(
            virtual_partitions[
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
                f"training partition."
            )

        # ----------------------------------------------------
        # Local scaler
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

        test_features = self.scaler.transform(
            test_features
        )

        local_training_features = (
            local_training_features
            .astype(np.float32)[..., np.newaxis]
        )

        validation_features = (
            validation_features
            .astype(np.float32)[..., np.newaxis]
        )

        test_features = (
            test_features
            .astype(np.float32)[..., np.newaxis]
        )

        self.X_train = local_training_features
        self.X_val = validation_features
        self.X_test = test_features

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
        # Local class weights
        # ----------------------------------------------------
        local_class_ids = np.unique(
            self.y_train
        )

        calculated_weights = compute_class_weight(
            class_weight="balanced",
            classes=local_class_ids,
            y=self.y_train,
        )

        self.class_weight = {
            int(class_id): float(weight)
            for class_id, weight in zip(
                local_class_ids,
                calculated_weights,
            )
        }

        # ----------------------------------------------------
        # Log local class distribution
        # ----------------------------------------------------
        class_ids, class_counts = np.unique(
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
                class_ids,
                class_counts,
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

        print("\n=== Client 1 Virtual Partition ===")
        print(f"Client ID                : {CLIENT_ID}")
        print(f"Original client          : {BASE_CLIENT_ID}")
        print(f"Dataset                  : {CSV_PATH}")
        print(f"Total original samples   : {len(all_labels)}")
        print(f"Number of features       : {all_features.shape[1]}")
        print(f"Number of classes        : {self.num_classes}")
        print(f"Class names              : {list(self.label_encoder.classes_)}")
        print(f"Virtual clients/group    : {NUM_VIRTUAL_CLIENTS}")
        print(
            f"Virtual client index     : "
            f"{VIRTUAL_CLIENT_INDEX + 1}/"
            f"{NUM_VIRTUAL_CLIENTS}"
        )
        print(f"Total federation clients : {TOTAL_EXPERIMENT_CLIENTS}")
        print(f"Dirichlet alpha          : {NONIID_ALPHA}")
        print(f"Complete training pool   : {len(training_pool_labels)}")
        print(f"Local training samples   : {len(self.y_train)}")
        print(f"Common validation samples: {len(self.y_val)}")
        print(f"Common test samples      : {len(self.y_test)}")
        print(f"Local classes present    : {len(local_class_ids)}")
        print(f"Local class distribution : {local_distribution}")
        print(f"Output directory         : {OUT_DIR}")

    # ========================================================
    # Shared-layer functions
    # ========================================================

    def get_shared(self):
        return (
            self.model
            .get_layer("shared_dense")
            .get_weights()
        )

    def set_shared(self, weights):
        if not is_valid_shared(weights):
            return False

        try:
            self.model.get_layer(
                "shared_dense"
            ).set_weights(list(weights))

            return True
        except Exception as exc:
            print(
                f"[{CLIENT_ID}] Failed to set "
                f"shared weights: {exc}"
            )

            return False

    def blend_weights(
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
    # gRPC communication
    # ========================================================

    def pull_global_shared(self):
        rpc_start = time.perf_counter()

        response = self.stub.GetSharedWeights(
            myproto_pb2.EmptyRequest(),
            metadata=[
                ("client_id", CLIENT_ID)
            ],
            timeout=RPC_TIMEOUT_SEC,
        )

        rpc_time = (
            time.perf_counter() - rpc_start
        )

        payload = (
            response.weights
            if response.weights
            else b""
        )

        bytes_received = int(len(payload))

        deserialize_start = time.perf_counter()

        weights = (
            safe_unpickle(payload)
            if payload
            else None
        )

        deserialize_time = (
            time.perf_counter()
            - deserialize_start
        )

        return {
            "weights": weights,
            "round": int(response.round),
            "bytes_received": bytes_received,
            "rpc_sec": float(rpc_time),
            "deserialization_sec": float(
                deserialize_time
            ),
        }

    def wait_for_aggregator(self):
        wait_start = time.perf_counter()

        print(
            f"[{CLIENT_ID}] Waiting for aggregator "
            f"at {SERVER_ADDRESS}..."
        )

        while True:
            try:
                pull_result = (
                    self.pull_global_shared()
                )

                print(
                    f"[{CLIENT_ID}] Aggregator reachable. "
                    f"Server round="
                    f"{pull_result['round']}."
                )

                return

            except Exception as exc:
                elapsed = (
                    time.perf_counter()
                    - wait_start
                )

                if elapsed > WAIT_FOR_SERVER_MAX_SEC:
                    raise RuntimeError(
                        f"Aggregator was not reachable "
                        f"after "
                        f"{WAIT_FOR_SERVER_MAX_SEC}s: "
                        f"{exc}"
                    ) from exc

                time.sleep(
                    WAIT_FOR_SERVER_SLEEP_SEC
                )

    def send_shared_update(
        self,
        shared_weights,
        num_samples,
    ):
        serialization_start = time.perf_counter()

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

        acknowledged = bool(
            getattr(
                acknowledgement,
                "ok",
                False,
            )
        )

        return {
            "ok": acknowledged,
            "bytes_uploaded": int(len(payload)),
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
        first_attempt = self.send_shared_update(
            shared_weights,
            num_samples,
        )

        if first_attempt["ok"]:
            return first_attempt

        print(
            f"[{CLIENT_ID}] First upload was "
            f"rejected. Resynchronizing once."
        )

        resync = self.pull_global_shared()

        if (
            int(resync["round"])
            != int(self.current_round)
        ):
            print(
                f"[{CLIENT_ID}] Round resync: "
                f"{self.current_round} -> "
                f"{resync['round']}"
            )

            self.current_round = int(
                resync["round"]
            )

        second_attempt = (
            self.send_shared_update(
                shared_weights,
                num_samples,
            )
        )

        second_attempt[
            "serialization_sec"
        ] += first_attempt[
            "serialization_sec"
        ]

        second_attempt[
            "upload_rpc_sec"
        ] += first_attempt[
            "upload_rpc_sec"
        ]

        second_attempt[
            "bytes_uploaded"
        ] += first_attempt[
            "bytes_uploaded"
        ]

        return second_attempt

    def wait_for_server_round(
        self,
        target_round,
    ):
        barrier_start = time.perf_counter()

        poll_count = 0
        poll_rpc_sec = 0.0
        poll_sleep_sec = 0.0
        deserialization_sec = 0.0

        total_poll_bytes = 0
        final_model_bytes = 0

        latest_weights = None
        latest_round = int(
            self.current_round
        )

        while True:
            pull_result = (
                self.pull_global_shared()
            )

            poll_count += 1

            poll_rpc_sec += float(
                pull_result["rpc_sec"]
            )

            deserialization_sec += float(
                pull_result[
                    "deserialization_sec"
                ]
            )

            total_poll_bytes += int(
                pull_result["bytes_received"]
            )

            latest_weights = (
                pull_result["weights"]
            )

            latest_round = int(
                pull_result["round"]
            )

            final_model_bytes = int(
                pull_result["bytes_received"]
            )

            elapsed = (
                time.perf_counter()
                - barrier_start
            )

            if latest_round >= int(target_round):
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
            class_weight=self.class_weight,
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
    # Metric logging
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
                sent_type,
                f"{self.gamma_local:.6f}",
                f"{self.gamma_global:.6f}",
                f"{delta_f1:+.6f}",
                now_string(),
            ])

    # ========================================================
    # Main strict-barrier loop
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

            print(
                f"\n[{CLIENT_ID}] "
                f"===== Round {round_id}/"
                f"{NUM_ROUNDS} "
                f"(server_round="
                f"{self.current_round}) ====="
            )

            # ------------------------------------------------
            # Align with the current server model
            # ------------------------------------------------
            alignment_pull = (
                self.pull_global_shared()
            )

            if (
                int(alignment_pull["round"])
                != int(self.current_round)
            ):
                print(
                    f"[{CLIENT_ID}] Aligning round "
                    f"{self.current_round} -> "
                    f"{alignment_pull['round']}"
                )

                self.current_round = int(
                    alignment_pull["round"]
                )

            if is_valid_shared(
                alignment_pull["weights"]
            ):
                self.set_shared(
                    alignment_pull["weights"]
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
            # Evaluate local model on validation data
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
            # Pull global candidate and blend
            # ------------------------------------------------
            mix_start = time.perf_counter()

            candidate_pull = (
                self.pull_global_shared()
            )

            global_weights = (
                candidate_pull["weights"]
            )

            if not is_valid_shared(
                global_weights
            ):
                global_weights = local_weights

            mixed_weights = self.blend_weights(
                local_weights,
                global_weights,
            )

            self.set_shared(mixed_weights)

            global_pull_and_mix_sec = (
                time.perf_counter()
                - mix_start
            )

            # ------------------------------------------------
            # Evaluate mixed candidate
            # ------------------------------------------------
            candidate_eval_start = (
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
                - candidate_eval_start
            )

            safety_gate_total_sec = (
                local_validation_eval_sec
                + global_pull_and_mix_sec
                + mixed_candidate_eval_sec
            )

            # ------------------------------------------------
            # Safety gate
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
                weights_to_send = mixed_weights
            else:
                sent_type = "local"
                weights_to_send = local_weights

                self.set_shared(local_weights)

            # ------------------------------------------------
            # Adaptive gamma
            # ------------------------------------------------
            candidate_delta_f1 = float(
                mixed_metrics["macro_f1"]
                - local_metrics["macro_f1"]
            )

            if ADAPT_GAMMA:
                (
                    self.gamma_local,
                    self.gamma_global,
                    candidate_delta_f1,
                ) = update_gamma_global(
                    self.gamma_global,
                    local_metrics["macro_f1"],
                    mixed_metrics["macro_f1"],
                )

            print(
                f"[{CLIENT_ID}] "
                f"Local Macro-F1="
                f"{local_metrics['macro_f1']:.6f} | "
                f"Mixed Macro-F1="
                f"{mixed_metrics['macro_f1']:.6f} | "
                f"Switch="
                f"{'ON' if switch_on else 'OFF'} | "
                f"Send={sent_type} | "
                f"gamma_global="
                f"{self.gamma_global:.3f} | "
                f"delta="
                f"{candidate_delta_f1:+.6f}"
            )

            # ------------------------------------------------
            # Send selected shared representation
            # ------------------------------------------------
            send_stats = (
                self.send_update_with_retry(
                    weights_to_send,
                    len(self.y_train),
                )
            )

            if not send_stats["ok"]:
                print(
                    f"[{CLIENT_ID}] Warning: update "
                    f"rejected after retry."
                )

            # ------------------------------------------------
            # Strict barrier
            # ------------------------------------------------
            target_round = (
                int(self.current_round) + 1
            )

            barrier_stats = (
                self.wait_for_server_round(
                    target_round
                )
            )

            if (
                int(barrier_stats["round"])
                >= target_round
            ):
                self.current_round = int(
                    barrier_stats["round"]
                )

                if is_valid_shared(
                    barrier_stats["weights"]
                ):
                    self.set_shared(
                        barrier_stats["weights"]
                    )

                print(
                    f"[{CLIENT_ID}] Barrier passed. "
                    f"New server round="
                    f"{self.current_round}, "
                    f"wait="
                    f"{barrier_stats['barrier_wait_sec']:.3f}s"
                )
            else:
                print(
                    f"[{CLIENT_ID}] Barrier timeout. "
                    f"Received round="
                    f"{barrier_stats['round']}, "
                    f"expected at least "
                    f"{target_round}."
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
                    int(barrier_stats["poll_count"]),
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
                    int(TOTAL_EXPERIMENT_CLIENTS),
                    int(len(self.y_train)),
                    int(len(self.y_val)),
                    int(len(self.y_test)),
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
                    int(barrier_stats["timed_out"]),
                    now_string(),
                ])

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------
            self.log_metric_stage(
                round_id=round_id,
                stage="LOCAL",
                metrics=local_metrics,
                training_history=training_history,
                switch_on=switch_on,
                sent_type=sent_type,
                delta_f1=candidate_delta_f1,
            )

            self.log_metric_stage(
                round_id=round_id,
                stage="MIXED_CANDIDATE",
                metrics=mixed_metrics,
                training_history=training_history,
                switch_on=switch_on,
                sent_type=sent_type,
                delta_f1=candidate_delta_f1,
            )

            self.log_metric_stage(
                round_id=round_id,
                stage="GLOBAL",
                metrics=global_metrics,
                training_history=training_history,
                switch_on=switch_on,
                sent_type=sent_type,
                delta_f1=candidate_delta_f1,
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
                f"[{CLIENT_ID}] "
                f"GLOBAL Macro-F1="
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
    # Final test
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
                    self.label_encoder.classes_
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

        print("\nSaved CSV files:")
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
    client = Client1CICToNIoTVirtualPFTL()
    client.run()