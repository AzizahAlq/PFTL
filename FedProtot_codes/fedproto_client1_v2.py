"""
FedProto Client 1 — CIC-ToN-IoT
================================
This client must be used with the FedProto aggregator, not the PTFL aggregator.
"""

# ============================================================
# Imports and deterministic configuration
# ============================================================

import os
import time
import csv
import json
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

import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import pandas as pd
import grpc

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

import myproto_pb2
import myproto_pb2_grpc


# ============================================================
# Experiment settings
# ============================================================

# ============================================================
# Fixed experiment configuration
# Everything is defined inside this file; no external environment
# variables are required.
# ============================================================

SERVER_ADDRESS = "localhost:50052"
CLIENT_ID = "client1"
NUM_ROUNDS = 20

LOCAL_EPOCHS = 1
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

PRIVATE_DIM = 16
PROTO_DIM = 8
LAMBDA_PROTO = 1.0
CLIP_NORM = 1.0

TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)

MAX_GRPC_MSG = 50 * 1024 * 1024
WAIT_FOR_SERVER_MAX_SEC = 600
WAIT_FOR_SERVER_SLEEP_SEC = 2.0
BARRIER_TIMEOUT_SEC = 1800
BARRIER_POLL_SEC = 1.0

# Convergence definition:
# First round whose validation Macro-F1 is >= 99% of final validation
# Macro-F1 for three consecutive rounds.
CONVERGENCE_RATIO = 0.99
CONVERGENCE_PATIENCE = 3


# ============================================================
# Paths
# ============================================================

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
    / "fedproto_logs"
    / "client1_ton_iot"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMM_LOG = OUT_DIR / f"{CLIENT_ID}_fedproto_comm_log.csv"
METRICS_LOG = OUT_DIR / f"{CLIENT_ID}_fedproto_metrics_log.csv"
CURVE_CSV = OUT_DIR / f"{CLIENT_ID}_fedproto_curve.csv"
PROTOTYPE_LOG = OUT_DIR / f"{CLIENT_ID}_prototype_log.csv"

FINAL_METRICS_CSV = OUT_DIR / f"{CLIENT_ID}_final_metrics.csv"
CLASSIFICATION_REPORT_CSV = OUT_DIR / f"{CLIENT_ID}_classification_report.csv"

CONFUSION_MATRIX_CSV = OUT_DIR / f"{CLIENT_ID}_confusion_matrix.csv"
CONFUSION_MATRIX_NORMALIZED_CSV = (
    OUT_DIR / f"{CLIENT_ID}_confusion_matrix_normalized.csv"
)
CONFUSION_MATRIX_PNG = OUT_DIR / f"{CLIENT_ID}_confusion_matrix_raw.png"
CONFUSION_MATRIX_NORMALIZED_PNG = (
    OUT_DIR / f"{CLIENT_ID}_confusion_matrix_normalized.png"
)

CONVERGENCE_CURVE_PNG = OUT_DIR / f"{CLIENT_ID}_convergence_curve.png"
CONVERGENCE_SUMMARY_JSON = OUT_DIR / f"{CLIENT_ID}_convergence_summary.json"

LOCAL_PROTOTYPES_FINAL_PKL = (
    OUT_DIR / f"{CLIENT_ID}_final_local_prototypes.pkl"
)
GLOBAL_PROTOTYPES_FINAL_PKL = (
    OUT_DIR / f"{CLIENT_ID}_final_global_prototypes.pkl"
)

MODEL_PATH = OUT_DIR / f"{CLIENT_ID}_fedproto_final_model.keras"
SCALER_PATH = OUT_DIR / f"{CLIENT_ID}_standard_scaler.pkl"


# ============================================================
# FedProto embedding visualization / representation analysis
# ============================================================

VISUALIZE_EMBEDDING = True
VIS_MAX_SAMPLES = 5000

# Keep projection randomness fixed for reproducibility.
TSNE_SEED = 123
UMAP_SEED = 123

# Client 1 has more classes than Client 3, so colors are generated
# deterministically from tab20 while the legend order follows local IDs.
EMBEDDINGS_CSV = (
    OUT_DIR / f"{CLIENT_ID}_fedproto_embeddings_seed{SEED}.csv"
)

TSNE_COORDINATES_CSV = (
    OUT_DIR / f"{CLIENT_ID}_fedproto_TSNE_coordinates_seed{SEED}.csv"
)

TSNE_PNG = (
    OUT_DIR / f"{CLIENT_ID}_fedproto_embedding_TSNE_seed{SEED}.png"
)

UMAP_COORDINATES_CSV = (
    OUT_DIR / f"{CLIENT_ID}_fedproto_UMAP_coordinates_seed{SEED}.csv"
)

UMAP_PNG = (
    OUT_DIR / f"{CLIENT_ID}_fedproto_embedding_UMAP_seed{SEED}.png"
)


# ============================================================
# Utilities
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_pickle_loads(payload: bytes, default=None):
    if not payload:
        return default
    try:
        return pickle.loads(payload)
    except Exception as exc:
        print(f"[{CLIENT_ID}] Warning: failed to unpickle payload: {exc}")
        return default


def validate_prototype_vector(prototype) -> bool:
    try:
        vector = np.asarray(prototype, dtype=np.float32)
    except Exception:
        return False

    return (
        vector.ndim == 1
        and vector.shape[0] == PROTO_DIM
        and np.all(np.isfinite(vector))
    )


def normalize_global_prototype_payload(payload) -> dict:
    if payload is None:
        return {}

    if (
        isinstance(payload, dict)
        and "prototypes" in payload
        and isinstance(payload["prototypes"], dict)
    ):
        payload = payload["prototypes"]

    if not isinstance(payload, dict):
        return {}

    normalized = {}

    for global_id_raw, entry in payload.items():
        try:
            global_id = int(global_id_raw)
        except Exception:
            continue

        if isinstance(entry, dict):
            prototype = entry.get("prototype")
            semantic_label = str(
                entry.get("semantic_label", f"GLOBAL_{global_id}")
            )
            count = int(entry.get("count", 0))
        else:
            prototype = entry
            semantic_label = f"GLOBAL_{global_id}"
            count = 0

        if not validate_prototype_vector(prototype):
            continue

        normalized[global_id] = {
            "semantic_label": semantic_label,
            "prototype": np.asarray(prototype, dtype=np.float32),
            "count": count,
        }

    return normalized


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))

    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
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


def ensure_csv_headers() -> None:
    if not COMM_LOG.exists():
        with COMM_LOG.open("w", newline="") as file:
            csv.writer(file).writerow([
                "seed",
                "round",
                "server_round_before_send",
                "server_round_after_barrier",
                "prototype_classes_sent",
                "prototype_values_sent",
                "payload_bytes",
                "rtt_sec",
                "barrier_wait_sec",
                "timestamp",
            ])

    if not METRICS_LOG.exists():
        with METRICS_LOG.open("w", newline="") as file:
            csv.writer(file).writerow([
                "seed",
                "round",
                "server_round",
                "train_total_loss",
                "train_classification_loss",
                "train_prototype_loss",
                "train_accuracy",
                "validation_accuracy",
                "validation_macro_precision",
                "validation_macro_recall",
                "validation_macro_f1",
                "validation_weighted_precision",
                "validation_weighted_recall",
                "validation_weighted_f1",
                "matched_prototype_samples",
                "processed_training_samples",
                "prototype_coverage_ratio",
                "available_global_classes_for_client",
                "local_train_time_sec",
                "prototype_compute_time_sec",
                "communication_rtt_sec",
                "barrier_wait_sec",
                "round_total_time_sec",
                "cumulative_time_sec",
                "timestamp",
            ])

    if not CURVE_CSV.exists():
        with CURVE_CSV.open("w", newline="") as file:
            csv.writer(file).writerow([
                "seed",
                "round",
                "validation_macro_f1",
                "validation_weighted_f1",
                "validation_accuracy",
                "classification_loss",
                "prototype_loss",
                "round_total_time_sec",
                "cumulative_time_sec",
            ])

    if not PROTOTYPE_LOG.exists():
        with PROTOTYPE_LOG.open("w", newline="") as file:
            csv.writer(file).writerow([
                "seed",
                "round",
                "local_id",
                "global_id",
                "semantic_label",
                "class_train_count",
                "prototype_l2_norm",
                "global_prototype_available_before_training",
                "timestamp",
            ])


# ============================================================
# Model
# ============================================================

def build_fedproto_model(
    input_shape: tuple,
    num_classes: int,
) -> tf.keras.Model:
    """
    Preserves the supplied Client 1 PTFL architecture.

    Important:
    The third convolution uses 64 filters, matching the supplied
    Client 1 implementation.
    """

    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv1D(
        64, 3, padding="valid", activation="relu", name="private_conv1"
    )(inputs)
    x = layers.BatchNormalization(name="private_bn1")(x)
    x = layers.MaxPooling1D(2, name="private_pool1")(x)
    x = layers.Dropout(0.25, name="private_dropout1")(x)

    x = layers.Conv1D(
        128, 3, padding="valid", activation="relu", name="private_conv2"
    )(x)
    x = layers.BatchNormalization(name="private_bn2")(x)
    x = layers.MaxPooling1D(2, name="private_pool2")(x)
    x = layers.Dropout(0.25, name="private_dropout2")(x)

    x = layers.Conv1D(
        128, 3, padding="valid", activation="relu", name="private_conv3"
    )(x)
    x = layers.BatchNormalization(name="private_bn3")(x)

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
        name=f"FedProto_{CLIENT_ID}_CNN",
    )


# ============================================================
# FedProto Client 1
# ============================================================

class FedProtoClient1:
    def __init__(self):
        ensure_csv_headers()

        self.current_round = 0
        self.global_prototypes = {}
        self.round_history = []
        self.experiment_start_time = time.perf_counter()

        self._load_local_global_mapping()
        self._load_data()

        self.model = build_fedproto_model(
            self.input_shape,
            self.num_classes,
        )

        self.optimizer = optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=CLIP_NORM,
        )

        self.classification_loss_fn = (
            tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction="none",
            )
        )

        self.channel = grpc.insecure_channel(
            SERVER_ADDRESS,
            options=[
                ("grpc.max_send_message_length", MAX_GRPC_MSG),
                ("grpc.max_receive_message_length", MAX_GRPC_MSG),
            ],
        )
        self.stub = myproto_pb2_grpc.AggregatorStub(self.channel)

        print("\n" + "=" * 80)
        print("FEDPROTO CLIENT 1 MODEL")
        print("=" * 80)
        self.model.summary()

        self.wait_for_aggregator_reachable()

        prototypes, server_round = self.pull_global_prototypes()
        self.current_round = int(server_round)
        self.global_prototypes = prototypes

        print(f"\n[{CLIENT_ID}] Initial server round: {self.current_round}")
        print(
            f"[{CLIENT_ID}] Initial valid global prototypes: "
            f"{len(self.global_prototypes)}"
        )

    # --------------------------------------------------------
    # Mapping
    # --------------------------------------------------------

    def _load_local_global_mapping(self) -> None:
        if not GLOBAL_MAPPING_PATH.exists():
            raise FileNotFoundError(
                f"Mapping file not found:\n{GLOBAL_MAPPING_PATH}"
            )

        with GLOBAL_MAPPING_PATH.open("r", encoding="utf-8") as file:
            all_mappings = json.load(file)

        if CLIENT_ID not in all_mappings:
            raise KeyError(
                f"{CLIENT_ID} is missing from:\n{GLOBAL_MAPPING_PATH}"
            )

        client_mapping = all_mappings[CLIENT_ID]
        raw_mapping = client_mapping.get("local_to_global")

        if not isinstance(raw_mapping, dict):
            raise KeyError(
                f"'local_to_global' is missing for {CLIENT_ID}."
            )

        self.local_to_global = {}
        self.local_to_semantic = {}
        self.global_to_local = {}

        for local_id_text, entry in raw_mapping.items():
            local_id = int(local_id_text)
            global_id = int(entry["global_id"])
            semantic_label = str(entry["semantic_label"])

            self.local_to_global[local_id] = global_id
            self.local_to_semantic[local_id] = semantic_label
            self.global_to_local[global_id] = local_id

        print("\n" + "=" * 80)
        print("CLIENT 1 LOCAL-TO-GLOBAL MAPPING")
        print("=" * 80)

        for local_id in sorted(self.local_to_global):
            print(
                f"Local {local_id:2d}"
                f" -> Global {self.local_to_global[local_id]:2d}"
                f" -> {self.local_to_semantic[local_id]}"
            )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    def _load_data(self) -> None:
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

        self.num_classes = len(self.local_to_global)

        expected_ids = set(range(self.num_classes))
        observed_ids = {int(v) for v in np.unique(y)}

        if observed_ids != expected_ids:
            raise ValueError(
                "Client 1 labels are not contiguous.\n"
                f"Expected: {sorted(expected_ids)}\n"
                f"Observed: {sorted(observed_ids)}"
            )

        for local_id in sorted(observed_ids):
            expected_semantic = self.local_to_semantic[local_id]
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

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.X_train = X_train[..., np.newaxis].astype(np.float32)
        self.X_val = X_val[..., np.newaxis].astype(np.float32)
        self.X_test = X_test[..., np.newaxis].astype(np.float32)

        self.y_train = y_train.astype(np.int32)
        self.y_val = y_val.astype(np.int32)
        self.y_test = y_test.astype(np.int32)

        self.input_shape = (self.X_train.shape[1], 1)

        classes = np.unique(self.y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.y_train,
        )

        self.class_weight = {
            int(class_id): float(weight)
            for class_id, weight in zip(classes, weights)
        }

        self.class_weight_tensor = tf.constant(
            [
                self.class_weight[class_id]
                for class_id in range(self.num_classes)
            ],
            dtype=tf.float32,
        )

        print("\n" + "=" * 80)
        print("FEDPROTO CLIENT 1 READINESS")
        print("=" * 80)
        print(f"Dataset:       {CSV_PATH}")
        print(f"Rows:          {len(X):,}")
        print(f"Features:      {X.shape[1]}")
        print(f"Classes:       {self.num_classes}")
        print(f"Prototype dim: {PROTO_DIM}")
        print(f"Lambda proto:  {LAMBDA_PROTO}")
        print(
            f"Splits: train={len(self.y_train):,}, "
            f"val={len(self.y_val):,}, "
            f"test={len(self.y_test):,}"
        )

        print("\nTraining class counts and weights:")
        train_counts = pd.Series(self.y_train).value_counts().sort_index()

        for local_id in range(self.num_classes):
            print(
                f"  {local_id:2d} "
                f"{self.local_to_semantic[local_id]:<30} "
                f"count={int(train_counts.get(local_id, 0)):<7d} "
                f"weight={self.class_weight[local_id]:.6f}"
            )

    # --------------------------------------------------------
    # RPC
    # --------------------------------------------------------

    def wait_for_aggregator_reachable(self) -> None:
        start = time.perf_counter()
        print(
            f"\n[{CLIENT_ID}] Waiting for FedProto aggregator "
            f"at {SERVER_ADDRESS}..."
        )

        while True:
            try:
                _, server_round = self.pull_global_prototypes()
                print(
                    f"[{CLIENT_ID}] Aggregator reachable "
                    f"(server_round={server_round})."
                )
                return
            except Exception as exc:
                if time.perf_counter() - start > WAIT_FOR_SERVER_MAX_SEC:
                    raise RuntimeError(
                        f"Aggregator not reachable after "
                        f"{WAIT_FOR_SERVER_MAX_SEC}s: {exc}"
                    ) from exc
                time.sleep(WAIT_FOR_SERVER_SLEEP_SEC)

    def pull_global_prototypes(self) -> tuple[dict, int]:
        response = self.stub.GetSharedWeights(
            myproto_pb2.EmptyRequest(),
            metadata=[("client_id", CLIENT_ID)],
        )

        payload = safe_pickle_loads(response.weights, default={})

        return (
            normalize_global_prototype_payload(payload),
            int(response.round),
        )

    def send_prototype_update(
        self,
        local_prototypes: dict,
    ) -> tuple[bool, int, float]:
        update = {
            "client_id": CLIENT_ID,
            "round": int(self.current_round),
            "proto_dim": PROTO_DIM,
            "prototypes": local_prototypes,
        }

        payload = pickle.dumps(
            update,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        request = myproto_pb2.SharedUpdate(
            weights=payload,
            round=int(self.current_round),
            num_samples=int(len(self.y_train)),
        )

        start = time.perf_counter()
        ack = self.stub.SendSharedUpdate(
            request,
            metadata=[("client_id", CLIENT_ID)],
        )
        rtt = time.perf_counter() - start

        return (
            bool(getattr(ack, "ok", False)),
            len(payload),
            rtt,
        )

    def send_update_retry_once(
        self,
        local_prototypes: dict,
    ) -> tuple[bool, int, float]:
        ok, bytes_sent, rtt = self.send_prototype_update(
            local_prototypes
        )

        if ok:
            return ok, bytes_sent, rtt

        _, server_round = self.pull_global_prototypes()

        if int(server_round) != int(self.current_round):
            print(
                f"[{CLIENT_ID}] Resync after rejection: "
                f"{self.current_round} -> {server_round}"
            )
            self.current_round = int(server_round)

        return self.send_prototype_update(local_prototypes)

    def wait_for_server_round(
        self,
        target_round: int,
        timeout_sec: float,
    ) -> tuple[dict, int, float]:
        start = time.perf_counter()

        while True:
            prototypes, server_round = self.pull_global_prototypes()

            if int(server_round) >= int(target_round):
                return (
                    prototypes,
                    int(server_round),
                    time.perf_counter() - start,
                )

            elapsed = time.perf_counter() - start
            if elapsed > timeout_sec:
                return prototypes, int(server_round), elapsed

            time.sleep(BARRIER_POLL_SEC)

    # --------------------------------------------------------
    # Global prototype targets
    # --------------------------------------------------------

    def create_global_target_matrix(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros(
            (self.num_classes, PROTO_DIM),
            dtype=np.float32,
        )
        availability = np.zeros(
            self.num_classes,
            dtype=np.float32,
        )

        for local_id in range(self.num_classes):
            global_id = self.local_to_global[local_id]

            if global_id not in self.global_prototypes:
                continue

            prototype = self.global_prototypes[global_id]["prototype"]

            if not validate_prototype_vector(prototype):
                continue

            matrix[local_id] = np.asarray(
                prototype,
                dtype=np.float32,
            )
            availability[local_id] = 1.0

        return matrix, availability

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

    def train_one_local_round(self) -> dict:
        target_matrix_np, availability_np = (
            self.create_global_target_matrix()
        )

        target_matrix = tf.constant(
            target_matrix_np,
            dtype=tf.float32,
        )
        availability = tf.constant(
            availability_np,
            dtype=tf.float32,
        )

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (self.X_train, self.y_train)
            )
            .shuffle(
                buffer_size=len(self.y_train),
                seed=SEED + int(self.current_round),
                reshuffle_each_iteration=True,
            )
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        proto_loss_sum = 0.0

        correct = 0
        processed = 0
        matched = 0

        for _epoch in range(LOCAL_EPOCHS):
            for x_batch, y_batch in dataset:
                y_batch = tf.cast(y_batch, tf.int32)

                with tf.GradientTape() as tape:
                    outputs = self.model(x_batch, training=True)
                    logits = outputs["logits"]
                    embeddings = outputs["embedding"]

                    per_sample_ce = self.classification_loss_fn(
                        y_batch,
                        logits,
                    )
                    sample_weights = tf.gather(
                        self.class_weight_tensor,
                        y_batch,
                    )

                    classification_loss = tf.reduce_mean(
                        per_sample_ce * sample_weights
                    )

                    batch_targets = tf.gather(
                        target_matrix,
                        y_batch,
                    )
                    batch_available = tf.gather(
                        availability,
                        y_batch,
                    )

                    per_sample_proto_distance = tf.reduce_mean(
                        tf.square(embeddings - batch_targets),
                        axis=1,
                    )

                    matched_count = tf.reduce_sum(
                        batch_available
                    )

                    prototype_loss = tf.where(
                        matched_count > 0,
                        tf.reduce_sum(
                            per_sample_proto_distance
                            * batch_available
                        )
                        / tf.maximum(matched_count, 1.0),
                        tf.constant(0.0, dtype=tf.float32),
                    )

                    total_loss = (
                        classification_loss
                        + LAMBDA_PROTO * prototype_loss
                    )

                gradients = tape.gradient(
                    total_loss,
                    self.model.trainable_variables,
                )

                gradient_pairs = [
                    (g, v)
                    for g, v in zip(
                        gradients,
                        self.model.trainable_variables,
                    )
                    if g is not None
                ]

                self.optimizer.apply_gradients(gradient_pairs)

                predicted = tf.argmax(
                    logits,
                    axis=1,
                    output_type=tf.int32,
                )

                batch_n = int(tf.shape(y_batch)[0].numpy())

                correct += int(
                    tf.reduce_sum(
                        tf.cast(
                            tf.equal(predicted, y_batch),
                            tf.int32,
                        )
                    ).numpy()
                )

                processed += batch_n
                matched += int(matched_count.numpy())

                total_loss_sum += float(total_loss.numpy()) * batch_n
                ce_loss_sum += float(classification_loss.numpy()) * batch_n
                proto_loss_sum += float(prototype_loss.numpy()) * batch_n

        denominator = max(processed, 1)

        return {
            "total_loss": total_loss_sum / denominator,
            "classification_loss": ce_loss_sum / denominator,
            "prototype_loss": proto_loss_sum / denominator,
            "accuracy": correct / denominator,
            "matched_prototype_samples": matched,
            "processed_samples": processed,
            "prototype_coverage_ratio": matched / denominator,
            "available_global_classes_for_client": int(
                np.sum(availability_np)
            ),
        }

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = (
            tf.data.Dataset
            .from_tensor_slices(X)
            .batch(BATCH_SIZE)
        )

        predictions = []
        probabilities = []
        embeddings = []

        for x_batch in dataset:
            outputs = self.model(x_batch, training=False)
            logits = outputs["logits"]
            batch_embeddings = outputs["embedding"]

            batch_probabilities = tf.nn.softmax(
                logits,
                axis=1,
            )
            batch_predictions = tf.argmax(
                batch_probabilities,
                axis=1,
                output_type=tf.int32,
            )

            predictions.append(batch_predictions.numpy())
            probabilities.append(batch_probabilities.numpy())
            embeddings.append(batch_embeddings.numpy())

        return (
            np.concatenate(predictions, axis=0),
            np.concatenate(probabilities, axis=0),
            np.concatenate(embeddings, axis=0),
        )

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        y_pred, _, _ = self.predict(X)
        result = calculate_metrics(y, y_pred)
        result["predictions"] = y_pred
        return result

    # --------------------------------------------------------
    # Prototype computation
    # --------------------------------------------------------

    def compute_local_prototypes(self) -> dict:
        _, _, embeddings = self.predict(self.X_train)

        local_prototypes = {}

        for local_id in range(self.num_classes):
            mask = self.y_train == local_id
            count = int(np.sum(mask))

            if count == 0:
                continue

            prototype = np.mean(
                embeddings[mask],
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)

            if not validate_prototype_vector(prototype):
                raise ValueError(
                    f"Invalid prototype for local class {local_id}."
                )

            global_id = self.local_to_global[local_id]

            local_prototypes[global_id] = {
                "semantic_label": self.local_to_semantic[local_id],
                "prototype": prototype,
                "count": count,
                "local_id": local_id,
            }

        return local_prototypes

    def log_local_prototypes(
        self,
        round_id: int,
        local_prototypes: dict,
    ) -> None:
        with PROTOTYPE_LOG.open("a", newline="") as file:
            writer = csv.writer(file)

            for global_id in sorted(local_prototypes):
                entry = local_prototypes[global_id]
                vector = np.asarray(
                    entry["prototype"],
                    dtype=np.float32,
                )

                writer.writerow([
                    SEED,
                    round_id,
                    entry["local_id"],
                    global_id,
                    entry["semantic_label"],
                    entry["count"],
                    f"{np.linalg.norm(vector):.8f}",
                    int(global_id in self.global_prototypes),
                    now_str(),
                ])

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    def log_round_metrics(
        self,
        round_id: int,
        server_round: int,
        train_result: dict,
        val_result: dict,
        local_train_time: float,
        prototype_compute_time: float,
        communication_rtt: float,
        barrier_wait: float,
        round_total_time: float,
        cumulative_time: float,
    ) -> None:
        with METRICS_LOG.open("a", newline="") as file:
            csv.writer(file).writerow([
                SEED,
                round_id,
                server_round,
                f"{train_result['total_loss']:.8f}",
                f"{train_result['classification_loss']:.8f}",
                f"{train_result['prototype_loss']:.8f}",
                f"{train_result['accuracy']:.8f}",
                f"{val_result['accuracy']:.8f}",
                f"{val_result['macro_precision']:.8f}",
                f"{val_result['macro_recall']:.8f}",
                f"{val_result['macro_f1']:.8f}",
                f"{val_result['weighted_precision']:.8f}",
                f"{val_result['weighted_recall']:.8f}",
                f"{val_result['weighted_f1']:.8f}",
                train_result["matched_prototype_samples"],
                train_result["processed_samples"],
                f"{train_result['prototype_coverage_ratio']:.8f}",
                train_result["available_global_classes_for_client"],
                f"{local_train_time:.4f}",
                f"{prototype_compute_time:.4f}",
                f"{communication_rtt:.4f}",
                f"{barrier_wait:.4f}",
                f"{round_total_time:.4f}",
                f"{cumulative_time:.4f}",
                now_str(),
            ])

        with CURVE_CSV.open("a", newline="") as file:
            csv.writer(file).writerow([
                SEED,
                round_id,
                f"{val_result['macro_f1']:.16f}",
                f"{val_result['weighted_f1']:.16f}",
                f"{val_result['accuracy']:.16f}",
                f"{train_result['classification_loss']:.16f}",
                f"{train_result['prototype_loss']:.16f}",
                f"{round_total_time:.6f}",
                f"{cumulative_time:.6f}",
            ])

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def run(self) -> None:
        for local_iteration in range(NUM_ROUNDS):
            round_id = local_iteration + 1
            round_start = time.perf_counter()

            pulled_prototypes, server_round = (
                self.pull_global_prototypes()
            )

            if int(server_round) != int(self.current_round):
                print(
                    f"[{CLIENT_ID}] Align round "
                    f"{self.current_round} -> {server_round}"
                )
                self.current_round = int(server_round)

            self.global_prototypes = pulled_prototypes

            print("\n" + "=" * 80)
            print(
                f"[{CLIENT_ID}] ROUND {round_id}/{NUM_ROUNDS} "
                f"| server_round={self.current_round}"
            )
            print(
                f"[{CLIENT_ID}] Global prototypes available: "
                f"{len(self.global_prototypes)}"
            )
            print("=" * 80)

            train_start = time.perf_counter()
            train_result = self.train_one_local_round()
            local_train_time = time.perf_counter() - train_start

            val_result = self.evaluate(
                self.X_val,
                self.y_val,
            )

            proto_start = time.perf_counter()
            local_prototypes = self.compute_local_prototypes()
            prototype_compute_time = time.perf_counter() - proto_start

            self.log_local_prototypes(
                round_id,
                local_prototypes,
            )

            print(
                f"[{CLIENT_ID}] Total loss={train_result['total_loss']:.6f} "
                f"| CE={train_result['classification_loss']:.6f} "
                f"| Proto={train_result['prototype_loss']:.6f}"
            )
            print(
                f"[{CLIENT_ID}] Validation Macro-F1="
                f"{val_result['macro_f1']:.6f} "
                f"| Weighted-F1={val_result['weighted_f1']:.6f} "
                f"| Accuracy={val_result['accuracy']:.6f}"
            )
            print(
                f"[{CLIENT_ID}] Prototype coverage="
                f"{train_result['prototype_coverage_ratio']:.4f} "
                f"| matching global classes="
                f"{train_result['available_global_classes_for_client']} "
                f"| classes sent={len(local_prototypes)}"
            )

            server_round_before = int(self.current_round)

            ok, bytes_sent, rtt = self.send_update_retry_once(
                local_prototypes
            )

            if not ok:
                print(
                    f"[{CLIENT_ID}] Prototype update rejected "
                    "after retry."
                )

            target_round = int(self.current_round) + 1

            new_prototypes, new_server_round, barrier_wait = (
                self.wait_for_server_round(
                    target_round,
                    BARRIER_TIMEOUT_SEC,
                )
            )

            if int(new_server_round) >= target_round:
                self.current_round = int(new_server_round)
                self.global_prototypes = new_prototypes

                print(
                    f"[{CLIENT_ID}] Barrier passed: "
                    f"{server_round_before} -> {self.current_round} "
                    f"(wait={barrier_wait:.2f}s)"
                )
            else:
                raise RuntimeError(
                    f"[{CLIENT_ID}] Barrier timeout: "
                    f"server_round={new_server_round}, "
                    f"expected >= {target_round}."
                )

            round_total_time = time.perf_counter() - round_start
            cumulative_time = (
                time.perf_counter() - self.experiment_start_time
            )

            with COMM_LOG.open("a", newline="") as file:
                csv.writer(file).writerow([
                    SEED,
                    round_id,
                    server_round_before,
                    self.current_round,
                    len(local_prototypes),
                    len(local_prototypes) * PROTO_DIM,
                    bytes_sent,
                    f"{rtt:.4f}",
                    f"{barrier_wait:.4f}",
                    now_str(),
                ])

            self.log_round_metrics(
                round_id,
                server_round_before,
                train_result,
                val_result,
                local_train_time,
                prototype_compute_time,
                rtt,
                barrier_wait,
                round_total_time,
                cumulative_time,
            )

            self.round_history.append({
                "round": round_id,
                "validation_macro_f1": val_result["macro_f1"],
                "validation_weighted_f1": val_result["weighted_f1"],
                "validation_accuracy": val_result["accuracy"],
                "classification_loss": train_result["classification_loss"],
                "prototype_loss": train_result["prototype_loss"],
                "round_total_time_sec": round_total_time,
                "cumulative_time_sec": cumulative_time,
            })

        self.finalize_experiment()

    # --------------------------------------------------------
    # Convergence
    # --------------------------------------------------------

    def determine_convergence(self) -> dict:
        if not self.round_history:
            return {
                "best_round": None,
                "best_validation_macro_f1": None,
                "final_validation_macro_f1": None,
                "convergence_round": None,
                "convergence_time_sec": None,
            }

        values = np.asarray(
            [
                row["validation_macro_f1"]
                for row in self.round_history
            ],
            dtype=np.float64,
        )

        best_index = int(np.argmax(values))
        final_value = float(values[-1])
        threshold = CONVERGENCE_RATIO * final_value

        convergence_index = None

        for start_index in range(len(values)):
            end_index = start_index + CONVERGENCE_PATIENCE

            if end_index > len(values):
                break

            if np.all(values[start_index:end_index] >= threshold):
                convergence_index = start_index
                break

        if convergence_index is None:
            convergence_round = None
            convergence_time = None
        else:
            convergence_round = int(
                self.round_history[convergence_index]["round"]
            )
            convergence_time = float(
                self.round_history[convergence_index][
                    "cumulative_time_sec"
                ]
            )

        return {
            "convergence_definition": (
                f"First round where validation Macro-F1 is at least "
                f"{CONVERGENCE_RATIO:.2%} of final validation Macro-F1 "
                f"for {CONVERGENCE_PATIENCE} consecutive rounds."
            ),
            "best_round": int(
                self.round_history[best_index]["round"]
            ),
            "best_validation_macro_f1": float(values[best_index]),
            "final_validation_macro_f1": final_value,
            "convergence_threshold_macro_f1": float(threshold),
            "convergence_round": convergence_round,
            "convergence_time_sec": convergence_time,
        }

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    def save_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        class_names = [
            self.local_to_semantic[i]
            for i in range(self.num_classes)
        ]
        labels = np.arange(self.num_classes)

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names,
        ).to_csv(CONFUSION_MATRIX_CSV)

        row_sums = matrix.sum(axis=1, keepdims=True)

        normalized = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums != 0,
        )

        pd.DataFrame(
            normalized,
            index=class_names,
            columns=class_names,
        ).to_csv(CONFUSION_MATRIX_NORMALIZED_CSV)

        # With 34 classes, annotating every cell makes the image unreadable.
        # The CSV files preserve every value. PNGs emphasize structure.
        size = 18

        plt.figure(figsize=(size, size))
        plt.imshow(matrix, interpolation="nearest", aspect="auto")
        plt.title("FedProto Client 1 — Raw Confusion Matrix")
        plt.colorbar()
        ticks = np.arange(self.num_classes)
        plt.xticks(ticks, class_names, rotation=90, fontsize=7)
        plt.yticks(ticks, class_names, fontsize=7)
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.tight_layout()
        plt.savefig(
            CONFUSION_MATRIX_PNG,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(size, size))
        plt.imshow(
            normalized,
            interpolation="nearest",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        plt.title(
            "FedProto Client 1 — Row-Normalized Confusion Matrix"
        )
        plt.colorbar()
        plt.xticks(ticks, class_names, rotation=90, fontsize=7)
        plt.yticks(ticks, class_names, fontsize=7)
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.tight_layout()
        plt.savefig(
            CONFUSION_MATRIX_NORMALIZED_PNG,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def save_convergence_curve(
        self,
        convergence_summary: dict,
    ) -> None:
        rounds = [row["round"] for row in self.round_history]
        macro_f1 = [
            row["validation_macro_f1"]
            for row in self.round_history
        ]
        weighted_f1 = [
            row["validation_weighted_f1"]
            for row in self.round_history
        ]
        accuracy = [
            row["validation_accuracy"]
            for row in self.round_history
        ]

        plt.figure(figsize=(9, 6))
        plt.plot(
            rounds,
            macro_f1,
            marker="o",
            label="Validation Macro-F1",
        )
        plt.plot(
            rounds,
            weighted_f1,
            marker="s",
            label="Validation Weighted-F1",
        )
        plt.plot(
            rounds,
            accuracy,
            marker="^",
            label="Validation Accuracy",
        )

        convergence_round = convergence_summary.get(
            "convergence_round"
        )

        if convergence_round is not None:
            plt.axvline(
                convergence_round,
                linestyle="--",
                label=f"Convergence round {convergence_round}",
            )

        plt.xlabel("Communication round")
        plt.ylabel("Score")
        plt.title("FedProto Client 1 Convergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            CONVERGENCE_CURVE_PNG,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # --------------------------------------------------------
    # FedProto prototype-embedding visualization
    # --------------------------------------------------------

    def visualize_prototype_embedding(
        self,
        max_samples: int = VIS_MAX_SAMPLES,
    ) -> None:
        """
        Save and visualize the final 8-D FedProto prototype_embedding
        using the held-out test set.

        Outputs:
        - raw 8-D embedding CSV
        - t-SNE coordinates CSV + PNG
        - UMAP coordinates CSV + PNG (when umap-learn is installed)

        IMPORTANT:
        Silhouette analysis should be computed from EMBEDDINGS_CSV
        (the original 8-D representation), not from the 2-D projections.
        """

        print("\n" + "=" * 80)
        print(f"[{CLIENT_ID}] FEDPROTO PROTOTYPE EMBEDDING VISUALIZATION")
        print("=" * 80)

        X_vis = self.X_test
        y_vis = self.y_test

        print(f"[{CLIENT_ID}] Original test samples: {len(X_vis)}")
        print(f"[{CLIENT_ID}] Number of classes: {self.num_classes}")

        # Stratified subsampling, identical principle to Client 3.
        if len(X_vis) > max_samples:
            all_idx = np.arange(len(y_vis))

            selected_idx, _ = train_test_split(
                all_idx,
                train_size=max_samples,
                random_state=SEED,
                stratify=y_vis,
            )

            X_vis = X_vis[selected_idx]
            y_vis = y_vis[selected_idx]

        print(f"[{CLIENT_ID}] Visualization samples: {len(X_vis)}")

        # predict() returns the final sample-level 8-D prototype_embedding.
        _, _, embeddings = self.predict(X_vis)

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        print(
            f"[{CLIENT_ID}] Prototype embedding shape: "
            f"{embeddings.shape}"
        )

        if embeddings.ndim != 2 or embeddings.shape[1] != PROTO_DIM:
            raise ValueError(
                f"Expected embedding shape (N, {PROTO_DIM}), "
                f"got {embeddings.shape}."
            )

        label_names = np.asarray(
            [
                str(self.local_to_semantic[int(label_id)]).strip()
                for label_id in y_vis
            ],
            dtype=object,
        )

        print(f"\n[{CLIENT_ID}] Visualization labels:")
        for name, count in zip(
            *np.unique(label_names, return_counts=True)
        ):
            print(f"  {name:<30}: {count}")

        # ====================================================
        # Save raw 8-D embeddings
        # ====================================================

        embedding_df = pd.DataFrame(
            embeddings,
            columns=[
                f"proto_dim_{i + 1}"
                for i in range(embeddings.shape[1])
            ],
        )

        embedding_df["label_id"] = y_vis
        embedding_df["label_name"] = label_names

        embedding_df.to_csv(
            EMBEDDINGS_CSV,
            index=False,
        )

        print(
            f"[{CLIENT_ID}] Raw FedProto embeddings saved: "
            f"{EMBEDDINGS_CSV}"
        )

        # Stable class order = local label ID order.
        class_ids = list(range(self.num_classes))
        class_names = [
            str(self.local_to_semantic[class_id]).strip()
            for class_id in class_ids
        ]

        # Deterministic colors. Client 1 may have >20 classes, so use
        # the continuous tab20 colormap rather than hard-coding six colors.
        cmap = plt.get_cmap("tab20")
        denom = max(len(class_names) - 1, 1)
        class_colors = {
            name: cmap(i / denom)
            for i, name in enumerate(class_names)
        }

        # ====================================================
        # t-SNE
        # ====================================================

        print(f"\n[{CLIENT_ID}] Running t-SNE...")

        if len(embeddings) > 31:
            tsne_perplexity = 30
        else:
            tsne_perplexity = max(
                5,
                min(30, len(embeddings) - 1),
            )

        tsne_model = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=TSNE_SEED,
        )

        tsne_2d = tsne_model.fit_transform(embeddings)

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
            f"FedProto Prototype Embedding - t-SNE\n"
            f"{CLIENT_ID} | {self.num_classes} Classes | "
            f"Train Seed={SEED} | t-SNE Seed={TSNE_SEED}"
        )
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            frameon=True,
        )

        plt.tight_layout()
        plt.savefig(
            TSNE_PNG,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"[{CLIENT_ID}] t-SNE figure saved: "
            f"{TSNE_PNG}"
        )
        print(
            f"[{CLIENT_ID}] t-SNE coordinates saved: "
            f"{TSNE_COORDINATES_CSV}"
        )

        # ====================================================
        # UMAP
        # ====================================================

        if UMAP_AVAILABLE:
            print(f"\n[{CLIENT_ID}] Running UMAP...")

            umap_model = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=UMAP_SEED,
            )

            umap_2d = umap_model.fit_transform(embeddings)

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
                f"FedProto Prototype Embedding - UMAP\n"
                f"{CLIENT_ID} | {self.num_classes} Classes | "
                f"Train Seed={SEED} | UMAP Seed={UMAP_SEED}"
            )
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")

            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=7,
                frameon=True,
            )

            plt.tight_layout()
            plt.savefig(
                UMAP_PNG,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            print(
                f"[{CLIENT_ID}] UMAP figure saved: "
                f"{UMAP_PNG}"
            )
            print(
                f"[{CLIENT_ID}] UMAP coordinates saved: "
                f"{UMAP_COORDINATES_CSV}"
            )
        else:
            print(
                f"[{CLIENT_ID}] UMAP skipped: install with "
                f"`pip install umap-learn`."
            )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    def finalize_experiment(self) -> None:
        print("\n" + "=" * 80)
        print(f"[{CLIENT_ID}] FINAL TEST EVALUATION")
        print("=" * 80)

        y_pred, _, _ = self.predict(self.X_test)
        final_metrics = calculate_metrics(self.y_test, y_pred)

        class_names = [
            self.local_to_semantic[i]
            for i in range(self.num_classes)
        ]
        labels = np.arange(self.num_classes)

        report_dict = classification_report(
            self.y_test,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_text = classification_report(
            self.y_test,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )

        pd.DataFrame(report_dict).transpose().to_csv(
            CLASSIFICATION_REPORT_CSV
        )

        self.save_confusion_matrices(
            self.y_test,
            y_pred,
        )

        convergence_summary = self.determine_convergence()
        total_time = (
            time.perf_counter() - self.experiment_start_time
        )

        average_round_time = (
            float(
                np.mean(
                    [
                        row["round_total_time_sec"]
                        for row in self.round_history
                    ]
                )
            )
            if self.round_history
            else None
        )

        summary = {
            "client_id": CLIENT_ID,
            "dataset": "CIC-ToN-IoT",
            "seed": SEED,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lambda_proto": LAMBDA_PROTO,
            "prototype_dimension": PROTO_DIM,
            "private_dimension": PRIVATE_DIM,
            "train_samples": int(len(self.y_train)),
            "validation_samples": int(len(self.y_val)),
            "test_samples": int(len(self.y_test)),
            "number_of_classes": int(self.num_classes),
            "final_test_metrics": final_metrics,
            "total_experiment_time_sec": float(total_time),
            "average_round_time_sec": average_round_time,
            **convergence_summary,
        }

        with CONVERGENCE_SUMMARY_JSON.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

        self.save_convergence_curve(convergence_summary)

        # Save the same representation-analysis artifacts used for Client 3.
        if VISUALIZE_EMBEDDING:
            self.visualize_prototype_embedding(
                max_samples=VIS_MAX_SAMPLES
            )

        # ----------------------------------------------------
        # Preserve results from all seeds in one CSV file.
        # If the current seed already exists, replace only that
        # seed/client/dataset row; otherwise append a new row.
        # ----------------------------------------------------
        final_row = pd.DataFrame([{
            "seed": SEED,
            "client_id": CLIENT_ID,
            "dataset": "CIC-ToN-IoT",
            **final_metrics,
            "best_round": convergence_summary["best_round"],
            "best_validation_macro_f1": (
                convergence_summary["best_validation_macro_f1"]
            ),
            "convergence_round": (
                convergence_summary["convergence_round"]
            ),
            "convergence_time_sec": (
                convergence_summary["convergence_time_sec"]
            ),
            "total_experiment_time_sec": total_time,
            "average_round_time_sec": average_round_time,
        }])

        final_columns = [
            "seed",
            "client_id",
            "dataset",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "best_round",
            "best_validation_macro_f1",
            "convergence_round",
            "convergence_time_sec",
            "total_experiment_time_sec",
            "average_round_time_sec",
        ]

        final_row = final_row.reindex(columns=final_columns)

        if FINAL_METRICS_CSV.exists() and FINAL_METRICS_CSV.stat().st_size > 0:
            try:
                existing_results = pd.read_csv(FINAL_METRICS_CSV)

                # Add any missing columns safely, preserving older files.
                for column in final_columns:
                    if column not in existing_results.columns:
                        existing_results[column] = np.nan

                existing_results = existing_results.reindex(
                    columns=final_columns
                )

                # Remove only a previous row for this exact experiment.
                same_experiment = (
                    pd.to_numeric(
                        existing_results["seed"], errors="coerce"
                    ).eq(SEED)
                    & existing_results["client_id"].astype(str).eq(CLIENT_ID)
                    & existing_results["dataset"].astype(str).eq(
                        "CIC-ToN-IoT"
                    )
                )

                existing_results = existing_results.loc[
                    ~same_experiment
                ].copy()

                combined_results = pd.concat(
                    [existing_results, final_row],
                    ignore_index=True,
                )
            except Exception as exc:
                print(
                    f"[{CLIENT_ID}] Warning: could not read existing "
                    f"final metrics CSV: {exc}. A backup will be created."
                )

                backup_path = FINAL_METRICS_CSV.with_name(
                    FINAL_METRICS_CSV.stem
                    + f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    + FINAL_METRICS_CSV.suffix
                )
                FINAL_METRICS_CSV.replace(backup_path)
                combined_results = final_row.copy()
        else:
            combined_results = final_row.copy()

        combined_results["seed"] = pd.to_numeric(
            combined_results["seed"], errors="coerce"
        )
        combined_results = combined_results.sort_values(
            by=["seed", "client_id", "dataset"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

        # Atomic save prevents a partially written CSV if interrupted.
        temporary_metrics_csv = FINAL_METRICS_CSV.with_suffix(".tmp")
        combined_results.to_csv(
            temporary_metrics_csv,
            index=False,
        )
        temporary_metrics_csv.replace(FINAL_METRICS_CSV)

        print(
            f"[{CLIENT_ID}] Preserved {len(combined_results)} final "
            f"metric row(s) in {FINAL_METRICS_CSV}"
        )

        final_local_prototypes = self.compute_local_prototypes()

        with LOCAL_PROTOTYPES_FINAL_PKL.open("wb") as file:
            pickle.dump(
                final_local_prototypes,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with GLOBAL_PROTOTYPES_FINAL_PKL.open("wb") as file:
            pickle.dump(
                self.global_prototypes,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self.model.save(MODEL_PATH)

        with SCALER_PATH.open("wb") as file:
            pickle.dump(
                self.scaler,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print("\nFinal test metrics:")
        for name, value in final_metrics.items():
            print(f"{name:<22}: {value:.6f}")

        print("\nClassification report:\n")
        print(report_text)

        print("\nConvergence:")
        print(
            f"Best validation Macro-F1: "
            f"{convergence_summary['best_validation_macro_f1']}"
        )
        print(f"Best round: {convergence_summary['best_round']}")
        print(
            f"Convergence round: "
            f"{convergence_summary['convergence_round']}"
        )
        print(
            f"Convergence time (sec): "
            f"{convergence_summary['convergence_time_sec']}"
        )
        print(f"Total experiment time (sec): {total_time:.4f}")
        print(f"Average round time (sec): {average_round_time}")

        print("\nSaved outputs:")
        for path in [
            COMM_LOG,
            METRICS_LOG,
            CURVE_CSV,
            PROTOTYPE_LOG,
            FINAL_METRICS_CSV,
            CLASSIFICATION_REPORT_CSV,
            CONFUSION_MATRIX_CSV,
            CONFUSION_MATRIX_NORMALIZED_CSV,
            CONFUSION_MATRIX_PNG,
            CONFUSION_MATRIX_NORMALIZED_PNG,
            CONVERGENCE_CURVE_PNG,
            CONVERGENCE_SUMMARY_JSON,
            LOCAL_PROTOTYPES_FINAL_PKL,
            GLOBAL_PROTOTYPES_FINAL_PKL,
            MODEL_PATH,
            SCALER_PATH,
            EMBEDDINGS_CSV,
            TSNE_COORDINATES_CSV,
            TSNE_PNG,
            UMAP_COORDINATES_CSV,
            UMAP_PNG,
        ]:
            print(path)


if __name__ == "__main__":
    FedProtoClient1().run()