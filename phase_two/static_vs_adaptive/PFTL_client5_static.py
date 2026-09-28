#!/usr/bin/env python3.10
# ============================================================
# client5_BCCC_NRC_2024_PURE_STATIC_PFTL_GAMMA_05.py
# (Matches Standalone model + unified 70/15/15 split)
# ============================================================

import os, time, csv, pickle, random
from datetime import datetime

SEED = int(os.environ.get("SEED", "32"))
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
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers

import myproto_pb2
import myproto_pb2_grpc

# -------------------------
# Config
# -------------------------
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "localhost:50051")
CLIENT_ID      = os.environ.get("CLIENT_ID", "client5")
NUM_ROUNDS     = int(os.environ.get("NUM_ROUNDS", "20"))

CSV_PATH  = os.environ.get(
    "CSV_PATH",
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/Multi_class_Datasets/D5_CIC-BCCC-NRC_2024_Balanced_Final.csv"
)
LABEL_COL = os.environ.get("LABEL_COL", "Label")

OUT_DIR = os.environ.get(
    "OUT_DIR",
    f"/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/static_pftl_logs/{CLIENT_ID}_bccc_nrc_2024"
)
os.makedirs(OUT_DIR, exist_ok=True)

COMM_LOG    = os.path.join(OUT_DIR, f"{CLIENT_ID}_comm_log.csv")
METRICS_LOG = os.path.join(OUT_DIR, f"{CLIENT_ID}_metrics_log.csv")
CURVE_CSV   = os.path.join(OUT_DIR, f"{CLIENT_ID}_local_vs_global_curve.csv")
FINAL_LOG   = os.path.join(OUT_DIR, "client_final_metrics.csv")

#  unified split rule
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15 / (1.0 - TEST_SIZE)  # 0.17647058823529413

EPOCHS_PER_ROUND = 1
BATCH_SIZE       = 256
LR               = 1e-3

PRIVATE_DIM = 16
SHARED_DIM  = 8

# Pure static-gamma PFTL configuration.
# No safety gate is used: the fixed local/global blend is always accepted.
# EPS remains only for backward-compatible CSV logging.
EPS = 0.0

GAMMA_GLOBAL_STATIC = float(os.environ.get("GAMMA_GLOBAL", "0.50"))
GAMMA_LOCAL_STATIC = 1.0 - GAMMA_GLOBAL_STATIC
GAMMA_MODE = "static"

# First round reaching 95% of the best accepted validation Macro-F1.
CONVERGENCE_FRACTION = 0.95

MAX_GRPC_MSG = 50 * 1024 * 1024

WAIT_SERVER_MAX_SEC     = 600
WAIT_SERVER_POLL_SEC    = 2.0
WAIT_ROUND_ADV_MAX_SEC  = 1800
WAIT_ROUND_ADV_POLL_SEC = 1.0

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def is_valid_shared(w) -> bool:
    return isinstance(w, (list, tuple)) and len(w) == 2

def safe_unpickle(b: bytes):
    if not b:
        return None
    try:
        return pickle.loads(b)
    except Exception:
        return None

def predict_labels(model, X_batch, num_classes):
    probs = model.predict(X_batch, batch_size=1024, verbose=0)
    if num_classes == 2:
        return (probs.reshape(-1) >= 0.5).astype(int)
    return np.argmax(probs, axis=1)

def eval_all_metrics(model, X_eval, y_eval, num_classes):
    y_hat = predict_labels(model, X_eval, num_classes)
    acc = float(accuracy_score(y_eval, y_hat))
    mp, mr, mf1, _ = precision_recall_fscore_support(y_eval, y_hat, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_eval, y_hat, average="weighted", zero_division=0)
    return acc, float(mp), float(mr), float(mf1), float(wp), float(wr), float(wf1)

def ensure_csv_headers():
    if not os.path.exists(COMM_LOG):
        with open(COMM_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed", "client_id", "gamma_mode", "round",
                "bytes_sent",
                "server_round_before_send", "server_round_after_barrier",
                "rtt_sec", "barrier_wait_sec", "timestamp"
            ])

    if not os.path.exists(METRICS_LOG):
        with open(METRICS_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed", "client_id", "gamma_mode", "round", "stage",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "accuracy",
                "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1",
                "switch_on", "eps_margin",
                "gamma_local", "gamma_global",
                "delta_macro_f1_mixed_minus_local",
                "sent_weights_type",
                "local_train_time_sec", "mix_time_sec",
                "round_total_time_sec", "elapsed_experiment_sec",
                "timestamp"
            ])

    if not os.path.exists(CURVE_CSV):
        with open(CURVE_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed", "client_id", "gamma_mode", "round",
                "local_macro_f1", "mixed_candidate_macro_f1",
                "accepted_macro_f1", "switch_on",
                "gamma_local", "gamma_global",
                "elapsed_experiment_sec", "timestamp"
            ])

    if not os.path.exists(FINAL_LOG):
        with open(FINAL_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "seed", "client_id", "dataset", "gamma_mode",
                "gamma_local", "gamma_global",
                "accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1",
                "best_round", "best_validation_macro_f1",
                "convergence_round", "convergence_time_sec",
                "total_experiment_time_sec", "average_round_time_sec",
                "timestamp"
            ])


def log_stage_row(
    seed, round_id, stage,
    train_loss, train_acc, val_loss, val_acc,
    acc, mp, mr, mf1, wp, wr, wf1,
    switch_on, gamma_local, gamma_global, delta_f1,
    sent_type,
    local_train_time, mix_time, round_total_time,
    elapsed_experiment_sec,
):
    with open(METRICS_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            int(seed), CLIENT_ID, GAMMA_MODE, int(round_id), str(stage),
            f"{float(train_loss):.6f}", f"{float(train_acc):.6f}",
            f"{float(val_loss):.6f}", f"{float(val_acc):.6f}",
            f"{float(acc):.6f}",
            f"{float(mp):.6f}", f"{float(mr):.6f}", f"{float(mf1):.6f}",
            f"{float(wp):.6f}", f"{float(wr):.6f}", f"{float(wf1):.6f}",
            int(switch_on), f"{float(EPS):.6f}",
            f"{float(gamma_local):.6f}", f"{float(gamma_global):.6f}",
            f"{float(delta_f1):+.6f}", str(sent_type),
            f"{float(local_train_time):.4f}",
            f"{float(mix_time):.4f}",
            f"{float(round_total_time):.4f}",
            f"{float(elapsed_experiment_sec):.4f}",
            now_str()
        ])


def upsert_final_metrics(row):
    """Preserve all other seeds and replace only the same seed/client/mode."""

    columns = [
        "seed", "client_id", "dataset", "gamma_mode",
        "gamma_local", "gamma_global",
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
        "best_round", "best_validation_macro_f1",
        "convergence_round", "convergence_time_sec",
        "total_experiment_time_sec", "average_round_time_sec",
        "timestamp"
    ]

    new_df = pd.DataFrame([row], columns=columns)

    # Load previous final results.
    if os.path.exists(FINAL_LOG) and os.path.getsize(FINAL_LOG) > 0:
        try:
            old_df = pd.read_csv(FINAL_LOG)
        except Exception as exc:
            print(f"[WARNING] Could not read existing final log: {exc}")
            old_df = pd.DataFrame(columns=columns)
    else:
        old_df = pd.DataFrame(columns=columns)

    # Add any missing columns without deleting existing rows.
    for column in columns:
        if column not in old_df.columns:
            old_df[column] = np.nan

    # Remove only an existing row with the same:
    # seed + client_id + gamma_mode.
    keep_mask = ~(
        (
            pd.to_numeric(
                old_df["seed"],
                errors="coerce"
            ) == int(row["seed"])
        )
        & (
            old_df["client_id"].astype(str)
            == str(row["client_id"])
        )
        & (
            old_df["gamma_mode"].astype(str)
            == str(row["gamma_mode"])
        )
    )

    # Keep all previous seeds, then append the current result.
    combined = pd.concat(
        [
            old_df.loc[keep_mask, columns],
            new_df
        ],
        ignore_index=True
    )

    # Ensure seeds are stored and sorted numerically.
    combined["seed"] = pd.to_numeric(
        combined["seed"],
        errors="coerce"
    ).astype("Int64")

    combined = combined.sort_values(
        by=["client_id", "seed"],
        ascending=[True, True],
        kind="stable"
    ).reset_index(drop=True)

    # Save safely.
    combined.to_csv(FINAL_LOG, index=False)

    print(
        f"[{row['client_id']}] Final metrics saved for "
        f"seed={row['seed']} in {FINAL_LOG}"
    )

def calculate_convergence(accepted_f1_history, elapsed_history):
    if not accepted_f1_history:
        return np.nan, np.nan, np.nan, np.nan

    values = np.asarray(accepted_f1_history, dtype=float)
    best_index = int(np.nanargmax(values))
    best_round = best_index + 1
    best_f1 = float(values[best_index])

    threshold = CONVERGENCE_FRACTION * best_f1
    convergence_index = next(
        (index for index, value in enumerate(values) if value >= threshold),
        best_index,
    )

    convergence_round = convergence_index + 1
    convergence_time = float(elapsed_history[convergence_index])

    return best_round, best_f1, convergence_round, convergence_time


#  MUST MATCH STANDALONE CLIENT5 MODEL EXACTLY
def build_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, padding="valid", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 3, padding="valid", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 3, padding="valid", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(PRIVATE_DIM, activation="relu", name="private_dense")(x)
    x = layers.Dense(SHARED_DIM,  activation="relu", name="shared_dense")(x)

    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="y_out")(x)
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    else:
        out = layers.Dense(num_classes, activation="softmax", name="y_out")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model = models.Model(inp, out, name=f"StaticPFTL_{CLIENT_ID}_CNN_MATCH_STANDALONE")
    model.compile(optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0), loss=loss, metrics=metrics)
    return model

class StaticPFTLClient5_LocalFirst_StrictBarrier:
    def __init__(self):
        ensure_csv_headers()

        self.dataset_name = os.environ.get("DATASET_NAME", "CIC-BCCC-NRC-2024")
        self.accepted_f1_history = []
        self.elapsed_history = []
        self.round_time_history = []

        self.current_round = 0
        self.gamma_global = float(GAMMA_GLOBAL_STATIC)
        self.gamma_local = float(GAMMA_LOCAL_STATIC)

        self.channel = grpc.insecure_channel(
            SERVER_ADDRESS,
            options=[
                ("grpc.max_send_message_length", MAX_GRPC_MSG),
                ("grpc.max_receive_message_length", MAX_GRPC_MSG),
            ],
        )
        self.stub = myproto_pb2_grpc.AggregatorStub(self.channel)

        self._load_data()
        self.model = build_cnn(self.input_shape, self.num_classes)

        print("\n==== Model Summary ====")
        self.model.summary()

        self.wait_for_server()

        # LOCAL-FIRST initial sync
        w, r = self.pull_global()
        self.current_round = int(r)
        if is_valid_shared(w):
            ok = self.set_shared_if_valid(w)
            print(f"[{CLIENT_ID}] Initial sync: loaded global (server_round={self.current_round}, set_shared={ok}).")
        else:
            print(f"[{CLIENT_ID}] Initial sync: server returned empty global (server_round={self.current_round}). "
                  f"Starting LOCAL training first.")

    def _load_data(self):
        df = pd.read_csv(CSV_PATH, low_memory=False)
        df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

        if LABEL_COL not in df.columns:
            raise ValueError(f"Label column '{LABEL_COL}' not found. Available columns:\n{list(df.columns)}")

        y_raw = (
            df[LABEL_COL].astype(str)
              .str.replace("\ufeff", "", regex=False)
              .str.replace("�", "-", regex=False)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
              .values
        )

        X_df = df.drop(columns=[LABEL_COL], errors="ignore").select_dtypes(include=[np.number]).copy()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)  # keep NaN for imputer
        X = X_df.values.astype(np.float32)

        self.le = LabelEncoder()
        y = self.le.fit_transform(y_raw).astype(int)
        self.num_classes = int(len(self.le.classes_))

        print("\n=== Client 5 Static PFTL Readiness Summary ===")
        print(f"CSV: {CSV_PATH}")
        print(f"Samples: {X.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {self.num_classes}")
        print("Top 15 label counts:")
        print(pd.Series(y_raw).value_counts().head(15))

        #  unified split rule
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=VAL_SIZE_FROM_TRAIN, random_state=SEED, stratify=y_train
        )

        # preprocessing (same as standalone)
        self.imputer = SimpleImputer(strategy="median")
        X_train = self.imputer.fit_transform(X_train)
        X_val   = self.imputer.transform(X_val)
        X_test  = self.imputer.transform(X_test)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)

        self.X_train = X_train[..., np.newaxis]
        self.X_val   = X_val[..., np.newaxis]
        self.X_test  = X_test[..., np.newaxis]
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.input_shape = (self.X_train.shape[1], 1)

        classes_idx = np.unique(self.y_train)
        cw = compute_class_weight(class_weight="balanced", classes=classes_idx, y=self.y_train)
        self.class_weight = {int(i): float(w) for i, w in zip(classes_idx, cw)}
        print("Class weights enabled.")

        print("\n===== SPLIT SIZES (70/15/15) =====")
        n = len(y)
        print("Train:", len(self.X_train), f"({len(self.X_train)/n:.3f})")
        print("Val  :", len(self.X_val),   f"({len(self.X_val)/n:.3f})")
        print("Test :", len(self.X_test),  f"({len(self.X_test)/n:.3f})")

        print("\nLogs:")
        print("  METRICS_LOG:", METRICS_LOG)
        print("  CURVE_CSV  :", CURVE_CSV)
        print("  COMM_LOG   :", COMM_LOG)

    def get_shared(self):
        return self.model.get_layer("shared_dense").get_weights()

    def set_shared_if_valid(self, w):
        if not is_valid_shared(w):
            return False
        try:
            if any(np.isnan(np.array(x)).any() for x in w):
                return False
            self.model.get_layer("shared_dense").set_weights(list(w))
            return True
        except Exception:
            return False

    def blend(self, local_w, global_w):
        gl = float(self.gamma_local)
        gg = float(self.gamma_global)
        return [gl * l + gg * g for l, g in zip(local_w, global_w)]

    # RPC
    def rpc_get(self):
        return self.stub.GetSharedWeights(myproto_pb2.EmptyRequest(), metadata=[("client_id", CLIENT_ID)])

    def rpc_send(self, payload, num_samples):
        return self.stub.SendSharedUpdate(
            myproto_pb2.SharedUpdate(weights=payload, round=int(self.current_round), num_samples=int(num_samples)),
            metadata=[("client_id", CLIENT_ID)]
        )

    def pull_global(self):
        resp = self.rpc_get()
        return safe_unpickle(resp.weights), int(resp.round)

    # waits
    def wait_for_server(self):
        print(f"[{CLIENT_ID}] Waiting for aggregator at {SERVER_ADDRESS} ...")
        t0 = time.perf_counter()
        while True:
            try:
                resp = self.rpc_get()
                _ = int(resp.round)
                print(f"[{CLIENT_ID}] Aggregator reachable (server_round={resp.round}).")
                return
            except Exception as e:
                if time.perf_counter() - t0 > WAIT_SERVER_MAX_SEC:
                    raise RuntimeError(f"[{CLIENT_ID}] Aggregator not reachable: {e}")
                time.sleep(WAIT_SERVER_POLL_SEC)

    def align_to_server(self):
        # Synchronize ONLY the server round.
        # Keep the client's retained personalized mixed state from
        # the previous round; do not overwrite it with server global weights.
        _w, r = self.pull_global()
        self.current_round = int(r)

    def wait_barrier_advance(self, before_round: int):
        t0 = time.perf_counter()
        target = int(before_round) + 1
        while True:
            # Pull only to observe server progress.
            # Do NOT install aggregated server global weights into the client model.
            _w, r = self.pull_global()
            if int(r) >= target:
                return int(r), float(time.perf_counter() - t0)

            if (time.perf_counter() - t0) > WAIT_ROUND_ADV_MAX_SEC:
                raise RuntimeError(
                    f"[{CLIENT_ID}] Barrier timeout: server_round still {r}, expected >= {target}. "
                    f"Check MIN_CLIENTS_TO_AGG and that all clients are running."
                )
            time.sleep(WAIT_ROUND_ADV_POLL_SEC)

    def train_one_epoch(self):
        hist = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=EPOCHS_PER_ROUND,
            batch_size=BATCH_SIZE,
            class_weight=self.class_weight,
            verbose=1
        )
        train_loss = float(hist.history["loss"][0])
        val_loss   = float(hist.history["val_loss"][0])
        train_acc  = float(hist.history.get("accuracy", [np.nan])[0])
        val_acc    = float(hist.history.get("val_accuracy", [np.nan])[0])
        return train_loss, train_acc, val_loss, val_acc

    def run(self):
        experiment_start = time.perf_counter()

        for local_iter in range(NUM_ROUNDS):
            round_id = local_iter + 1
            round_start = time.perf_counter()

            # Synchronize ONLY the server round.
            # Keep the client's retained post-training shared state.
            self.align_to_server()

            print(
                f"\n[{CLIENT_ID}] ===== Round {round_id}/{NUM_ROUNDS} | "
                f"server_round={self.current_round} ====="
            )

            # ============================================================
            # SAME LOGIC AS FIXED CLIENT 1
            #
            # Round 1:
            #   Local training first; no pre-training blend.
            #
            # Round >= 2:
            #   retained post-training client weights from round r-1
            #       + newest server global weights
            #       -> fixed 0.5/0.5 blend
            #       -> SET blended shared weights
            #       -> LOCAL TRAINING
            #       -> SEND post-training shared weights
            # ============================================================

            mix_time = 0.0
            pretrain_mixed_mf1 = np.nan
            mixed_acc = mixed_mp = mixed_mr = np.nan
            mixed_wp = mixed_wr = mixed_wf1 = np.nan

            if round_id >= 2:
                retained_w = self.get_shared()

                mix_t0 = time.perf_counter()
                global_w, _ = self.pull_global()
                if not is_valid_shared(global_w):
                    global_w = retained_w

                start_w = self.blend(retained_w, global_w)
                self.set_shared_if_valid(start_w)
                mix_time = time.perf_counter() - mix_t0

                (
                    mixed_acc,
                    mixed_mp,
                    mixed_mr,
                    pretrain_mixed_mf1,
                    mixed_wp,
                    mixed_wr,
                    mixed_wf1,
                ) = eval_all_metrics(
                    self.model,
                    self.X_val,
                    self.y_val,
                    self.num_classes,
                )

                print(
                    f"[{CLIENT_ID}] Pre-train static blend applied | "
                    f"gamma_local={self.gamma_local:.3f} | "
                    f"gamma_global={self.gamma_global:.3f} | "
                    f"pretrain_mixed_MacroF1={pretrain_mixed_mf1:.6f}"
                )
            else:
                print(
                    f"[{CLIENT_ID}] Round 1: local training first; "
                    f"no pre-training blend."
                )

            # Local training starts from:
            # Round 1  -> initial synchronized/local state
            # Round >=2 -> newly blended shared weights
            train_t0 = time.perf_counter()
            train_loss, train_acc, val_loss, val_acc = self.train_one_epoch()
            local_train_time = time.perf_counter() - train_t0

            trained_w = self.get_shared()

            (
                local_acc,
                local_mp,
                local_mr,
                local_mf1,
                local_wp,
                local_wr,
                local_wf1,
            ) = eval_all_metrics(
                self.model,
                self.X_val,
                self.y_val,
                self.num_classes,
            )

            self.gamma_global = float(GAMMA_GLOBAL_STATIC)
            self.gamma_local = float(GAMMA_LOCAL_STATIC)

            # Static PFTL: no safety gate.
            # Send the POST-TRAINING shared weights.
            switch_on = 0 if round_id == 1 else 1
            sent_type = (
                "local_trained"
                if round_id == 1
                else "blended_then_trained"
            )
            weights_to_send = trained_w
            accepted_mf1 = float(local_mf1)

            if round_id >= 2 and np.isfinite(pretrain_mixed_mf1):
                delta_f1 = float(local_mf1 - pretrain_mixed_mf1)
            else:
                delta_f1 = 0.0

            print(
                f"[{CLIENT_ID}] Post-train MacroF1={local_mf1:.6f} | "
                f"Send={sent_type} | "
                f"gamma_local={self.gamma_local:.3f} | "
                f"gamma_global={self.gamma_global:.3f}"
            )

            # Send the post-training shared weights.
            payload = pickle.dumps(weights_to_send)
            bytes_sent = int(len(payload))
            server_round_before = int(self.current_round)

            send_t0 = time.perf_counter()
            while True:
                ack = self.rpc_send(
                    payload,
                    num_samples=self.X_train.shape[0]
                )
                if bool(getattr(ack, "ok", False)):
                    break

                # Retry after synchronizing round number only.
                # Do NOT install server global weights here.
                self.align_to_server()
                server_round_before = int(self.current_round)

            rtt_sec = float(time.perf_counter() - send_t0)

            # Strict barrier: wait for server advancement only.
            # Do NOT overwrite the client's retained post-training state.
            new_srv_round, waited_sec = self.wait_barrier_advance(
                server_round_before
            )
            self.current_round = int(new_srv_round)

            with open(COMM_LOG, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(SEED), CLIENT_ID, GAMMA_MODE, int(round_id),
                    int(bytes_sent),
                    int(server_round_before), int(self.current_round),
                    f"{rtt_sec:.6f}",
                    f"{waited_sec:.6f}",
                    now_str()
                ])

            round_total_time = time.perf_counter() - round_start
            elapsed_experiment_sec = (
                time.perf_counter() - experiment_start
            )

            self.accepted_f1_history.append(accepted_mf1)
            self.elapsed_history.append(elapsed_experiment_sec)
            self.round_time_history.append(round_total_time)

            # Existing curve schema retained.
            # mixed_candidate_macro_f1 now means the PRE-TRAINING blended
            # initialization for rounds >=2; blank in round 1.
            with open(CURVE_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    int(SEED), CLIENT_ID, GAMMA_MODE, int(round_id),
                    f"{local_mf1:.16f}",
                    "" if not np.isfinite(pretrain_mixed_mf1)
                    else f"{pretrain_mixed_mf1:.16f}",
                    f"{accepted_mf1:.16f}",
                    int(switch_on),
                    f"{self.gamma_local:.16f}",
                    f"{self.gamma_global:.16f}",
                    f"{elapsed_experiment_sec:.6f}",
                    now_str(),
                ])

            # Post-training metrics
            log_stage_row(
                SEED, round_id, "POST_TRAIN",
                train_loss, train_acc, val_loss, val_acc,
                local_acc, local_mp, local_mr, local_mf1,
                local_wp, local_wr, local_wf1,
                switch_on, self.gamma_local, self.gamma_global,
                delta_f1, sent_type,
                local_train_time, mix_time, round_total_time,
                elapsed_experiment_sec,
            )

            # Optional pre-training blend metrics for rounds >=2.
            if round_id >= 2:
                log_stage_row(
                    SEED, round_id, "PRETRAIN_MIXED",
                    train_loss, train_acc, val_loss, val_acc,
                    mixed_acc, mixed_mp, mixed_mr, pretrain_mixed_mf1,
                    mixed_wp, mixed_wr, mixed_wf1,
                    switch_on, self.gamma_local, self.gamma_global,
                    delta_f1, sent_type,
                    local_train_time, mix_time, round_total_time,
                    elapsed_experiment_sec,
                )

            print(
                f"[{CLIENT_ID}] Barrier passed: "
                f"{server_round_before} -> {self.current_round} "
                f"(wait={waited_sec:.2f}s)"
            )

        print(f"\n[{CLIENT_ID}] ===== FINAL TEST EVALUATION =====")
        y_pred = predict_labels(self.model, self.X_test, self.num_classes)

        acc = float(accuracy_score(self.y_test, y_pred))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average="macro", zero_division=0
        )
        w_p, w_r, w_f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average="weighted", zero_division=0
        )

        print("\n===== STATIC PFTL FINAL TEST RESULTS =====")
        print(f"Accuracy        : {acc:.6f}")
        print(f"Macro-Precision : {macro_p:.6f}")
        print(f"Macro-Recall    : {macro_r:.6f}")
        print(f"Macro-F1        : {macro_f1:.6f}")

        print("\n===== WEIGHTED (support-weighted) METRICS =====")
        print(f"Weighted-Precision : {w_p:.6f}")
        print(f"Weighted-Recall    : {w_r:.6f}")
        print(f"Weighted-F1        : {w_f1:.6f}")

        print("\n==== Classification Report ====")
        print(
            classification_report(
                self.y_test,
                y_pred,
                target_names=self.le.classes_,
                zero_division=0
            )
        )

        print("\n==== Confusion Matrix ====")
        print(confusion_matrix(self.y_test, y_pred))

        total_experiment_time = time.perf_counter() - experiment_start
        average_round_time = (
            float(np.mean(self.round_time_history))
            if self.round_time_history
            else np.nan
        )

        (
            best_round,
            best_validation_macro_f1,
            convergence_round,
            convergence_time_sec,
        ) = calculate_convergence(
            self.accepted_f1_history,
            self.elapsed_history,
        )

        final_row = {
            "seed": int(SEED),
            "client_id": CLIENT_ID,
            "dataset": self.dataset_name,
            "gamma_mode": GAMMA_MODE,
            "gamma_local": float(self.gamma_local),
            "gamma_global": float(self.gamma_global),
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(w_p),
            "weighted_recall": float(w_r),
            "weighted_f1": float(w_f1),
            "best_round": int(best_round),
            "best_validation_macro_f1": float(
                best_validation_macro_f1
            ),
            "convergence_round": float(convergence_round),
            "convergence_time_sec": float(convergence_time_sec),
            "total_experiment_time_sec": float(total_experiment_time),
            "average_round_time_sec": float(average_round_time),
            "timestamp": now_str(),
        }

        upsert_final_metrics(final_row)

        print("\n===== CONVERGENCE / TIMING =====")
        print(f"Best validation round       : {best_round}")
        print(
            f"Best validation Macro-F1    : "
            f"{best_validation_macro_f1:.6f}"
        )
        print(f"Convergence round           : {convergence_round}")
        print(f"Convergence time (s)        : {convergence_time_sec:.3f}")
        print(f"Total experiment time (s)   : {total_experiment_time:.3f}")
        print(f"Average round time (s)      : {average_round_time:.3f}")

        print("\n===== SAVED CSV FILES =====")
        print("Curve CSV  :", CURVE_CSV)
        print("Metrics CSV:", METRICS_LOG)
        print("Comm CSV   :", COMM_LOG)
        print("Final CSV  :", FINAL_LOG)

if __name__ == "__main__":
    ensure_csv_headers()
    StaticPFTLClient5_LocalFirst_StrictBarrier().run()