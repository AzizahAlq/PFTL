#!/usr/bin/env python3.10

# ============================================================

# PFTL Client (Design-B + STRICT BARRIER)

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



SEED = 512



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

# Standard libraries

# ============================================================



import pandas as pd

import grpc



# ============================================================

# Scikit-learn

# ============================================================



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.utils.class_weight import compute_class_weight



from sklearn.metrics import (

    accuracy_score,

    precision_recall_fscore_support,

    classification_report,

    confusion_matrix,

)



# ============================================================

# TensorFlow / Keras

# ============================================================



from tensorflow.keras import layers, models, optimizers



# ============================================================

# Generated proto modules

# ============================================================



import myproto_pb2

import myproto_pb2_grpc



# ============================================================

# PFTL Settings

# ============================================================



SERVER_ADDRESS = os.environ.get(

    "SERVER_ADDRESS",

    "localhost:50051"

)



CLIENT_ID = os.environ.get(

    "CLIENT_ID",

    "client1"

)



NUM_ROUNDS = int(

    os.environ.get(

        "NUM_ROUNDS",

        "20"

    )

)



# ============================================================

# Representation dimensions

# ============================================================



PRIVATE_DIM = 16

SHARED_DIM = 8



# ============================================================

# Data / Paths

# ============================================================



CSV_PATH = os.environ.get(

    "CSV_PATH",

    "/nfs/aalqahtani/PFTL_Multi_class/"

    "M_PFTL_codes/Multi_class_Datasets/"

    "D1_CIC-ToN-IoT_new.csv"

)



LABEL_COL = os.environ.get(

    "LABEL_COL",

    "Attack"

)



OUT_DIR = os.environ.get(

    "OUT_DIR",

    "/nfs/aalqahtani/PFTL_Multi_class/"

    "M_PFTL_codes/ptfl_logs/"

    "client1_ton_iot_v2"

)



os.makedirs(

    OUT_DIR,

    exist_ok=True

)



# ============================================================

# Output files

# ============================================================



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
# Final epsilon-sensitivity results
# ============================================================

FINAL_RESULTS_CSV = os.path.join(
    OUT_DIR,
    f"{CLIENT_ID}_epsilon_sensitivity_final_results.csv"
)


# ============================================================

# Split style: 70 / 15 / 15

# ============================================================



TEST_SIZE = 0.15



VAL_SIZE_FROM_TRAIN = 0.17647058823529413



# ============================================================

# Training configuration

# ============================================================



EPOCHS_PER_ROUND = 1

BATCH_SIZE = 1024

LR = 1e-3



# ============================================================

# Safety Switch + Adaptive Gamma

# ============================================================



ADAPT_GAMMA = True



EPS = float(
    os.environ.get(
        "EPS",
        "0.0" 
    )
)



GAMMA_GLOBAL_INIT = 0.5



ETA = 0.05

TAU = 0.05

GAMMA_MIN = 0.00

GAMMA_MAX = 1.00



# ============================================================

# Networking + waiting

# ============================================================



MAX_GRPC_MSG = 50 * 1024 * 1024



WAIT_FOR_SERVER_MAX_SEC = 600

WAIT_FOR_SERVER_SLEEP_SEC = 2.0



BARRIER_TIMEOUT_SEC = 900

BARRIER_POLL_SEC = 1.0



# ============================================================

# Utility functions

# ============================================================



def now_str():

    return datetime.now().strftime(

        "%Y-%m-%d %H:%M:%S"

    )





def clip(x, lo, hi):

    return float(

        max(

            lo,

            min(

                hi,

                x

            )

        )

    )





def safe_unpickle_weights(b: bytes):

    try:

        return pickle.loads(b)

    except Exception:

        return None





def is_valid_shared(weights_obj) -> bool:

    return (

        isinstance(

            weights_obj,

            (list, tuple)

        )

        and

        len(weights_obj) == 2

    )



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

            probs.reshape(-1) >= 0.5

        ).astype(int)



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



    mp, mr, mf1, _ = (

        precision_recall_fscore_support(

            y_eval,

            y_hat,

            average="macro",

            zero_division=0

        )

    )



    wp, wr, wf1, _ = (

        precision_recall_fscore_support(

            y_eval,

            y_hat,

            average="weighted",

            zero_division=0

        )

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

# Adaptive Gamma

# ============================================================



def apply_unified_safety_adaptation(

    gamma_global,

    f1_local,

    f1_mixed,

    eps=EPS,

    eta=ETA,

    tau=TAU,

    adapt_gamma=ADAPT_GAMMA

):

    # Raw Mixed-vs-Local Macro-F1 difference.

    delta = float(

        f1_mixed - f1_local

    )



    # ONE unified signal for both:

    #   1) safety-gate decision

    #   2) gamma adaptation

    #

    # Positive margin  -> Mixed exceeds Local by at least EPS.

    # Negative margin  -> Mixed does not satisfy the safety gate.

    gate_margin = float(

        delta - float(eps)

    )



    switch_on = int(

        gate_margin >= 0.0

    )



    if adapt_gamma:

        step = (

            float(eta)

            *

            float(

                np.tanh(

                    gate_margin

                    /

                    max(

                        float(tau),

                        1e-8

                    )

                )

            )

        )



        gamma_global = clip(

            float(gamma_global) + step,

            GAMMA_MIN,

            GAMMA_MAX

        )

    else:

        gamma_global = float(

            gamma_global

        )



    gamma_local = (

        1.0 - gamma_global

    )



    return (

        gamma_local,

        gamma_global,

        delta,

        gate_margin,

        switch_on

    )



# ============================================================

# CSV setup

# ============================================================



def ensure_csv_headers():



    if not os.path.exists(COMM_LOG):

        with open(

            COMM_LOG,

            "w",

            newline=""

        ) as f:

            csv.writer(f).writerow(

                [

                    "seed",

                    "round",

                    "bytes_sent",

                    "rtt_sec",

                    "barrier_wait_sec",

                    "timestamp",

                ]

            )



    if not os.path.exists(METRICS_LOG):

        with open(

            METRICS_LOG,

            "w",

            newline=""

        ) as f:

            csv.writer(f).writerow(

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



    if not os.path.exists(CURVE_CSV):

        with open(

            CURVE_CSV,

            "w",

            newline=""

        ) as f:

            csv.writer(f).writerow(

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



        csv.writer(f).writerow(

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
# Save final test result for each seed / epsilon
# ============================================================

def save_final_test_result(seed, epsilon, accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1):
    file_exists = os.path.exists(FINAL_RESULTS_CSV)
    with open(FINAL_RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["seed", "epsilon", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "timestamp"])
        writer.writerow([int(seed), float(epsilon), f"{float(accuracy):.6f}", f"{float(macro_precision):.6f}", f"{float(macro_recall):.6f}", f"{float(macro_f1):.6f}", f"{float(weighted_precision):.6f}", f"{float(weighted_recall):.6f}", f"{float(weighted_f1):.6f}", now_str()])
    print("\n==========================================")
    print(f"[{CLIENT_ID}] FINAL TEST RESULT SAVED")
    print("==========================================")
    print(f"Seed     : {seed}")
    print(f"Epsilon  : {epsilon}")
    print(f"Macro-F1 : {macro_f1:.6f}")
    print(f"File     : {FINAL_RESULTS_CSV}")


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



    x = layers.Dropout(

        0.25

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



    x = layers.Dropout(

        0.25

    )(x)



    x = layers.Conv1D(

        128,

        3,

        padding="valid",

        activation="relu"

    )(x)



    x = layers.BatchNormalization()(x)



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



    if num_classes == 2:



        out = layers.Dense(

            1,

            activation="sigmoid",

            name="y_out"

        )(x)



        loss = "binary_crossentropy"



        metrics = [

            tf.keras.metrics.BinaryAccuracy(

                name="accuracy"

            )

        ]



    else:



        out = layers.Dense(

            num_classes,

            activation="softmax",

            name="y_out"

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

        inp,

        out,

        name=f"PFTL_{CLIENT_ID}_CNN"

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



class PTFLClientDesignB_StrictBarrier:



    def __init__(self):



        ensure_csv_headers()



        self.current_round = 0



        self.gamma_global = float(

            GAMMA_GLOBAL_INIT

        )



        self.gamma_local = (

            1.0

            -

            self.gamma_global

        )



        self.channel = grpc.insecure_channel(

            SERVER_ADDRESS,



            options=[

                (

                    "grpc.max_send_message_length",

                    MAX_GRPC_MSG

                ),

                (

                    "grpc.max_receive_message_length",

                    MAX_GRPC_MSG

                ),

            ],

        )



        self.stub = (

            myproto_pb2_grpc

            .AggregatorStub(

                self.channel

            )

        )



        self._load_data()



        self.model = build_cnn(

            self.input_shape,

            self.num_classes

        )



        print(

            "\n==== Model Summary ===="

        )



        self.model.summary()



        self.wait_for_aggregator_reachable()



        w, r = self.pull_global_shared()



        self.current_round = int(r)



        if is_valid_shared(w):



            self.set_shared(w)



            print(

                f"[{CLIENT_ID}] "

                f"Initial sync: loaded server global "

                f"(server_round={r})."

            )



        else:



            print(

                f"[{CLIENT_ID}] "

                f"Initial sync: server has no global yet "

                f"(server_round={r}). "

                f"I will start LOCAL training and send "

                f"my shared weights after Round 1."

            )



    # ========================================================

    # gRPC helpers

    # ========================================================



    def wait_for_aggregator_reachable(

        self

    ):



        t0 = time.perf_counter()



        print(

            f"[{CLIENT_ID}] "

            f"Waiting for aggregator at "

            f"{SERVER_ADDRESS} ..."

        )



        while True:



            try:



                _w, r = (

                    self.pull_global_shared()

                )



                print(

                    f"[{CLIENT_ID}] "

                    f"Aggregator reachable "

                    f"(server_round={r})."

                )



                return



            except Exception as e:



                if (

                    time.perf_counter() - t0

                    >

                    WAIT_FOR_SERVER_MAX_SEC

                ):



                    raise RuntimeError(

                        f"[{CLIENT_ID}] "

                        f"Aggregator not reachable after "

                        f"{WAIT_FOR_SERVER_MAX_SEC}s: {e}"

                    )



                time.sleep(

                    WAIT_FOR_SERVER_SLEEP_SEC

                )



    def pull_global_shared(

        self

    ):



        resp = (

            self.stub.GetSharedWeights(

                myproto_pb2.EmptyRequest(),



                metadata=[

                    (

                        "client_id",

                        CLIENT_ID

                    )

                ],

            )

        )



        w = safe_unpickle_weights(

            resp.weights

        )



        return (

            w,

            int(resp.round)

        )



    def send_shared_update(

        self,

        shared_weights,

        num_samples: int

    ):



        payload = pickle.dumps(

            shared_weights

        )



        req = myproto_pb2.SharedUpdate(

            weights=payload,

            round=int(

                self.current_round

            ),

            num_samples=int(

                num_samples

            ),

        )



        t0 = time.perf_counter()



        ack = self.stub.SendSharedUpdate(

            req,



            metadata=[

                (

                    "client_id",

                    CLIENT_ID

                )

            ],

        )



        rtt = (

            time.perf_counter()

            -

            t0

        )



        ok = bool(

            getattr(

                ack,

                "ok",

                False

            )

        )



        return (

            ok,

            len(payload),

            rtt

        )



    def send_update_retry_once(

        self,

        shared_weights,

        num_samples: int

    ):



        (

            ok,

            bytes_sent,

            rtt

        ) = self.send_shared_update(

            shared_weights,

            num_samples

        )



        if ok:



            return (

                ok,

                bytes_sent,

                rtt

            )



        _w, srv_round = (

            self.pull_global_shared()

        )



        if (

            int(srv_round)

            !=

            int(self.current_round)

        ):



            print(

                f"[{CLIENT_ID}] "

                f"Resync after reject: "

                f"{self.current_round} "

                f"-> {srv_round}"

            )



            self.current_round = int(

                srv_round

            )



        (

            ok2,

            bytes_sent2,

            rtt2

        ) = self.send_shared_update(

            shared_weights,

            num_samples

        )



        return (

            ok2,

            bytes_sent2,

            rtt2

        )



    def wait_for_server_round(

        self,

        target_round: int,

        timeout_sec: float

    ):



        t0 = time.perf_counter()



        while True:



            w, r = (

                self.pull_global_shared()

            )



            if (

                int(r)

                >=

                int(target_round)

            ):



                return (

                    w,

                    int(r),

                    (

                        time.perf_counter()

                        -

                        t0

                    )

                )



            if (

                time.perf_counter() - t0

                >

                timeout_sec

            ):



                return (

                    w,

                    int(r),

                    (

                        time.perf_counter()

                        -

                        t0

                    )

                )



            time.sleep(

                BARRIER_POLL_SEC

            )



    # ========================================================

    # Data loading

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

                f"Label column '{LABEL_COL}' not found. "

                f"Available columns:\n"

                f"{list(df.columns)}"

            )



        y_raw = (

            df[LABEL_COL]

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

                columns=[LABEL_COL],

                errors="ignore"

            )

            .select_dtypes(

                include=[np.number]

            )

            .copy()

        )



        X_df = (

            X_df

            .replace(

                [np.inf, -np.inf],

                np.nan

            )

            .fillna(0.0)

        )



        X = X_df.values.astype(

            np.float32

        )



        self.le = LabelEncoder()



        y = (

            self.le

            .fit_transform(

                y_raw

            )

            .astype(int)

        )



        self.num_classes = int(

            len(

                self.le.classes_

            )

        )



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



            test_size=VAL_SIZE_FROM_TRAIN,



            random_state=SEED,



            stratify=y_train

        )



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



        X_train = (

            X_train[

                ...,

                np.newaxis

            ]

        )



        X_val = (

            X_val[

                ...,

                np.newaxis

            ]

        )



        X_test = (

            X_test[

                ...,

                np.newaxis

            ]

        )



        self.X_train = X_train

        self.X_val = X_val

        self.X_test = X_test



        self.y_train = y_train

        self.y_val = y_val

        self.y_test = y_test



        self.input_shape = (

            X_train.shape[1],

            1

        )



        classes_idx = np.unique(

            self.y_train

        )



        cw = compute_class_weight(

            class_weight="balanced",



            classes=classes_idx,



            y=self.y_train

        )



        self.class_weight = {

            int(i): float(w)

            for i, w

            in zip(

                classes_idx,

                cw

            )

        }



        print(

            "\n=== PFTL Client Readiness Summary ==="

        )



        print(

            f"Client   : {CLIENT_ID}"

        )



        print(

            f"CSV      : {CSV_PATH}"

        )



        print(

            f"Samples  : {X.shape[0]}"

        )



        print(

            f"Features : {X.shape[1]}"

        )



        print(

            f"Classes  : {self.num_classes}"

        )



        print(

            "Class names:",

            list(

                self.le.classes_

            )

        )



        print(

            "Splits:",

            len(self.X_train),

            len(self.X_val),

            len(self.X_test)

        )



    # ========================================================

    # Shared layer helpers

    # ========================================================



    def get_shared(

        self

    ):



        return (

            self.model

            .get_layer(

                "shared_dense"

            )

            .get_weights()

        )



    def set_shared(

        self,

        w

    ):



        if not is_valid_shared(w):

            return



        if any(

            np.isnan(x).any()

            for x in w

        ):

            return



        self.model.get_layer(

            "shared_dense"

        ).set_weights(

            list(w)

        )



    def blend(

        self,

        local_w,

        global_w

    ):



        gl = float(

            self.gamma_local

        )



        gg = float(

            self.gamma_global

        )



        return [

            (

                gl * l

                +

                gg * g

            )



            for l, g

            in zip(

                local_w,

                global_w

            )

        ]



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



            epochs=EPOCHS_PER_ROUND,



            batch_size=BATCH_SIZE,



            class_weight=(

                self.class_weight

            ),



            verbose=1,

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

    # Main training loop

    # ========================================================



    def run(

        self

    ):



        for local_iter in range(

            NUM_ROUNDS

        ):



            round_id = (

                local_iter + 1

            )



            round_start = (

                time.perf_counter()

            )



            (

                w_srv,

                srv_round

            ) = (

                self.pull_global_shared()

            )



            if (

                int(srv_round)

                !=

                int(self.current_round)

            ):



                print(

                    f"[{CLIENT_ID}] "

                    f"Align current_round "

                    f"{self.current_round} "

                    f"-> {srv_round}"

                )



                self.current_round = int(

                    srv_round

                )



            # Keep the selected client model from the previous round.

            # Synchronize the server round only; do not overwrite the

            # retained LOCAL/MIXED shared parameters with w_srv here.



            print(

                f"\n[{CLIENT_ID}] "

                f"===== Round "

                f"{round_id}/{NUM_ROUNDS} "

                f"(server_round="

                f"{self.current_round}) "

                f"====="

            )



            t0 = (

                time.perf_counter()

            )



            (

                train_loss,

                train_acc,

                val_loss,

                val_acc

            ) = (

                self.train_one_epoch()

            )



            local_train_time = (

                time.perf_counter()

                -

                t0

            )



            local_w = (

                self.get_shared()

            )



            (

                local_acc,

                local_mp,

                local_mr,

                local_mf1,



                local_wp,

                local_wr,

                local_wf1

            ) = eval_all_metrics(



                self.model,



                self.X_val,



                self.y_val,



                self.num_classes

            )



            t1 = (

                time.perf_counter()

            )



            (

                global_w,

                _

            ) = (

                self.pull_global_shared()

            )



            if not is_valid_shared(

                global_w

            ):



                global_w = (

                    local_w

                )



            mixed_w = self.blend(



                local_w,



                global_w

            )



            self.set_shared(

                mixed_w

            )



            mix_time = (

                time.perf_counter()

                -

                t1

            )



            (

                mixed_acc,

                mixed_mp,

                mixed_mr,

                mixed_mf1,



                mixed_wp,

                mixed_wr,

                mixed_wf1

            ) = eval_all_metrics(



                self.model,



                self.X_val,



                self.y_val,



                self.num_classes

            )



            # ====================================================

            # Unified safety gate + adaptive gamma

            # ====================================================

            # The SAME epsilon-adjusted margin controls BOTH:

            #   - whether MIXED is accepted

            #   - whether gamma_global increases or decreases

            #

            # gate_margin = (Mixed Macro-F1 - Local Macro-F1) - EPS

            (

                self.gamma_local,

                self.gamma_global,

                delta_f1,

                gate_margin,

                switch_on

            ) = apply_unified_safety_adaptation(

                self.gamma_global,

                local_mf1,

                mixed_mf1,

                eps=EPS,

                eta=ETA,

                tau=TAU,

                adapt_gamma=ADAPT_GAMMA

            )



            if switch_on:



                sent_type = (

                    "mixed"

                )



                weights_to_send = (

                    mixed_w

                )



            else:



                self.set_shared(

                    local_w

                )



                sent_type = (

                    "local"

                )



                weights_to_send = (

                    local_w

                )



            print(



                f"[{CLIENT_ID}] "



                f"MacroF1 "

                f"Local={local_mf1:.6f} | "



                f"Mixed={mixed_mf1:.6f} | "



                f"Switch="

                f"{'ON' if switch_on else 'OFF'} "



                f"(eps={EPS}) | "



                f"Send={sent_type} | "



                f"gamma_global="

                f"{self.gamma_global:.3f} "



                f"(Delta="

                f"{delta_f1:+.6f}, "

                f"Margin="

                f"{gate_margin:+.6f})"

            )



            with open(

                CURVE_CSV,

                "a",

                newline=""

            ) as f:



                csv.writer(f).writerow(

                    [

                        int(SEED),



                        int(round_id),



                        f"{local_mf1:.16f}",



                        f"{mixed_mf1:.16f}",



                        int(switch_on),



                        f"{self.gamma_global:.16f}",

                    ]

                )



            (

                ok,

                bytes_sent,

                rtt

            ) = self.send_update_retry_once(



                weights_to_send,



                self.X_train.shape[0]

            )



            if not ok:



                print(

                    f"[{CLIENT_ID}] "

                    f"SendSharedUpdate rejected "

                    f"(ok=False) even after retry. "

                    f"Will resync next loop."

                )



            target_round = (

                int(

                    self.current_round

                )

                +

                1

            )



            (

                w_new,

                new_round,

                waited

            ) = self.wait_for_server_round(



                target_round,



                timeout_sec=(

                    BARRIER_TIMEOUT_SEC

                )

            )



            barrier_wait = (

                waited

            )



            if (

                int(new_round)

                >=

                target_round

            ):



                self.current_round = int(

                    new_round

                )



                # Barrier advances the round only.

                # Keep the selected LOCAL or MIXED model as the client's

                # starting model for the next round; do not install w_new.



                print(

                    f"[{CLIENT_ID}] "

                    f"Barrier passed -> "

                    f"server_round="

                    f"{self.current_round} "

                    f"(waited "

                    f"{barrier_wait:.1f}s)"

                )



            else:



                print(

                    f"[{CLIENT_ID}] "

                    f"Barrier TIMEOUT: "

                    f"server_round="

                    f"{new_round}, "

                    f"expected>="

                    f"{target_round}"

                )



            with open(

                COMM_LOG,

                "a",

                newline=""

            ) as f:



                csv.writer(f).writerow(

                    [

                        int(SEED),



                        int(round_id),



                        int(bytes_sent),



                        f"{float(rtt):.3f}",



                        f"{float(barrier_wait):.3f}",



                        now_str(),

                    ]

                )



            round_total_time = (



                time.perf_counter()

                -

                round_start

            )



            log_stage_row(



                SEED,



                round_id,



                "LOCAL",



                train_loss,

                train_acc,



                val_loss,

                val_acc,



                local_acc,



                local_mp,

                local_mr,

                local_mf1,



                local_wp,

                local_wr,

                local_wf1,



                switch_on,



                EPS,



                self.gamma_local,

                self.gamma_global,



                delta_f1,



                sent_type,



                local_train_time,



                mix_time,



                round_total_time,

            )



            log_stage_row(



                SEED,



                round_id,



                "GLOBAL",



                train_loss,

                train_acc,



                val_loss,

                val_acc,



                mixed_acc,



                mixed_mp,

                mixed_mr,

                mixed_mf1,



                mixed_wp,

                mixed_wr,

                mixed_wf1,



                switch_on,



                EPS,



                self.gamma_local,

                self.gamma_global,



                delta_f1,



                sent_type,



                local_train_time,



                mix_time,



                round_total_time,

            )



        # ====================================================

        # FINAL TEST EVALUATION

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

            "\n===== FINAL PFTL RESULTS ====="

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



        # Save one summary row for this completed seed/epsilon run.
        save_final_test_result(
            seed=SEED, epsilon=EPS, accuracy=acc,
            macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f1,
            weighted_precision=w_p, weighted_recall=w_r, weighted_f1=w_f1,
        )

        print(

            "\n==== Classification Report ===="

        )



        print(

            classification_report(



                self.y_test,



                y_pred,



                target_names=(

                    self.le.classes_

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



        print(

            "\n===== SAVED OUTPUTS ====="

        )



        print(

            "Curve CSV   :",

            CURVE_CSV

        )



        print(

            "Metrics CSV :",

            METRICS_LOG

        )



        print(

            "Comm CSV    :",

            COMM_LOG

        )



# ============================================================

# Main

# ============================================================



if __name__ == "__main__":

    PTFLClientDesignB_StrictBarrier().run()