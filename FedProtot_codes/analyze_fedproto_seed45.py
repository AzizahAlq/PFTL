#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""
Analyze all FedProto clients for one seed (default: 45)
======================================================

This script automatically reads the outputs produced by:

    fedproto_client1.py
    fedproto_client2.py
    fedproto_client3.py
    fedproto_client4.py
    fedproto_client5.py
    fedproto_client6.py
    fedproto_aggregator.py

It creates:

1. Combined final-performance table for all six clients.
2. Convergence summary.
3. Communication summary.
4. Prototype-behavior summary.
5. Per-round combined metrics.
6. Macro averages across clients.
7. Weighted averages across clients using test-set size.
8. Confusion-matrix summaries.
9. Publication-ready plots.
10. One JSON report containing the main conclusions.

Expected project structure:

/nfs/aalqahtani/Proto_PFTL_Multi_class/M_PFTL_codes/
├── fedproto_logs/
│   ├── client1_ton_iot/
│   ├── client2_cic_iot_2023/
│   ├── client3_unsw_nb15/
│   ├── client4_cic_ids_2017/
│   ├── client5_cic_bccc_nrc_2024/
│   └── client6_cic_iot_idad_2024/
└── fedproto_server_logs/

Run:

    cd /nfs/aalqahtani/Proto_PFTL_Multi_class/M_PFTL_codes
    python analyze_fedproto_seed45.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Fixed configuration
# ============================================================

SEED = 45

BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "M_PFTL_codes"
)

FEDPROTO_LOG_ROOT = BASE_DIR / "fedproto_logs"
SERVER_LOG_DIR = BASE_DIR / "fedproto_server_logs"

OUTPUT_DIR = BASE_DIR / "fedproto_analysis" / f"seed_{SEED}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENTS = {
    "client1": {
        "dataset": "CIC-ToN-IoT",
        "directory": FEDPROTO_LOG_ROOT / "client1_ton_iot",
        "local_classes": 10,
    },
    "client2": {
        "dataset": "CIC-IoT-2023",
        "directory": FEDPROTO_LOG_ROOT / "client2_cic_iot_2023",
        "local_classes": 34,
    },
    "client3": {
        "dataset": "UNSW-NB15",
        "directory": FEDPROTO_LOG_ROOT / "client3_unsw_nb15",
        "local_classes": 6,
    },
    "client4": {
        "dataset": "CIC-IDS-2017",
        "directory": FEDPROTO_LOG_ROOT / "client4_cic_ids_2017",
        "local_classes": 14,
    },
    "client5": {
        "dataset": "CIC-BCCC-NRC-2024",
        "directory": FEDPROTO_LOG_ROOT / "client5_cic_bccc_nrc_2024",
        "local_classes": 15,
    },
    "client6": {
        "dataset": "CIC-IoT-IDAD-2024",
        "directory": FEDPROTO_LOG_ROOT / "client6_cic_iot_idad_2024",
        "local_classes": 15,
    },
}


# ============================================================
# Output files
# ============================================================

FINAL_PERFORMANCE_CSV = OUTPUT_DIR / "all_clients_final_performance.csv"
CONVERGENCE_CSV = OUTPUT_DIR / "all_clients_convergence.csv"
COMMUNICATION_CSV = OUTPUT_DIR / "all_clients_communication.csv"
PROTOTYPE_SUMMARY_CSV = OUTPUT_DIR / "all_clients_prototype_summary.csv"
ROUND_METRICS_CSV = OUTPUT_DIR / "all_clients_round_metrics.csv"
CONFUSION_SUMMARY_CSV = OUTPUT_DIR / "all_clients_confusion_summary.csv"
AGGREGATE_SUMMARY_CSV = OUTPUT_DIR / "fedproto_seed45_aggregate_summary.csv"
SERVER_SUMMARY_CSV = OUTPUT_DIR / "server_aggregation_summary.csv"
ANALYSIS_JSON = OUTPUT_DIR / "fedproto_seed45_analysis.json"

PLOT_FINAL_MACRO_F1 = OUTPUT_DIR / "final_macro_f1_by_client.png"
PLOT_FINAL_WEIGHTED_F1 = OUTPUT_DIR / "final_weighted_f1_by_client.png"
PLOT_CONVERGENCE = OUTPUT_DIR / "validation_macro_f1_convergence.png"
PLOT_TOTAL_TIME = OUTPUT_DIR / "total_experiment_time_by_client.png"
PLOT_COMMUNICATION = OUTPUT_DIR / "communication_bytes_by_client.png"
PLOT_PROTO_LOSS = OUTPUT_DIR / "prototype_loss_by_round.png"
PLOT_CE_LOSS = OUTPUT_DIR / "classification_loss_by_round.png"
PLOT_COVERAGE = OUTPUT_DIR / "prototype_coverage_by_round.png"


# ============================================================
# Generic helpers
# ============================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return pd.DataFrame()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return {}


def numeric_value(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")

    return result if math.isfinite(result) else float("nan")


def first_existing(directory: Path, candidates: list[str]) -> Path | None:
    for filename in candidates:
        path = directory / filename
        if path.exists():
            return path
    return None


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def safe_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=0)) if not values.empty else float("nan")


def safe_last(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def safe_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else float("nan")


def safe_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


# ============================================================
# Client file readers
# ============================================================

def read_client_final_metrics(
    client_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(config["directory"])

    final_path = first_existing(
        directory,
        [
            f"{client_id}_final_metrics.csv",
            f"{client_id}_fedproto_final_metrics.csv",
        ],
    )

    convergence_path = first_existing(
        directory,
        [
            f"{client_id}_convergence_summary.json",
            f"{client_id}_fedproto_convergence_summary.json",
        ],
    )

    row: dict[str, Any] = {
        "seed": SEED,
        "client_id": client_id,
        "dataset": config["dataset"],
        "local_classes": config["local_classes"],
    }

    if final_path is not None:
        df = safe_read_csv(final_path)

        if not df.empty:
            first = df.iloc[0].to_dict()
            row.update(first)

    convergence = (
        safe_read_json(convergence_path)
        if convergence_path is not None
        else {}
    )

    final_test_metrics = convergence.get("final_test_metrics", {})

    # Fill values from JSON when missing from CSV.
    for key in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]:
        if key not in row or pd.isna(row.get(key)):
            if key in final_test_metrics:
                row[key] = final_test_metrics[key]

    for key in [
        "best_round",
        "best_validation_macro_f1",
        "convergence_round",
        "convergence_time_sec",
        "total_experiment_time_sec",
        "average_round_time_sec",
        "train_samples",
        "validation_samples",
        "test_samples",
        "number_of_classes",
        "lambda_proto",
        "prototype_dimension",
    ]:
        if key not in row or pd.isna(row.get(key)):
            if key in convergence:
                row[key] = convergence[key]

    row["final_metrics_file"] = str(final_path) if final_path else ""
    row["convergence_file"] = (
        str(convergence_path) if convergence_path else ""
    )

    return row


def read_client_round_metrics(
    client_id: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    directory = Path(config["directory"])

    path = first_existing(
        directory,
        [
            f"{client_id}_fedproto_metrics_log.csv",
            f"{client_id}_metrics_log.csv",
        ],
    )

    if path is None:
        return pd.DataFrame()

    df = safe_read_csv(path)

    if df.empty:
        return df

    df.insert(0, "dataset", config["dataset"])
    df.insert(0, "client_id", client_id)

    return df


def read_client_communication(
    client_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(config["directory"])

    path = first_existing(
        directory,
        [
            f"{client_id}_fedproto_comm_log.csv",
            f"{client_id}_comm_log.csv",
        ],
    )

    result = {
        "seed": SEED,
        "client_id": client_id,
        "dataset": config["dataset"],
        "communication_rounds": 0,
        "total_payload_bytes_sent": float("nan"),
        "mean_payload_bytes_per_round": float("nan"),
        "min_payload_bytes_per_round": float("nan"),
        "max_payload_bytes_per_round": float("nan"),
        "total_rtt_sec": float("nan"),
        "mean_rtt_sec": float("nan"),
        "total_barrier_wait_sec": float("nan"),
        "mean_barrier_wait_sec": float("nan"),
        "communication_file": str(path) if path else "",
    }

    if path is None:
        return result

    df = safe_read_csv(path)

    if df.empty:
        return result

    bytes_column = None

    for candidate in [
        "payload_bytes",
        "prototype_payload_bytes",
        "bytes_sent",
    ]:
        if candidate in df.columns:
            bytes_column = candidate
            break

    rtt_column = None
    for candidate in [
        "rtt_sec",
        "communication_rtt_sec",
    ]:
        if candidate in df.columns:
            rtt_column = candidate
            break

    barrier_column = None
    for candidate in [
        "barrier_wait_sec",
        "barrier_wait_time_sec",
    ]:
        if candidate in df.columns:
            barrier_column = candidate
            break

    result["communication_rounds"] = int(len(df))

    if bytes_column:
        values = pd.to_numeric(df[bytes_column], errors="coerce")
        result["total_payload_bytes_sent"] = float(values.sum())
        result["mean_payload_bytes_per_round"] = float(values.mean())
        result["min_payload_bytes_per_round"] = float(values.min())
        result["max_payload_bytes_per_round"] = float(values.max())

    if rtt_column:
        values = pd.to_numeric(df[rtt_column], errors="coerce")
        result["total_rtt_sec"] = float(values.sum())
        result["mean_rtt_sec"] = float(values.mean())

    if barrier_column:
        values = pd.to_numeric(df[barrier_column], errors="coerce")
        result["total_barrier_wait_sec"] = float(values.sum())
        result["mean_barrier_wait_sec"] = float(values.mean())

    return result


def read_client_prototype_summary(
    client_id: str,
    config: dict[str, Any],
    round_df: pd.DataFrame,
) -> dict[str, Any]:
    directory = Path(config["directory"])

    prototype_log_path = first_existing(
        directory,
        [
            f"{client_id}_prototype_log.csv",
            f"{client_id}_fedproto_prototype_log.csv",
        ],
    )

    result = {
        "seed": SEED,
        "client_id": client_id,
        "dataset": config["dataset"],
        "local_classes": config["local_classes"],
        "mean_prototype_loss": float("nan"),
        "final_prototype_loss": float("nan"),
        "minimum_prototype_loss": float("nan"),
        "maximum_prototype_loss": float("nan"),
        "mean_prototype_coverage": float("nan"),
        "final_prototype_coverage": float("nan"),
        "mean_matching_global_classes": float("nan"),
        "final_matching_global_classes": float("nan"),
        "mean_prototype_norm": float("nan"),
        "final_round_mean_prototype_norm": float("nan"),
        "prototype_log_file": (
            str(prototype_log_path) if prototype_log_path else ""
        ),
    }

    if not round_df.empty:
        if "train_prototype_loss" in round_df.columns:
            result["mean_prototype_loss"] = safe_mean(
                round_df["train_prototype_loss"]
            )
            result["final_prototype_loss"] = safe_last(
                round_df["train_prototype_loss"]
            )
            result["minimum_prototype_loss"] = safe_min(
                round_df["train_prototype_loss"]
            )
            result["maximum_prototype_loss"] = safe_max(
                round_df["train_prototype_loss"]
            )

        if "prototype_coverage_ratio" in round_df.columns:
            result["mean_prototype_coverage"] = safe_mean(
                round_df["prototype_coverage_ratio"]
            )
            result["final_prototype_coverage"] = safe_last(
                round_df["prototype_coverage_ratio"]
            )

        if "available_global_classes_for_client" in round_df.columns:
            result["mean_matching_global_classes"] = safe_mean(
                round_df["available_global_classes_for_client"]
            )
            result["final_matching_global_classes"] = safe_last(
                round_df["available_global_classes_for_client"]
            )

    if prototype_log_path is not None:
        proto_df = safe_read_csv(prototype_log_path)

        if not proto_df.empty and "prototype_l2_norm" in proto_df.columns:
            result["mean_prototype_norm"] = safe_mean(
                proto_df["prototype_l2_norm"]
            )

            if "round" in proto_df.columns:
                round_numbers = pd.to_numeric(
                    proto_df["round"],
                    errors="coerce",
                )
                max_round = round_numbers.max()

                final_rows = proto_df[
                    round_numbers == max_round
                ]

                result["final_round_mean_prototype_norm"] = safe_mean(
                    final_rows["prototype_l2_norm"]
                )

    return result


def read_confusion_summary(
    client_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(config["directory"])

    path = first_existing(
        directory,
        [
            f"{client_id}_confusion_matrix.csv",
            f"{client_id}_fedproto_confusion_matrix.csv",
        ],
    )

    result = {
        "seed": SEED,
        "client_id": client_id,
        "dataset": config["dataset"],
        "total_test_samples_from_confusion_matrix": float("nan"),
        "correct_predictions": float("nan"),
        "errors": float("nan"),
        "confusion_accuracy": float("nan"),
        "most_confused_true_class": "",
        "most_confused_predicted_class": "",
        "largest_off_diagonal_count": float("nan"),
        "confusion_matrix_file": str(path) if path else "",
    }

    if path is None:
        return result

    df = safe_read_csv(path)

    if df.empty:
        return result

    # The first CSV column stores row labels.
    if df.shape[1] > 1:
        row_names = df.iloc[:, 0].astype(str).tolist()
        matrix_df = df.iloc[:, 1:].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0)
        col_names = [str(column) for column in matrix_df.columns]
    else:
        return result

    matrix = matrix_df.to_numpy(dtype=np.float64)

    total = float(matrix.sum())
    correct = float(np.trace(matrix))
    errors = total - correct

    result["total_test_samples_from_confusion_matrix"] = total
    result["correct_predictions"] = correct
    result["errors"] = errors
    result["confusion_accuracy"] = (
        correct / total if total > 0 else float("nan")
    )

    if matrix.shape[0] == matrix.shape[1] and matrix.size > 0:
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, -np.inf)

        flat_index = int(np.argmax(off_diagonal))
        row_index, column_index = np.unravel_index(
            flat_index,
            off_diagonal.shape,
        )

        largest = off_diagonal[row_index, column_index]

        if np.isfinite(largest):
            result["most_confused_true_class"] = (
                row_names[row_index]
                if row_index < len(row_names)
                else str(row_index)
            )
            result["most_confused_predicted_class"] = (
                col_names[column_index]
                if column_index < len(col_names)
                else str(column_index)
            )
            result["largest_off_diagonal_count"] = float(largest)

    return result


# ============================================================
# Aggregate statistics
# ============================================================

def build_aggregate_summary(
    final_df: pd.DataFrame,
    communication_df: pd.DataFrame,
    prototype_df: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "best_validation_macro_f1",
        "total_experiment_time_sec",
        "average_round_time_sec",
    ]

    rows = []

    for metric in metric_columns:
        if metric not in final_df.columns:
            continue

        values = pd.to_numeric(
            final_df[metric],
            errors="coerce",
        )

        rows.append({
            "scope": "client_macro_average",
            "metric": metric,
            "value": float(values.mean()),
            "std_across_clients": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        })

    # Test-sample weighted performance.
    if "test_samples" in final_df.columns:
        weights = pd.to_numeric(
            final_df["test_samples"],
            errors="coerce",
        ).fillna(0)

        for metric in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
        ]:
            if metric not in final_df.columns:
                continue

            values = pd.to_numeric(
                final_df[metric],
                errors="coerce",
            )

            mask = (
                values.notna()
                & weights.notna()
                & (weights > 0)
            )

            weighted_value = (
                float(
                    np.average(
                        values[mask],
                        weights=weights[mask],
                    )
                )
                if mask.any()
                else float("nan")
            )

            rows.append({
                "scope": "test_sample_weighted_average",
                "metric": metric,
                "value": weighted_value,
                "std_across_clients": float("nan"),
                "minimum": float("nan"),
                "maximum": float("nan"),
            })

    if not communication_df.empty:
        for metric in [
            "total_payload_bytes_sent",
            "mean_payload_bytes_per_round",
            "total_rtt_sec",
            "total_barrier_wait_sec",
        ]:
            if metric not in communication_df.columns:
                continue

            values = pd.to_numeric(
                communication_df[metric],
                errors="coerce",
            )

            rows.append({
                "scope": "communication_total_all_clients",
                "metric": metric,
                "value": float(values.sum()),
                "std_across_clients": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })

    if not prototype_df.empty:
        for metric in [
            "mean_prototype_loss",
            "final_prototype_loss",
            "mean_prototype_coverage",
            "final_prototype_coverage",
        ]:
            if metric not in prototype_df.columns:
                continue

            values = pd.to_numeric(
                prototype_df[metric],
                errors="coerce",
            )

            rows.append({
                "scope": "prototype_client_average",
                "metric": metric,
                "value": float(values.mean()),
                "std_across_clients": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            })

    return pd.DataFrame(rows)


# ============================================================
# Server logs
# ============================================================

def read_server_summary() -> pd.DataFrame:
    path = SERVER_LOG_DIR / "server_fedproto_aggregation_log.csv"

    df = safe_read_csv(path)

    if df.empty:
        return df

    numeric_columns = [
        "server_round_after_aggregation",
        "num_clients_used",
        "total_client_training_samples",
        "number_of_global_prototypes",
        "number_of_shared_classes",
        "number_of_single_contributor_classes",
        "bytes_received_round",
        "bytes_sent_last_pull",
        "aggregation_time_sec",
        "round_duration_sec",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# Plotting
# ============================================================

def plot_bar(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if df.empty or y_column not in df.columns:
        return

    plot_df = df[[x_column, y_column]].copy()
    plot_df[y_column] = pd.to_numeric(
        plot_df[y_column],
        errors="coerce",
    )
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df[x_column], plot_df[y_column])
    plt.title(title)
    plt.xlabel("Client")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_round_lines(
    round_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if round_df.empty or metric not in round_df.columns:
        return

    plt.figure(figsize=(10, 6))

    plotted = False

    for client_id, client_df in round_df.groupby("client_id"):
        if "round" not in client_df.columns:
            continue

        x = pd.to_numeric(
            client_df["round"],
            errors="coerce",
        )
        y = pd.to_numeric(
            client_df[metric],
            errors="coerce",
        )

        valid = x.notna() & y.notna()

        if not valid.any():
            continue

        plt.plot(
            x[valid],
            y[valid],
            marker="o",
            label=client_id,
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Communication round")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    final_rows = []
    communication_rows = []
    prototype_rows = []
    confusion_rows = []
    all_round_frames = []

    print_section("READING FEDPROTO OUTPUTS FOR ALL SIX CLIENTS")

    for client_id, config in CLIENTS.items():
        directory = Path(config["directory"])

        print(f"\n{client_id}: {config['dataset']}")
        print(f"Directory: {directory}")

        if not directory.exists():
            print("[WARNING] Directory does not exist.")

        final_row = read_client_final_metrics(
            client_id,
            config,
        )
        final_rows.append(final_row)

        round_df = read_client_round_metrics(
            client_id,
            config,
        )

        if not round_df.empty:
            all_round_frames.append(round_df)

        communication_rows.append(
            read_client_communication(
                client_id,
                config,
            )
        )

        prototype_rows.append(
            read_client_prototype_summary(
                client_id,
                config,
                round_df,
            )
        )

        confusion_rows.append(
            read_confusion_summary(
                client_id,
                config,
            )
        )

    final_df = pd.DataFrame(final_rows)
    communication_df = pd.DataFrame(communication_rows)
    prototype_df = pd.DataFrame(prototype_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    round_df = (
        pd.concat(
            all_round_frames,
            ignore_index=True,
        )
        if all_round_frames
        else pd.DataFrame()
    )

    # Ensure a useful column order.
    final_columns = [
        "seed",
        "client_id",
        "dataset",
        "local_classes",
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
        "train_samples",
        "validation_samples",
        "test_samples",
        "number_of_classes",
        "lambda_proto",
        "prototype_dimension",
        "final_metrics_file",
        "convergence_file",
    ]

    final_df = final_df.reindex(
        columns=[
            column
            for column in final_columns
            if column in final_df.columns
        ]
    )

    convergence_columns = [
        "seed",
        "client_id",
        "dataset",
        "best_round",
        "best_validation_macro_f1",
        "convergence_round",
        "convergence_time_sec",
        "total_experiment_time_sec",
        "average_round_time_sec",
    ]

    convergence_df = final_df.reindex(
        columns=[
            column
            for column in convergence_columns
            if column in final_df.columns
        ]
    )

    aggregate_df = build_aggregate_summary(
        final_df,
        communication_df,
        prototype_df,
    )

    server_df = read_server_summary()

    # Save tables.
    final_df.to_csv(
        FINAL_PERFORMANCE_CSV,
        index=False,
    )
    convergence_df.to_csv(
        CONVERGENCE_CSV,
        index=False,
    )
    communication_df.to_csv(
        COMMUNICATION_CSV,
        index=False,
    )
    prototype_df.to_csv(
        PROTOTYPE_SUMMARY_CSV,
        index=False,
    )
    confusion_df.to_csv(
        CONFUSION_SUMMARY_CSV,
        index=False,
    )
    round_df.to_csv(
        ROUND_METRICS_CSV,
        index=False,
    )
    aggregate_df.to_csv(
        AGGREGATE_SUMMARY_CSV,
        index=False,
    )
    server_df.to_csv(
        SERVER_SUMMARY_CSV,
        index=False,
    )

    # Plots.
    plot_bar(
        final_df,
        "client_id",
        "macro_f1",
        "FedProto Final Macro-F1 — Seed 45",
        "Test Macro-F1",
        PLOT_FINAL_MACRO_F1,
    )

    plot_bar(
        final_df,
        "client_id",
        "weighted_f1",
        "FedProto Final Weighted-F1 — Seed 45",
        "Test Weighted-F1",
        PLOT_FINAL_WEIGHTED_F1,
    )

    plot_bar(
        final_df,
        "client_id",
        "total_experiment_time_sec",
        "FedProto Total Experiment Time — Seed 45",
        "Seconds",
        PLOT_TOTAL_TIME,
    )

    plot_bar(
        communication_df,
        "client_id",
        "total_payload_bytes_sent",
        "FedProto Total Prototype Payload Sent — Seed 45",
        "Bytes",
        PLOT_COMMUNICATION,
    )

    plot_round_lines(
        round_df,
        "validation_macro_f1",
        "FedProto Validation Macro-F1 Convergence — Seed 45",
        "Validation Macro-F1",
        PLOT_CONVERGENCE,
    )

    plot_round_lines(
        round_df,
        "train_prototype_loss",
        "FedProto Prototype Loss — Seed 45",
        "Prototype Loss",
        PLOT_PROTO_LOSS,
    )

    plot_round_lines(
        round_df,
        "train_classification_loss",
        "FedProto Classification Loss — Seed 45",
        "Classification Loss",
        PLOT_CE_LOSS,
    )

    plot_round_lines(
        round_df,
        "prototype_coverage_ratio",
        "FedProto Prototype Coverage — Seed 45",
        "Coverage Ratio",
        PLOT_COVERAGE,
    )

    # Main machine-readable analysis.
    analysis = {
        "seed": SEED,
        "clients_expected": len(CLIENTS),
        "clients_with_final_results": int(
            final_df["macro_f1"].notna().sum()
            if "macro_f1" in final_df.columns
            else 0
        ),
        "client_macro_average": {},
        "test_sample_weighted_average": {},
        "best_client_by_test_macro_f1": None,
        "lowest_client_by_test_macro_f1": None,
        "communication_total_bytes_sent_all_clients": None,
        "output_directory": str(OUTPUT_DIR),
    }

    for metric in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]:
        if metric in final_df.columns:
            values = pd.to_numeric(
                final_df[metric],
                errors="coerce",
            )
            analysis["client_macro_average"][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }

    if (
        "test_samples" in final_df.columns
        and "macro_f1" in final_df.columns
    ):
        weights = pd.to_numeric(
            final_df["test_samples"],
            errors="coerce",
        )
        values = pd.to_numeric(
            final_df["macro_f1"],
            errors="coerce",
        )
        valid = values.notna() & weights.notna() & (weights > 0)

        if valid.any():
            analysis["test_sample_weighted_average"]["macro_f1"] = float(
                np.average(
                    values[valid],
                    weights=weights[valid],
                )
            )

    if "macro_f1" in final_df.columns:
        macro_values = pd.to_numeric(
            final_df["macro_f1"],
            errors="coerce",
        )

        if macro_values.notna().any():
            best_index = macro_values.idxmax()
            worst_index = macro_values.idxmin()

            analysis["best_client_by_test_macro_f1"] = {
                "client_id": final_df.loc[best_index, "client_id"],
                "dataset": final_df.loc[best_index, "dataset"],
                "macro_f1": float(macro_values.loc[best_index]),
            }

            analysis["lowest_client_by_test_macro_f1"] = {
                "client_id": final_df.loc[worst_index, "client_id"],
                "dataset": final_df.loc[worst_index, "dataset"],
                "macro_f1": float(macro_values.loc[worst_index]),
            }

    if "total_payload_bytes_sent" in communication_df.columns:
        analysis[
            "communication_total_bytes_sent_all_clients"
        ] = float(
            pd.to_numeric(
                communication_df["total_payload_bytes_sent"],
                errors="coerce",
            ).sum()
        )

    with ANALYSIS_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            analysis,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Console report.
    print_section("FINAL PERFORMANCE")

    display_columns = [
        column
        for column in [
            "client_id",
            "dataset",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "best_round",
            "convergence_round",
            "total_experiment_time_sec",
        ]
        if column in final_df.columns
    ]

    print(
        final_df[display_columns].to_string(
            index=False
        )
    )

    print_section("COMMUNICATION")

    communication_display = [
        column
        for column in [
            "client_id",
            "communication_rounds",
            "total_payload_bytes_sent",
            "mean_payload_bytes_per_round",
            "mean_rtt_sec",
            "mean_barrier_wait_sec",
        ]
        if column in communication_df.columns
    ]

    print(
        communication_df[
            communication_display
        ].to_string(index=False)
    )

    print_section("PROTOTYPE SUMMARY")

    prototype_display = [
        column
        for column in [
            "client_id",
            "local_classes",
            "mean_prototype_loss",
            "final_prototype_loss",
            "mean_prototype_coverage",
            "final_prototype_coverage",
            "final_matching_global_classes",
        ]
        if column in prototype_df.columns
    ]

    print(
        prototype_df[
            prototype_display
        ].to_string(index=False)
    )

    print_section("SAVED ANALYSIS FILES")

    for path in [
        FINAL_PERFORMANCE_CSV,
        CONVERGENCE_CSV,
        COMMUNICATION_CSV,
        PROTOTYPE_SUMMARY_CSV,
        ROUND_METRICS_CSV,
        CONFUSION_SUMMARY_CSV,
        AGGREGATE_SUMMARY_CSV,
        SERVER_SUMMARY_CSV,
        ANALYSIS_JSON,
        PLOT_FINAL_MACRO_F1,
        PLOT_FINAL_WEIGHTED_F1,
        PLOT_CONVERGENCE,
        PLOT_TOTAL_TIME,
        PLOT_COMMUNICATION,
        PLOT_PROTO_LOSS,
        PLOT_CE_LOSS,
        PLOT_COVERAGE,
    ]:
        print(path)

    print("\nFedProto seed-45 analysis completed successfully.")


if __name__ == "__main__":
    main()