#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# COMPLETE PFTL MULTI-CLASS SCALABILITY ANALYSIS
# ============================================================
#
# Reads recursively:
#   *_communication.csv
#   *_scalability.csv
#   *_convergence_curve.csv
#   *_metrics.csv
#   *_partition.csv
#
# Supports federation sizes:
#   6, 12, 30, 60, 90
#
# Main outputs:
#   system_scalability_per_client.csv
#   system_scalability_per_seed.csv
#   system_scalability_summary.csv
#
#   convergence_scalability_per_client.csv
#   convergence_scalability_summary.csv
#
#   partition_scalability_per_client.csv
#   partition_scalability_summary.csv
#
#   system_scalability_table.tex
#   convergence_scalability_table.tex
#   combined_scalability_table.tex
#
# Notes:
#   1. metrics.csv contains LOCAL, MIXED_CANDIDATE, and GLOBAL
#      rows for each round. It is used to repair missing values
#      in convergence_curve.csv.
#
#   2. metrics.csv is not treated as a final-test metrics file.
#
#   3. Server aggregation time is not available in the supplied
#      client logs. It is therefore not reported.
#
# ============================================================

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/scalability_logs"
)

OUTPUT_DIR = ROOT_DIR / "scalability_analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENT_COUNTS = [6, 12, 30, 60, 90]

EXPECTED_ROUNDS = 20

# Convergence:
# First round reaching 95% of the best observed Macro-F1,
# with subsequent values remaining above 93% of the best.
CONVERGENCE_THRESHOLD = 0.95
CONVERGENCE_TOLERANCE = 0.02


# ============================================================
# Basic utility functions
# ============================================================

def safe_numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric Series or an all-NaN Series."""

    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(numeric.mean())


def safe_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    if len(numeric) == 1:
        return 0.0

    return float(numeric.std(ddof=1))


def safe_min(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(numeric.min())


def safe_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(numeric.max())


def safe_sum(values: pd.Series) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return np.nan

    return float(numeric.sum())


def format_float(
    value: float,
    decimals: int = 3,
) -> str:
    if pd.isna(value):
        return "--"

    return f"{value:.{decimals}f}"


def format_integer(value: float) -> str:
    if pd.isna(value):
        return "--"

    return f"{value:,.0f}"


def read_csv_safely(
    file_path: Path,
) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(file_path)
    except Exception as exc:
        print(
            f"WARNING: Could not read {file_path}: {exc}",
            file=sys.stderr,
        )
        return None

    if frame.empty:
        print(
            f"WARNING: Empty CSV skipped: {file_path}",
            file=sys.stderr,
        )
        return None

    return frame


# ============================================================
# Experiment discovery
# ============================================================

def extract_seed_from_directory_name(
    directory_name: str,
) -> Optional[int]:
    match = re.search(
        r"seed_(\d+)",
        directory_name,
    )

    if match is None:
        return None

    return int(match.group(1))


def find_experiment_directories(
    client_count: int,
) -> list[Path]:
    patterns = [
        f"clients_{client_count}_fullsize_noniid_alpha_*",
        f"clients_{client_count}_*",
    ]

    results: set[Path] = set()

    for pattern in patterns:
        for path in ROOT_DIR.glob(pattern):
            if path.is_dir():
                results.add(path)

    return sorted(results)


def find_files(
    experiment_dir: Path,
    suffix: str,
) -> list[Path]:
    return sorted(
        experiment_dir.rglob(f"*{suffix}")
    )


def infer_client_id(
    file_path: Path,
    suffix: str,
) -> str:
    return file_path.name.replace(
        suffix,
        "",
    )


def add_common_columns(
    frame: pd.DataFrame,
    file_path: Path,
    suffix: str,
    expected_clients: int,
    directory_seed: Optional[int],
) -> pd.DataFrame:
    data = frame.copy()

    if "client_id" not in data.columns:
        data["client_id"] = infer_client_id(
            file_path,
            suffix,
        )

    if "seed" not in data.columns:
        data["seed"] = directory_seed

    if "total_experiment_clients" not in data.columns:
        data["total_experiment_clients"] = expected_clients

    data["federation_size"] = expected_clients
    data["source_file"] = str(file_path)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

    return data


# ============================================================
# Repeated execution handling
# ============================================================

def assign_run_ids(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect multiple executions in a CSV.

    Example:
        rounds 1 ... 14, 1 ... 20

    produces:
        run 0: 1 ... 14
        run 1: 1 ... 20
    """

    data = frame.copy()

    if "timestamp" in data.columns:
        data = data.sort_values(
            "timestamp",
            kind="stable",
        ).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    rounds = pd.to_numeric(
        data["round"],
        errors="coerce",
    )

    previous = rounds.shift(1)

    new_run = (
        rounds.lt(previous)
        | (
            rounds.eq(1)
            & previous.notna()
            & previous.ne(1)
        )
    ).fillna(False)

    data["detected_run_id"] = new_run.cumsum()

    return data


def select_latest_complete_run(
    frame: pd.DataFrame,
    expected_rounds: int = EXPECTED_ROUNDS,
) -> pd.DataFrame:
    """
    Select the latest complete 20-round run.

    If none is complete, select the latest run having the
    greatest number of unique rounds.
    """

    if frame.empty or "round" not in frame.columns:
        return frame.copy()

    data = assign_run_ids(frame)

    candidates = []

    expected_set = set(
        range(1, expected_rounds + 1)
    )

    for run_id, run in data.groupby(
        "detected_run_id",
        sort=True,
    ):
        observed = set(
            pd.to_numeric(
                run["round"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        latest_timestamp = (
            run["timestamp"].max()
            if "timestamp" in run.columns
            else pd.NaT
        )

        candidates.append(
            {
                "run_id": run_id,
                "complete": expected_set.issubset(observed),
                "round_count": len(observed),
                "latest_timestamp": latest_timestamp,
            }
        )

    candidate_table = pd.DataFrame(candidates)

    complete = candidate_table[
        candidate_table["complete"]
    ]

    if not complete.empty:
        selected_run_id = (
            complete
            .sort_values(
                ["latest_timestamp", "run_id"],
                kind="stable",
            )
            .iloc[-1]["run_id"]
        )
    else:
        selected_run_id = (
            candidate_table
            .sort_values(
                [
                    "round_count",
                    "latest_timestamp",
                    "run_id",
                ],
                kind="stable",
            )
            .iloc[-1]["run_id"]
        )

        print(
            "WARNING: No complete run found. "
            "Using the latest run with the most rounds.",
            file=sys.stderr,
        )

    selected = data[
        data["detected_run_id"] == selected_run_id
    ].copy()

    selected = selected.drop(
        columns=["detected_run_id"],
        errors="ignore",
    )

    selected["round"] = pd.to_numeric(
        selected["round"],
        errors="coerce",
    )

    selected = selected.drop_duplicates(
        subset=["round"],
        keep="last",
    )

    return selected.sort_values(
        "round",
        kind="stable",
    ).reset_index(drop=True)


def clean_standard_logs(
    logs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean logs having one row per round.
    """

    if logs.empty:
        return logs.copy()

    cleaned = []

    for (seed, client_id), group in logs.groupby(
        ["seed", "client_id"],
        sort=False,
        dropna=False,
    ):
        selected = select_latest_complete_run(
            group,
            expected_rounds=EXPECTED_ROUNDS,
        )

        selected["seed"] = seed
        selected["client_id"] = client_id

        cleaned.append(selected)

    if not cleaned:
        return pd.DataFrame()

    return pd.concat(
        cleaned,
        ignore_index=True,
    )


# ============================================================
# Stage-metrics repeated-run handling
# ============================================================

def clean_stage_metrics_runs(
    logs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean metrics.csv logs.

    Each round contains:
        LOCAL
        MIXED_CANDIDATE
        GLOBAL

    A new run is detected when stage=LOCAL and round=1
    occurs again.
    """

    if logs.empty:
        return logs.copy()

    cleaned_groups = []

    for (seed, client_id), group in logs.groupby(
        ["seed", "client_id"],
        sort=False,
        dropna=False,
    ):
        group = group.copy()

        if "timestamp" in group.columns:
            group = group.sort_values(
                ["timestamp", "round", "stage"],
                kind="stable",
            ).reset_index(drop=True)
        else:
            group = group.reset_index(drop=True)

        group["round"] = pd.to_numeric(
            group["round"],
            errors="coerce",
        )

        group["stage"] = (
            group["stage"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        run_start = (
            group["round"].eq(1)
            & group["stage"].eq("LOCAL")
        )

        group["detected_run_id"] = (
            run_start.cumsum() - 1
        )

        candidates = []

        expected_set = set(
            range(1, EXPECTED_ROUNDS + 1)
        )

        for run_id, run in group.groupby(
            "detected_run_id",
            sort=True,
        ):
            local_rounds = set(
                run.loc[
                    run["stage"].eq("LOCAL"),
                    "round",
                ]
                .dropna()
                .astype(int)
                .tolist()
            )

            latest_timestamp = (
                run["timestamp"].max()
                if "timestamp" in run.columns
                else pd.NaT
            )

            candidates.append(
                {
                    "run_id": run_id,
                    "complete":
                        expected_set.issubset(local_rounds),
                    "round_count": len(local_rounds),
                    "latest_timestamp": latest_timestamp,
                }
            )

        candidate_table = pd.DataFrame(candidates)

        complete = candidate_table[
            candidate_table["complete"]
        ]

        if not complete.empty:
            selected_run_id = (
                complete
                .sort_values(
                    ["latest_timestamp", "run_id"],
                    kind="stable",
                )
                .iloc[-1]["run_id"]
            )
        else:
            selected_run_id = (
                candidate_table
                .sort_values(
                    [
                        "round_count",
                        "latest_timestamp",
                        "run_id",
                    ],
                    kind="stable",
                )
                .iloc[-1]["run_id"]
            )

        selected = group[
            group["detected_run_id"]
            == selected_run_id
        ].copy()

        selected = selected.drop(
            columns=["detected_run_id"],
            errors="ignore",
        )

        selected = selected.drop_duplicates(
            subset=["round", "stage"],
            keep="last",
        )

        cleaned_groups.append(selected)

    if not cleaned_groups:
        return pd.DataFrame()

    return pd.concat(
        cleaned_groups,
        ignore_index=True,
    )


# ============================================================
# Client coverage validation
# ============================================================

def validate_client_coverage(
    logs: pd.DataFrame,
    expected_clients: int,
    log_name: str,
    rows_per_round: int = 1,
) -> None:
    if logs.empty:
        print(
            f"WARNING: {log_name} logs are empty."
        )
        return

    observed_clients = logs[
        "client_id"
    ].nunique()

    seed_count = logs["seed"].nunique()

    expected_rows = (
        expected_clients
        * EXPECTED_ROUNDS
        * seed_count
        * rows_per_round
    )

    observed_rows = len(logs)

    if observed_clients == expected_clients:
        print(
            f"{log_name}: correctly found "
            f"{observed_clients} clients."
        )
    else:
        print(
            f"WARNING: {log_name}: expected "
            f"{expected_clients} clients, found "
            f"{observed_clients}."
        )

    if observed_rows == expected_rows:
        print(
            f"{log_name}: correctly found "
            f"{observed_rows} rows."
        )
    else:
        print(
            f"WARNING: {log_name}: expected "
            f"{expected_rows} rows, found "
            f"{observed_rows}."
        )


# ============================================================
# Communication logs
# ============================================================

def load_communication_logs(
    experiment_dir: Path,
    expected_clients: int,
    directory_seed: Optional[int],
) -> pd.DataFrame:
    frames = []

    for file_path in find_files(
        experiment_dir,
        "_communication.csv",
    ):
        frame = read_csv_safely(file_path)

        if frame is None or "round" not in frame.columns:
            continue

        frame = add_common_columns(
            frame,
            file_path,
            "_communication.csv",
            expected_clients,
            directory_seed,
        )

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(
        frames,
        ignore_index=True,
    )

    numeric_columns = [
        "seed",
        "round",
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
    ]

    for column in numeric_columns:
        if column in logs.columns:
            logs[column] = pd.to_numeric(
                logs[column],
                errors="coerce",
            )

    matching = logs[
        logs["total_experiment_clients"]
        == expected_clients
    ]

    if not matching.empty:
        logs = matching.copy()

    logs = clean_standard_logs(logs)

    logs["shared_exchange_bytes"] = (
        safe_numeric(
            logs,
            "bytes_uploaded",
        ).fillna(0)
        + safe_numeric(
            logs,
            "final_model_bytes_downloaded",
        ).fillna(0)
    )

    logs["rpc_time_sec"] = (
        safe_numeric(
            logs,
            "upload_rpc_sec",
        ).fillna(0)
        + safe_numeric(
            logs,
            "poll_rpc_sec",
        ).fillna(0)
    )

    return logs


# ============================================================
# Scalability logs
# ============================================================

def load_scalability_logs(
    experiment_dir: Path,
    expected_clients: int,
    directory_seed: Optional[int],
) -> pd.DataFrame:
    frames = []

    for file_path in find_files(
        experiment_dir,
        "_scalability.csv",
    ):
        frame = read_csv_safely(file_path)

        if frame is None or "round" not in frame.columns:
            continue

        frame = add_common_columns(
            frame,
            file_path,
            "_scalability.csv",
            expected_clients,
            directory_seed,
        )

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(
        frames,
        ignore_index=True,
    )

    numeric_columns = [
        "seed",
        "round",
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
    ]

    for column in numeric_columns:
        if column in logs.columns:
            logs[column] = pd.to_numeric(
                logs[column],
                errors="coerce",
            )

    matching = logs[
        logs["total_experiment_clients"]
        == expected_clients
    ]

    if not matching.empty:
        logs = matching.copy()

    logs = clean_standard_logs(logs)

    return logs


# ============================================================
# Convergence curve logs
# ============================================================

def load_convergence_logs(
    experiment_dir: Path,
    expected_clients: int,
    directory_seed: Optional[int],
) -> pd.DataFrame:
    frames = []

    for file_path in find_files(
        experiment_dir,
        "_convergence_curve.csv",
    ):
        frame = read_csv_safely(file_path)

        if frame is None or "round" not in frame.columns:
            continue

        frame = add_common_columns(
            frame,
            file_path,
            "_convergence_curve.csv",
            expected_clients,
            directory_seed,
        )

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(
        frames,
        ignore_index=True,
    )

    numeric_columns = [
        "seed",
        "round",
        "local_macro_f1",
        "mixed_candidate_macro_f1",
        "post_aggregation_global_macro_f1",
        "switch_on",
        "gamma_global",
        "elapsed_experiment_sec",
        "total_experiment_clients",
    ]

    for column in numeric_columns:
        if column in logs.columns:
            logs[column] = pd.to_numeric(
                logs[column],
                errors="coerce",
            )

    if "total_experiment_clients" in logs.columns:
        matching = logs[
            logs["total_experiment_clients"]
            == expected_clients
        ]

        if not matching.empty:
            logs = matching.copy()

    logs = clean_standard_logs(logs)

    return logs


# ============================================================
# Stage-based metrics logs
# ============================================================

def load_stage_metrics_logs(
    experiment_dir: Path,
    expected_clients: int,
    directory_seed: Optional[int],
) -> pd.DataFrame:
    frames = []

    for file_path in find_files(
        experiment_dir,
        "_metrics.csv",
    ):
        frame = read_csv_safely(file_path)

        if frame is None:
            continue

        required = {
            "round",
            "stage",
            "macro_f1",
        }

        missing = required.difference(frame.columns)

        if missing:
            print(
                f"WARNING: Skipping {file_path}; "
                f"missing {sorted(missing)}",
                file=sys.stderr,
            )
            continue

        frame = add_common_columns(
            frame,
            file_path,
            "_metrics.csv",
            expected_clients,
            directory_seed,
        )

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(
        frames,
        ignore_index=True,
    )

    numeric_columns = [
        "seed",
        "round",
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
        "gamma_local",
        "gamma_global",
        "candidate_delta_macro_f1",
    ]

    for column in numeric_columns:
        if column in logs.columns:
            logs[column] = pd.to_numeric(
                logs[column],
                errors="coerce",
            )

    logs["stage"] = (
        logs["stage"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    logs = clean_stage_metrics_runs(logs)

    return logs


# ============================================================
# Partition logs
# ============================================================

def assign_partition_run_ids(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect repeated executions appended to one *_partition.csv file.

    A partition execution writes exactly one row per observed class and the
    class IDs are emitted in ascending order by np.unique(...). If the same
    client is executed again, another ascending class-ID sequence is appended
    to the same CSV. Therefore a new run starts whenever class_id is less than
    or equal to the previous class_id.

    This prevents an appended file such as
        0,1,...,9,0,1,...,9
    from being interpreted as 20 positive classes and twice the true number
    of partition samples.
    """
    if frame.empty:
        return frame.copy()

    data = frame.copy().reset_index(drop=True)
    data["_original_order"] = np.arange(len(data), dtype=int)

    if "class_id" not in data.columns:
        data["detected_partition_run_id"] = 0
        return data

    data["class_id"] = pd.to_numeric(
        data["class_id"],
        errors="coerce",
    )

    # Preserve file append order. Timestamp has only second-level resolution in
    # the client logs, so it must not be the sole ordering key.
    class_ids = data["class_id"]
    previous = class_ids.shift(1)

    new_run = (
        class_ids.notna()
        & previous.notna()
        & class_ids.le(previous)
    ).fillna(False)

    data["detected_partition_run_id"] = new_run.cumsum()
    return data


def select_latest_complete_partition_run(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep one partition execution for one client/file.

    Preference order:
      1. Runs with the largest number of unique class IDs (normally a complete
         partition dump).
      2. Among those, the latest run in file append order.

    Because the partition generator enforces a minimum number of samples for
    every class, a complete run should contain all local class IDs.
    """
    if frame.empty:
        return frame.copy()

    data = assign_partition_run_ids(frame)

    if "detected_partition_run_id" not in data.columns:
        return data

    candidates = []

    for run_id, run in data.groupby(
        "detected_partition_run_id",
        sort=True,
    ):
        unique_classes = (
            pd.to_numeric(run.get("class_id"), errors="coerce")
            .dropna()
            .nunique()
            if "class_id" in run.columns
            else len(run)
        )

        latest_order = int(run["_original_order"].max())
        latest_timestamp = (
            run["timestamp"].max()
            if "timestamp" in run.columns
            else pd.NaT
        )

        candidates.append({
            "run_id": int(run_id),
            "unique_classes": int(unique_classes),
            "latest_order": latest_order,
            "latest_timestamp": latest_timestamp,
        })

    candidate_table = pd.DataFrame(candidates)

    max_classes = int(candidate_table["unique_classes"].max())
    best = candidate_table[
        candidate_table["unique_classes"] == max_classes
    ]

    selected_run_id = int(
        best.sort_values(
            ["latest_order", "run_id"],
            kind="stable",
        ).iloc[-1]["run_id"]
    )

    selected = data[
        data["detected_partition_run_id"] == selected_run_id
    ].copy()

    # Defensive guarantee: exactly one row per class in the selected run.
    if "class_id" in selected.columns:
        selected = selected.drop_duplicates(
            subset=["class_id"],
            keep="last",
        )

    selected = selected.drop(
        columns=[
            "detected_partition_run_id",
            "_original_order",
        ],
        errors="ignore",
    )

    return selected.reset_index(drop=True)


def clean_partition_logs(
    logs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove repeated partition dumps that were appended by rerunning clients.

    Cleaning is performed per federation size, seed, client ID, and source
    file so that independent clients/seeds are never merged.
    """
    if logs.empty:
        return logs.copy()

    cleaned = []

    group_columns = [
        column
        for column in [
            "federation_size",
            "seed",
            "client_id",
            "source_file",
        ]
        if column in logs.columns
    ]

    for _, group in logs.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        cleaned.append(
            select_latest_complete_partition_run(group)
        )

    if not cleaned:
        return pd.DataFrame()

    result = pd.concat(
        cleaned,
        ignore_index=True,
    )

    # A second defensive de-duplication protects against the same logical
    # client being discovered through more than one identical source path.
    dedupe_keys = [
        column
        for column in [
            "federation_size",
            "seed",
            "client_id",
            "class_id",
        ]
        if column in result.columns
    ]

    if len(dedupe_keys) >= 4:
        if "timestamp" in result.columns:
            result = result.sort_values(
                "timestamp",
                kind="stable",
            )

        result = result.drop_duplicates(
            subset=dedupe_keys,
            keep="last",
        )

    return result.reset_index(drop=True)


def load_partition_logs(
    experiment_dir: Path,
    expected_clients: int,
    directory_seed: Optional[int],
) -> pd.DataFrame:
    frames = []

    for file_path in find_files(
        experiment_dir,
        "_partition.csv",
    ):
        frame = read_csv_safely(file_path)

        if frame is None:
            continue

        frame = add_common_columns(
            frame,
            file_path,
            "_partition.csv",
            expected_clients,
            directory_seed,
        )

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(
        frames,
        ignore_index=True,
    )

    numeric_columns = [
        "seed",
        "virtual_client_index",
        "num_virtual_clients",
        "total_experiment_clients",
        "noniid_alpha",
        "full_dataset_size_multiplier",
        "minimum_samples_per_class_per_client",
        "class_id",
        "sample_count",
    ]

    for column in numeric_columns:
        if column in logs.columns:
            logs[column] = pd.to_numeric(
                logs[column],
                errors="coerce",
            )

    # Keep only rows belonging to the federation size currently being
    # analyzed. This mirrors the filtering already used for the other logs.
    if "total_experiment_clients" in logs.columns:
        matching = logs[
            logs["total_experiment_clients"] == expected_clients
        ]

        if not matching.empty:
            logs = matching.copy()

    # IMPORTANT: partition CSVs are append-only. A rerun can therefore leave
    # several complete class-distribution dumps in the same file. Clean those
    # repeats before computing sample totals, positive-class counts, or entropy.
    logs = clean_partition_logs(logs)

    return logs


# ============================================================
# Reconstruct convergence from metrics.csv
# ============================================================

def build_convergence_from_stage_metrics(
    metrics_logs: pd.DataFrame,
) -> pd.DataFrame:
    if metrics_logs.empty:
        return pd.DataFrame()

    keys = [
        "federation_size",
        "seed",
        "client_id",
        "round",
    ]

    wide = metrics_logs.pivot_table(
        index=keys,
        columns="stage",
        values="macro_f1",
        aggfunc="last",
    ).reset_index()

    wide.columns.name = None

    wide = wide.rename(
        columns={
            "LOCAL": "local_macro_f1",
            "MIXED_CANDIDATE":
                "mixed_candidate_macro_f1",
            "GLOBAL":
                "post_aggregation_global_macro_f1",
        }
    )

    local_rows = metrics_logs[
        metrics_logs["stage"].eq("LOCAL")
    ].copy()

    metadata_columns = [
        column
        for column in [
            *keys,
            "switch_on",
            "gamma_global",
            "timestamp",
        ]
        if column in local_rows.columns
    ]

    metadata = local_rows[
        metadata_columns
    ].drop_duplicates(
        subset=keys,
        keep="last",
    )

    reconstructed = wide.merge(
        metadata,
        on=keys,
        how="left",
    )

    return reconstructed


def repair_convergence_logs(
    convergence_logs: pd.DataFrame,
    metrics_logs: pd.DataFrame,
) -> pd.DataFrame:
    reconstructed = build_convergence_from_stage_metrics(
        metrics_logs
    )

    if reconstructed.empty:
        return convergence_logs.copy()

    if convergence_logs.empty:
        print(
            "Using metrics.csv to reconstruct all convergence logs."
        )
        repaired = reconstructed.copy()
    else:
        keys = [
            "federation_size",
            "seed",
            "client_id",
            "round",
        ]

        repaired = convergence_logs.merge(
            reconstructed,
            on=keys,
            how="outer",
            suffixes=("", "_metrics"),
        )

        repair_columns = [
            "local_macro_f1",
            "mixed_candidate_macro_f1",
            "post_aggregation_global_macro_f1",
            "switch_on",
            "gamma_global",
            "timestamp",
        ]

        for column in repair_columns:
            metrics_column = f"{column}_metrics"

            if metrics_column not in repaired.columns:
                continue

            if column not in repaired.columns:
                repaired[column] = repaired[
                    metrics_column
                ]
            else:
                repaired[column] = repaired[
                    column
                ].combine_first(
                    repaired[metrics_column]
                )

            repaired = repaired.drop(
                columns=[metrics_column],
                errors="ignore",
            )

    repaired["candidate_minus_local"] = (
        safe_numeric(
            repaired,
            "mixed_candidate_macro_f1",
        )
        - safe_numeric(
            repaired,
            "local_macro_f1",
        )
    )

    repaired["global_minus_local"] = (
        safe_numeric(
            repaired,
            "post_aggregation_global_macro_f1",
        )
        - safe_numeric(
            repaired,
            "local_macro_f1",
        )
    )

    keys = [
        "federation_size",
        "seed",
        "client_id",
        "round",
    ]

    return repaired.sort_values(
        keys,
        kind="stable",
    ).reset_index(drop=True)


# ============================================================
# Diagnostics
# ============================================================

def diagnose_convergence_logs(
    convergence_logs: pd.DataFrame,
) -> None:
    print()
    print("=" * 90)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 90)

    required = [
        "local_macro_f1",
        "mixed_candidate_macro_f1",
        "post_aggregation_global_macro_f1",
    ]

    problem_count = 0

    for keys, group in convergence_logs.groupby(
        ["federation_size", "seed", "client_id"],
        sort=True,
        dropna=False,
    ):
        clients, seed, client_id = keys
        problems = []

        for column in required:
            if column not in group.columns:
                problems.append(
                    f"{column}=missing"
                )
                continue

            valid = pd.to_numeric(
                group[column],
                errors="coerce",
            ).notna().sum()

            if valid == 0:
                problems.append(
                    f"{column}=all-NA"
                )
            elif valid < EXPECTED_ROUNDS:
                problems.append(
                    f"{column}={valid}/{EXPECTED_ROUNDS}"
                )

        if problems:
            problem_count += 1

            print(
                f"clients={clients}, seed={seed}, "
                f"client={client_id}: "
                + ", ".join(problems)
            )

    if problem_count == 0:
        print(
            "All convergence clients contain valid LOCAL, "
            "MIXED_CANDIDATE, and GLOBAL values."
        )


# ============================================================
# Convergence calculation
# ============================================================

def calculate_convergence_round(
    rounds: pd.Series,
    values: pd.Series,
) -> float:
    data = pd.DataFrame(
        {
            "round": pd.to_numeric(
                rounds,
                errors="coerce",
            ),
            "value": pd.to_numeric(
                values,
                errors="coerce",
            ),
        }
    ).dropna(
        subset=["round", "value"]
    )

    if data.empty:
        return np.nan

    data = data.sort_values(
        "round",
        kind="stable",
    )

    best_value = safe_max(
        data["value"]
    )

    if pd.isna(best_value):
        return np.nan

    target = (
        CONVERGENCE_THRESHOLD
        * best_value
    )

    lower_bound = (
        CONVERGENCE_THRESHOLD
        - CONVERGENCE_TOLERANCE
    ) * best_value

    for _, row in data.iterrows():
        current_round = float(
            row["round"]
        )

        current_value = float(
            row["value"]
        )

        if current_value < target:
            continue

        remaining = data.loc[
            data["round"] >= current_round,
            "value",
        ]

        if (
            not remaining.empty
            and (remaining >= lower_bound).all()
        ):
            return current_round

    best_rows = data[
        data["value"] == best_value
    ]

    if best_rows.empty:
        return np.nan

    return float(
        best_rows.iloc[0]["round"]
    )


def get_best_value_and_round(
    group: pd.DataFrame,
    metric_column: str,
) -> tuple[float, float]:
    if metric_column not in group.columns:
        return np.nan, np.nan

    values = pd.to_numeric(
        group[metric_column],
        errors="coerce",
    )

    valid = values.notna()

    if not valid.any():
        return np.nan, np.nan

    valid_group = group.loc[
        valid
    ].copy()

    valid_group[metric_column] = pd.to_numeric(
        valid_group[metric_column],
        errors="coerce",
    )

    best_index = valid_group[
        metric_column
    ].idxmax()

    return (
        float(
            valid_group.loc[
                best_index,
                metric_column,
            ]
        ),
        float(
            valid_group.loc[
                best_index,
                "round",
            ]
        ),
    )


def get_final_valid_value(
    group: pd.DataFrame,
    metric_column: str,
) -> float:
    if metric_column not in group.columns:
        return np.nan

    values = pd.to_numeric(
        group[metric_column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.iloc[-1])


# ============================================================
# System summary per client
# ============================================================

def summarize_system_per_client(
    scalability_logs: pd.DataFrame,
    communication_logs: pd.DataFrame,
) -> pd.DataFrame:
    primary = (
        scalability_logs.copy()
        if not scalability_logs.empty
        else communication_logs.copy()
    )

    if primary.empty:
        return pd.DataFrame()

    rows = []

    for keys, group in primary.groupby(
        ["federation_size", "seed", "client_id"],
        sort=True,
        dropna=False,
    ):
        clients, seed, client_id = keys

        communication_group = pd.DataFrame()

        if not communication_logs.empty:
            communication_group = communication_logs[
                (
                    communication_logs["federation_size"]
                    == clients
                )
                & (
                    communication_logs["seed"]
                    == seed
                )
                & (
                    communication_logs["client_id"]
                    == client_id
                )
            ]

        if communication_group.empty:
            communication_group = group

        round_times = safe_numeric(
            group,
            "round_total_sec",
        )

        rows.append(
            {
                "clients": int(clients),
                "seed": int(seed),
                "client_id": str(client_id),

                "rounds":
                    int(group["round"].nunique()),

                "train_samples":
                    safe_max(
                        safe_numeric(
                            group,
                            "train_samples",
                        )
                    ),

                "validation_samples":
                    safe_max(
                        safe_numeric(
                            group,
                            "validation_samples",
                        )
                    ),

                "test_samples":
                    safe_max(
                        safe_numeric(
                            group,
                            "test_samples",
                        )
                    ),

                "bytes_uploaded_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "bytes_uploaded",
                        )
                    ),

                "final_download_bytes_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "final_model_bytes_downloaded",
                        )
                    ),

                "shared_exchange_bytes_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "shared_exchange_bytes",
                        )
                    ),

                "application_bytes_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "total_application_bytes",
                        )
                    ),

                "application_bytes_total":
                    safe_sum(
                        safe_numeric(
                            communication_group,
                            "total_application_bytes",
                        )
                    ),

                "poll_bytes_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "total_poll_bytes_downloaded",
                        )
                    ),

                "poll_count_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "poll_count",
                        )
                    ),

                "upload_rpc_sec_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "upload_rpc_sec",
                        )
                    ),

                "poll_rpc_sec_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "poll_rpc_sec",
                        )
                    ),

                "rpc_time_sec_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "rpc_time_sec",
                        )
                    ),

                "barrier_wait_sec_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "barrier_wait_sec",
                        )
                    ),

                "poll_sleep_sec_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "poll_sleep_sec",
                        )
                    ),

                "serialization_sec_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "serialization_sec",
                        )
                    ),

                "deserialization_sec_mean":
                    safe_mean(
                        safe_numeric(
                            communication_group,
                            "deserialization_sec",
                        )
                    ),

                "local_training_sec_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "local_training_sec",
                        )
                    ),

                "local_validation_sec_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "local_validation_eval_sec",
                        )
                    ),

                "mixed_candidate_eval_sec_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "mixed_candidate_eval_sec",
                        )
                    ),

                "global_validation_sec_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "global_validation_eval_sec",
                        )
                    ),

                "safety_gate_sec_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "safety_gate_total_sec",
                        )
                    ),

                "non_training_overhead_sec_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "non_training_overhead_sec",
                        )
                    ),

                "non_training_overhead_percent_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "non_training_overhead_percent",
                        )
                    ),

                "round_time_sec_mean":
                    safe_mean(round_times),

                "round_time_sec_std":
                    safe_std(round_times),

                "client_total_time_sec":
                    safe_sum(round_times),

                "barrier_timeout_count":
                    int(
                        safe_numeric(
                            communication_group,
                            "barrier_timeout",
                        )
                        .fillna(0)
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Convergence summary per client
# ============================================================

def summarize_convergence_per_client(
    convergence_logs: pd.DataFrame,
) -> pd.DataFrame:
    if convergence_logs.empty:
        return pd.DataFrame()

    rows = []

    for keys, group in convergence_logs.groupby(
        ["federation_size", "seed", "client_id"],
        sort=True,
        dropna=False,
    ):
        clients, seed, client_id = keys

        group = group.copy()

        group["round"] = pd.to_numeric(
            group["round"],
            errors="coerce",
        )

        group = group.sort_values(
            "round",
            kind="stable",
        ).reset_index(drop=True)

        for column in [
            "local_macro_f1",
            "mixed_candidate_macro_f1",
            "post_aggregation_global_macro_f1",
            "switch_on",
            "gamma_global",
            "elapsed_experiment_sec",
            "candidate_minus_local",
            "global_minus_local",
        ]:
            if column in group.columns:
                group[column] = pd.to_numeric(
                    group[column],
                    errors="coerce",
                )

        best_local, best_local_round = (
            get_best_value_and_round(
                group,
                "local_macro_f1",
            )
        )

        best_candidate, best_candidate_round = (
            get_best_value_and_round(
                group,
                "mixed_candidate_macro_f1",
            )
        )

        best_global, best_global_round = (
            get_best_value_and_round(
                group,
                "post_aggregation_global_macro_f1",
            )
        )

        local_series = safe_numeric(
            group,
            "local_macro_f1",
        )

        candidate_series = safe_numeric(
            group,
            "mixed_candidate_macro_f1",
        )

        global_series = safe_numeric(
            group,
            "post_aggregation_global_macro_f1",
        )

        gamma_series = safe_numeric(
            group,
            "gamma_global",
        )

        switch_series = safe_numeric(
            group,
            "switch_on",
        ).dropna()

        rows.append(
            {
                "clients": int(clients),
                "seed": int(seed),
                "client_id": str(client_id),

                "rounds":
                    int(group["round"].nunique()),

                "valid_local_count":
                    int(local_series.notna().sum()),

                "valid_candidate_count":
                    int(candidate_series.notna().sum()),

                "valid_global_count":
                    int(global_series.notna().sum()),

                "local_convergence_round":
                    calculate_convergence_round(
                        group["round"],
                        local_series,
                    ),

                "candidate_convergence_round":
                    calculate_convergence_round(
                        group["round"],
                        candidate_series,
                    ),

                "global_convergence_round":
                    calculate_convergence_round(
                        group["round"],
                        global_series,
                    ),

                "best_local_macro_f1":
                    best_local,

                "best_local_round":
                    best_local_round,

                "best_candidate_macro_f1":
                    best_candidate,

                "best_candidate_round":
                    best_candidate_round,

                "best_global_macro_f1":
                    best_global,

                "best_global_round":
                    best_global_round,

                "final_local_macro_f1":
                    get_final_valid_value(
                        group,
                        "local_macro_f1",
                    ),

                "final_candidate_macro_f1":
                    get_final_valid_value(
                        group,
                        "mixed_candidate_macro_f1",
                    ),

                "final_global_macro_f1":
                    get_final_valid_value(
                        group,
                        "post_aggregation_global_macro_f1",
                    ),

                "gamma_global_mean":
                    safe_mean(gamma_series),

                "gamma_global_std":
                    safe_std(gamma_series),

                "gamma_global_min":
                    safe_min(gamma_series),

                "gamma_global_max":
                    safe_max(gamma_series),

                "gamma_global_final":
                    get_final_valid_value(
                        group,
                        "gamma_global",
                    ),

                "switch_on_count":
                    int(switch_series.sum())
                    if not switch_series.empty
                    else 0,

                "switch_on_rate":
                    float(switch_series.mean())
                    if not switch_series.empty
                    else np.nan,

                "candidate_improvement_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "candidate_minus_local",
                        )
                    ),

                "global_improvement_mean":
                    safe_mean(
                        safe_numeric(
                            group,
                            "global_minus_local",
                        )
                    ),

                "elapsed_experiment_sec":
                    get_final_valid_value(
                        group,
                        "elapsed_experiment_sec",
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Partition summary
# ============================================================

def summarize_partition_per_client(
    partition_logs: pd.DataFrame,
) -> pd.DataFrame:
    if partition_logs.empty:
        return pd.DataFrame()

    rows = []

    for keys, group in partition_logs.groupby(
        ["federation_size", "seed", "client_id"],
        sort=True,
        dropna=False,
    ):
        clients, seed, client_id = keys

        # Defensive check. After load_partition_logs() there must be at most
        # one row per class for this logical client. If an old/malformed log
        # still contains duplicate class rows, keep only the latest row rather
        # than summing repeated partition executions.
        if "class_id" in group.columns:
            if "timestamp" in group.columns:
                group = group.sort_values(
                    "timestamp",
                    kind="stable",
                )

            group = group.drop_duplicates(
                subset=["class_id"],
                keep="last",
            )

        sample_counts = safe_numeric(
            group,
            "sample_count",
        )

        positive_classes = int(
            (sample_counts > 0).sum()
        )

        total_samples = safe_sum(
            sample_counts
        )

        proportions = (
            sample_counts / total_samples
            if (
                not pd.isna(total_samples)
                and total_samples > 0
            )
            else pd.Series(
                np.nan,
                index=group.index,
            )
        )

        entropy = np.nan

        valid_proportions = proportions[
            proportions > 0
        ].dropna()

        if not valid_proportions.empty:
            entropy = float(
                -np.sum(
                    valid_proportions
                    * np.log(valid_proportions)
                )
            )

        rows.append(
            {
                "clients": int(clients),
                "seed": int(seed),
                "client_id": str(client_id),

                "base_client_id":
                    group[
                        "base_client_id"
                    ].iloc[0]
                    if "base_client_id"
                    in group.columns
                    else "",

                "partition_method":
                    group[
                        "partition_method"
                    ].iloc[0]
                    if "partition_method"
                    in group.columns
                    else "",

                "noniid_alpha":
                    safe_mean(
                        safe_numeric(
                            group,
                            "noniid_alpha",
                        )
                    ),

                "total_partition_samples":
                    total_samples,

                "number_of_classes":
                    int(
                        group["class_id"].nunique()
                    )
                    if "class_id"
                    in group.columns
                    else len(group),

                "positive_classes":
                    positive_classes,

                "minimum_class_samples":
                    safe_min(sample_counts),

                "maximum_class_samples":
                    safe_max(sample_counts),

                "mean_class_samples":
                    safe_mean(sample_counts),

                "class_distribution_entropy":
                    entropy,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Per-seed summary
# ============================================================

def summarize_system_per_seed(
    system_per_client: pd.DataFrame,
    convergence_per_client: pd.DataFrame,
) -> pd.DataFrame:
    if system_per_client.empty:
        return pd.DataFrame()

    rows = []

    for keys, group in system_per_client.groupby(
        ["clients", "seed"],
        sort=True,
    ):
        clients, seed = keys

        convergence_group = pd.DataFrame()

        if not convergence_per_client.empty:
            convergence_group = (
                convergence_per_client[
                    (
                        convergence_per_client["clients"]
                        == clients
                    )
                    & (
                        convergence_per_client["seed"]
                        == seed
                    )
                ]
            )

        elapsed_time = (
            safe_max(
                convergence_group[
                    "elapsed_experiment_sec"
                ]
            )
            if not convergence_group.empty
            else np.nan
        )

        if pd.isna(elapsed_time):
            elapsed_time = safe_max(
                group["client_total_time_sec"]
            )

        rows.append(
            {
                "clients": int(clients),
                "seed": int(seed),

                "observed_clients":
                    int(group["client_id"].nunique()),

                "shared_exchange_bytes_per_client_round":
                    safe_mean(
                        group[
                            "shared_exchange_bytes_mean"
                        ]
                    ),

                "application_bytes_per_client_round":
                    safe_mean(
                        group[
                            "application_bytes_mean"
                        ]
                    ),

                "total_application_bytes":
                    safe_sum(
                        group[
                            "application_bytes_total"
                        ]
                    ),

                "rpc_time_sec":
                    safe_mean(
                        group["rpc_time_sec_mean"]
                    ),

                "sync_time_sec":
                    safe_mean(
                        group[
                            "barrier_wait_sec_mean"
                        ]
                    ),

                "poll_sleep_sec":
                    safe_mean(
                        group[
                            "poll_sleep_sec_mean"
                        ]
                    ),

                "local_training_sec":
                    safe_mean(
                        group[
                            "local_training_sec_mean"
                        ]
                    ),

                "safety_gate_sec":
                    safe_mean(
                        group[
                            "safety_gate_sec_mean"
                        ]
                    ),

                "non_training_overhead_sec":
                    safe_mean(
                        group[
                            "non_training_overhead_sec_mean"
                        ]
                    ),

                "non_training_overhead_percent":
                    safe_mean(
                        group[
                            "non_training_overhead_percent_mean"
                        ]
                    ),

                "round_time_sec":
                    safe_mean(
                        group[
                            "round_time_sec_mean"
                        ]
                    ),

                "total_time_sec":
                    elapsed_time,

                "local_convergence_round":
                    safe_mean(
                        convergence_group[
                            "local_convergence_round"
                        ]
                    )
                    if not convergence_group.empty
                    else np.nan,

                "global_convergence_round":
                    safe_mean(
                        convergence_group[
                            "global_convergence_round"
                        ]
                    )
                    if not convergence_group.empty
                    else np.nan,

                "final_local_macro_f1":
                    safe_mean(
                        convergence_group[
                            "final_local_macro_f1"
                        ]
                    )
                    if not convergence_group.empty
                    else np.nan,

                "final_global_macro_f1":
                    safe_mean(
                        convergence_group[
                            "final_global_macro_f1"
                        ]
                    )
                    if not convergence_group.empty
                    else np.nan,

                "final_gamma_global":
                    safe_mean(
                        convergence_group[
                            "gamma_global_final"
                        ]
                    )
                    if not convergence_group.empty
                    else np.nan,

                "global_acceptance_rate":
                    safe_mean(
                        convergence_group[
                            "switch_on_rate"
                        ]
                    )
                    if not convergence_group.empty
                    else np.nan,

                "barrier_timeout_count":
                    int(
                        group[
                            "barrier_timeout_count"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Final federation-size summaries
# ============================================================

def summarize_system_across_seeds(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    if per_seed.empty:
        return pd.DataFrame()

    rows = []

    for clients, group in per_seed.groupby(
        "clients",
        sort=True,
    ):
        rows.append(
            {
                "clients": int(clients),

                "seeds":
                    int(group["seed"].nunique()),

                "observed_clients":
                    int(
                        round(
                            safe_mean(
                                group[
                                    "observed_clients"
                                ]
                            )
                        )
                    ),

                "shared_exchange_bytes_per_client_round":
                    safe_mean(
                        group[
                            "shared_exchange_bytes_per_client_round"
                        ]
                    ),

                "application_bytes_per_client_round":
                    safe_mean(
                        group[
                            "application_bytes_per_client_round"
                        ]
                    ),

                "rpc_time_sec":
                    safe_mean(
                        group["rpc_time_sec"]
                    ),

                "sync_time_sec":
                    safe_mean(
                        group["sync_time_sec"]
                    ),

                "local_training_sec":
                    safe_mean(
                        group["local_training_sec"]
                    ),

                "safety_gate_sec":
                    safe_mean(
                        group["safety_gate_sec"]
                    ),

                "non_training_overhead_sec":
                    safe_mean(
                        group[
                            "non_training_overhead_sec"
                        ]
                    ),

                "non_training_overhead_percent":
                    safe_mean(
                        group[
                            "non_training_overhead_percent"
                        ]
                    ),

                "round_time_sec":
                    safe_mean(
                        group["round_time_sec"]
                    ),

                "total_time_sec":
                    safe_mean(
                        group["total_time_sec"]
                    ),

                "local_convergence_round":
                    safe_mean(
                        group[
                            "local_convergence_round"
                        ]
                    ),

                "global_convergence_round":
                    safe_mean(
                        group[
                            "global_convergence_round"
                        ]
                    ),

                "final_local_macro_f1":
                    safe_mean(
                        group[
                            "final_local_macro_f1"
                        ]
                    ),

                "final_global_macro_f1":
                    safe_mean(
                        group[
                            "final_global_macro_f1"
                        ]
                    ),

                "final_gamma_global":
                    safe_mean(
                        group[
                            "final_gamma_global"
                        ]
                    ),

                "global_acceptance_rate":
                    safe_mean(
                        group[
                            "global_acceptance_rate"
                        ]
                    ),

                "total_application_bytes":
                    safe_mean(
                        group[
                            "total_application_bytes"
                        ]
                    ),

                "barrier_timeout_count":
                    int(
                        group[
                            "barrier_timeout_count"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "clients"
    ).reset_index(drop=True)


def summarize_convergence_across_clients(
    convergence_per_client: pd.DataFrame,
) -> pd.DataFrame:
    if convergence_per_client.empty:
        return pd.DataFrame()

    rows = []

    for clients, group in convergence_per_client.groupby(
        "clients",
        sort=True,
    ):
        rows.append(
            {
                "clients": int(clients),

                "seeds":
                    int(group["seed"].nunique()),

                "client_seed_rows":
                    len(group),

                "local_convergence_round_mean":
                    safe_mean(
                        group[
                            "local_convergence_round"
                        ]
                    ),

                "local_convergence_round_std":
                    safe_std(
                        group[
                            "local_convergence_round"
                        ]
                    ),

                "global_convergence_round_mean":
                    safe_mean(
                        group[
                            "global_convergence_round"
                        ]
                    ),

                "global_convergence_round_std":
                    safe_std(
                        group[
                            "global_convergence_round"
                        ]
                    ),

                "best_local_macro_f1_mean":
                    safe_mean(
                        group[
                            "best_local_macro_f1"
                        ]
                    ),

                "final_local_macro_f1_mean":
                    safe_mean(
                        group[
                            "final_local_macro_f1"
                        ]
                    ),

                "final_local_macro_f1_std":
                    safe_std(
                        group[
                            "final_local_macro_f1"
                        ]
                    ),

                "final_global_macro_f1_mean":
                    safe_mean(
                        group[
                            "final_global_macro_f1"
                        ]
                    ),

                "gamma_global_final_mean":
                    safe_mean(
                        group[
                            "gamma_global_final"
                        ]
                    ),

                "gamma_global_final_std":
                    safe_std(
                        group[
                            "gamma_global_final"
                        ]
                    ),

                "global_acceptance_rate_mean":
                    safe_mean(
                        group[
                            "switch_on_rate"
                        ]
                    ),

                "global_acceptance_rate_std":
                    safe_std(
                        group[
                            "switch_on_rate"
                        ]
                    ),

                "elapsed_experiment_sec_max":
                    safe_max(
                        group[
                            "elapsed_experiment_sec"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "clients"
    ).reset_index(drop=True)


def summarize_partitions(
    partition_per_client: pd.DataFrame,
) -> pd.DataFrame:
    if partition_per_client.empty:
        return pd.DataFrame()

    rows = []

    for clients, group in partition_per_client.groupby(
        "clients",
        sort=True,
    ):
        rows.append(
            {
                "clients": int(clients),

                "observed_clients":
                    int(group["client_id"].nunique()),

                "noniid_alpha":
                    safe_mean(
                        group["noniid_alpha"]
                    ),

                "mean_partition_samples":
                    safe_mean(
                        group[
                            "total_partition_samples"
                        ]
                    ),

                "std_partition_samples":
                    safe_std(
                        group[
                            "total_partition_samples"
                        ]
                    ),

                "mean_positive_classes":
                    safe_mean(
                        group[
                            "positive_classes"
                        ]
                    ),

                "mean_minimum_class_samples":
                    safe_mean(
                        group[
                            "minimum_class_samples"
                        ]
                    ),

                "mean_maximum_class_samples":
                    safe_mean(
                        group[
                            "maximum_class_samples"
                        ]
                    ),

                "mean_class_entropy":
                    safe_mean(
                        group[
                            "class_distribution_entropy"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "clients"
    ).reset_index(drop=True)


# ============================================================
# LaTeX tables
# ============================================================

def write_system_latex(
    summary: pd.DataFrame,
) -> None:
    output_path = (
        OUTPUT_DIR
        / "system_scalability_table.tex"
    )

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\small",
        r"\caption{System-level scalability of adaptive PFTL with increasing numbers of participating clients. Shared exchange represents the direct shared-layer upload and download, while application traffic includes synchronization polling. RPC time is the combined upload and polling RPC time. Results are averaged across clients, rounds, and available seeds.}",
        r"\label{tab:system_scalability}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"\textbf{Clients}",
        r"& \textbf{Shared Exchange}",
        r"& \textbf{App.\ Traffic}",
        r"& \textbf{RPC}",
        r"& \textbf{Sync.}",
        r"& \textbf{Training}",
        r"& \textbf{Round}",
        r"& \textbf{Total} \\",
        r"",
        r"& \textbf{(B/client/round)}",
        r"& \textbf{(B/client/round)}",
        r"& \textbf{(s)}",
        r"& \textbf{(s)}",
        r"& \textbf{(s)}",
        r"& \textbf{(s)}",
        r"& \textbf{(s)} \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{int(row['clients'])}"
            f" & {format_integer(row['shared_exchange_bytes_per_client_round'])}"
            f" & {format_integer(row['application_bytes_per_client_round'])}"
            f" & {format_float(row['rpc_time_sec'])}"
            f" & {format_float(row['sync_time_sec'])}"
            f" & {format_float(row['local_training_sec'])}"
            f" & {format_float(row['round_time_sec'])}"
            f" & {format_float(row['total_time_sec'])}"
            r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_convergence_latex(
    summary: pd.DataFrame,
) -> None:
    output_path = (
        OUTPUT_DIR
        / "convergence_scalability_table.tex"
    )

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\small",
        r"\caption{Convergence and adaptive-personalization behavior of PFTL with increasing federation sizes. Convergence denotes the first round reaching at least 95\% of the best Macro-F1 while remaining within a 2\% tolerance thereafter. Global acceptance is the percentage of rounds in which the mixed local--global candidate was selected.}",
        r"\label{tab:convergence_scalability}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"\textbf{Clients}",
        r"& \textbf{Local Conv.}",
        r"& \textbf{Global Conv.}",
        r"& \textbf{Final Local F1}",
        r"& \textbf{Final Global F1}",
        r"& \textbf{Final $\gamma_g$}",
        r"& \textbf{Global Accept.} \\",
        r"",
        r"& \textbf{(round)}",
        r"& \textbf{(round)}",
        r"&",
        r"&",
        r"&",
        r"& \textbf{(\%)} \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        acceptance = (
            row["global_acceptance_rate_mean"]
            * 100.0
            if not pd.isna(
                row[
                    "global_acceptance_rate_mean"
                ]
            )
            else np.nan
        )

        lines.append(
            f"{int(row['clients'])}"
            f" & {format_float(row['local_convergence_round_mean'], 1)}"
            f" & {format_float(row['global_convergence_round_mean'], 1)}"
            f" & {format_float(row['final_local_macro_f1_mean'], 4)}"
            f" & {format_float(row['final_global_macro_f1_mean'], 4)}"
            f" & {format_float(row['gamma_global_final_mean'], 3)}"
            f" & {format_float(acceptance, 1)}"
            r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_combined_latex(
    summary: pd.DataFrame,
) -> None:
    output_path = (
        OUTPUT_DIR
        / "combined_scalability_table.tex"
    )

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\small",
        r"\caption{Combined system and convergence scalability of adaptive PFTL. Communication denotes mean application-layer traffic per client per round.}",
        r"\label{tab:combined_scalability}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"\textbf{Clients}",
        r"& \textbf{Comm.}",
        r"& \textbf{RPC}",
        r"& \textbf{Sync.}",
        r"& \textbf{Round}",
        r"& \textbf{Total}",
        r"& \textbf{Conv.}",
        r"& \textbf{Final F1}",
        r"& \textbf{Accept.} \\",
        r"",
        r"& \textbf{(B/client/round)}",
        r"& \textbf{(s)}",
        r"& \textbf{(s)}",
        r"& \textbf{(s)}",
        r"& \textbf{(s)}",
        r"& \textbf{(round)}",
        r"&",
        r"& \textbf{(\%)} \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        acceptance = (
            row["global_acceptance_rate"]
            * 100.0
            if not pd.isna(
                row["global_acceptance_rate"]
            )
            else np.nan
        )

        lines.append(
            f"{int(row['clients'])}"
            f" & {format_integer(row['application_bytes_per_client_round'])}"
            f" & {format_float(row['rpc_time_sec'])}"
            f" & {format_float(row['sync_time_sec'])}"
            f" & {format_float(row['round_time_sec'])}"
            f" & {format_float(row['total_time_sec'])}"
            f" & {format_float(row['local_convergence_round'], 1)}"
            f" & {format_float(row['final_local_macro_f1'], 4)}"
            f" & {format_float(acceptance, 1)}"
            r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 90)
    print("PFTL MULTI-CLASS SCALABILITY ANALYSIS")
    print("=" * 90)
    print(f"Root directory  : {ROOT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Expected rounds : {EXPECTED_ROUNDS}")
    print(f"Client counts   : {CLIENT_COUNTS}")
    print()

    all_communication = []
    all_scalability = []
    all_convergence = []
    all_stage_metrics = []
    all_partitions = []

    processed_experiments = []

    for expected_clients in CLIENT_COUNTS:
        experiment_directories = (
            find_experiment_directories(
                expected_clients
            )
        )

        if not experiment_directories:
            print(
                f"WARNING: No directory found for "
                f"{expected_clients} clients.",
                file=sys.stderr,
            )
            continue

        for experiment_dir in experiment_directories:
            directory_seed = (
                extract_seed_from_directory_name(
                    experiment_dir.name
                )
            )

            print("-" * 90)
            print(
                f"Federation size : {expected_clients}"
            )
            print(
                f"Seed            : {directory_seed}"
            )
            print(
                f"Directory       : {experiment_dir}"
            )

            communication = load_communication_logs(
                experiment_dir,
                expected_clients,
                directory_seed,
            )

            scalability = load_scalability_logs(
                experiment_dir,
                expected_clients,
                directory_seed,
            )

            convergence = load_convergence_logs(
                experiment_dir,
                expected_clients,
                directory_seed,
            )

            stage_metrics = load_stage_metrics_logs(
                experiment_dir,
                expected_clients,
                directory_seed,
            )

            partitions = load_partition_logs(
                experiment_dir,
                expected_clients,
                directory_seed,
            )

            validate_client_coverage(
                scalability,
                expected_clients,
                "Scalability",
            )

            validate_client_coverage(
                communication,
                expected_clients,
                "Communication",
            )

            validate_client_coverage(
                convergence,
                expected_clients,
                "Convergence",
            )

            validate_client_coverage(
                stage_metrics,
                expected_clients,
                "Stage metrics",
                rows_per_round=3,
            )

            if not communication.empty:
                all_communication.append(
                    communication
                )

            if not scalability.empty:
                all_scalability.append(
                    scalability
                )

            if not convergence.empty:
                all_convergence.append(
                    convergence
                )

            if not stage_metrics.empty:
                all_stage_metrics.append(
                    stage_metrics
                )

            if not partitions.empty:
                all_partitions.append(
                    partitions
                )

            processed_experiments.append(
                {
                    "clients": expected_clients,
                    "seed": directory_seed,
                    "directory": str(experiment_dir),

                    "communication_rows":
                        len(communication),

                    "scalability_rows":
                        len(scalability),

                    "convergence_rows":
                        len(convergence),

                    "stage_metrics_rows":
                        len(stage_metrics),

                    "partition_rows":
                        len(partitions),

                    "communication_clients":
                        communication[
                            "client_id"
                        ].nunique()
                        if not communication.empty
                        else 0,

                    "scalability_clients":
                        scalability[
                            "client_id"
                        ].nunique()
                        if not scalability.empty
                        else 0,

                    "convergence_clients":
                        convergence[
                            "client_id"
                        ].nunique()
                        if not convergence.empty
                        else 0,

                    "metrics_clients":
                        stage_metrics[
                            "client_id"
                        ].nunique()
                        if not stage_metrics.empty
                        else 0,
                }
            )

    communication_logs = (
        pd.concat(
            all_communication,
            ignore_index=True,
        )
        if all_communication
        else pd.DataFrame()
    )

    scalability_logs = (
        pd.concat(
            all_scalability,
            ignore_index=True,
        )
        if all_scalability
        else pd.DataFrame()
    )

    convergence_logs = (
        pd.concat(
            all_convergence,
            ignore_index=True,
        )
        if all_convergence
        else pd.DataFrame()
    )

    stage_metrics_logs = (
        pd.concat(
            all_stage_metrics,
            ignore_index=True,
        )
        if all_stage_metrics
        else pd.DataFrame()
    )

    partition_logs = (
        pd.concat(
            all_partitions,
            ignore_index=True,
        )
        if all_partitions
        else pd.DataFrame()
    )

    if (
        communication_logs.empty
        and scalability_logs.empty
    ):
        raise RuntimeError(
            "No valid communication or scalability logs found."
        )

    # Repair missing convergence values using metrics.csv.
    convergence_logs = repair_convergence_logs(
        convergence_logs,
        stage_metrics_logs,
    )

    diagnose_convergence_logs(
        convergence_logs
    )

    # Generate summaries.
    system_per_client = summarize_system_per_client(
        scalability_logs,
        communication_logs,
    )

    convergence_per_client = (
        summarize_convergence_per_client(
            convergence_logs
        )
    )

    partition_per_client = (
        summarize_partition_per_client(
            partition_logs
        )
    )

    system_per_seed = summarize_system_per_seed(
        system_per_client,
        convergence_per_client,
    )

    system_summary = (
        summarize_system_across_seeds(
            system_per_seed
        )
    )

    convergence_summary = (
        summarize_convergence_across_clients(
            convergence_per_client
        )
    )

    partition_summary = summarize_partitions(
        partition_per_client
    )

    # Save processed directory information.
    pd.DataFrame(
        processed_experiments
    ).to_csv(
        OUTPUT_DIR
        / "processed_experiment_directories.csv",
        index=False,
    )

    # Save cleaned logs.
    if not communication_logs.empty:
        communication_logs.to_csv(
            OUTPUT_DIR
            / "cleaned_communication_logs.csv",
            index=False,
        )

    if not scalability_logs.empty:
        scalability_logs.to_csv(
            OUTPUT_DIR
            / "cleaned_scalability_logs.csv",
            index=False,
        )

    if not convergence_logs.empty:
        convergence_logs.to_csv(
            OUTPUT_DIR
            / "cleaned_repaired_convergence_logs.csv",
            index=False,
        )

    if not stage_metrics_logs.empty:
        stage_metrics_logs.to_csv(
            OUTPUT_DIR
            / "cleaned_stage_metrics_logs.csv",
            index=False,
        )

    if not partition_logs.empty:
        partition_logs.to_csv(
            OUTPUT_DIR
            / "cleaned_partition_logs.csv",
            index=False,
        )

    # Save summaries.
    system_per_client.to_csv(
        OUTPUT_DIR
        / "system_scalability_per_client.csv",
        index=False,
    )

    system_per_seed.to_csv(
        OUTPUT_DIR
        / "system_scalability_per_seed.csv",
        index=False,
    )

    system_summary.to_csv(
        OUTPUT_DIR
        / "system_scalability_summary.csv",
        index=False,
    )

    convergence_per_client.to_csv(
        OUTPUT_DIR
        / "convergence_scalability_per_client.csv",
        index=False,
    )

    convergence_summary.to_csv(
        OUTPUT_DIR
        / "convergence_scalability_summary.csv",
        index=False,
    )

    partition_per_client.to_csv(
        OUTPUT_DIR
        / "partition_scalability_per_client.csv",
        index=False,
    )

    partition_summary.to_csv(
        OUTPUT_DIR
        / "partition_scalability_summary.csv",
        index=False,
    )

    # Generate LaTeX.
    if not system_summary.empty:
        write_system_latex(
            system_summary
        )

        write_combined_latex(
            system_summary
        )

    if not convergence_summary.empty:
        write_convergence_latex(
            convergence_summary
        )

    # Print results.
    print()
    print("=" * 90)
    print("SYSTEM SCALABILITY SUMMARY")
    print("=" * 90)

    if not system_summary.empty:
        display_columns = [
            "clients",
            "seeds",
            "observed_clients",
            "shared_exchange_bytes_per_client_round",
            "application_bytes_per_client_round",
            "rpc_time_sec",
            "sync_time_sec",
            "local_training_sec",
            "round_time_sec",
            "total_time_sec",
            "local_convergence_round",
            "final_local_macro_f1",
            "global_acceptance_rate",
        ]

        print(
            system_summary[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    print()
    print("=" * 90)
    print("CONVERGENCE SCALABILITY SUMMARY")
    print("=" * 90)

    if not convergence_summary.empty:
        print(
            convergence_summary.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    print()
    print("=" * 90)
    print("PARTITION SUMMARY")
    print("=" * 90)

    if not partition_summary.empty:
        print(
            partition_summary.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    for output_file in sorted(
        OUTPUT_DIR.iterdir()
    ):
        if output_file.is_file():
            print(output_file)

    print()
    print(
        "PFTL scalability analysis completed successfully."
    )


if __name__ == "__main__":
    main()