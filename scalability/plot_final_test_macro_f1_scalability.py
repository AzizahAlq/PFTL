#!/usr/bin/env python3.10
# ============================================================
# Plot PFTL FINAL-TEST Macro-F1 scalability
# ============================================================
#
# IMPORTANT:
# Final-test Macro-F1 is measured once after all communication
# rounds. Therefore, it must NOT be presented as a convergence
# curve over rounds.
#
# This script parses FINAL TEST EVALUATION blocks from launcher
# output files and plots:
#
#   Mean Final-Test Macro-F1 vs Total Number of Clients
#
# The error bars show ±1 population standard deviation across
# all participating clients at each federation size.
#
# Expected launcher log structure:
#
#   scalability_launcher_logs_fullsize_noniid/
#       clients_6/
#           client1_ton_iot_v1.out
#           ...
#       clients_12/
#           client1_ton_iot_v1.out
#           client1_ton_iot_v2.out
#           ...
#
# Outputs:
#   final_test_macro_f1_vs_clients.png
#   final_test_performance_summary.csv
#   final_test_client_results.csv
# ============================================================

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths and experiment sizes
# ============================================================

LAUNCHER_LOG_ROOT = Path(
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/"
    "scalability_launcher_logs_fullsize_noniid"
)

OUTPUT_DIR = Path(
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/scalability_analysis_results/"
    "final_test_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLIENT_SCALES = [
    6,
    12,
     30,
     60,
    # 90,
    # 120,
]


# ============================================================
# Parsing expressions
# ============================================================

FINAL_TEST_BLOCK = re.compile(
    r"\[(?P<client_id>[^\]]+)\]\s*"
    r"===== FINAL TEST EVALUATION ====="
    r"(?P<body>.*?)(?="
    r"\[[^\]]+\]\s*===== FINAL TEST EVALUATION ====="
    r"|$)",
    re.DOTALL,
)

METRIC_PATTERNS = {
    "accuracy": re.compile(
        r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    ),
    "macro_precision": re.compile(
        r"Macro-Precision\s*:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    ),
    "macro_recall": re.compile(
        r"Macro-Recall\s*:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    ),
    "macro_f1": re.compile(
        r"Macro-F1\s*:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    ),
}


# ============================================================
# Parsing helpers
# ============================================================

def parse_output_file(
    output_path: Path,
    total_clients: int,
) -> list[dict[str, object]]:
    """
    Parse FINAL TEST EVALUATION blocks from one launcher output file.
    """
    try:
        text = output_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as exc:
        print(
            f"[WARNING] Could not read {output_path}: {exc}"
        )
        return []

    records: list[dict[str, object]] = []

    for block_match in FINAL_TEST_BLOCK.finditer(text):
        client_id = block_match.group(
            "client_id"
        ).strip()

        body = block_match.group("body")

        record: dict[str, object] = {
            "total_clients": int(total_clients),
            "client_id": client_id,
            "source_file": str(output_path),
        }

        for metric_name, metric_pattern in (
            METRIC_PATTERNS.items()
        ):
            metric_match = metric_pattern.search(body)

            record[metric_name] = (
                float(metric_match.group(1))
                if metric_match
                else np.nan
            )

        if pd.notna(record["macro_f1"]):
            records.append(record)

    return records


def collect_scale_results(
    total_clients: int,
) -> pd.DataFrame:
    """
    Collect all final-test results for one federation size.
    """
    scale_dir = (
        LAUNCHER_LOG_ROOT
        / f"clients_{total_clients}"
    )

    if not scale_dir.exists():
        print(
            f"[WARNING] Missing launcher directory: "
            f"{scale_dir}"
        )
        return pd.DataFrame()

    output_files = sorted(
        list(scale_dir.rglob("*.out"))
        + list(scale_dir.rglob("*.log"))
        + list(scale_dir.rglob("*.txt"))
    )

    if not output_files:
        print(
            f"[WARNING] No launcher output files found in "
            f"{scale_dir}"
        )
        return pd.DataFrame()

    records: list[dict[str, object]] = []

    for output_file in output_files:
        records.extend(
            parse_output_file(
                output_file,
                total_clients,
            )
        )

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)

    # If the same client appears more than once, retain the last result.
    frame = frame.drop_duplicates(
        subset=[
            "total_clients",
            "client_id",
        ],
        keep="last",
    )

    return frame


# ============================================================
# Summary
# ============================================================

def summarize_final_test(
    client_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate federation-level final-test means and population
    standard deviations across participating clients.
    """
    if client_results.empty:
        return pd.DataFrame()

    summary_rows = []

    for total_clients, group in client_results.groupby(
        "total_clients"
    ):
        row = {
            "total_clients": int(total_clients),
            "observed_clients": int(
                group["client_id"].nunique()
            ),
        }

        for metric in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ):
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            row[f"mean_{metric}"] = (
                float(values.mean())
                if not values.empty
                else np.nan
            )

            # Population standard deviation across all clients.
            row[f"std_{metric}"] = (
                float(values.std(ddof=0))
                if not values.empty
                else np.nan
            )

            row[f"min_{metric}"] = (
                float(values.min())
                if not values.empty
                else np.nan
            )

            row[f"max_{metric}"] = (
                float(values.max())
                if not values.empty
                else np.nan
            )

        summary_rows.append(row)

    return (
        pd.DataFrame(summary_rows)
        .sort_values("total_clients")
        .reset_index(drop=True)
    )


# ============================================================
# Figure
# ============================================================

def plot_final_test_macro_f1(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Draw mean final-test Macro-F1 against federation size.

    Error bars represent ±1 population standard deviation across
    participating clients.
    """
    plot_frame = summary[
        [
            "total_clients",
            "mean_macro_f1",
            "std_macro_f1",
        ]
    ].dropna().sort_values(
        "total_clients"
    )

    if plot_frame.empty:
        raise RuntimeError(
            "No valid final-test Macro-F1 results were found."
        )

    x = plot_frame[
        "total_clients"
    ].to_numpy()

    y = plot_frame[
        "mean_macro_f1"
    ].to_numpy()

    error = plot_frame[
        "std_macro_f1"
    ].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(
        figsize=(7.6, 5.2)
    )

    ax.errorbar(
        x,
        y,
        yerr=error,
        marker="o",
        markersize=8,
        linewidth=2,
        capsize=6,
        capthick=1.5,
    )

    # Add mean values above the points.
    for x_value, y_value in zip(x, y):
        ax.annotate(
            f"{y_value:.4f}",
            xy=(x_value, y_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_xlabel(
        "Total Number of Clients",
        fontsize=12,
    )

    ax.set_ylabel(
        "Mean Final-Test Macro-F1",
        fontsize=12,
    )

    ax.set_title(
        "PFTL Final-Test Performance Scalability\n"
        r"Dirichlet non-IID, $\alpha=0.3$",
        fontsize=13,
    )

    ax.set_xticks(x)

    # Automatically use a useful vertical range.
    lower = max(
        0.0,
        float(
            np.min(y - error)
            - 0.04
        ),
    )

    upper = min(
        1.0,
        float(
            np.max(y + error)
            + 0.04
        ),
    )

    if upper - lower < 0.15:
        midpoint = (
            upper + lower
        ) / 2.0

        lower = max(
            0.0,
            midpoint - 0.10,
        )

        upper = min(
            1.0,
            midpoint + 0.10,
        )

    ax.set_ylim(
        lower,
        upper,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    all_results = []

    for total_clients in CLIENT_SCALES:
        print(
            f"\nCollecting final-test results for "
            f"{total_clients} clients..."
        )

        frame = collect_scale_results(
            total_clients
        )

        if frame.empty:
            print(
                f"[WARNING] No valid final-test results "
                f"for {total_clients} clients."
            )
            continue

        observed = int(
            frame["client_id"].nunique()
        )

        print(
            f"Observed final-test clients: {observed}"
        )

        if observed != total_clients:
            print(
                f"[WARNING] Expected {total_clients} clients "
                f"but parsed {observed} final-test results."
            )

        all_results.append(frame)

    if not all_results:
        raise RuntimeError(
            "No final-test results were found."
        )

    client_results = pd.concat(
        all_results,
        ignore_index=True,
        sort=False,
    )

    summary = summarize_final_test(
        client_results
    )

    client_csv = (
        OUTPUT_DIR
        / "final_test_client_results.csv"
    )

    summary_csv = (
        OUTPUT_DIR
        / "final_test_performance_summary.csv"
    )

    figure_path = (
        OUTPUT_DIR
        / "final_test_macro_f1_vs_clients.png"
    )

    client_results.to_csv(
        client_csv,
        index=False,
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    plot_final_test_macro_f1(
        summary,
        figure_path,
    )

    print(
        "\n============================================================"
    )
    print(
        "Final-test scalability analysis completed."
    )

    print(
        f"Client results: {client_csv}"
    )

    print(
        f"Summary table : {summary_csv}"
    )

    print(
        f"Figure        : {figure_path}"
    )

    preview_columns = [
        "total_clients",
        "observed_clients",
        "mean_accuracy",
        "mean_macro_precision",
        "mean_macro_recall",
        "mean_macro_f1",
        "std_macro_f1",
    ]

    print(
        "\nFinal-test summary:"
    )

    print(
        summary[
            preview_columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()