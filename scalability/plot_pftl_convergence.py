#!/usr/bin/env python3.10
# ============================================================
# Plot PFTL convergence for 6, 12, 30, and 60 clients
# ============================================================
#
# Reads all:
#   *_convergence_curve.csv
#
# from:
#   clients_6_fullsize_noniid_alpha_0.3_seed_123
#   clients_12_fullsize_noniid_alpha_0.3_seed_123
#
# Generates:
#   convergence_6_clients.png
#   convergence_12_clients.png
#   convergence_all_client_scales.png
#   convergence_summary.csv
#
# The main curve is the mean post-aggregation GLOBAL Macro-F1
# across all participating clients at each communication round.
# The shaded region represents ±1 standard deviation across clients.
# ============================================================

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/scalability_logs/"
)

OUTPUT_DIR = Path(
    "/nfs/aalqahtani/PFTL_Multi_class/"
    "M_PFTL_codes/scalability_analysis_results/"
    "convergence_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENTS = {
    6: ROOT_DIR / (
        "clients_6_fullsize_noniid_alpha_0.3_seed_123"
    ),
    12: ROOT_DIR / (
        "clients_12_fullsize_noniid_alpha_0.3_seed_123"
    ),
    30: ROOT_DIR / (
        "clients_30_fullsize_noniid_alpha_0.3_seed_123"
    ),
    60: ROOT_DIR / (
        "clients_60_fullsize_noniid_alpha_0.3_seed_123"
    ),
    90: ROOT_DIR / (
        "clients_90_fullsize_noniid_alpha_0.3_seed_123"
    ),
}


# ============================================================
# Helpers
# ============================================================

def infer_client_id(path: Path) -> str:
    match = re.search(
        r"(client\d+_[A-Za-z0-9_]+_v\d+)",
        path.name,
    )

    if match:
        return match.group(1)

    return path.stem.replace(
        "_convergence_curve",
        "",
    )


def load_convergence_files(
    experiment_dir: Path,
) -> pd.DataFrame:
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}"
        )

    files = sorted(
        experiment_dir.rglob(
            "*_convergence_curve.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No convergence CSV files found in: {experiment_dir}"
        )

    frames = []

    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(
                f"[WARNING] Could not read {path}: {exc}"
            )
            continue

        frame = frame.copy()

        # Standardize Client 1's column name.
        if (
            "post_aggregation_global_macro_f1" not in frame.columns
            and "global_macro_f1" in frame.columns
        ):
            frame = frame.rename(
                columns={
                    "global_macro_f1":
                    "post_aggregation_global_macro_f1"
                }
            )

        required = {
            "seed",
            "round",
            "local_macro_f1",
            "mixed_candidate_macro_f1",
            "post_aggregation_global_macro_f1",
        }

        missing = required - set(frame.columns)

        if missing:
            print(
                f"[WARNING] Skipping {path}; missing columns: "
                f"{sorted(missing)}"
            )
            continue

        frame["client_id"] = infer_client_id(path)
        frame["source_file"] = str(path)

        numeric_columns = [
            "seed",
            "round",
            "local_macro_f1",
            "mixed_candidate_macro_f1",
            "post_aggregation_global_macro_f1",
            "gamma_global",
            "switch_on",
        ]

        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        frame = frame[frame["seed"] == 123].copy()

        if frame.empty:
            print(
                f"[WARNING] Skipping {path}; no rows for seed 123."
            )
            continue

        frame = frame.dropna(
            subset=[
                "round",
                "local_macro_f1",
                "mixed_candidate_macro_f1",
                "post_aggregation_global_macro_f1",
            ]
        )

        if frame.empty:
            print(
                f"[WARNING] Skipping {path}; no valid rows remain."
            )
            continue

        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No valid convergence files found in {experiment_dir}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return combined


def summarize_convergence(
    frame: pd.DataFrame,
    total_clients: int,
) -> pd.DataFrame:
    metrics = [
        "local_macro_f1",
        "mixed_candidate_macro_f1",
        "post_aggregation_global_macro_f1",
    ]

    aggregation = {}

    for metric in metrics:
        aggregation[metric] = [
            "mean",
            "std",
            "min",
            "max",
            "count",
        ]

    summary = (
        frame.groupby(
            "round",
            as_index=False,
        )
        .agg(aggregation)
    )

    summary.columns = [
        "_".join(
            str(item)
            for item in column
            if str(item)
        ).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    summary["total_clients"] = int(total_clients)

    return summary


def add_mean_std_curve(
    ax,
    summary: pd.DataFrame,
    mean_column: str,
    std_column: str,
    label: str,
):
    rounds = summary["round"].to_numpy()
    means = summary[mean_column].to_numpy()
    stds = summary[std_column].fillna(0.0).to_numpy()

    ax.plot(
        rounds,
        means,
        marker="o",
        linewidth=2,
        markersize=4,
        label=label,
    )

    ax.fill_between(
        rounds,
        means - stds,
        means + stds,
        alpha=0.18,
    )


# ============================================================
# Separate figure for each client scale
# ============================================================

def plot_single_experiment(
    summary: pd.DataFrame,
    total_clients: int,
    output_path: Path,
):
    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    add_mean_std_curve(
        ax,
        summary,
        "local_macro_f1_mean",
        "local_macro_f1_std",
        "Local model",
    )

    add_mean_std_curve(
        ax,
        summary,
        "mixed_candidate_macro_f1_mean",
        "mixed_candidate_macro_f1_std",
        "Mixed candidate",
    )

    add_mean_std_curve(
        ax,
        summary,
        "post_aggregation_global_macro_f1_mean",
        "post_aggregation_global_macro_f1_std",
        "Post-aggregation personalized model",
    )

    ax.set_xlabel(
        "Communication Round",
        fontsize=12,
    )

    ax.set_ylabel(
        "Mean Validation Macro-F1",
        fontsize=12,
    )

    ax.set_title(
        f"PFTL Convergence with {total_clients} Clients\n"
        r"Dirichlet non-IID, $\alpha=0.3$",
        fontsize=13,
    )

    ax.set_xticks(
        summary["round"].astype(int)
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    # Use a common valid metric range.
    ax.set_ylim(
        0.0,
        0.60,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Combined comparison across all client scales
# ============================================================

def plot_comparison(
    summaries: dict[int, pd.DataFrame],
    output_path: Path,
):
    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    for total_clients, summary in sorted(
        summaries.items()
    ):
        rounds = summary["round"].to_numpy()

        mean_values = summary[
            "post_aggregation_global_macro_f1_mean"
        ].to_numpy()

        std_values = summary[
            "post_aggregation_global_macro_f1_std"
        ].fillna(0.0).to_numpy()

        ax.plot(
            rounds,
            mean_values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=f"{total_clients} clients",
        )

        ax.fill_between(
            rounds,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
        )

    ax.set_xlabel(
        "Communication Round",
        fontsize=12,
    )

    ax.set_ylabel(
        "Mean Validation Macro-F1",
        fontsize=12,
    )

    ax.set_title(
        "PFTL Convergence Comparison Across Client Scales\n"
        r"Dirichlet non-IID, $\alpha=0.3$",
        fontsize=13,
    )

    all_rounds = sorted(
        set(
            int(value)
            for summary in summaries.values()
            for value in summary["round"]
        )
    )

    ax.set_xticks(all_rounds)

    ax.set_ylim(
        0.0,
        0.60,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="best",
        frameon=True,
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

def main():
    summaries = []
    summary_map = {}

    for total_clients, experiment_dir in sorted(
        EXPERIMENTS.items()
    ):
        print(
            f"\nLoading {total_clients}-client experiment:"
        )
        print(experiment_dir)

        frame = load_convergence_files(
            experiment_dir
        )

        observed_clients = int(
            frame["client_id"].nunique()
        )

        observed_rounds = int(
            frame["round"].nunique()
        )

        print(
            f"Observed clients: {observed_clients}"
        )
        print(
            f"Observed rounds : {observed_rounds}"
        )

        if observed_clients != total_clients:
            raise RuntimeError(
                f"Expected {total_clients} clients but found "
                f"{observed_clients}. The convergence figure "
                f"would be incomplete."
            )

        expected_rows = total_clients * observed_rounds
        if len(frame) != expected_rows:
            print(
                f"[WARNING] Expected approximately {expected_rows} "
                f"client-round rows but found {len(frame)}."
            )

        summary = summarize_convergence(
            frame,
            total_clients,
        )

        summaries.append(summary)
        summary_map[total_clients] = summary

        output_path = (
            OUTPUT_DIR
            / f"convergence_{total_clients}_clients.png"
        )

        plot_single_experiment(
            summary,
            total_clients,
            output_path,
        )

        print(
            f"Saved: {output_path}"
        )

    combined_summary = pd.concat(
        summaries,
        ignore_index=True,
        sort=False,
    )

    summary_csv = (
        OUTPUT_DIR
        / "convergence_summary.csv"
    )

    combined_summary.to_csv(
        summary_csv,
        index=False,
    )

    comparison_path = (
        OUTPUT_DIR
        / "convergence_all_client_scales.png"
    )

    plot_comparison(
        summary_map,
        comparison_path,
    )

    print(
        "\n============================================================"
    )
    print(
        "Convergence analysis completed."
    )
    print(
        f"Summary CSV: {summary_csv}"
    )
    print(
        f"Comparison figure: {comparison_path}"
    )


if __name__ == "__main__":
    main()