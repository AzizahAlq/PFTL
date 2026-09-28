import os
import numpy as np
import pandas as pd

# ============================================================
# CLIENT 1 — READ EXISTING LOGS ONLY
# ============================================================

BASE_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "ptfl_logs/client1_ton_iot_v2"
)

METRICS_LOG = os.path.join(
    BASE_DIR,
    "client1_metrics_log.csv"
)

FINAL_RESULTS = os.path.join(
    BASE_DIR,
    "client1_epsilon_sensitivity_final_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "client1_epsilon_sensitivity_summary.csv"
)

EPSILON_VALUES = [0.0, 0.001, 0.003, 0.01]


# ============================================================
# 1. READ LOGS
# ============================================================

metrics = pd.read_csv(METRICS_LOG)
final = pd.read_csv(FINAL_RESULTS)

print("Metrics rows:", len(metrics))
print("Final-test rows:", len(final))


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

metrics.columns = metrics.columns.str.strip()
final.columns = final.columns.str.strip()

# Support either naming convention:
# metrics log -> eps_margin
# final results -> epsilon or eps_margin

if "epsilon" in final.columns and "eps_margin" not in final.columns:
    final = final.rename(columns={"epsilon": "eps_margin"})


# ============================================================
# 3. CONVERT IMPORTANT COLUMNS TO NUMERIC
# ============================================================

for col in [
    "seed",
    "round",
    "switch_on",
    "eps_margin",
    "macro_f1",
]:
    if col in metrics.columns:
        metrics[col] = pd.to_numeric(
            metrics[col],
            errors="coerce"
        )

for col in [
    "seed",
    "eps_margin",
    "macro_f1",
    "accuracy",
]:
    if col in final.columns:
        final[col] = pd.to_numeric(
            final[col],
            errors="coerce"
        )


# ============================================================
# 4. KEEP ONLY EPSILON VALUES OF INTEREST
# ============================================================

metrics = metrics[
    metrics["eps_margin"].apply(
        lambda x: any(
            np.isclose(x, eps)
            for eps in EPSILON_VALUES
        )
    )
].copy()

final = final[
    final["eps_margin"].apply(
        lambda x: any(
            np.isclose(x, eps)
            for eps in EPSILON_VALUES
        )
    )
].copy()


# ============================================================
# 5. COUNT ONE GATE DECISION PER ROUND
# ============================================================

# Your metrics file can contain several stages for the same round.
# We must NOT count switch_on multiple times.
#
# Prefer GLOBAL stage because this represents the result after
# the local-vs-mixed/global gate decision.

if "stage" in metrics.columns:

    stages = (
        metrics["stage"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (stages == "GLOBAL").any():

        gate = metrics[
            stages == "GLOBAL"
        ].copy()

    else:
        # If GLOBAL does not exist, keep one row per
        # seed × epsilon × round.
        gate = metrics.copy()

else:
    gate = metrics.copy()


# ============================================================
# 6. REMOVE DUPLICATE RUN RECORDS
# ============================================================

gate = gate.dropna(
    subset=[
        "seed",
        "round",
        "eps_margin",
        "switch_on"
    ]
)

gate = gate.drop_duplicates(
    subset=[
        "seed",
        "eps_margin",
        "round"
    ],
    keep="last"
)


# ============================================================
# 7. CALCULATE switch_on = 1 / 0
# ============================================================

gate_summary = (
    gate
    .groupby("eps_margin")
    .agg(
        seeds=("seed", "nunique"),
        total_decisions=("switch_on", "count"),

        switch_on_1=(
            "switch_on",
            lambda x: int((x == 1).sum())
        ),

        switch_on_0=(
            "switch_on",
            lambda x: int((x == 0).sum())
        ),
    )
    .reset_index()
)

gate_summary["switch_rate_percent"] = (
    gate_summary["switch_on_1"]
    / gate_summary["total_decisions"]
    * 100
)


# ============================================================
# 8. FINAL TEST RESULTS
# ============================================================

final = final.dropna(
    subset=[
        "seed",
        "eps_margin",
        "macro_f1"
    ]
)

# If same seed/epsilon appears more than once,
# use latest logged result.
final = final.drop_duplicates(
    subset=[
        "seed",
        "eps_margin"
    ],
    keep="last"
)


performance = (
    final
    .groupby("eps_margin")
    .agg(
        final_test_seeds=(
            "seed",
            "nunique"
        ),

        macro_f1_mean=(
            "macro_f1",
            "mean"
        ),

        macro_f1_sd=(
            "macro_f1",
            "std"
        ),

        macro_f1_min=(
            "macro_f1",
            "min"
        ),

        macro_f1_max=(
            "macro_f1",
            "max"
        ),
    )
    .reset_index()
)


# Accuracy if available
if "accuracy" in final.columns:

    accuracy_summary = (
        final
        .groupby("eps_margin")
        .agg(
            accuracy_mean=(
                "accuracy",
                "mean"
            ),

            accuracy_sd=(
                "accuracy",
                "std"
            ),
        )
        .reset_index()
    )

    performance = performance.merge(
        accuracy_summary,
        on="eps_margin",
        how="left"
    )


# ============================================================
# 9. MERGE GATE + FINAL TEST RESULTS
# ============================================================

summary = gate_summary.merge(
    performance,
    on="eps_margin",
    how="outer"
)

summary = summary.sort_values(
    "eps_margin"
).reset_index(drop=True)


# ============================================================
# 10. DIFFERENCE FROM DEFAULT EPSILON = 0.003
# ============================================================

default_row = summary[
    np.isclose(
        summary["eps_margin"],
        0.003
    )
]

if not default_row.empty:

    default_f1 = (
        default_row[
            "macro_f1_mean"
        ].iloc[0]
    )

    summary[
        "delta_f1_vs_eps_0.003"
    ] = (
        summary["macro_f1_mean"]
        - default_f1
    )

else:

    summary[
        "delta_f1_vs_eps_0.003"
    ] = np.nan


# ============================================================
# 11. ROUND DISPLAY VALUES
# ============================================================

for col in [
    "switch_rate_percent",
    "macro_f1_mean",
    "macro_f1_sd",
    "macro_f1_min",
    "macro_f1_max",
    "accuracy_mean",
    "accuracy_sd",
    "delta_f1_vs_eps_0.003",
]:

    if col in summary.columns:

        summary[col] = (
            summary[col]
            .astype(float)
            .round(6)
        )


# ============================================================
# 12. SAVE
# ============================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# 13. PRINT FINAL CLIENT 1 TABLE
# ============================================================

print("\n")
print("=" * 110)
print("CLIENT 1 — EPSILON SENSITIVITY ANALYSIS")
print("=" * 110)

display_columns = [
    "eps_margin",
    "seeds",
    "total_decisions",
    "switch_on_1",
    "switch_on_0",
    "switch_rate_percent",
    "final_test_seeds",
    "macro_f1_mean",
    "macro_f1_sd",
    "delta_f1_vs_eps_0.003",
]

display_columns = [
    c for c in display_columns
    if c in summary.columns
]

print(
    summary[
        display_columns
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_SUMMARY)


# ============================================================
# 14. CHECK EXPECTED COMPLETENESS
# ============================================================

print("\n")
print("=" * 110)
print("COMPLETENESS CHECK")
print("=" * 110)

for eps in EPSILON_VALUES:

    row = summary[
        np.isclose(
            summary["eps_margin"],
            eps
        )
    ]

    if row.empty:

        print(
            f"ε={eps}: NO DATA"
        )

        continue

    n_final = row[
        "final_test_seeds"
    ].iloc[0]

    n_gate = row[
        "seeds"
    ].iloc[0]

    print(
        f"ε={eps:<6} | "
        f"Gate seeds={n_gate} | "
        f"Final-test seeds={n_final}"
    )