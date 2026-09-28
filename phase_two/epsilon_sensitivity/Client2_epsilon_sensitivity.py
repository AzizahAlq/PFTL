import os
import numpy as np
import pandas as pd

# ============================================================
# CLIENT 2 — EPSILON SENSITIVITY FROM EXISTING LOGS ONLY
# ============================================================

BASE_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "ptfl_logs/client2_v2_localfirst_strict"
)

METRICS_LOG = os.path.join(
    BASE_DIR,
    "client2_metrics_log.csv"
)

FINAL_RESULTS = os.path.join(
    BASE_DIR,
    "client2_epsilon_sensitivity_final_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "client2_epsilon_sensitivity_summary.csv"
)

EPSILON_VALUES = [0.0, 0.001, 0.003, 0.01]


# ============================================================
# READ FILES
# ============================================================

metrics = pd.read_csv(METRICS_LOG)
final = pd.read_csv(FINAL_RESULTS)

metrics.columns = metrics.columns.str.strip()
final.columns = final.columns.str.strip()

print(f"Metrics rows    : {len(metrics)}")
print(f"Final-test rows : {len(final)}")


# ============================================================
# NORMALIZE EPSILON COLUMN
# ============================================================

if "epsilon" in final.columns and "eps_margin" not in final.columns:
    final = final.rename(
        columns={"epsilon": "eps_margin"}
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in [
    "seed",
    "round",
    "switch_on",
    "eps_margin",
    "macro_f1"
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
    "accuracy"
]:
    if col in final.columns:
        final[col] = pd.to_numeric(
            final[col],
            errors="coerce"
        )


# ============================================================
# KEEP ONLY THE FOUR EPSILON SETTINGS
# ============================================================

def valid_epsilon(x):
    if pd.isna(x):
        return False

    return any(
        np.isclose(float(x), eps)
        for eps in EPSILON_VALUES
    )


metrics = metrics[
    metrics["eps_margin"].apply(valid_epsilon)
].copy()

final = final[
    final["eps_margin"].apply(valid_epsilon)
].copy()


# ============================================================
# EXTRACT ONE GATE DECISION PER ROUND
# ============================================================

# Prefer GLOBAL rows because the decision to accept/reject
# the mixed/global candidate is represented there.

if "stage" in metrics.columns:

    stage_upper = (
        metrics["stage"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (stage_upper == "GLOBAL").any():

        gate = metrics[
            stage_upper == "GLOBAL"
        ].copy()

    else:

        gate = metrics.copy()

else:

    gate = metrics.copy()


# ============================================================
# REMOVE INVALID / DUPLICATE ROUND RECORDS
# ============================================================

gate = gate.dropna(
    subset=[
        "round",
        "eps_margin",
        "switch_on"
    ]
)

# One gate decision for each epsilon × round.
# If a run was repeated, keep the latest logged record.

duplicate_columns = [
    "eps_margin",
    "round"
]

if "seed" in gate.columns:
    duplicate_columns.insert(0, "seed")

gate = gate.drop_duplicates(
    subset=duplicate_columns,
    keep="last"
)


# ============================================================
# COUNT ON / OFF
# ============================================================

gate_summary = (
    gate
    .groupby("eps_margin")
    .agg(
        total_rounds=(
            "switch_on",
            "count"
        ),

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


# ============================================================
# ACCEPTANCE RATE
# ============================================================

gate_summary["acceptance_rate_percent"] = (
    gate_summary["switch_on_1"]
    / gate_summary["total_rounds"]
    * 100.0
)


# ============================================================
# FINAL TEST RESULTS
# ============================================================

final = final.dropna(
    subset=[
        "eps_margin",
        "macro_f1"
    ]
)

# If the same epsilon was run more than once,
# use the latest result for this sensitivity sweep.

if "seed" in final.columns:

    final = final.drop_duplicates(
        subset=[
            "seed",
            "eps_margin"
        ],
        keep="last"
    )

else:

    final = final.drop_duplicates(
        subset=["eps_margin"],
        keep="last"
    )


# Because this is a fixed-seed epsilon sensitivity analysis,
# report the final test value for each epsilon rather than SD.

performance = (
    final
    .groupby("eps_margin")
    .agg(
        final_macro_f1=(
            "macro_f1",
            "last"
        )
    )
    .reset_index()
)


# Add final accuracy if available

if "accuracy" in final.columns:

    accuracy = (
        final
        .groupby("eps_margin")
        .agg(
            final_accuracy=(
                "accuracy",
                "last"
            )
        )
        .reset_index()
    )

    performance = performance.merge(
        accuracy,
        on="eps_margin",
        how="left"
    )


# ============================================================
# MERGE GATE BEHAVIOR + FINAL PERFORMANCE
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
# DIFFERENCE RELATIVE TO epsilon = 0.003
# ============================================================

reference = summary[
    np.isclose(
        summary["eps_margin"],
        0.003
    )
]

if not reference.empty:

    reference_f1 = float(
        reference["final_macro_f1"].iloc[0]
    )

    summary["delta_f1_vs_eps_0.003"] = (
        summary["final_macro_f1"]
        - reference_f1
    )

else:

    summary["delta_f1_vs_eps_0.003"] = np.nan


# ============================================================
# ROUND VALUES FOR DISPLAY
# ============================================================

for col in [
    "acceptance_rate_percent",
    "final_macro_f1",
    "final_accuracy",
    "delta_f1_vs_eps_0.003"
]:

    if col in summary.columns:

        summary[col] = (
            pd.to_numeric(
                summary[col],
                errors="coerce"
            )
            .round(6)
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n")
print("=" * 100)
print("CLIENT 2 — EPSILON SENSITIVITY ANALYSIS")
print("=" * 100)

columns = [
    "eps_margin",
    "total_rounds",
    "switch_on_1",
    "switch_on_0",
    "acceptance_rate_percent",
    "final_macro_f1",
    "delta_f1_vs_eps_0.003"
]

columns = [
    col for col in columns
    if col in summary.columns
]

print(
    summary[columns].to_string(
        index=False
    )
)


# ============================================================
# CHECK EACH EPSILON
# ============================================================

print("\n")
print("=" * 100)
print("EPSILON CHECK")
print("=" * 100)

for eps in EPSILON_VALUES:

    row = summary[
        np.isclose(
            summary["eps_margin"],
            eps
        )
    ]

    if row.empty:

        print(
            f"ε={eps:<6} -> NO DATA"
        )

        continue

    r = row.iloc[0]

    print(
        f"ε={eps:<6} | "
        f"Rounds={int(r['total_rounds'])} | "
        f"ON={int(r['switch_on_1'])} | "
        f"OFF={int(r['switch_on_0'])} | "
        f"Rate={r['acceptance_rate_percent']:.2f}% | "
        f"Final Macro-F1={r['final_macro_f1']:.6f}"
    )


print("\nSummary saved to:")
print(OUTPUT_SUMMARY)