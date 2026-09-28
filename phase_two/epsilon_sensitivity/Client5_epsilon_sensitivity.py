import os
import numpy as np
import pandas as pd

# ============================================================
# CLIENT 5 — EPSILON SENSITIVITY ANALYSIS
# READ EXISTING LOGS ONLY
# ============================================================

BASE_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "ptfl_logs/client5_v2_localfirst_strict"
)

METRICS_LOG = os.path.join(
    BASE_DIR,
    "client5_metrics_log.csv"
)

FINAL_RESULTS = os.path.join(
    BASE_DIR,
    "client5_epsilon_sensitivity_final_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "client5_epsilon_sensitivity_summary.csv"
)

EPSILON_VALUES = [0.0, 0.001, 0.003, 0.01]


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(METRICS_LOG):
    raise FileNotFoundError(
        f"Client 5 metrics log not found:\n{METRICS_LOG}"
    )

if not os.path.exists(FINAL_RESULTS):
    raise FileNotFoundError(
        f"Client 5 final-results file not found:\n{FINAL_RESULTS}"
    )


# ============================================================
# READ LOGS
# ============================================================

metrics = pd.read_csv(METRICS_LOG)
final = pd.read_csv(FINAL_RESULTS)

metrics.columns = metrics.columns.str.strip()
final.columns = final.columns.str.strip()

print("\nReading CLIENT 5 logs:")
print(f"Metrics file    : {METRICS_LOG}")
print(f"Final file      : {FINAL_RESULTS}")
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
# CHECK REQUIRED COLUMNS
# ============================================================

required_metrics = [
    "round",
    "switch_on",
    "eps_margin"
]

required_final = [
    "eps_margin",
    "macro_f1"
]

for col in required_metrics:
    if col not in metrics.columns:
        raise ValueError(
            f"Missing column '{col}' in:\n{METRICS_LOG}"
        )

for col in required_final:
    if col not in final.columns:
        raise ValueError(
            f"Missing column '{col}' in:\n{FINAL_RESULTS}"
        )


# ============================================================
# CONVERT TO NUMERIC
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
# KEEP ONLY THE FOUR EPSILON VALUES
# ============================================================

def is_valid_epsilon(value):

    if pd.isna(value):
        return False

    return any(
        np.isclose(
            float(value),
            eps,
            atol=1e-9
        )
        for eps in EPSILON_VALUES
    )


metrics = metrics[
    metrics["eps_margin"].apply(is_valid_epsilon)
].copy()

final = final[
    final["eps_margin"].apply(is_valid_epsilon)
].copy()


# ============================================================
# SELECT GATE-DECISION ROWS
# ============================================================

if "stage" in metrics.columns:

    stage_upper = (
        metrics["stage"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print("\nAvailable stages:")
    print(stage_upper.value_counts(dropna=False))

    if (stage_upper == "GLOBAL").any():

        gate = metrics[
            stage_upper == "GLOBAL"
        ].copy()

        print("\nUsing GLOBAL rows for gate decisions.")

    else:

        gate = metrics.copy()

        print(
            "\nWARNING: GLOBAL stage not found. "
            "Using available rows."
        )

else:

    gate = metrics.copy()

    print(
        "\nWARNING: 'stage' column not found. "
        "Using available rows."
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

gate = gate.dropna(
    subset=[
        "round",
        "eps_margin",
        "switch_on"
    ]
)


# ============================================================
# ONE GATE DECISION PER ROUND
# ============================================================

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
# COUNT ACCEPTED / REJECTED ROUNDS
# ============================================================

gate_summary = (
    gate
    .groupby(
        "eps_margin",
        as_index=False
    )
    .agg(

        total_rounds=(
            "switch_on",
            "count"
        ),

        accepted_rounds=(
            "switch_on",
            lambda x: int(
                (x == 1).sum()
            )
        ),

        rejected_rounds=(
            "switch_on",
            lambda x: int(
                (x == 0).sum()
            )
        )
    )
)


# ============================================================
# ACCEPTANCE RATE
# ============================================================

gate_summary[
    "acceptance_rate_percent"
] = (
    gate_summary["accepted_rounds"]
    /
    gate_summary["total_rounds"]
    *
    100.0
)


# ============================================================
# PREPARE FINAL TEST RESULTS
# ============================================================

final = final.dropna(
    subset=[
        "eps_margin",
        "macro_f1"
    ]
)


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


# ============================================================
# FINAL MACRO-F1
# ============================================================

performance = (
    final
    .groupby(
        "eps_margin",
        as_index=False
    )
    .agg(
        final_macro_f1=(
            "macro_f1",
            "last"
        )
    )
)


# ============================================================
# MERGE RESULTS
# ============================================================

summary = gate_summary.merge(
    performance,
    on="eps_margin",
    how="outer"
)

summary = (
    summary
    .sort_values("eps_margin")
    .reset_index(drop=True)
)


# ============================================================
# ROUND DISPLAY VALUES
# ============================================================

summary["acceptance_rate_percent"] = (
    summary["acceptance_rate_percent"]
    .round(2)
)

summary["final_macro_f1"] = (
    summary["final_macro_f1"]
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
print("=" * 95)

print(
    "CLIENT 5 — EPSILON SENSITIVITY ANALYSIS"
)

print("=" * 95)


display_columns = [
    "eps_margin",
    "total_rounds",
    "accepted_rounds",
    "rejected_rounds",
    "acceptance_rate_percent",
    "final_macro_f1"
]


print(
    summary[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# EPSILON CHECK
# ============================================================

print("\n")
print("=" * 95)

print(
    "CLIENT 5 — EPSILON CHECK"
)

print("=" * 95)


for eps in EPSILON_VALUES:

    row = summary[
        np.isclose(
            summary["eps_margin"],
            eps,
            atol=1e-9
        )
    ]

    if row.empty:

        print(
            f"ε={eps:<6} | NO DATA"
        )

        continue

    r = row.iloc[0]

    total = int(
        r["total_rounds"]
    )

    accepted = int(
        r["accepted_rounds"]
    )

    rejected = int(
        r["rejected_rounds"]
    )

    rate = float(
        r["acceptance_rate_percent"]
    )

    macro_f1 = float(
        r["final_macro_f1"]
    )

    print(
        f"ε={eps:<6} | "
        f"Accepted={accepted:2d}/{total:2d} | "
        f"Rejected={rejected:2d} | "
        f"Rate={rate:6.2f}% | "
        f"Final Macro-F1={macro_f1:.6f}"
    )


# ============================================================
# CONSISTENCY CHECK
# ============================================================

print("\n")
print("=" * 95)

print(
    "CONSISTENCY CHECK"
)

print("=" * 95)


for _, row in summary.iterrows():

    eps = row["eps_margin"]

    total = int(
        row["total_rounds"]
    )

    accepted = int(
        row["accepted_rounds"]
    )

    rejected = int(
        row["rejected_rounds"]
    )

    if accepted + rejected == total:
        status = "OK"
    else:
        status = "CHECK LOG"

    print(
        f"ε={eps:<6} | "
        f"Accepted + Rejected = {accepted + rejected} | "
        f"Total = {total} | "
        f"{status}"
    )


# ============================================================
# SAVE LOCATION
# ============================================================

print("\n")
print("=" * 95)

print(
    "CLIENT 5 SUMMARY SAVED TO:"
)

print("=" * 95)

print(OUTPUT_SUMMARY)

print("\nDone.")