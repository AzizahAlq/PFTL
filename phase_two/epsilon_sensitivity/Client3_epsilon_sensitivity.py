import os
import numpy as np
import pandas as pd

# ============================================================
# CLIENT 3 — EPSILON SENSITIVITY ANALYSIS
# READ EXISTING LOGS ONLY
# ============================================================

CLIENT_ID = "client3"

BASE_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "ptfl_logs/client3_unsw_v2"
)

METRICS_LOG = os.path.join(
    BASE_DIR,
    "client3_metrics_log.csv"
)

FINAL_RESULTS = os.path.join(
    BASE_DIR,
    "client3_epsilon_sensitivity_final_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "client3_epsilon_sensitivity_summary.csv"
)

EPSILON_VALUES = [0.0, 0.001, 0.003, 0.01]


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(METRICS_LOG):
    raise FileNotFoundError(
        f"Client 3 metrics log not found:\n{METRICS_LOG}"
    )

if not os.path.exists(FINAL_RESULTS):
    raise FileNotFoundError(
        f"Client 3 final-results file not found:\n{FINAL_RESULTS}"
    )


# ============================================================
# READ CLIENT 3 LOGS
# ============================================================

metrics = pd.read_csv(METRICS_LOG)
final = pd.read_csv(FINAL_RESULTS)

metrics.columns = metrics.columns.str.strip()
final.columns = final.columns.str.strip()

print("\nReading CLIENT 3 logs:")
print(f"Metrics file    : {METRICS_LOG}")
print(f"Final file      : {FINAL_RESULTS}")
print(f"Metrics rows    : {len(metrics)}")
print(f"Final-test rows : {len(final)}")


# ============================================================
# NORMALIZE EPSILON COLUMN
# ============================================================

# Metrics file normally uses eps_margin.
# Final-results file may use epsilon.

if "epsilon" in final.columns and "eps_margin" not in final.columns:
    final = final.rename(
        columns={
            "epsilon": "eps_margin"
        }
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
# CONVERT COLUMNS TO NUMERIC
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
# KEEP ONLY EPSILON VALUES USED IN SENSITIVITY ANALYSIS
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
    metrics["eps_margin"].apply(
        is_valid_epsilon
    )
].copy()


final = final[
    final["eps_margin"].apply(
        is_valid_epsilon
    )
].copy()


# ============================================================
# IDENTIFY THE GATE-DECISION ROW
# ============================================================

# Client logs can contain multiple rows for each round,
# for example LOCAL and GLOBAL.
#
# We need ONE switch_on decision per round.
#
# Prefer GLOBAL rows because they contain the result
# of the local-vs-mixed/global safety-gate decision.

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
# REMOVE DUPLICATE ROUND RECORDS
# ============================================================

# Fixed-seed epsilon sensitivity:
# each epsilon should contribute one decision per round.
#
# If a run was repeated and duplicate rows exist,
# keep the latest logged value.

duplicate_columns = [
    "eps_margin",
    "round"
]

if "seed" in gate.columns:
    duplicate_columns.insert(
        0,
        "seed"
    )


gate = gate.drop_duplicates(
    subset=duplicate_columns,
    keep="last"
)


# ============================================================
# COUNT switch_on = 1 AND switch_on = 0
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

        switch_on_1=(
            "switch_on",
            lambda x: int(
                (x == 1).sum()
            )
        ),

        switch_on_0=(
            "switch_on",
            lambda x: int(
                (x == 0).sum()
            )
        )
    )
)


# ============================================================
# CALCULATE ACCEPTANCE RATE
# ============================================================

gate_summary[
    "acceptance_rate_percent"
] = (

    gate_summary["switch_on_1"]
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


# ============================================================
# REMOVE DUPLICATE FINAL RESULTS
# ============================================================

# We are doing a fixed-seed epsilon sensitivity analysis.
# Therefore, each epsilon should have one final-test result.
#
# If the same epsilon was logged more than once,
# use the latest result for that seed.

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
        subset=[
            "eps_margin"
        ],
        keep="last"
    )


# ============================================================
# FINAL MACRO-F1 FOR EACH EPSILON
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
# OPTIONAL: FINAL ACCURACY
# ============================================================

if "accuracy" in final.columns:

    accuracy_summary = (
        final
        .groupby(
            "eps_margin",
            as_index=False
        )
        .agg(

            final_accuracy=(
                "accuracy",
                "last"
            )

        )
    )

    performance = performance.merge(
        accuracy_summary,
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


summary = (
    summary
    .sort_values(
        "eps_margin"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# DELTA MACRO-F1 RELATIVE TO ε = 0.003
# ============================================================

reference = summary[
    np.isclose(
        summary["eps_margin"],
        0.003,
        atol=1e-9
    )
]


if not reference.empty:

    reference_f1 = float(
        reference[
            "final_macro_f1"
        ].iloc[0]
    )

    summary[
        "delta_f1_vs_eps_0.003"
    ] = (

        summary[
            "final_macro_f1"
        ]
        -
        reference_f1
    )

else:

    summary[
        "delta_f1_vs_eps_0.003"
    ] = np.nan


# ============================================================
# ROUND DISPLAY VALUES
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
# SAVE CLIENT 3 SUMMARY
# ============================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# PRINT CLIENT 3 FINAL TABLE
# ============================================================

print("\n")
print("=" * 105)

print(
    "CLIENT 3 — EPSILON SENSITIVITY ANALYSIS"
)

print("=" * 105)


display_columns = [

    "eps_margin",
    "total_rounds",
    "switch_on_1",
    "switch_on_0",
    "acceptance_rate_percent",
    "final_macro_f1",
    "delta_f1_vs_eps_0.003"

]


display_columns = [

    col
    for col in display_columns
    if col in summary.columns

]


print(

    summary[
        display_columns
    ].to_string(
        index=False
    )

)


# ============================================================
# DETAILED EPSILON CHECK
# ============================================================

print("\n")
print("=" * 105)

print(
    "CLIENT 3 — EPSILON CHECK"
)

print("=" * 105)


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
            f"ε={eps:<6} -> NO DATA"
        )

        continue


    r = row.iloc[0]


    total_rounds = int(
        r["total_rounds"]
    )

    on_count = int(
        r["switch_on_1"]
    )

    off_count = int(
        r["switch_on_0"]
    )

    rate = float(
        r["acceptance_rate_percent"]
    )

    macro_f1 = float(
        r["final_macro_f1"]
    )


    print(

        f"ε={eps:<6} | "
        f"Rounds={total_rounds:2d} | "
        f"ON={on_count:2d} | "
        f"OFF={off_count:2d} | "
        f"Rate={rate:6.2f}% | "
        f"Final Macro-F1={macro_f1:.6f}"

    )


# ============================================================
# SIMPLE CONSISTENCY CHECK
# ============================================================

print("\n")
print("=" * 105)

print(
    "CONSISTENCY CHECK"
)

print("=" * 105)


for _, row in summary.iterrows():

    eps = row["eps_margin"]

    total = int(
        row["total_rounds"]
    )

    on = int(
        row["switch_on_1"]
    )

    off = int(
        row["switch_on_0"]
    )


    if on + off == total:

        status = "OK"

    else:

        status = "CHECK LOG"


    print(

        f"ε={eps:<6} | "
        f"ON + OFF = {on + off} | "
        f"Total rounds = {total} | "
        f"{status}"

    )


# ============================================================
# SHOW BEST OBSERVED MACRO-F1
# ============================================================

valid_performance = summary.dropna(
    subset=[
        "final_macro_f1"
    ]
)


if not valid_performance.empty:

    best_index = (
        valid_performance[
            "final_macro_f1"
        ]
        .idxmax()
    )

    best_row = summary.loc[
        best_index
    ]

    print("\n")
    print("=" * 105)

    print(
        "CLIENT 3 — HIGHEST OBSERVED FINAL MACRO-F1"
    )

    print("=" * 105)

    print(
        f"Epsilon        : {best_row['eps_margin']}"
    )

    print(
        f"Final Macro-F1 : "
        f"{best_row['final_macro_f1']:.6f}"
    )

    print(
        f"Switch ON      : "
        f"{int(best_row['switch_on_1'])}"
    )

    print(
        f"Switch OFF     : "
        f"{int(best_row['switch_on_0'])}"
    )

    print(
        f"Acceptance     : "
        f"{best_row['acceptance_rate_percent']:.2f}%"
    )


# ============================================================
# OUTPUT LOCATION
# ============================================================

print("\n")
print("=" * 105)

print(
    "CLIENT 3 SUMMARY SAVED TO:"
)

print("=" * 105)

print(
    OUTPUT_SUMMARY
)

print("\nDone.")