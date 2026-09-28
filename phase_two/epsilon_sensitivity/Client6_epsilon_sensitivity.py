import os
import numpy as np
import pandas as pd

# ============================================================
# CLIENT 6 — EPSILON SENSITIVITY ANALYSIS
# READ EXISTING LOGS ONLY
# ============================================================

BASE_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "ptfl_logs/client6_intelligent_v2"
)

CURVE_LOG = os.path.join(
    BASE_DIR,
    "client6_local_vs_global_curve.csv"
)

FINAL_RESULTS = os.path.join(
    BASE_DIR,
    "client6_epsilon_sensitivity_final_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "client6_epsilon_sensitivity_summary.csv"
)

# ============================================================
# IMPORTANT:
# Chronological order in which epsilon runs were executed
# ============================================================

EPSILON_RUN_ORDER = [
    0.003,
    0.001,
    0.010,
    0.000
]

EXPECTED_ROUNDS = 20


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(CURVE_LOG):
    raise FileNotFoundError(
        f"Client 6 curve log not found:\n{CURVE_LOG}"
    )

if not os.path.exists(FINAL_RESULTS):
    raise FileNotFoundError(
        f"Client 6 final-results file not found:\n{FINAL_RESULTS}"
    )


# ============================================================
# READ FILES
# ============================================================

curve = pd.read_csv(CURVE_LOG)
final = pd.read_csv(FINAL_RESULTS)

curve.columns = curve.columns.str.strip()
final.columns = final.columns.str.strip()

print("\nReading CLIENT 6 logs:")
print(f"Curve file      : {CURVE_LOG}")
print(f"Final file      : {FINAL_RESULTS}")
print(f"Curve rows      : {len(curve)}")
print(f"Final-test rows : {len(final)}")

print("\nCurve columns:")
print(curve.columns.tolist())

print("\nFinal-result columns:")
print(final.columns.tolist())


# ============================================================
# NORMALIZE EPSILON COLUMN
# ============================================================

if "epsilon" in final.columns and "eps_margin" not in final.columns:

    final = final.rename(
        columns={
            "epsilon": "eps_margin"
        }
    )


# ============================================================
# CHECK REQUIRED CURVE COLUMNS
# ============================================================

required_curve_columns = [
    "round",
    "switch_on"
]

for col in required_curve_columns:

    if col not in curve.columns:

        raise ValueError(
            f"Missing required column '{col}' "
            f"in:\n{CURVE_LOG}"
        )


# ============================================================
# CHECK REQUIRED FINAL-RESULT COLUMNS
# ============================================================

required_final_columns = [
    "eps_margin",
    "macro_f1"
]

for col in required_final_columns:

    if col not in final.columns:

        raise ValueError(
            f"Missing required column '{col}' "
            f"in:\n{FINAL_RESULTS}"
        )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

for col in [
    "seed",
    "round",
    "local_macro_f1",
    "mixed_macro_f1",
    "switch_on",
    "gamma_global"
]:

    if col in curve.columns:

        curve[col] = pd.to_numeric(
            curve[col],
            errors="coerce"
        )


for col in [
    "seed",
    "eps_margin",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1"
]:

    if col in final.columns:

        final[col] = pd.to_numeric(
            final[col],
            errors="coerce"
        )


# ============================================================
# REMOVE ROWS WITHOUT ROUND / SWITCH INFORMATION
# ============================================================

curve = curve.dropna(
    subset=[
        "round",
        "switch_on"
    ]
).copy()

curve = curve.reset_index(
    drop=True
)


# ============================================================
# SHOW ROUND DISTRIBUTION
# ============================================================

print("\n")
print("=" * 100)
print("CLIENT 6 — ROUND DISTRIBUTION")
print("=" * 100)

print(
    curve["round"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# FIND COMPLETE ROUND 1 -> 20 BLOCKS
# ============================================================

blocks = []

expected_sequence = list(
    range(
        1,
        EXPECTED_ROUNDS + 1
    )
)


for start in range(len(curve)):

    # A valid experiment must start at round 1
    if int(curve.iloc[start]["round"]) != 1:
        continue

    end = start + EXPECTED_ROUNDS

    # Not enough rows left
    if end > len(curve):
        continue

    block = curve.iloc[
        start:end
    ].copy()

    actual_sequence = (
        block["round"]
        .astype(int)
        .tolist()
    )

    # Accept only exact 1 -> 20 sequence
    if actual_sequence == expected_sequence:

        blocks.append({
            "start": start,
            "end": end - 1,
            "data": block
        })


# ============================================================
# REMOVE OVERLAPPING BLOCKS
# ============================================================

non_overlapping_blocks = []

last_end = -1

for block in blocks:

    if block["start"] > last_end:

        non_overlapping_blocks.append(
            block
        )

        last_end = block["end"]


blocks = non_overlapping_blocks


# ============================================================
# DISPLAY DETECTED BLOCKS
# ============================================================

print("\n")
print("=" * 100)
print("CLIENT 6 — DETECTED COMPLETE 20-ROUND RUNS")
print("=" * 100)

print(
    f"Complete 20-round blocks found: {len(blocks)}"
)


for i, block in enumerate(
    blocks,
    start=1
):

    data = block["data"]

    print(
        f"Run {i}: "
        f"rows {block['start']}–{block['end']} | "
        f"rounds "
        f"{int(data['round'].iloc[0])}–"
        f"{int(data['round'].iloc[-1])}"
    )


# ============================================================
# REQUIRE EXACTLY FOUR COMPLETE EPSILON RUNS
# ============================================================

if len(blocks) != 4:

    print("\n")
    print("=" * 100)
    print("WARNING")
    print("=" * 100)

    print(
        f"Expected 4 complete epsilon runs, "
        f"but found {len(blocks)}."
    )

    print(
        "\nThe program will NOT assign epsilon values "
        "automatically because doing so could produce "
        "incorrect switch counts."
    )

    print(
        "\nPlease inspect the rows printed below."
    )

    print("\nFull curve log:")
    print(
        curve.to_string()
    )

    raise RuntimeError(
        "Could not safely identify exactly four "
        "complete 20-round Client 6 runs."
    )


# ============================================================
# ASSIGN EPSILON ACCORDING TO CHRONOLOGICAL RUN ORDER
# ============================================================

run_results = []

used_indices = set()


for run_number, (eps, block) in enumerate(
    zip(
        EPSILON_RUN_ORDER,
        blocks
    ),
    start=1
):

    data = block["data"].copy()

    start = block["start"]
    end = block["end"]

    used_indices.update(
        range(
            start,
            end + 1
        )
    )

    # --------------------------------------------------------
    # SWITCH COUNTS
    # --------------------------------------------------------

    accepted = int(
        (
            data["switch_on"] == 1
        ).sum()
    )

    rejected = int(
        (
            data["switch_on"] == 0
        ).sum()
    )

    total = len(data)

    acceptance_rate = (
        accepted
        /
        total
        *
        100.0
    )

    run_results.append({

        "run_number": run_number,

        "eps_margin": eps,

        "row_range": (
            f"{start}-{end}"
        ),

        "total_rounds": total,

        "accepted_rounds": accepted,

        "rejected_rounds": rejected,

        "acceptance_rate_percent": (
            acceptance_rate
        )
    })


gate_summary = pd.DataFrame(
    run_results
)


# ============================================================
# PRINT SWITCH ANALYSIS BEFORE MERGING PERFORMANCE
# ============================================================

print("\n")
print("=" * 100)
print("CLIENT 6 — SWITCH ANALYSIS BY RUN ORDER")
print("=" * 100)

print(
    gate_summary.to_string(
        index=False
    )
)


# ============================================================
# IDENTIFY EXTRA / OLD ROWS
# ============================================================

extra_indices = [
    i
    for i in range(len(curve))
    if i not in used_indices
]


print("\n")
print("=" * 100)
print("CLIENT 6 — EXTRA / UNASSIGNED ROWS")
print("=" * 100)


if len(extra_indices) == 0:

    print("No extra rows.")

else:

    print(
        f"Extra rows found: {len(extra_indices)}"
    )

    print(
        "\nThese rows are NOT used in the epsilon "
        "switch analysis:"
    )

    print(
        curve.iloc[
            extra_indices
        ].to_string()
    )


# ============================================================
# PREPARE FINAL MACRO-F1 RESULTS
# ============================================================

final = final.dropna(
    subset=[
        "eps_margin",
        "macro_f1"
    ]
).copy()


# Keep only our four epsilon values
def valid_epsilon(value):

    return any(
        np.isclose(
            float(value),
            eps,
            atol=1e-9
        )
        for eps in EPSILON_RUN_ORDER
    )


final = final[
    final["eps_margin"].apply(
        valid_epsilon
    )
].copy()


# ============================================================
# IF DUPLICATES EXIST, KEEP LATEST FINAL RESULT
# ============================================================

if "timestamp" in final.columns:

    final = final.sort_values(
        "timestamp"
    )


final = final.drop_duplicates(
    subset=[
        "eps_margin"
    ],
    keep="last"
)


# ============================================================
# FINAL MACRO-F1 TABLE
# ============================================================

performance = final[
    [
        "eps_margin",
        "macro_f1"
    ]
].copy()

performance = performance.rename(
    columns={
        "macro_f1": "final_macro_f1"
    }
)


# ============================================================
# MERGE SWITCH COUNTS WITH FINAL MACRO-F1
# ============================================================

summary = gate_summary.merge(
    performance,
    on="eps_margin",
    how="left"
)


# ============================================================
# SORT TABLE IN NUMERIC EPSILON ORDER
# ============================================================

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
# ROUND DISPLAY VALUES
# ============================================================

summary[
    "acceptance_rate_percent"
] = (

    summary[
        "acceptance_rate_percent"
    ].round(2)

)

summary[
    "final_macro_f1"
] = (

    summary[
        "final_macro_f1"
    ].round(6)

)


# ============================================================
# CONSISTENCY CHECK
# ============================================================

print("\n")
print("=" * 100)
print("CLIENT 6 — CONSISTENCY CHECK")
print("=" * 100)


all_ok = True


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

    if (
        accepted + rejected == total
        and total == EXPECTED_ROUNDS
    ):

        status = "OK"

    else:

        status = "CHECK LOG"

        all_ok = False


    print(
        f"ε={eps:<6} | "
        f"Accepted + Rejected="
        f"{accepted + rejected} | "
        f"Total={total} | "
        f"{status}"
    )


# ============================================================
# FINAL CLIENT 6 TABLE
# ============================================================

print("\n")
print("=" * 100)
print("CLIENT 6 — EPSILON SENSITIVITY ANALYSIS")
print("=" * 100)


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
# PAPER-FRIENDLY FORMAT
# ============================================================

print("\n")
print("=" * 100)
print("CLIENT 6 — PAPER TABLE FORMAT")
print("=" * 100)


for _, row in summary.iterrows():

    eps = row["eps_margin"]

    accepted = int(
        row["accepted_rounds"]
    )

    total = int(
        row["total_rounds"]
    )

    rate = float(
        row["acceptance_rate_percent"]
    )

    f1 = float(
        row["final_macro_f1"]
    )

    print(
        f"ε={eps:<6} | "
        f"Switch ON={accepted}/{total} | "
        f"Acceptance={rate:.0f}% | "
        f"Final Macro-F1={f1:.6f}"
    )


# ============================================================
# SAVE ONLY IF CONSISTENCY CHECK PASSES
# ============================================================

if all_ok:

    output_columns = [
        "eps_margin",
        "total_rounds",
        "accepted_rounds",
        "rejected_rounds",
        "acceptance_rate_percent",
        "final_macro_f1"
    ]

    summary[
        output_columns
    ].to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print("\n")
    print("=" * 100)
    print("CLIENT 6 SUMMARY SAVED TO:")
    print("=" * 100)

    print(
        OUTPUT_SUMMARY
    )

else:

    print(
        "\nSummary NOT saved because the "
        "consistency check failed."
    )


print("\nDone.")