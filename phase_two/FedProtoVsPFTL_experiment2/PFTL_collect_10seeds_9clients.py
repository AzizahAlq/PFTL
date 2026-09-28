#PFTL_collect_10seeds_9clients.py
from pathlib import Path
import pandas as pd

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "FedProtoVsPFTL/pftl_logs"
)

# Same 10 seeds used for FedProto
SEEDS = [32, 123, 12, 999, 308, 5, 77, 45, 101, 190]

# ============================================================
# Collect results
# ============================================================

all_clients = []

for client_num in range(1, 10):

    csv_path = (
        BASE_DIR
        / f"client{client_num}_ton_iot_aligned_20k"
        / f"client{client_num}_final_test_results.csv"
    )

    if not csv_path.exists():
        print(f"\nClient {client_num}: FILE NOT FOUND")
        print(csv_path)
        continue

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "seed",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print(
            f"\nClient {client_num}: "
            f"MISSING COLUMNS {missing_columns}"
        )
        continue

    # --------------------------------------------------------
    # Keep only requested seeds
    # --------------------------------------------------------

    df["seed"] = pd.to_numeric(
        df["seed"],
        errors="coerce"
    )

    df = df[
        df["seed"].isin(SEEDS)
    ].copy()

    # Check for duplicate seeds
    duplicate_seeds = (
        df.loc[
            df["seed"].duplicated(keep=False),
            "seed"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if duplicate_seeds:
        print(
            f"\nWARNING Client {client_num}: "
            f"duplicate seeds found: {sorted(duplicate_seeds)}"
        )

    # --------------------------------------------------------
    # Keep requested seed order
    # --------------------------------------------------------

    df["seed"] = df["seed"].astype(int)

    df["seed_order"] = df["seed"].map(
        {seed: i for i, seed in enumerate(SEEDS)}
    )

    df = (
        df.sort_values("seed_order")
        .drop(columns="seed_order")
        .reset_index(drop=True)
    )

    # Add client ID
    df.insert(
        0,
        "client",
        f"client{client_num}"
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"\n===== CLIENT {client_num} =====")
    print(f"Seeds found: {len(df)}/10")

    print(
        df[
            [
                "seed",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Check missing seeds
    # --------------------------------------------------------

    found_seeds = set(df["seed"].tolist())
    missing_seeds = set(SEEDS) - found_seeds

    if missing_seeds:
        print(
            "Missing seeds:",
            sorted(missing_seeds)
        )
    else:
        print("All 10 seeds found.")

    all_clients.append(df)


# ============================================================
# Combine all 9 clients
# ============================================================

if not all_clients:
    raise RuntimeError(
        "No PFTL result files were successfully loaded."
    )

combined = pd.concat(
    all_clients,
    ignore_index=True
)

# ============================================================
# Save combined results
# ============================================================

output = (
    BASE_DIR
    / "pftl_9clients_10seeds.csv"
)

combined.to_csv(
    output,
    index=False
)

# ============================================================
# Final summary
# ============================================================

print("\n======================================")
print("PFTL COLLECTION COMPLETE")
print("======================================")
print(f"Clients collected: {combined['client'].nunique()}/9")
print(f"Total rows:        {len(combined)}/90")
print()
print("Saved combined results:")
print(output)
print("======================================")