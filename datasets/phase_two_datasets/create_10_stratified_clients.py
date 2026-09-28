#!/usr/bin/env python3
# ============================================================
# Create 10 Disjoint Stratified Virtual Clients
# from Client 1: CIC-ToN-IoT
# ============================================================
#
# Original dataset:
#   D1_CIC-ToN-IoT_new.csv
#
# Original size:
#   200,000 samples
#
# Number of classes:
#   10
#
# Expected size per virtual client:
#   Approximately 20,000 samples
#
# Expected class distribution per client:
#   Benign       ≈ 8,525
#   xss          ≈ 8,257
#   password     ≈ 1,319
#   injection    ≈ 1,082
#   ransomware   ≈   479
#   scanning     ≈   143
#   backdoor     ≈   109
#   mitm         ≈    52
#   ddos         ≈    20
#   dos          ≈ 14–15
#
# Output files:
#   ton_iot_client01.csv
#   ton_iot_client02.csv
#   ...
#   ton_iot_client10.csv
#
# Properties:
#   - Stratified by the Attack column
#   - All 10 classes are included in every client
#   - No duplicated samples
#   - No overlap between clients
#   - Every original sample is assigned exactly once
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "Multi_class_Datasets/D1_CIC-ToN-IoT_new.csv"
)

OUTPUT_DIR = (
    "/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/"
    "Multi_class_Datasets/ton_iot_10_virtual_clients"
)

LABEL_COL = "Attack"

CLIENT_PREFIX = "ton_iot_client"

NUM_CLIENTS = 10
SEED = 45

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Clean Column Names
# ============================================================

def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


# ============================================================
# Clean Attack Labels
# ============================================================

def clean_attack_labels(series):
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("�", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================
# Inspect Original Dataset
# ============================================================

def inspect_dataset(df):

    print("\n" + "=" * 80)
    print("ORIGINAL DATASET")
    print("=" * 80)

    print(f"Rows          : {len(df):,}")
    print(f"Columns       : {df.shape[1]}")
    print(f"Total classes : {df[LABEL_COL].nunique()}")

    print("\nOriginal class distribution:")

    original_distribution = pd.DataFrame({
        "samples": df[LABEL_COL].value_counts(),
        "percentage": (
            df[LABEL_COL]
            .value_counts(normalize=True)
            .mul(100)
            .round(4)
        )
    })

    print(original_distribution.to_string())

    print("\nFirst 10 rows:")
    print(df.head(10).to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0]

    print("\nMissing values:")

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False).to_string())


# ============================================================
# Verify Class Coverage
# ============================================================

def verify_class_coverage(df):

    class_counts = df[LABEL_COL].value_counts()

    insufficient_classes = class_counts[
        class_counts < NUM_CLIENTS
    ]

    print("\n" + "=" * 80)
    print("CLASS COVERAGE CHECK")
    print("=" * 80)

    if insufficient_classes.empty:
        print(
            f"All classes contain at least {NUM_CLIENTS} samples."
        )
        print(
            f"Therefore, all {NUM_CLIENTS} virtual clients can "
            "receive every class without replication."
        )
    else:
        print(
            "The following classes contain fewer samples than "
            f"the number of clients ({NUM_CLIENTS}):"
        )

        print(insufficient_classes.to_string())

        raise ValueError(
            "A disjoint partition cannot place these classes "
            "in every client."
        )


# ============================================================
# Verify No Overlap
# ============================================================

def verify_partition(client_index_sets, original_size):

    total_assigned = sum(
        len(index_set)
        for index_set in client_index_sets
    )

    unique_indices = set().union(
        *client_index_sets
    )

    overlap_count = (
        total_assigned - len(unique_indices)
    )

    missing_count = (
        original_size - len(unique_indices)
    )

    print("\n" + "=" * 80)
    print("PARTITION VERIFICATION")
    print("=" * 80)

    print(f"Original samples       : {original_size:,}")
    print(f"Total assigned samples : {total_assigned:,}")
    print(f"Unique assigned rows   : {len(unique_indices):,}")
    print(f"Overlapping rows       : {overlap_count:,}")
    print(f"Missing rows           : {missing_count:,}")

    if overlap_count != 0:
        raise RuntimeError(
            f"Overlap detected: {overlap_count} duplicated rows."
        )

    if missing_count != 0:
        raise RuntimeError(
            f"Incomplete partition: {missing_count} rows missing."
        )

    print("\nVerification passed:")
    print("- No overlap")
    print("- No duplicated samples")
    print("- No missing samples")
    print("- Every original row assigned exactly once")


# ============================================================
# Create 10 Stratified Virtual Clients
# ============================================================

def create_virtual_clients(df):

    labels = df[LABEL_COL].to_numpy()

    dummy_features = np.zeros(
        shape=(len(df), 1),
        dtype=np.uint8
    )

    stratified_splitter = StratifiedKFold(
        n_splits=NUM_CLIENTS,
        shuffle=True,
        random_state=SEED
    )

    client_index_sets = []
    client_summary_rows = []
    class_distribution_dict = {}

    print("\n" + "=" * 80)
    print("CREATING 10 CIC-ToN-IoT VIRTUAL CLIENTS")
    print("=" * 80)

    split_iterator = stratified_splitter.split(
        dummy_features,
        labels
    )

    for client_number, (_, client_indices) in enumerate(
        split_iterator,
        start=1
    ):

        client_id = (
            f"{CLIENT_PREFIX}{client_number:02d}"
        )

        client_df = df.iloc[
            client_indices
        ].copy()

        # Preserve the original source index for verification.
        client_df.insert(
            0,
            "source_row_id",
            client_df.index.astype(int)
        )

        client_df.reset_index(
            drop=True,
            inplace=True
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{client_id}.csv"
        )

        client_df.to_csv(
            output_path,
            index=False
        )

        original_indices = set(
            client_df[
                "source_row_id"
            ].astype(int).tolist()
        )

        client_index_sets.append(
            original_indices
        )

        class_counts = (
            client_df[LABEL_COL]
            .value_counts()
            .sort_index()
        )

        class_distribution_dict[client_id] = {
            str(class_name): int(class_count)
            for class_name, class_count
            in class_counts.items()
        }

        client_summary_rows.append({
            "client_id": client_id,
            "samples": int(len(client_df)),
            "number_of_classes": int(
                client_df[LABEL_COL].nunique()
            ),
            "output_file": output_path
        })

        print("\n" + "-" * 80)
        print(client_id)
        print("-" * 80)

        print(f"Samples : {len(client_df):,}")
        print(
            f"Classes : "
            f"{client_df[LABEL_COL].nunique()}"
        )

        print("\nClass distribution:")
        print(class_counts.to_string())

        print(f"\nSaved to:\n{output_path}")

    # Verify that the partitions are disjoint.
    verify_partition(
        client_index_sets=client_index_sets,
        original_size=len(df)
    )

    # ========================================================
    # Save Client Summary
    # ========================================================

    client_summary_df = pd.DataFrame(
        client_summary_rows
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "ton_iot_clients_summary.csv"
    )

    client_summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # Save Class Distribution Matrix
    # ========================================================

    distribution_df = (
        pd.DataFrame(
            class_distribution_dict
        )
        .fillna(0)
        .astype(int)
    )

    distribution_df.index.name = LABEL_COL

    distribution_path = os.path.join(
        OUTPUT_DIR,
        "ton_iot_class_distribution_by_client.csv"
    )

    distribution_df.to_csv(
        distribution_path
    )

    # ========================================================
    # Save Partition Metadata
    # ========================================================

    metadata = {
        "source_dataset": CSV_PATH,
        "dataset_name": "CIC-ToN-IoT",
        "client_prefix": CLIENT_PREFIX,
        "label_column": LABEL_COL,
        "number_of_clients": NUM_CLIENTS,
        "number_of_classes": int(
            df[LABEL_COL].nunique()
        ),
        "seed": SEED,
        "partition_method": (
            "Disjoint stratified partition using StratifiedKFold"
        ),
        "original_samples": int(len(df)),
        "samples_per_client_approximately": int(
            len(df) / NUM_CLIENTS
        ),
        "no_overlap": True,
        "all_samples_assigned": True,
        "training_feature_note": (
            "When Attack is the target, remove Attack, Label, "
            "and source_row_id from the model input features."
        )
    }

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "ton_iot_partition_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)
    print("FINAL CLIENT SUMMARY")
    print("=" * 80)

    print(
        client_summary_df[
            [
                "client_id",
                "samples",
                "number_of_classes"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(
        f"Virtual clients:\n"
        f"{OUTPUT_DIR}/ton_iot_client01.csv\n"
        "...\n"
        f"{OUTPUT_DIR}/ton_iot_client10.csv"
    )

    print(f"\nSummary:\n{summary_path}")
    print(
        f"\nClass distribution:\n"
        f"{distribution_path}"
    )
    print(f"\nMetadata:\n{metadata_path}")


# ============================================================
# Main
# ============================================================

def main():

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{CSV_PATH}"
        )

    print("=" * 80)
    print("LOADING CIC-ToN-IoT DATASET")
    print("=" * 80)

    print(f"Dataset path:\n{CSV_PATH}")

    df = pd.read_csv(
        CSV_PATH,
        low_memory=False
    )

    df = clean_column_names(df)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' was not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df[LABEL_COL] = clean_attack_labels(
        df[LABEL_COL]
    )

    if df[LABEL_COL].isna().any():
        raise ValueError(
            f"The target column '{LABEL_COL}' contains "
            "missing values."
        )

    inspect_dataset(df)

    verify_class_coverage(df)

    create_virtual_clients(df)


if __name__ == "__main__":
    main()