#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Client 1 label inspection and FedProto mapping
Dataset: CIC-ToN-IoT

This script:
1. Loads the original CSV.
2. Detects the multi-class label column.
3. Prints the original label distribution and percentages.
4. Removes rows with missing or empty attack labels.
5. Maps the original labels to standardized semantic labels.
6. Creates local numeric class IDs for the client classifier.
7. Prints the mapped distribution.
8. Saves:
   - mapped dataset CSV
   - original label distribution CSV
   - label mapping JSON

The original dataset is not overwritten.
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# Paths
# ============================================================
# ============================================================
DATA_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/M_PFTL_codes/Multi_class_Datasets"
)

INPUT_CSV = DATA_DIR / "D1_CIC-ToN-IoT_new.csv"

OUTPUT_CSV = (
    DATA_DIR
    / "D1_CIC-ToN-IoT_FedProto_mapped.csv"
)

DISTRIBUTION_CSV = (
    DATA_DIR
    / "D1_CIC-ToN-IoT_original_label_distribution.csv"
)

MAPPED_DISTRIBUTION_CSV = (
    DATA_DIR
    / "D1_CIC-ToN-IoT_mapped_label_distribution.csv"
)

MAPPING_JSON = (
    DATA_DIR
    / "D1_CIC-ToN-IoT_label_mapping.json"
)


# ============================================================
# Client 1 canonical class order
# ============================================================
# These IDs are used only inside Client 1's private classifier.
#
# FedProto exchanges semantic class names and prototype vectors,
# not these local numeric IDs.
# ============================================================
CANONICAL_CLASSES = [
    "BENIGN",
    "XSS",
    "PASSWORD",
    "INJECTION",
    "RANSOMWARE",
    "SCANNING",
    "BACKDOOR",
    "MITM",
    "DDOS",
    "DOS",
]


# ============================================================
# Original-label aliases -> semantic FedProto labels
# ============================================================
LABEL_ALIASES = {
    # Benign
    "benign": "BENIGN",
    "normal": "BENIGN",
    "benign traffic": "BENIGN",

    # XSS
    "xss": "XSS",
    "cross site scripting": "XSS",
    "cross-site scripting": "XSS",
    "web attack xss": "XSS",

    # Password
    "password": "PASSWORD",
    "password attack": "PASSWORD",
    "password attacks": "PASSWORD",

    # Injection
    "injection": "INJECTION",

    # Ransomware
    "ransomware": "RANSOMWARE",

    # Scanning
    "scanning": "SCANNING",
    "scan": "SCANNING",

    # Backdoor
    "backdoor": "BACKDOOR",
    "backdoor malware": "BACKDOOR",

    # MITM
    "mitm": "MITM",
    "mitm attack": "MITM",
    "man in the middle": "MITM",
    "man-in-the-middle": "MITM",
    "mitm arp spoofing": "MITM",

    # DDoS
    "ddos": "DDOS",
    "distributed denial of service": "DDOS",

    # DoS
    "dos": "DOS",
    "denial of service": "DOS",
}


# ============================================================
# Helper functions
# ============================================================
def print_separator(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalize_text(value: object) -> str:
    """
    Normalize label text before alias matching.
    """
    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = " ".join(text.split())

    return text


def find_label_column(df: pd.DataFrame) -> str:
    """
    Detect the multi-class label column.

    The binary_label column is deliberately excluded.
    """
    preferred_columns = [
        "Attack",
        "attack",
        "Category",
        "category",
        "Class",
        "class",
        "Type",
        "type",
        "Label",
        "label",
    ]

    for column in preferred_columns:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not detect the multi-class label column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def clean_missing_labels(
    df: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate valid rows from rows with missing or empty labels.

    Returns:
        cleaned_df
        removed_rows_df
    """
    cleaned = df.copy()

    # Convert common string representations of missing values
    # into pandas missing values.
    cleaned[label_column] = cleaned[label_column].replace(
        [
            "",
            " ",
            "nan",
            "NaN",
            "None",
            "NONE",
            "null",
            "NULL",
            "<MISSING>",
        ],
        pd.NA,
    )

    # Also remove labels containing only whitespace.
    whitespace_mask = (
        cleaned[label_column]
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
    )

    cleaned.loc[
        whitespace_mask,
        label_column,
    ] = pd.NA

    missing_mask = cleaned[label_column].isna()

    removed_rows = cleaned.loc[missing_mask].copy()
    cleaned = cleaned.loc[~missing_mask].copy()

    return cleaned, removed_rows


def build_distribution(
    series: pd.Series,
    label_name: str,
) -> pd.DataFrame:
    """
    Build count and percentage table for a label series.
    """
    counts = (
        series
        .fillna("<MISSING>")
        .astype(str)
        .value_counts(dropna=False)
    )

    percentages = (
        series
        .fillna("<MISSING>")
        .astype(str)
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .mul(100)
    )

    distribution = pd.DataFrame({
        label_name: counts.index,
        "count": counts.values,
        "percentage": [
            float(percentages[label])
            for label in counts.index
        ],
    })

    return distribution


# ============================================================
# Main process
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "Input dataset was not found:\n"
            f"{INPUT_CSV}\n\n"
            "Check whether your directory really contains a "
            "space after 'Proto_PFTL_Multi_class'."
        )

    print(f"Loading dataset:\n{INPUT_CSV}")

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    # Remove accidental spaces around column names.
    df.columns = df.columns.str.strip()

    print_separator("DATASET INFORMATION")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumn names:")

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(f"{index:3d}. {column}")

    # --------------------------------------------------------
    # Detect multi-class label column
    # --------------------------------------------------------
    label_column = find_label_column(df)

    print_separator("DETECTED LABEL COLUMN")
    print(label_column)

    # --------------------------------------------------------
    # Inspect original distribution before removing anything
    # --------------------------------------------------------
    original_distribution = build_distribution(
        df[label_column],
        label_name="original_label",
    )

    print_separator("ORIGINAL LABEL DISTRIBUTION")

    print(
        original_distribution[
            ["original_label", "count"]
        ].to_string(index=False)
    )

    print_separator("ORIGINAL LABEL PERCENTAGES")

    print(
        original_distribution[
            ["original_label", "percentage"]
        ]
        .round({"percentage": 4})
        .to_string(index=False)
    )

    print_separator("ORIGINAL UNIQUE LABELS")

    original_unique_labels = sorted(
        df[label_column]
        .fillna("<MISSING>")
        .astype(str)
        .unique()
        .tolist()
    )

    for index, label in enumerate(
        original_unique_labels,
        start=1,
    ):
        print(f"{index:3d}. {label}")

    original_distribution.to_csv(
        DISTRIBUTION_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Remove missing-label rows
    # --------------------------------------------------------
    rows_before_cleaning = len(df)

    df, removed_rows = clean_missing_labels(
        df,
        label_column,
    )

    rows_after_cleaning = len(df)
    removed_count = len(removed_rows)

    print_separator("MISSING-LABEL CLEANING")

    print(f"Rows before cleaning: {rows_before_cleaning:,}")
    print(f"Rows removed:         {removed_count:,}")
    print(f"Rows remaining:       {rows_after_cleaning:,}")

    if removed_count > 0:
        removed_path = (
            DATA_DIR
            / "D1_CIC-ToN-IoT_removed_missing_label_rows.csv"
        )

        removed_rows.to_csv(
            removed_path,
            index=False,
        )

        print(
            "\nRemoved rows were saved for audit at:\n"
            f"{removed_path}"
        )

    # --------------------------------------------------------
    # Normalize and map labels
    # --------------------------------------------------------
    normalized_source = (
        df[label_column]
        .map(normalize_text)
    )

    df["semantic_label"] = (
        normalized_source
        .map(LABEL_ALIASES)
    )

    # --------------------------------------------------------
    # Stop if any valid labels remain unmapped
    # --------------------------------------------------------
    unknown_mask = df["semantic_label"].isna()

    if unknown_mask.any():
        unknown_summary = (
            df.loc[
                unknown_mask,
                label_column,
            ]
            .astype(str)
            .value_counts()
        )

        print_separator("ERROR: UNMAPPED LABELS FOUND")

        print(unknown_summary.to_string())

        print(
            "\nThe mapped dataset was not saved because "
            "some valid labels were not included in "
            "LABEL_ALIASES."
        )

        raise ValueError(
            "Mapping stopped because unmapped labels were found."
        )

    # --------------------------------------------------------
    # Validate semantic classes
    # --------------------------------------------------------
    found_classes = sorted(
        df["semantic_label"]
        .unique()
        .tolist()
    )

    unexpected_classes = sorted(
        set(found_classes)
        - set(CANONICAL_CLASSES)
    )

    if unexpected_classes:
        raise ValueError(
            "Unexpected semantic classes were generated:\n"
            f"{unexpected_classes}"
        )

    missing_expected_classes = sorted(
        set(CANONICAL_CLASSES)
        - set(found_classes)
    )

    if missing_expected_classes:
        print_separator(
            "WARNING: EXPECTED CLASSES NOT FOUND"
        )

        for semantic_label in missing_expected_classes:
            print(f"- {semantic_label}")

    # --------------------------------------------------------
    # Create fixed local numeric IDs
    # --------------------------------------------------------
    semantic_to_local_id = {
        semantic_label: class_id
        for class_id, semantic_label
        in enumerate(CANONICAL_CLASSES)
    }

    local_id_to_semantic = {
        class_id: semantic_label
        for semantic_label, class_id
        in semantic_to_local_id.items()
    }

    df["local_label_id"] = (
        df["semantic_label"]
        .map(semantic_to_local_id)
        .astype("int32")
    )

    # Preserve source label text.
    df["original_label"] = (
        df[label_column]
        .astype(str)
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------
    first_columns = [
        "original_label",
        "semantic_label",
        "local_label_id",
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in first_columns
    ]

    df = df[
        first_columns
        + remaining_columns
    ]

    # --------------------------------------------------------
    # Mapped label distribution
    # --------------------------------------------------------
    mapped_distribution = (
        df.groupby(
            [
                "local_label_id",
                "semantic_label",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="count")
        .sort_values("local_label_id")
        .reset_index(drop=True)
    )

    mapped_distribution["percentage"] = (
        mapped_distribution["count"]
        .div(len(df))
        .mul(100)
        .round(4)
    )

    print_separator("MAPPED LABEL DISTRIBUTION")

    print(
        mapped_distribution
        .to_string(index=False)
    )

    print_separator("CLIENT 1 LOCAL LABEL MAPPING")

    for class_id, semantic_label in (
        local_id_to_semantic.items()
    ):
        class_count = int(
            (
                df["local_label_id"]
                == class_id
            ).sum()
        )

        print(
            f"{class_id:2d} -> "
            f"{semantic_label:<15} "
            f"count={class_count:,}"
        )

    # --------------------------------------------------------
    # Final consistency checks
    # --------------------------------------------------------
    if df["semantic_label"].isna().any():
        raise ValueError(
            "Internal error: semantic_label contains missing values."
        )

    if df["local_label_id"].isna().any():
        raise ValueError(
            "Internal error: local_label_id contains missing values."
        )

    if len(df) != mapped_distribution["count"].sum():
        raise ValueError(
            "Mapped class counts do not equal the dataset row count."
        )

    if df["local_label_id"].nunique() != len(found_classes):
        raise ValueError(
            "Number of numeric classes does not match "
            "the number of semantic classes."
        )

    # --------------------------------------------------------
    # Save mapped dataset
    # --------------------------------------------------------
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    mapped_distribution.to_csv(
        MAPPED_DISTRIBUTION_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Save mapping metadata
    # --------------------------------------------------------
    mapping_info = {
        "client_id": "client1",
        "dataset": "CIC-ToN-IoT",
        "source_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "source_label_column": label_column,
        "rows_before_cleaning": int(
            rows_before_cleaning
        ),
        "rows_removed_for_missing_label": int(
            removed_count
        ),
        "rows_after_cleaning": int(
            rows_after_cleaning
        ),
        "number_of_local_classes": int(
            len(found_classes)
        ),
        "canonical_class_order": (
            CANONICAL_CLASSES
        ),
        "semantic_to_local_id": (
            semantic_to_local_id
        ),
        "local_id_to_semantic": {
            str(class_id): semantic_label
            for class_id, semantic_label
            in local_id_to_semantic.items()
        },
        "original_label_counts_before_cleaning": {
            str(row["original_label"]): int(
                row["count"]
            )
            for _, row
            in original_distribution.iterrows()
        },
        "mapped_class_counts": {
            str(row["semantic_label"]): int(
                row["count"]
            )
            for _, row
            in mapped_distribution.iterrows()
        },
        "mapped_class_percentages": {
            str(row["semantic_label"]): float(
                row["percentage"]
            )
            for _, row
            in mapped_distribution.iterrows()
        },
        "label_aliases": LABEL_ALIASES,
    }

    with open(
        MAPPING_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            mapping_info,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final output summary
    # --------------------------------------------------------
    print_separator("SAVED FILES")

    print(
        "Original label distribution:\n"
        f"{DISTRIBUTION_CSV}"
    )

    print(
        "\nMapped label distribution:\n"
        f"{MAPPED_DISTRIBUTION_CSV}"
    )

    print(
        "\nMapped dataset:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        "\nMapping metadata:\n"
        f"{MAPPING_JSON}"
    )

    print_separator("FINAL SUMMARY")

    print(f"Original rows: {rows_before_cleaning:,}")
    print(f"Removed rows:  {removed_count:,}")
    print(f"Saved rows:    {len(df):,}")
    print(
        "Saved classes: "
        f"{df['semantic_label'].nunique()}"
    )

    print("\nMapping completed successfully.")


if __name__ == "__main__":
    main()