#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Client 3 label inspection and FedProto mapping
Dataset: UNSW-NB15

This script:
1. Loads the original CSV.
2. Uses attack_cat as the multi-class label column.
3. Prints the original class distribution and percentages.
4. Maps the six classes to standardized FedProto semantic labels.
5. Creates fixed local numeric label IDs.
6. Saves:
   - mapped dataset CSV
   - original distribution CSV
   - mapped distribution CSV
   - mapping JSON
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# Paths
# ============================================================
DATA_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "M_PFTL_codes/Multi_class_Datasets"
)

INPUT_CSV = (
    DATA_DIR
    / "D3_UNSW_NB15_ALL_MERGED_final.csv"
)

OUTPUT_CSV = (
    DATA_DIR
    / "D3_UNSW-NB15_FedProto_mapped.csv"
)

ORIGINAL_DISTRIBUTION_CSV = (
    DATA_DIR
    / "D3_UNSW-NB15_original_label_distribution.csv"
)

MAPPED_DISTRIBUTION_CSV = (
    DATA_DIR
    / "D3_UNSW-NB15_mapped_label_distribution.csv"
)

MAPPING_JSON = (
    DATA_DIR
    / "D3_UNSW-NB15_label_mapping.json"
)


# ============================================================
# Fixed local class order for Client 3
# ============================================================
CANONICAL_CLASSES = [
    "BENIGN",
    "EXPLOITS",
    "FUZZERS",
    "RECONNAISSANCE",
    "DOS",
    "GENERIC",
]


# ============================================================
# Source label aliases -> FedProto semantic labels
# ============================================================
LABEL_ALIASES = {
    "normal": "BENIGN",
    "benign": "BENIGN",

    "exploits": "EXPLOITS",
    "exploit": "EXPLOITS",

    "fuzzers": "FUZZERS",
    "fuzzer": "FUZZERS",

    "reconnaissance": "RECONNAISSANCE",
    "recon": "RECONNAISSANCE",

    "dos": "DOS",
    "denial of service": "DOS",

    "generic": "GENERIC",
}


def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = " ".join(text.split())
    return text


def build_distribution(
    series: pd.Series,
    label_name: str,
) -> pd.DataFrame:
    cleaned = (
        series
        .fillna("<MISSING>")
        .astype(str)
        .str.strip()
    )

    counts = cleaned.value_counts(dropna=False)

    percentages = (
        cleaned
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

    return pd.DataFrame({
        label_name: counts.index,
        "count": counts.values,
        "percentage": [
            float(percentages[label])
            for label in counts.index
        ],
    })


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input dataset was not found:\n{INPUT_CSV}"
        )

    print(f"Loading dataset:\n{INPUT_CSV}")

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    df.columns = df.columns.str.strip()

    print_separator("DATASET INFORMATION")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumn names:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:3d}. {column}")

    label_column = "attack_cat"

    if label_column not in df.columns:
        raise ValueError(
            f"Expected label column '{label_column}' was not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    print_separator("DETECTED LABEL COLUMN")
    print(label_column)

    # ========================================================
    # Original distribution
    # ========================================================
    original_distribution = build_distribution(
        df[label_column],
        "original_label",
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

    # ========================================================
    # Missing-label check
    # ========================================================
    missing_mask = (
        df[label_column].isna()
        | df[label_column].astype(str).str.strip().eq("")
    )

    missing_count = int(missing_mask.sum())

    print_separator("MISSING-LABEL CHECK")
    print(f"Missing or empty labels: {missing_count:,}")

    if missing_count > 0:
        removed_rows_path = (
            DATA_DIR
            / "D3_UNSW-NB15_removed_missing_label_rows.csv"
        )

        df.loc[missing_mask].to_csv(
            removed_rows_path,
            index=False,
        )

        df = df.loc[~missing_mask].copy()

        print(
            "\nRemoved rows saved to:\n"
            f"{removed_rows_path}"
        )

    # ========================================================
    # Map labels
    # ========================================================
    normalized_labels = (
        df[label_column]
        .map(normalize_text)
    )

    df["semantic_label"] = (
        normalized_labels
        .map(LABEL_ALIASES)
    )

    unmapped_mask = df["semantic_label"].isna()

    if unmapped_mask.any():
        unmapped_summary = (
            df.loc[
                unmapped_mask,
                label_column,
            ]
            .astype(str)
            .value_counts()
        )

        print_separator("ERROR: UNMAPPED LABELS FOUND")
        print(unmapped_summary.to_string())

        raise ValueError(
            "Mapping stopped because unmapped labels were found."
        )

    # ========================================================
    # Create local numeric IDs
    # ========================================================
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

    df["original_label"] = (
        df[label_column]
        .astype(str)
    )

    # ========================================================
    # Reorder columns
    # ========================================================
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

    df = df[first_columns + remaining_columns]

    # ========================================================
    # Mapped distribution
    # ========================================================
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
    print(mapped_distribution.to_string(index=False))

    print_separator("CLIENT 3 LOCAL LABEL MAPPING")

    for class_id, semantic_label in (
        local_id_to_semantic.items()
    ):
        class_count = int(
            (df["local_label_id"] == class_id).sum()
        )

        print(
            f"{class_id:2d} -> "
            f"{semantic_label:<18} "
            f"count={class_count:,}"
        )

    # ========================================================
    # Save outputs
    # ========================================================
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    original_distribution.to_csv(
        ORIGINAL_DISTRIBUTION_CSV,
        index=False,
    )

    mapped_distribution.to_csv(
        MAPPED_DISTRIBUTION_CSV,
        index=False,
    )

    mapping_info = {
        "client_id": "client3",
        "dataset": "UNSW-NB15",
        "source_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "source_label_column": label_column,
        "rows_before_cleaning": int(
            len(df) + missing_count
        ),
        "rows_removed_for_missing_label": int(
            missing_count
        ),
        "rows_after_cleaning": int(
            len(df)
        ),
        "number_of_local_classes": int(
            df["semantic_label"].nunique()
        ),
        "canonical_class_order": CANONICAL_CLASSES,
        "semantic_to_local_id": (
            semantic_to_local_id
        ),
        "local_id_to_semantic": {
            str(class_id): semantic_label
            for class_id, semantic_label
            in local_id_to_semantic.items()
        },
        "original_label_counts": {
            str(row["original_label"]): int(row["count"])
            for _, row
            in original_distribution.iterrows()
        },
        "mapped_class_counts": {
            str(row["semantic_label"]): int(row["count"])
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

    print_separator("SAVED FILES")

    print(
        "Original distribution:\n"
        f"{ORIGINAL_DISTRIBUTION_CSV}"
    )

    print(
        "\nMapped distribution:\n"
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

    print(f"Saved rows:    {len(df):,}")
    print(
        f"Saved classes: "
        f"{df['semantic_label'].nunique()}"
    )

    print("\nClient 3 mapping completed successfully.")


if __name__ == "__main__":
    main()