#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Client 5 label inspection and FedProto mapping
Dataset: CIC-BCCC-NRC 2024

This script:
1. Loads the original CSV.
2. Detects the multi-class label column.
3. Prints the original label distribution and percentages.
4. Removes rows with missing labels.
5. Maps the 15 original labels to standardized semantic labels.
6. Creates fixed local numeric IDs.
7. Saves:
   - mapped dataset
   - original distribution
   - mapped distribution
   - mapping metadata JSON
   - removed missing-label rows, if any
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
    / "D5_CIC-BCCC-NRC_2024_Balanced_Final.csv"
)

OUTPUT_CSV = (
    DATA_DIR
    / "D5_CIC-BCCC-NRC-2024_FedProto_mapped.csv"
)

ORIGINAL_DISTRIBUTION_CSV = (
    DATA_DIR
    / "D5_CIC-BCCC-NRC-2024_original_label_distribution.csv"
)

MAPPED_DISTRIBUTION_CSV = (
    DATA_DIR
    / "D5_CIC-BCCC-NRC-2024_mapped_label_distribution.csv"
)

MAPPING_JSON = (
    DATA_DIR
    / "D5_CIC-BCCC-NRC-2024_label_mapping.json"
)

REMOVED_ROWS_CSV = (
    DATA_DIR
    / "D5_CIC-BCCC-NRC-2024_removed_missing_label_rows.csv"
)


# ============================================================
# Fixed local class order for Client 5
# ============================================================
CANONICAL_CLASSES = [
    "BENIGN",
    "DDOS_ICMP_FLOOD",
    "DDOS_UDP_FLOOD",
    "DOS_ICMP_FLOOD",
    "DOS_TCP_FLOOD",
    "DOS_UDP_FLOOD",
    "MITM_ARP_SPOOFING",
    "MQTT_DDOS_PUBLISH_FLOOD",
    "MQTT_DOS_CONNECT_FLOOD",
    "MQTT_DOS_PUBLISH_FLOOD",
    "MQTT_MALFORMED",
    "RECON_OS_SCAN",
    "RECON_PING_SWEEP",
    "RECON_PORT_SCAN",
    "VULNERABILITY_SCAN",
]


# ============================================================
# Source-label aliases -> semantic FedProto labels
# ============================================================
LABEL_ALIASES = {
    # Benign
    "benign traffic": "BENIGN",
    "benign": "BENIGN",
    "normal": "BENIGN",

    # DDoS
    "ddos icmp flood": "DDOS_ICMP_FLOOD",
    "ddos icmpflood": "DDOS_ICMP_FLOOD",

    "ddos udp flood": "DDOS_UDP_FLOOD",
    "ddos udpflood": "DDOS_UDP_FLOOD",

    # DoS
    "dos icmp flood": "DOS_ICMP_FLOOD",
    "dos icmpflood": "DOS_ICMP_FLOOD",

    "dos tcp flood": "DOS_TCP_FLOOD",
    "dos tcpflood": "DOS_TCP_FLOOD",

    "dos udp flood": "DOS_UDP_FLOOD",
    "dos udpflood": "DOS_UDP_FLOOD",

    # MITM
    "mitm arp spoofing": "MITM_ARP_SPOOFING",
    "mitm arpspoofing": "MITM_ARP_SPOOFING",
    "mitm arp spoof": "MITM_ARP_SPOOFING",

    # MQTT DDoS
    "mqtt ddos publish flood": "MQTT_DDOS_PUBLISH_FLOOD",
    "mqtt ddos publishflood": "MQTT_DDOS_PUBLISH_FLOOD",

    # MQTT DoS
    "mqtt dos connect flood": "MQTT_DOS_CONNECT_FLOOD",
    "mqtt dos connectflood": "MQTT_DOS_CONNECT_FLOOD",

    "mqtt dos publish flood": "MQTT_DOS_PUBLISH_FLOOD",
    "mqtt dos publishflood": "MQTT_DOS_PUBLISH_FLOOD",

    # MQTT malformed
    "mqtt malformed": "MQTT_MALFORMED",
    "mqttmalformed": "MQTT_MALFORMED",

    # Recon
    "recon os scan": "RECON_OS_SCAN",
    "recon osscan": "RECON_OS_SCAN",

    "recon ping sweep": "RECON_PING_SWEEP",
    "recon pingsweep": "RECON_PING_SWEEP",

    "recon port scan": "RECON_PORT_SCAN",
    "recon portscan": "RECON_PORT_SCAN",

    "recon vulnerability scan": "VULNERABILITY_SCAN",
    "recon vulnerabilityscan": "VULNERABILITY_SCAN",
    "vulnerability scan": "VULNERABILITY_SCAN",
    "vulnerabilityscan": "VULNERABILITY_SCAN",
}


# ============================================================
# Helpers
# ============================================================
def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_text(value: object) -> str:
    """
    Normalize source label text before alias matching.
    """
    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    text = " ".join(text.split())

    return text


def find_label_column(df: pd.DataFrame) -> str:
    """
    Detect the multi-class label column.
    """
    preferred_columns = [
        "Label",
        "label",
        "Attack",
        "attack",
        "Attack Type",
        "attack_type",
        "Category",
        "category",
        "Class",
        "class",
        "Type",
        "type",
    ]

    for column in preferred_columns:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not detect the multi-class label column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def build_distribution(
    series: pd.Series,
    label_name: str,
) -> pd.DataFrame:
    """
    Build label counts and percentages.
    """
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


def remove_missing_labels(
    df: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove missing and empty labels.
    """
    cleaned_df = df.copy()

    missing_strings = [
        "",
        " ",
        "nan",
        "NaN",
        "none",
        "None",
        "NONE",
        "null",
        "NULL",
        "<MISSING>",
    ]

    cleaned_df[label_column] = (
        cleaned_df[label_column]
        .replace(missing_strings, pd.NA)
    )

    whitespace_only = (
        cleaned_df[label_column]
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
    )

    cleaned_df.loc[
        whitespace_only,
        label_column,
    ] = pd.NA

    missing_mask = cleaned_df[label_column].isna()

    removed_rows = cleaned_df.loc[missing_mask].copy()
    cleaned_df = cleaned_df.loc[~missing_mask].copy()

    return cleaned_df, removed_rows


# ============================================================
# Main
# ============================================================
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

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        print(f"{index:3d}. {column}")

    # ========================================================
    # Detect label column
    # ========================================================
    label_column = find_label_column(df)

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

    print_separator("ORIGINAL UNIQUE LABELS")

    unique_labels = sorted(
        df[label_column]
        .fillna("<MISSING>")
        .astype(str)
        .unique()
        .tolist()
    )

    for index, label in enumerate(
        unique_labels,
        start=1,
    ):
        print(f"{index:3d}. {label}")

    original_distribution.to_csv(
        ORIGINAL_DISTRIBUTION_CSV,
        index=False,
    )

    # ========================================================
    # Remove missing labels
    # ========================================================
    rows_before = len(df)

    df, removed_rows = remove_missing_labels(
        df,
        label_column,
    )

    rows_after = len(df)
    removed_count = len(removed_rows)

    print_separator("MISSING-LABEL CLEANING")

    print(f"Rows before cleaning: {rows_before:,}")
    print(f"Rows removed:         {removed_count:,}")
    print(f"Rows remaining:       {rows_after:,}")

    if removed_count > 0:
        removed_rows.to_csv(
            REMOVED_ROWS_CSV,
            index=False,
        )

        print(
            "\nRemoved rows saved to:\n"
            f"{REMOVED_ROWS_CSV}"
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

    # ========================================================
    # Check unmapped labels
    # ========================================================
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

        print(
            "\nThe mapped dataset was not saved."
        )

        print(
            "\nAdd the exact normalized labels above "
            "to LABEL_ALIASES and rerun the script."
        )

        raise ValueError(
            "Mapping stopped because unmapped labels were found."
        )

    # ========================================================
    # Validate semantic classes
    # ========================================================
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
        print_separator("WARNING: EXPECTED CLASSES NOT FOUND")

        for semantic_label in missing_expected_classes:
            print(f"- {semantic_label}")

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

    df = df[
        first_columns
        + remaining_columns
    ]

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

    print(
        mapped_distribution
        .to_string(index=False)
    )

    print_separator("CLIENT 5 LOCAL LABEL MAPPING")

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
            f"{semantic_label:<30} "
            f"count={class_count:,}"
        )

    # ========================================================
    # Consistency checks
    # ========================================================
    if df["semantic_label"].isna().any():
        raise ValueError(
            "semantic_label contains missing values."
        )

    if df["local_label_id"].isna().any():
        raise ValueError(
            "local_label_id contains missing values."
        )

    if mapped_distribution["count"].sum() != len(df):
        raise ValueError(
            "Mapped class counts do not equal dataset rows."
        )

    # ========================================================
    # Save outputs
    # ========================================================
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    mapped_distribution.to_csv(
        MAPPED_DISTRIBUTION_CSV,
        index=False,
    )

    mapping_info = {
        "client_id": "client5",
        "dataset": "CIC-BCCC-NRC-2024",
        "source_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "source_label_column": label_column,
        "rows_before_cleaning": int(rows_before),
        "rows_removed_for_missing_label": int(
            removed_count
        ),
        "rows_after_cleaning": int(rows_after),
        "number_of_local_classes": int(
            df["semantic_label"].nunique()
        ),
        "canonical_class_order": CANONICAL_CLASSES,
        "semantic_to_local_id": semantic_to_local_id,
        "local_id_to_semantic": {
            str(class_id): semantic_label
            for class_id, semantic_label
            in local_id_to_semantic.items()
        },
        "original_label_counts": {
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

    # ========================================================
    # Final output
    # ========================================================
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

    print(f"Original rows: {rows_before:,}")
    print(f"Removed rows:  {removed_count:,}")
    print(f"Saved rows:    {len(df):,}")
    print(
        f"Saved classes: "
        f"{df['semantic_label'].nunique()}"
    )

    print("\nClient 5 mapping completed successfully.")


if __name__ == "__main__":
    main()