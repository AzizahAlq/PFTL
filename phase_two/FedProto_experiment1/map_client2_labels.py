#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Client 2 label inspection and FedProto semantic mapping
Dataset: CIC-IoT-2023

The script:
1. Loads the original CSV.
2. Detects the multi-class label column.
3. Prints the original class distribution and percentages.
4. Removes rows with missing labels.
5. Maps labels to standardized FedProto semantic labels.
6. Assigns fixed local numeric IDs for Client 2.
7. Saves the mapped dataset, distributions, and mapping metadata.
8. Stops safely if any unexpected label is found.
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
    / "D2_cic_iot_2023_200k_keep_selected_new.csv"
)

OUTPUT_CSV = (
    DATA_DIR
    / "D2_CIC-IoT-2023_FedProto_mapped.csv"
)

ORIGINAL_DISTRIBUTION_CSV = (
    DATA_DIR
    / "D2_CIC-IoT-2023_original_label_distribution.csv"
)

MAPPED_DISTRIBUTION_CSV = (
    DATA_DIR
    / "D2_CIC-IoT-2023_mapped_label_distribution.csv"
)

MAPPING_JSON = (
    DATA_DIR
    / "D2_CIC-IoT-2023_label_mapping.json"
)

REMOVED_ROWS_CSV = (
    DATA_DIR
    / "D2_CIC-IoT-2023_removed_missing_label_rows.csv"
)


# ============================================================
# Fixed local class order for Client 2
# ============================================================
CANONICAL_CLASSES = [
    "DDOS_ICMP_FLOOD",
    "DDOS_UDP_FLOOD",
    "DDOS_TCP_FLOOD",
    "DDOS_PSHACK_FLOOD",
    "DDOS_SYN_FLOOD",
    "DDOS_RSTFIN_FLOOD",
    "DDOS_SYNONYMOUS_IP_FLOOD",
    "DOS_UDP_FLOOD",
    "DOS_TCP_FLOOD",
    "DOS_SYN_FLOOD",
    "DDOS_ICMP_FRAGMENTATION",
    "VULNERABILITY_SCAN",
    "MITM_ARP_SPOOFING",
    "DDOS_ACK_FRAGMENTATION",
    "DDOS_UDP_FRAGMENTATION",
    "BENIGN",
    "MIRAI_GREETH_FLOOD",
    "MIRAI_UDP_PLAIN",
    "MIRAI_GREIP_FLOOD",
    "DNS_SPOOFING",
    "RECON_HOST_DISCOVERY",
    "RECON_OS_SCAN",
    "RECON_PORT_SCAN",
    "DOS_HTTP_FLOOD",
    "DDOS_HTTP_FLOOD",
    "DDOS_SLOWLORIS",
    "DICTIONARY_BRUTE_FORCE",
    "BROWSER_HIJACKING",
    "SQL_INJECTION",
    "COMMAND_INJECTION",
    "XSS",
    "BACKDOOR_MALWARE",
    "RECON_PING_SWEEP",
    "UPLOADING_ATTACK",
]


# ============================================================
# Source-label aliases -> semantic FedProto labels
# ============================================================
LABEL_ALIASES = {
    # DDoS floods
    "ddos icmp flood": "DDOS_ICMP_FLOOD",
    "ddos udp flood": "DDOS_UDP_FLOOD",
    "ddos tcp flood": "DDOS_TCP_FLOOD",
    "ddos pshack flood": "DDOS_PSHACK_FLOOD",
    "ddos psh ack flood": "DDOS_PSHACK_FLOOD",
    "ddos syn flood": "DDOS_SYN_FLOOD",
    "ddos rstfin flood": "DDOS_RSTFIN_FLOOD",
    "ddos rst fin flood": "DDOS_RSTFIN_FLOOD",
    "ddos synonymousip flood": "DDOS_SYNONYMOUS_IP_FLOOD",
    "ddos synonymous ip flood": "DDOS_SYNONYMOUS_IP_FLOOD",

    # DoS floods
    "dos udp flood": "DOS_UDP_FLOOD",
    "dos tcp flood": "DOS_TCP_FLOOD",
    "dos syn flood": "DOS_SYN_FLOOD",
    "dos http flood": "DOS_HTTP_FLOOD",

    # Fragmentation attacks
    "ddos icmp fragmentation": "DDOS_ICMP_FRAGMENTATION",
    "ddos ack fragmentation": "DDOS_ACK_FRAGMENTATION",
    "ddos udp fragmentation": "DDOS_UDP_FRAGMENTATION",

    # Scanning and reconnaissance
    "vulnerability scan": "VULNERABILITY_SCAN",
    "recon vulnerability scan": "VULNERABILITY_SCAN",
    "recon host discovery": "RECON_HOST_DISCOVERY",
    "recon os scan": "RECON_OS_SCAN",
    "recon port scan": "RECON_PORT_SCAN",
    "recon ping sweep": "RECON_PING_SWEEP",

    # MITM
    "mitm arp spoofing": "MITM_ARP_SPOOFING",
    "mitm arp spoof": "MITM_ARP_SPOOFING",

    # Benign
    "benign": "BENIGN",
    "normal": "BENIGN",
    "benign traffic": "BENIGN",

    # Mirai
    "mirai greeth flood": "MIRAI_GREETH_FLOOD",
    "mirai greethflood": "MIRAI_GREETH_FLOOD",
    "mirai udp plain": "MIRAI_UDP_PLAIN",
    "mirai udpplain": "MIRAI_UDP_PLAIN",
    "mirai greip flood": "MIRAI_GREIP_FLOOD",
    "mirai greipflood": "MIRAI_GREIP_FLOOD",

    # Spoofing
    "dns spoofing": "DNS_SPOOFING",

    # HTTP DDoS
    "ddos http flood": "DDOS_HTTP_FLOOD",
    "ddos slowloris": "DDOS_SLOWLORIS",
    "ddos slow loris": "DDOS_SLOWLORIS",

    # Brute force
    "dictionary brute force": "DICTIONARY_BRUTE_FORCE",
    "dictionary bruteforce": "DICTIONARY_BRUTE_FORCE",

    # Web and malware attacks
    "browser hijacking": "BROWSER_HIJACKING",
    "sql injection": "SQL_INJECTION",
    "command injection": "COMMAND_INJECTION",
    "xss": "XSS",
    "cross site scripting": "XSS",
    "backdoor malware": "BACKDOOR_MALWARE",
    "backdoor": "BACKDOOR_MALWARE",
    "uploading attack": "UPLOADING_ATTACK",
}

LABEL_ALIASES.update({
    "ddos rstfinflood": "DDOS_RSTFIN_FLOOD",
    "vulnerabilityscan": "VULNERABILITY_SCAN",
    "mitm arpspoofing": "MITM_ARP_SPOOFING",

    "recon hostdiscovery": "RECON_HOST_DISCOVERY",
    "recon osscan": "RECON_OS_SCAN",
    "recon portscan": "RECON_PORT_SCAN",
    "recon pingsweep": "RECON_PING_SWEEP",

    "dictionarybruteforce": "DICTIONARY_BRUTE_FORCE",
    "browserhijacking": "BROWSER_HIJACKING",
    "sqlinjection": "SQL_INJECTION",
    "commandinjection": "COMMAND_INJECTION",
})
def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_text(value: object) -> str:
    """
    Normalize the source label before matching it against aliases.
    """
    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    text = " ".join(text.split())

    return text


def find_label_column(df: pd.DataFrame) -> str:
    """
    Detect the multi-class attack-label column.
    """
    preferred_columns = [
        "Attack",
        "attack",
        "Label",
        "label",
        "Category",
        "category",
        "Class",
        "class",
        "Type",
        "type",
        "Attack Type",
        "attack_type",
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
    label_column_name: str,
) -> pd.DataFrame:
    """
    Generate label counts and percentages.
    """
    cleaned_series = (
        series
        .fillna("<MISSING>")
        .astype(str)
    )

    counts = cleaned_series.value_counts(dropna=False)

    percentages = (
        cleaned_series
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

    return pd.DataFrame({
        label_column_name: counts.index,
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
    Remove missing and empty labels and return removed rows separately.
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


def main() -> None:
    # ========================================================
    # Load dataset
    # ========================================================
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

    original_labels = sorted(
        df[label_column]
        .fillna("<MISSING>")
        .astype(str)
        .unique()
        .tolist()
    )

    for index, label in enumerate(
        original_labels,
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

    removed_count = len(removed_rows)
    rows_after = len(df)

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
            "\nRemoved rows were saved to:\n"
            f"{REMOVED_ROWS_CSV}"
        )

    # ========================================================
    # Map source labels
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
            "\nAdd the exact source labels above to "
            "LABEL_ALIASES and run the script again."
        )

        raise ValueError(
            "Mapping stopped because unmapped labels were found."
        )

    # ========================================================
    # Validate generated semantic labels
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

        for label in missing_expected_classes:
            print(f"- {label}")

    # ========================================================
    # Create Client 2 local numeric IDs
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

    print(
        mapped_distribution
        .to_string(index=False)
    )

    print_separator("CLIENT 2 LOCAL LABEL MAPPING")

    for class_id, semantic_label in (
        local_id_to_semantic.items()
    ):
        class_count = int(
            (
                df["local_label_id"] == class_id
            ).sum()
        )

        print(
            f"{class_id:2d} -> "
            f"{semantic_label:<30} "
            f"count={class_count:,}"
        )

    # ========================================================
    # Final consistency checks
    # ========================================================
    if df["semantic_label"].isna().any():
        raise ValueError(
            "semantic_label still contains missing values."
        )

    if df["local_label_id"].isna().any():
        raise ValueError(
            "local_label_id still contains missing values."
        )

    if mapped_distribution["count"].sum() != len(df):
        raise ValueError(
            "Mapped distribution count does not match "
            "the saved dataset size."
        )

    if df["semantic_label"].nunique() != len(found_classes):
        raise ValueError(
            "Semantic class-count consistency check failed."
        )

    # ========================================================
    # Save mapped dataset and distribution
    # ========================================================
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    mapped_distribution.to_csv(
        MAPPED_DISTRIBUTION_CSV,
        index=False,
    )

    # ========================================================
    # Save metadata JSON
    # ========================================================
    mapping_info = {
        "client_id": "client2",
        "dataset": "CIC-IoT-2023",
        "source_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "source_label_column": label_column,
        "rows_before_cleaning": int(rows_before),
        "rows_removed_for_missing_label": int(removed_count),
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

    # ========================================================
    # Final output
    # ========================================================
    print_separator("SAVED FILES")

    print(
        "Original label distribution:\n"
        f"{ORIGINAL_DISTRIBUTION_CSV}"
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

    print(f"Original rows: {rows_before:,}")
    print(f"Removed rows:  {removed_count:,}")
    print(f"Saved rows:    {len(df):,}")
    print(
        f"Saved classes: "
        f"{df['semantic_label'].nunique()}"
    )

    print("\nClient 2 mapping completed successfully.")


if __name__ == "__main__":
    main()