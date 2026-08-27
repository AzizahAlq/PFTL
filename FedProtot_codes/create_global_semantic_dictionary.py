#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create the global semantic label dictionary for the six-client
FedProto multi-class IDS experiment.

The script:

1. Reads each client's label-mapping JSON file.
2. Extracts all semantic class names.
3. Creates one stable global ID for every unique semantic class.
4. Records which clients contain each class.
5. Separates shared classes from client-specific classes.
6. Creates local-to-global ID mappings for every client.
7. Saves:
   - global_semantic_labels.json
   - global_semantic_labels.csv
   - shared_semantic_labels.csv
   - client_local_to_global_ids.json

Important:
- Local label IDs remain client-specific.
- Global IDs are used only for prototype exchange.
- Only exactly matching semantic names are treated as the same class.
"""

from pathlib import Path
from collections import defaultdict
import json
import pandas as pd


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "M_PFTL_codes"
)

DATA_DIR = BASE_DIR / "Multi_class_Datasets"

OUTPUT_DIR = BASE_DIR / "FedProto_Global_Mapping"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Client mapping JSON files
# ============================================================
CLIENT_MAPPING_FILES = {
    "client1": (
        DATA_DIR
        / "D1_CIC-ToN-IoT_label_mapping.json"
    ),
    "client2": (
        DATA_DIR
        / "D2_CIC-IoT-2023_label_mapping.json"
    ),
    "client3": (
        DATA_DIR
        / "D3_UNSW-NB15_label_mapping.json"
    ),
    "client4": (
        DATA_DIR
        / "D4_CIC-IDS-2017_label_mapping.json"
    ),
    "client5": (
        DATA_DIR
        / "D5_CIC-BCCC-NRC-2024_label_mapping.json"
    ),
    "client6": (
        DATA_DIR
        / "D6_CIC-IoT-IDAD-2024_label_mapping.json"
    ),
}


# ============================================================
# Output files
# ============================================================
GLOBAL_JSON = (
    OUTPUT_DIR
    / "global_semantic_labels.json"
)

GLOBAL_CSV = (
    OUTPUT_DIR
    / "global_semantic_labels.csv"
)

SHARED_CSV = (
    OUTPUT_DIR
    / "shared_semantic_labels.csv"
)

LOCAL_TO_GLOBAL_JSON = (
    OUTPUT_DIR
    / "client_local_to_global_ids.json"
)

OVERLAP_MATRIX_CSV = (
    OUTPUT_DIR
    / "semantic_label_client_overlap_matrix.csv"
)


def print_separator(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required mapping file was not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_local_mapping(mapping_data: dict) -> dict[int, str]:
    """
    Return:
        {
            local_id: semantic_label
        }

    The mapping scripts save local IDs as string keys in JSON,
    so they are converted back to integers here.
    """
    raw_mapping = mapping_data.get(
        "local_id_to_semantic"
    )

    if raw_mapping is None:
        raise KeyError(
            "The mapping JSON does not contain "
            "'local_id_to_semantic'."
        )

    return {
        int(local_id): str(semantic_label).strip()
        for local_id, semantic_label
        in raw_mapping.items()
    }


def create_stable_global_order(
    semantic_labels: set[str],
) -> list[str]:
    """
    Keep BENIGN as global ID 0, then sort all remaining
    semantic labels alphabetically.

    This makes the global dictionary deterministic.
    """
    labels = set(semantic_labels)

    ordered = []

    if "BENIGN" in labels:
        ordered.append("BENIGN")
        labels.remove("BENIGN")

    ordered.extend(sorted(labels))

    return ordered


def main() -> None:
    # --------------------------------------------------------
    # Load all client mappings
    # --------------------------------------------------------
    client_local_mappings = {}
    client_datasets = {}
    semantic_to_clients = defaultdict(list)
    all_semantic_labels = set()

    print_separator("LOADING CLIENT MAPPING FILES")

    for client_id, mapping_path in (
        CLIENT_MAPPING_FILES.items()
    ):
        mapping_data = load_json(mapping_path)

        local_mapping = extract_local_mapping(
            mapping_data
        )

        client_local_mappings[client_id] = (
            local_mapping
        )

        client_datasets[client_id] = (
            mapping_data.get(
                "dataset",
                "Unknown dataset",
            )
        )

        client_classes = set(
            local_mapping.values()
        )

        if len(client_classes) != len(local_mapping):
            raise ValueError(
                f"{client_id} has duplicate semantic labels "
                "assigned to different local IDs."
            )

        all_semantic_labels.update(
            client_classes
        )

        for semantic_label in sorted(
            client_classes
        ):
            semantic_to_clients[
                semantic_label
            ].append(client_id)

        print(
            f"{client_id:<8} | "
            f"{client_datasets[client_id]:<25} | "
            f"classes={len(client_classes):2d} | "
            f"{mapping_path.name}"
        )

    # --------------------------------------------------------
    # Create deterministic global IDs
    # --------------------------------------------------------
    global_class_order = (
        create_stable_global_order(
            all_semantic_labels
        )
    )

    semantic_to_global_id = {
        semantic_label: global_id
        for global_id, semantic_label
        in enumerate(global_class_order)
    }

    global_id_to_semantic = {
        global_id: semantic_label
        for semantic_label, global_id
        in semantic_to_global_id.items()
    }

    # --------------------------------------------------------
    # Build local-to-global mappings
    # --------------------------------------------------------
    client_local_to_global = {}

    for client_id, local_mapping in (
        client_local_mappings.items()
    ):
        local_to_global = {}

        for local_id, semantic_label in (
            local_mapping.items()
        ):
            local_to_global[str(local_id)] = {
                "semantic_label": semantic_label,
                "global_id": int(
                    semantic_to_global_id[
                        semantic_label
                    ]
                ),
            }

        client_local_to_global[
            client_id
        ] = {
            "dataset": client_datasets[
                client_id
            ],
            "number_of_local_classes": len(
                local_mapping
            ),
            "local_to_global": local_to_global,
        }

    # --------------------------------------------------------
    # Build global table
    # --------------------------------------------------------
    global_rows = []

    for global_id, semantic_label in (
        global_id_to_semantic.items()
    ):
        clients = sorted(
            semantic_to_clients[
                semantic_label
            ]
        )

        global_rows.append({
            "global_id": global_id,
            "semantic_label": semantic_label,
            "number_of_clients": len(clients),
            "is_shared": len(clients) >= 2,
            "clients": ", ".join(clients),
        })

    global_df = pd.DataFrame(
        global_rows
    ).sort_values("global_id")

    shared_df = (
        global_df[
            global_df["is_shared"]
        ]
        .copy()
        .sort_values(
            [
                "number_of_clients",
                "global_id",
            ],
            ascending=[False, True],
        )
    )

    unique_df = (
        global_df[
            ~global_df["is_shared"]
        ]
        .copy()
        .sort_values("global_id")
    )

    # --------------------------------------------------------
    # Build overlap matrix
    # --------------------------------------------------------
    client_ids = list(
        CLIENT_MAPPING_FILES.keys()
    )

    overlap_rows = []

    for semantic_label in global_class_order:
        row = {
            "global_id": semantic_to_global_id[
                semantic_label
            ],
            "semantic_label": semantic_label,
        }

        present_clients = set(
            semantic_to_clients[
                semantic_label
            ]
        )

        for client_id in client_ids:
            row[client_id] = int(
                client_id in present_clients
            )

        row["number_of_clients"] = len(
            present_clients
        )

        overlap_rows.append(row)

    overlap_df = pd.DataFrame(
        overlap_rows
    )

    # --------------------------------------------------------
    # Save JSON dictionary
    # --------------------------------------------------------
    global_dictionary = {
        "description": (
            "Global semantic label dictionary for the "
            "six-client FedProto heterogeneous "
            "multi-class IDS experiment."
        ),
        "number_of_clients": len(
            CLIENT_MAPPING_FILES
        ),
        "number_of_global_classes": len(
            global_class_order
        ),
        "number_of_shared_classes": int(
            len(shared_df)
        ),
        "number_of_single_client_classes": int(
            len(unique_df)
        ),
        "id_assignment_rule": (
            "BENIGN receives global ID 0; all remaining "
            "semantic labels are sorted alphabetically."
        ),
        "semantic_to_global_id": (
            semantic_to_global_id
        ),
        "global_id_to_semantic": {
            str(global_id): semantic_label
            for global_id, semantic_label
            in global_id_to_semantic.items()
        },
        "semantic_label_clients": {
            semantic_label: sorted(clients)
            for semantic_label, clients
            in semantic_to_clients.items()
        },
    }

    with GLOBAL_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            global_dictionary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with LOCAL_TO_GLOBAL_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            client_local_to_global,
            file,
            indent=2,
            ensure_ascii=False,
        )

    global_df.to_csv(
        GLOBAL_CSV,
        index=False,
    )

    shared_df.to_csv(
        SHARED_CSV,
        index=False,
    )

    overlap_df.to_csv(
        OVERLAP_MATRIX_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print_separator(
        "GLOBAL SEMANTIC LABEL DICTIONARY"
    )

    print(
        global_df.to_string(
            index=False
        )
    )

    print_separator(
        "SHARED CLASSES ELIGIBLE FOR "
        "CROSS-CLIENT PROTOTYPE AGGREGATION"
    )

    if shared_df.empty:
        print(
            "No semantic label is shared by "
            "multiple clients."
        )
    else:
        print(
            shared_df[
                [
                    "global_id",
                    "semantic_label",
                    "number_of_clients",
                    "clients",
                ]
            ].to_string(index=False)
        )

    print_separator(
        "SINGLE-CLIENT CLASSES"
    )

    if unique_df.empty:
        print(
            "Every class occurs in at least two clients."
        )
    else:
        print(
            unique_df[
                [
                    "global_id",
                    "semantic_label",
                    "clients",
                ]
            ].to_string(index=False)
        )

    print_separator(
        "CLIENT LOCAL ID TO GLOBAL ID MAPPINGS"
    )

    for client_id in client_ids:
        print(
            f"\n{client_id} — "
            f"{client_datasets[client_id]}"
        )

        local_mapping = (
            client_local_to_global[
                client_id
            ]["local_to_global"]
        )

        for local_id in sorted(
            local_mapping,
            key=lambda value: int(value),
        ):
            entry = local_mapping[
                local_id
            ]

            print(
                f"  local {int(local_id):2d}"
                f" -> global "
                f"{entry['global_id']:2d}"
                f" -> "
                f"{entry['semantic_label']}"
            )

    print_separator("SAVED FILES")

    print(
        f"Global JSON dictionary:\n{GLOBAL_JSON}"
    )

    print(
        f"\nGlobal label table:\n{GLOBAL_CSV}"
    )

    print(
        f"\nShared-label table:\n{SHARED_CSV}"
    )

    print(
        "\nClient local-to-global mappings:\n"
        f"{LOCAL_TO_GLOBAL_JSON}"
    )

    print(
        f"\nClient overlap matrix:\n"
        f"{OVERLAP_MATRIX_CSV}"
    )

    print_separator("FINAL SUMMARY")

    print(
        "Clients:               "
        f"{len(CLIENT_MAPPING_FILES)}"
    )

    print(
        "Global semantic labels:"
        f" {len(global_class_order)}"
    )

    print(
        "Shared labels:         "
        f" {len(shared_df)}"
    )

    print(
        "Single-client labels:  "
        f" {len(unique_df)}"
    )

    print(
        "\nGlobal semantic dictionary "
        "created successfully."
    )


if __name__ == "__main__":
    main()