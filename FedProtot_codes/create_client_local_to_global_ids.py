#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create client_local_to_global_ids.json for the six-client
FedProto multi-class IDS experiment.

Each client keeps its own local_label_id, while FedProto uses
a common global_id to aggregate prototypes for matching
semantic classes.
"""

from pathlib import Path
import json


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

OUTPUT_JSON = OUTPUT_DIR / "client_local_to_global_ids.json"

GLOBAL_LABEL_JSON = OUTPUT_DIR / "global_semantic_labels.json"


# ============================================================
# Six client mapping files
# ============================================================
CLIENT_MAPPING_FILES = {
    "client1": DATA_DIR / "D1_CIC-ToN-IoT_label_mapping.json",

    "client2": DATA_DIR / "D2_CIC-IoT-2023_label_mapping.json",

    "client3": DATA_DIR / "D3_UNSW-NB15_label_mapping.json",

    "client4": DATA_DIR / "D4_CIC-IDS-2017_label_mapping.json",

    "client5": (
        DATA_DIR
        / "D5_CIC-BCCC-NRC-2024_label_mapping.json"
    ),

    "client6": (
        DATA_DIR
        / "D6_CIC-IoT-IDAD-2024_label_mapping.json"
    ),
}


def load_json(path: Path) -> dict:
    """Load and validate a JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file was not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_local_to_semantic(mapping_data: dict) -> dict[int, str]:
    """
    Extract:
        local ID -> semantic label

    Supports both mapping formats:
      local_id_to_semantic
      semantic_to_local_id
    """

    if "local_id_to_semantic" in mapping_data:
        return {
            int(local_id): str(semantic_label).strip()
            for local_id, semantic_label
            in mapping_data["local_id_to_semantic"].items()
        }

    if "semantic_to_local_id" in mapping_data:
        return {
            int(local_id): str(semantic_label).strip()
            for semantic_label, local_id
            in mapping_data["semantic_to_local_id"].items()
        }

    raise KeyError(
        "Mapping JSON must contain either "
        "'local_id_to_semantic' or "
        "'semantic_to_local_id'."
    )


def create_global_dictionary(
    all_semantic_labels: set[str],
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Assign deterministic global IDs.

    BENIGN is assigned global ID 0.
    Remaining labels are sorted alphabetically.
    """

    remaining_labels = set(all_semantic_labels)
    ordered_labels = []

    if "BENIGN" in remaining_labels:
        ordered_labels.append("BENIGN")
        remaining_labels.remove("BENIGN")

    ordered_labels.extend(
        sorted(remaining_labels)
    )

    semantic_to_global_id = {
        semantic_label: global_id
        for global_id, semantic_label
        in enumerate(ordered_labels)
    }

    global_id_to_semantic = {
        global_id: semantic_label
        for semantic_label, global_id
        in semantic_to_global_id.items()
    }

    return (
        semantic_to_global_id,
        global_id_to_semantic,
    )


def main() -> None:
    print("=" * 90)
    print("LOADING CLIENT LABEL MAPPINGS")
    print("=" * 90)

    client_data = {}
    all_semantic_labels = set()

    # --------------------------------------------------------
    # Load all six client mapping files
    # --------------------------------------------------------
    for client_id, mapping_path in CLIENT_MAPPING_FILES.items():
        mapping_data = load_json(mapping_path)

        local_to_semantic = extract_local_to_semantic(
            mapping_data
        )

        semantic_labels = set(
            local_to_semantic.values()
        )

        if len(semantic_labels) != len(local_to_semantic):
            raise ValueError(
                f"{client_id} contains duplicate semantic labels "
                "assigned to multiple local IDs."
            )

        all_semantic_labels.update(
            semantic_labels
        )

        client_data[client_id] = {
            "dataset": mapping_data.get(
                "dataset",
                "Unknown dataset",
            ),
            "source_mapping_file": str(mapping_path),
            "local_to_semantic": local_to_semantic,
        }

        print(
            f"{client_id:<8} "
            f"dataset={client_data[client_id]['dataset']:<25} "
            f"classes={len(local_to_semantic):2d}"
        )

    # --------------------------------------------------------
    # Load existing global dictionary or recreate it
    # --------------------------------------------------------
    if GLOBAL_LABEL_JSON.exists():
        global_data = load_json(
            GLOBAL_LABEL_JSON
        )

        semantic_to_global_id = {
            str(label): int(global_id)
            for label, global_id
            in global_data[
                "semantic_to_global_id"
            ].items()
        }

        print(
            "\nUsing existing global semantic dictionary:\n"
            f"{GLOBAL_LABEL_JSON}"
        )

        missing_global_labels = sorted(
            all_semantic_labels
            - set(semantic_to_global_id)
        )

        if missing_global_labels:
            raise ValueError(
                "The existing global dictionary is missing "
                "these semantic labels:\n"
                + "\n".join(missing_global_labels)
            )

    else:
        (
            semantic_to_global_id,
            global_id_to_semantic,
        ) = create_global_dictionary(
            all_semantic_labels
        )

        global_data = {
            "number_of_global_classes": len(
                semantic_to_global_id
            ),
            "semantic_to_global_id": (
                semantic_to_global_id
            ),
            "global_id_to_semantic": {
                str(global_id): semantic_label
                for global_id, semantic_label
                in global_id_to_semantic.items()
            },
        }

        with GLOBAL_LABEL_JSON.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                global_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nCreated new global semantic dictionary:\n"
            f"{GLOBAL_LABEL_JSON}"
        )

    # --------------------------------------------------------
    # Create local-to-global mappings
    # --------------------------------------------------------
    output_data = {}

    for client_id, info in client_data.items():
        local_to_global = {}

        for local_id in sorted(
            info["local_to_semantic"]
        ):
            semantic_label = (
                info["local_to_semantic"][
                    local_id
                ]
            )

            global_id = int(
                semantic_to_global_id[
                    semantic_label
                ]
            )

            local_to_global[str(local_id)] = {
                "global_id": global_id,
                "semantic_label": semantic_label,
            }

        output_data[client_id] = {
            "dataset": info["dataset"],
            "number_of_local_classes": len(
                local_to_global
            ),
            "source_mapping_file": (
                info["source_mapping_file"]
            ),
            "local_to_global": local_to_global,
        }

    # --------------------------------------------------------
    # Save final JSON
    # --------------------------------------------------------
    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Display mappings
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("CLIENT LOCAL ID TO GLOBAL ID MAPPINGS")
    print("=" * 90)

    for client_id, client_info in output_data.items():
        print(
            f"\n{client_id} — "
            f"{client_info['dataset']}"
        )

        local_mapping = (
            client_info["local_to_global"]
        )

        for local_id_text in sorted(
            local_mapping,
            key=lambda value: int(value),
        ):
            entry = local_mapping[
                local_id_text
            ]

            print(
                f"  Local {int(local_id_text):2d}"
                f" -> Global {entry['global_id']:2d}"
                f" -> {entry['semantic_label']}"
            )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    print(f"Clients: {len(output_data)}")
    print(
        "Global semantic labels: "
        f"{len(semantic_to_global_id)}"
    )

    for client_id, client_info in output_data.items():
        print(
            f"{client_id}: "
            f"{client_info['number_of_local_classes']} "
            "local classes"
        )

    print("\nSaved file:")
    print(OUTPUT_JSON)

    print(
        "\nclient_local_to_global_ids.json "
        "created successfully."
    )


if __name__ == "__main__":
    main()