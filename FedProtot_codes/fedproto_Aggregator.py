#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

"""
FedProto Aggregator — Strict Six-Client Barrier
================================================

This server is compatible with:
    fedproto_client1.py
    fedproto_client2.py
    fedproto_client3.py
    fedproto_client4.py
    fedproto_client5.py
    fedproto_client6.py

All settings are defined inside this file.

Strict synchronization:
- The server accepts an update only when:
      client_round == server_round
- Each client contributes at most once per round.
- The server advances only after all six clients contribute.
- Clients wait at the barrier until server_round increases.

FedProto aggregation:
- Clients send class prototypes keyed by global semantic class ID.
- Prototypes are aggregated separately for each global class.
- Aggregation is weighted by the number of samples belonging to that class:
      p_global,c = sum_k(n_k,c * p_k,c) / sum_k(n_k,c)
- Classes available on only one client are preserved unchanged for that round.
- A class absent from all updates in a round retains its previous global prototype.

Expected client payload:
{
    "client_id": "client1",
    "round": 0,
    "proto_dim": 8,
    "prototypes": {
        global_id: {
            "semantic_label": "BENIGN",
            "prototype": np.ndarray(shape=(8,), dtype=float32),
            "count": 1000,
            "local_id": 0
        }
    }
}
"""

import os
import time
import csv
import json
import pickle
import threading
from pathlib import Path
from datetime import datetime
from concurrent import futures
from collections import defaultdict

import numpy as np
import grpc

import myproto_pb2
import myproto_pb2_grpc


# ============================================================
# Fixed server configuration
# ============================================================

SERVER_ADDRESS = "0.0.0.0:50052"

EXPECTED_CLIENTS = {
    "client1",
    "client2",
    "client3",
    "client4",
    "client5",
    "client6",
}

MIN_CLIENTS_TO_AGG = 6
PROTO_DIM = 8
MAX_GRPC_MSG = 50 * 1024 * 1024
MAX_WORKERS = 32

BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "M_PFTL_codes"
)

OUT_DIR = (
    BASE_DIR
    / "fedproto_server_logs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_MAPPING_PATH = (
    BASE_DIR
    / "FedProto_Global_Mapping"
    / "global_semantic_labels.json"
)

AGG_LOG = OUT_DIR / "server_fedproto_aggregation_log.csv"
UPDATE_LOG = OUT_DIR / "server_fedproto_update_log.csv"
PULL_LOG = OUT_DIR / "server_fedproto_pull_log.csv"
CLASS_LOG = OUT_DIR / "server_fedproto_class_log.csv"

LATEST_PKL = OUT_DIR / "global_prototypes_latest.pkl"
LATEST_JSON = OUT_DIR / "global_prototypes_latest_summary.json"

SEED = 190
np.random.seed(SEED)


# ============================================================
# Utilities
# ============================================================

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_csv(path: Path, header: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="") as file:
            csv.writer(file).writerow(header)


def safe_client_id(context) -> str:
    metadata = dict(context.invocation_metadata())
    return str(metadata.get("client_id", "")).strip()


def validate_vector(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)

    if array.ndim != 1:
        raise ValueError(
            f"Prototype must be one-dimensional; got shape {array.shape}."
        )

    if array.shape[0] != PROTO_DIM:
        raise ValueError(
            f"Prototype dimension mismatch: expected {PROTO_DIM}, "
            f"got {array.shape[0]}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("Prototype contains NaN or infinite values.")

    return array


def load_global_semantic_dictionary() -> tuple[dict[int, str], dict[str, int]]:
    if not GLOBAL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Global semantic dictionary not found:\n{GLOBAL_MAPPING_PATH}"
        )

    with GLOBAL_MAPPING_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    raw_id_to_semantic = data.get("global_id_to_semantic", {})
    raw_semantic_to_id = data.get("semantic_to_global_id", {})

    id_to_semantic = {
        int(global_id): str(label)
        for global_id, label in raw_id_to_semantic.items()
    }

    semantic_to_id = {
        str(label): int(global_id)
        for label, global_id in raw_semantic_to_id.items()
    }

    if not id_to_semantic or not semantic_to_id:
        raise ValueError(
            "Global semantic dictionary is missing the expected mappings."
        )

    return id_to_semantic, semantic_to_id


def save_latest(
    round_id: int,
    global_prototypes: dict,
) -> None:
    obj = {
        "round": int(round_id),
        "proto_dim": PROTO_DIM,
        "global_prototypes": global_prototypes,
        "timestamp": timestamp(),
    }

    with LATEST_PKL.open("wb") as file:
        pickle.dump(
            obj,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    summary = {
        "round": int(round_id),
        "proto_dim": PROTO_DIM,
        "number_of_global_prototypes": len(global_prototypes),
        "classes": {
            str(global_id): {
                "semantic_label": entry["semantic_label"],
                "count": int(entry.get("count", 0)),
                "contributors": list(entry.get("contributors", [])),
                "prototype_l2_norm": float(
                    np.linalg.norm(
                        np.asarray(entry["prototype"], dtype=np.float32)
                    )
                ),
            }
            for global_id, entry in sorted(global_prototypes.items())
        },
        "timestamp": timestamp(),
    }

    with LATEST_JSON.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_round_snapshot(
    round_id: int,
    global_prototypes: dict,
) -> Path:
    path = OUT_DIR / f"global_prototypes_round_{round_id:03d}.pkl"

    obj = {
        "round": int(round_id),
        "proto_dim": PROTO_DIM,
        "global_prototypes": global_prototypes,
        "timestamp": timestamp(),
    }

    with path.open("wb") as file:
        pickle.dump(
            obj,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return path


# ============================================================
# FedProto aggregator
# ============================================================

class FedProtoAggregator(
    myproto_pb2_grpc.AggregatorServicer
):
    def __init__(self):
        self.lock = threading.Lock()

        self.id_to_semantic, self.semantic_to_id = (
            load_global_semantic_dictionary()
        )

        self.current_round = 0

        # Returned to clients:
        # {
        #   global_id: {
        #       semantic_label,
        #       prototype,
        #       count,
        #       contributors
        #   }
        # }
        self.global_prototypes = {}

        # Strict per-round state
        self.seen_clients = set()
        self.client_payloads = {}

        self.bytes_received_round = 0
        self.total_client_samples_round = 0
        self.round_start_time = None
        self.last_bytes_sent = 0

        ensure_csv(
            AGG_LOG,
            [
                "server_round_after_aggregation",
                "clients_used",
                "num_clients_used",
                "total_client_training_samples",
                "number_of_global_prototypes",
                "number_of_shared_classes",
                "number_of_single_contributor_classes",
                "bytes_received_round",
                "bytes_sent_last_pull",
                "aggregation_time_sec",
                "round_duration_sec",
                "snapshot_file",
                "timestamp",
            ],
        )

        ensure_csv(
            UPDATE_LOG,
            [
                "server_round",
                "client_id",
                "client_round",
                "client_training_samples",
                "prototype_classes_received",
                "prototype_values_received",
                "bytes_received",
                "accepted",
                "reason",
                "timestamp",
            ],
        )

        ensure_csv(
            PULL_LOG,
            [
                "server_round",
                "client_id",
                "global_prototype_classes_sent",
                "prototype_values_sent",
                "bytes_sent",
                "timestamp",
            ],
        )

        ensure_csv(
            CLASS_LOG,
            [
                "server_round_after_aggregation",
                "global_id",
                "semantic_label",
                "contributors",
                "number_of_contributors",
                "total_class_samples",
                "prototype_l2_norm",
                "retained_from_previous_round",
                "timestamp",
            ],
        )

        print("=" * 80)
        print("FEDPROTO STRICT-BARRIER AGGREGATOR")
        print("=" * 80)
        print(f"Address:                 {SERVER_ADDRESS}")
        print(f"Expected clients:        {sorted(EXPECTED_CLIENTS)}")
        print(f"Clients required/round:  {MIN_CLIENTS_TO_AGG}")
        print(f"Prototype dimension:     {PROTO_DIM}")
        print(f"Global semantic labels:  {len(self.id_to_semantic)}")
        print(f"Output directory:        {OUT_DIR}")
        print("Aggregation: class-count-weighted prototype mean")
        print("Initial global prototypes: empty")
        print("=" * 80)

    # --------------------------------------------------------
    # State and logging
    # --------------------------------------------------------

    def _reset_round_state(self) -> None:
        self.seen_clients.clear()
        self.client_payloads.clear()
        self.bytes_received_round = 0
        self.total_client_samples_round = 0
        self.round_start_time = None

    def _log_update(
        self,
        client_id: str,
        client_round: int,
        num_samples: int,
        num_classes: int,
        bytes_received: int,
        accepted: bool,
        reason: str,
    ) -> None:
        with UPDATE_LOG.open("a", newline="") as file:
            csv.writer(file).writerow([
                self.current_round,
                client_id,
                client_round,
                num_samples,
                num_classes,
                num_classes * PROTO_DIM,
                bytes_received,
                int(accepted),
                reason,
                timestamp(),
            ])

    # --------------------------------------------------------
    # Payload validation
    # --------------------------------------------------------

    def _parse_update(
        self,
        request,
        metadata_client_id: str,
    ) -> tuple[dict, int, int]:
        try:
            payload = pickle.loads(request.weights)
        except Exception as exc:
            raise ValueError(
                f"Unable to unpickle prototype payload: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("Prototype payload must be a dictionary.")

        payload_client_id = str(
            payload.get("client_id", metadata_client_id)
        ).strip()

        if not payload_client_id:
            raise ValueError("Missing client_id.")

        if payload_client_id != metadata_client_id:
            raise ValueError(
                "client_id mismatch between metadata and payload."
            )

        payload_round = int(
            payload.get("round", request.round)
        )

        payload_proto_dim = int(
            payload.get("proto_dim", -1)
        )

        if payload_proto_dim != PROTO_DIM:
            raise ValueError(
                f"proto_dim mismatch: expected {PROTO_DIM}, "
                f"got {payload_proto_dim}."
            )

        raw_prototypes = payload.get("prototypes")

        if not isinstance(raw_prototypes, dict):
            raise ValueError(
                "Payload field 'prototypes' must be a dictionary."
            )

        parsed = {}

        for global_id_raw, entry in raw_prototypes.items():
            global_id = int(global_id_raw)

            if global_id not in self.id_to_semantic:
                raise ValueError(
                    f"Unknown global semantic ID: {global_id}."
                )

            if not isinstance(entry, dict):
                raise ValueError(
                    f"Prototype entry for global ID {global_id} "
                    "must be a dictionary."
                )

            semantic_label = str(
                entry.get(
                    "semantic_label",
                    self.id_to_semantic[global_id],
                )
            ).strip()

            expected_semantic = self.id_to_semantic[global_id]

            if semantic_label != expected_semantic:
                raise ValueError(
                    f"Semantic mismatch for global ID {global_id}: "
                    f"expected '{expected_semantic}', "
                    f"got '{semantic_label}'."
                )

            class_count = int(entry.get("count", 0))

            if class_count <= 0:
                raise ValueError(
                    f"Class count for global ID {global_id} "
                    "must be positive."
                )

            prototype = validate_vector(
                entry.get("prototype")
            )

            parsed[global_id] = {
                "semantic_label": semantic_label,
                "prototype": prototype,
                "count": class_count,
                "local_id": int(entry.get("local_id", -1)),
            }

        if not parsed:
            raise ValueError(
                "The client update contains no valid prototypes."
            )

        request_num_samples = int(request.num_samples)

        if request_num_samples <= 0:
            raise ValueError(
                "request.num_samples must be positive."
            )

        return parsed, payload_round, request_num_samples

    # --------------------------------------------------------
    # Class-wise FedProto aggregation
    # --------------------------------------------------------

    def _aggregate_round(self) -> dict:
        """
        Aggregate each global semantic class independently.

        The weighting uses class counts, not each client's total
        training-set size.
        """

        class_entries = defaultdict(list)

        for client_id, client_data in self.client_payloads.items():
            for global_id, entry in client_data["prototypes"].items():
                class_entries[global_id].append({
                    "client_id": client_id,
                    "prototype": entry["prototype"],
                    "count": int(entry["count"]),
                    "semantic_label": entry["semantic_label"],
                })

        new_global = {}

        # Aggregate classes seen this round.
        for global_id, entries in class_entries.items():
            total_count = sum(entry["count"] for entry in entries)

            weighted_sum = np.zeros(
                PROTO_DIM,
                dtype=np.float64,
            )

            contributors = []

            for entry in entries:
                weighted_sum += (
                    entry["prototype"].astype(np.float64)
                    * float(entry["count"])
                )
                contributors.append(entry["client_id"])

            prototype = (
                weighted_sum / float(total_count)
            ).astype(np.float32)

            new_global[global_id] = {
                "semantic_label": self.id_to_semantic[global_id],
                "prototype": prototype,
                "count": int(total_count),
                "contributors": sorted(contributors),
            }

        # Retain any class absent from every client in this round.
        for global_id, previous_entry in self.global_prototypes.items():
            if global_id not in new_global:
                new_global[global_id] = {
                    "semantic_label": previous_entry["semantic_label"],
                    "prototype": np.asarray(
                        previous_entry["prototype"],
                        dtype=np.float32,
                    ),
                    "count": int(previous_entry.get("count", 0)),
                    "contributors": list(
                        previous_entry.get("contributors", [])
                    ),
                    "retained_from_previous_round": True,
                }

        return new_global

    def _finalize_round(self) -> None:
        aggregation_start = time.perf_counter()

        previous_ids = set(self.global_prototypes)
        new_global = self._aggregate_round()

        aggregation_time = (
            time.perf_counter() - aggregation_start
        )

        self.global_prototypes = new_global
        self.current_round += 1

        snapshot = save_round_snapshot(
            self.current_round,
            self.global_prototypes,
        )

        save_latest(
            self.current_round,
            self.global_prototypes,
        )

        shared_classes = 0
        single_contributor_classes = 0

        for global_id, entry in sorted(
            self.global_prototypes.items()
        ):
            contributors = list(
                entry.get("contributors", [])
            )
            number_of_contributors = len(contributors)

            if number_of_contributors >= 2:
                shared_classes += 1
            elif number_of_contributors == 1:
                single_contributor_classes += 1

            retained = int(
                entry.get(
                    "retained_from_previous_round",
                    False,
                )
                or (
                    global_id in previous_ids
                    and global_id not in {
                        gid
                        for client_data in self.client_payloads.values()
                        for gid in client_data["prototypes"]
                    }
                )
            )

            with CLASS_LOG.open("a", newline="") as file:
                csv.writer(file).writerow([
                    self.current_round,
                    global_id,
                    entry["semantic_label"],
                    ",".join(contributors),
                    number_of_contributors,
                    int(entry.get("count", 0)),
                    f"{float(np.linalg.norm(entry['prototype'])):.8f}",
                    retained,
                    timestamp(),
                ])

        round_duration = (
            time.perf_counter() - self.round_start_time
            if self.round_start_time is not None
            else aggregation_time
        )

        used_clients = sorted(self.seen_clients)

        with AGG_LOG.open("a", newline="") as file:
            csv.writer(file).writerow([
                self.current_round,
                ",".join(used_clients),
                len(used_clients),
                self.total_client_samples_round,
                len(self.global_prototypes),
                shared_classes,
                single_contributor_classes,
                self.bytes_received_round,
                self.last_bytes_sent,
                f"{aggregation_time:.6f}",
                f"{round_duration:.6f}",
                snapshot.name,
                timestamp(),
            ])

        print(
            f"[SERVER] AGGREGATION COMPLETE -> round={self.current_round} | "
            f"clients={len(used_clients)} | "
            f"global_classes={len(self.global_prototypes)} | "
            f"shared_classes={shared_classes} | "
            f"single-client_classes={single_contributor_classes} | "
            f"duration={round_duration:.2f}s"
        )

        self._reset_round_state()

    # --------------------------------------------------------
    # gRPC methods
    # --------------------------------------------------------

    def SendSharedUpdate(self, request, context):
        metadata_client_id = safe_client_id(context)
        bytes_received = len(request.weights)
        request_round = int(request.round)
        request_num_samples = int(request.num_samples)

        if metadata_client_id not in EXPECTED_CLIENTS:
            self._log_update(
                metadata_client_id or "missing",
                request_round,
                request_num_samples,
                0,
                bytes_received,
                False,
                "unknown_or_missing_client_id",
            )
            return myproto_pb2.Ack(ok=False)

        try:
            parsed_prototypes, payload_round, num_samples = (
                self._parse_update(
                    request,
                    metadata_client_id,
                )
            )
        except Exception as exc:
            self._log_update(
                metadata_client_id,
                request_round,
                request_num_samples,
                0,
                bytes_received,
                False,
                f"invalid_payload: {exc}",
            )

            print(
                f"[SERVER] REJECT {metadata_client_id}: {exc}"
            )
            return myproto_pb2.Ack(ok=False)

        with self.lock:
            if request_round != self.current_round:
                self._log_update(
                    metadata_client_id,
                    request_round,
                    num_samples,
                    len(parsed_prototypes),
                    bytes_received,
                    False,
                    "request_round_mismatch",
                )
                return myproto_pb2.Ack(ok=False)

            if payload_round != self.current_round:
                self._log_update(
                    metadata_client_id,
                    payload_round,
                    num_samples,
                    len(parsed_prototypes),
                    bytes_received,
                    False,
                    "payload_round_mismatch",
                )
                return myproto_pb2.Ack(ok=False)

            if metadata_client_id in self.seen_clients:
                # Idempotent duplicate: acknowledge without counting twice.
                self._log_update(
                    metadata_client_id,
                    request_round,
                    num_samples,
                    len(parsed_prototypes),
                    bytes_received,
                    True,
                    "duplicate_already_accepted",
                )
                return myproto_pb2.Ack(ok=True)

            if self.round_start_time is None:
                self.round_start_time = time.perf_counter()

            self.seen_clients.add(metadata_client_id)

            self.client_payloads[metadata_client_id] = {
                "prototypes": parsed_prototypes,
                "num_samples": num_samples,
            }

            self.bytes_received_round += bytes_received
            self.total_client_samples_round += num_samples

            self._log_update(
                metadata_client_id,
                request_round,
                num_samples,
                len(parsed_prototypes),
                bytes_received,
                True,
                "accepted",
            )

            print(
                f"[SERVER] ACCEPT {metadata_client_id} "
                f"round={self.current_round} | "
                f"classes={len(parsed_prototypes)} | "
                f"clients={len(self.seen_clients)}/{MIN_CLIENTS_TO_AGG}"
            )

            if len(self.seen_clients) >= MIN_CLIENTS_TO_AGG:
                missing = EXPECTED_CLIENTS - self.seen_clients

                if missing:
                    print(
                        "[SERVER] Waiting despite threshold because "
                        f"expected clients are missing: {sorted(missing)}"
                    )
                else:
                    self._finalize_round()

            return myproto_pb2.Ack(ok=True)

    def GetSharedWeights(self, request, context):
        client_id = safe_client_id(context) or "unknown"

        with self.lock:
            # Clients' normalizer accepts either the raw mapping or a
            # {"prototypes": mapping} wrapper. Use a wrapper with metadata.
            response_object = {
                "round": int(self.current_round),
                "proto_dim": PROTO_DIM,
                "prototypes": self.global_prototypes,
            }

            payload = pickle.dumps(
                response_object,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

            server_round = int(self.current_round)
            bytes_sent = len(payload)
            self.last_bytes_sent = bytes_sent
            number_of_classes = len(self.global_prototypes)

        with PULL_LOG.open("a", newline="") as file:
            csv.writer(file).writerow([
                server_round,
                client_id,
                number_of_classes,
                number_of_classes * PROTO_DIM,
                bytes_sent,
                timestamp(),
            ])

        # This response class matches the supplied PTFL server/protobuf.
        return myproto_pb2.SharedWeights(
            weights=payload,
            round=server_round,
        )


# ============================================================
# Server entry point
# ============================================================

def serve() -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ),
        options=[
            (
                "grpc.max_send_message_length",
                MAX_GRPC_MSG,
            ),
            (
                "grpc.max_receive_message_length",
                MAX_GRPC_MSG,
            ),
        ],
    )

    servicer = FedProtoAggregator()

    myproto_pb2_grpc.add_AggregatorServicer_to_server(
        servicer,
        server,
    )

    bound_port = server.add_insecure_port(
        SERVER_ADDRESS
    )

    if bound_port == 0:
        raise RuntimeError(
            f"Failed to bind gRPC server to {SERVER_ADDRESS}."
        )

    server.start()

    print(f"[SERVER] FedProto server running on {SERVER_ADDRESS}")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()