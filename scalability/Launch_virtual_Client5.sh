#!/bin/bash

NUM_VIRTUAL_CLIENTS=9

CLIENT_SCRIPT="/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/client5_BCCC_NRC_2024_PFTL_virtual.py"

TOTAL_EXPERIMENT_CLIENTS=$((6 * NUM_VIRTUAL_CLIENTS))

LAUNCH_LOG_DIR="/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/scalability_launcher_logs/clients_${TOTAL_EXPERIMENT_CLIENTS}/client5_bccc_nrc_2024"

mkdir -p "$LAUNCH_LOG_DIR"

PIDS=()

for ((i=0; i<NUM_VIRTUAL_CLIENTS; i++))
do
    CLIENT_NUMBER=$((i + 1))
    CLIENT_ID="client5_bccc_nrc_2024_v${CLIENT_NUMBER}"

    python3.10 "$CLIENT_SCRIPT" "$i" \
        > "${LAUNCH_LOG_DIR}/${CLIENT_ID}.out" 2>&1 &

    PIDS+=("$!")

    sleep 0.1
done

for PID in "${PIDS[@]}"
do
    wait "$PID"
done

echo "All ${NUM_VIRTUAL_CLIENTS} Client 5 virtual clients finished."