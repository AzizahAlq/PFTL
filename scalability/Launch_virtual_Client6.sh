#!/bin/bash

# Must match NUM_VIRTUAL_CLIENTS inside the Python file.
NUM_VIRTUAL_CLIENTS=9

CLIENT_SCRIPT="/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/client6_IDAD_2024_PFTL_virtual.py"

TOTAL_EXPERIMENT_CLIENTS=$((6 * NUM_VIRTUAL_CLIENTS))

LAUNCH_LOG_DIR="/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/scalability_launcher_logs/clients_${TOTAL_EXPERIMENT_CLIENTS}/client6_idad_2024"

mkdir -p "$LAUNCH_LOG_DIR"

if [[ ! -f "$CLIENT_SCRIPT" ]]; then
    echo "[ERROR] Client 6 script not found:"
    echo "$CLIENT_SCRIPT"
    exit 1
fi

case "$NUM_VIRTUAL_CLIENTS" in
    1|5|9|15|20)
        ;;
    *)
        echo "[ERROR] NUM_VIRTUAL_CLIENTS must be 1, 5, 9, 15, or 20."
        exit 1
        ;;
esac

PIDS=()
CLIENT_IDS=()

echo "============================================================"
echo "Launching Client 6 IDAD-2024 virtual clients"
echo "Virtual clients          : $NUM_VIRTUAL_CLIENTS"
echo "Expected federation size : $TOTAL_EXPERIMENT_CLIENTS"
echo "Client script            : $CLIENT_SCRIPT"
echo "Launcher logs            : $LAUNCH_LOG_DIR"
echo "============================================================"

for ((i=0; i<NUM_VIRTUAL_CLIENTS; i++))
do
    CLIENT_NUMBER=$((i + 1))
    CLIENT_ID="client6_idad_2024_v${CLIENT_NUMBER}"

    OUTPUT_FILE="${LAUNCH_LOG_DIR}/${CLIENT_ID}.out"

    echo "Starting ${CLIENT_ID} with virtual index ${i}"

    python3.10 "$CLIENT_SCRIPT" "$i" \
        > "$OUTPUT_FILE" 2>&1 &

    PIDS+=("$!")
    CLIENT_IDS+=("$CLIENT_ID")

    sleep 0.1
done

FAILED=0

for index in "${!PIDS[@]}"
do
    PID="${PIDS[$index]}"
    CLIENT_ID="${CLIENT_IDS[$index]}"

    if wait "$PID"; then
        echo "[OK] ${CLIENT_ID} finished."
    else
        EXIT_CODE=$?
        echo "[ERROR] ${CLIENT_ID} failed with exit code ${EXIT_CODE}."
        FAILED=$((FAILED + 1))
    fi
done

if [[ "$FAILED" -gt 0 ]]; then
    echo "${FAILED} Client 6 virtual process(es) failed."
    exit 1
fi

echo "All ${NUM_VIRTUAL_CLIENTS} Client 6 virtual clients finished."