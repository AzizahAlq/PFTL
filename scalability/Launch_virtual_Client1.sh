#!/bin/bash

# ============================================================
# Launch virtual clients for:
# Client 1 - CIC-ToN-IoT
#
# Set NUM_VIRTUAL_CLIENTS to:
#   1  -> Client 1 contributes 1 client
#   5  -> Client 1 contributes 5 clients
#   9  -> Client 1 contributes 9 clients
#   15 -> Client 1 contributes 15 clients
#   20 -> Client 1 contributes 20 clients
#
# IMPORTANT:
# NUM_VIRTUAL_CLIENTS here must match NUM_VIRTUAL_CLIENTS
# inside client1_CNN_CIC-ToN-IoT_PFTL_virtual.py
# ============================================================

set -u

# ----------------------------
# Client 1 Python code
# ----------------------------
CLIENT_SCRIPT="/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/client1_CNN_CIC-ToN-IoT_PFTL_virtual.py"

# ----------------------------
# Change only this value
# ----------------------------
NUM_VIRTUAL_CLIENTS=9

# Total federation size when all six original clients use
# the same number of virtual clients:
TOTAL_EXPERIMENT_CLIENTS=$((6 * NUM_VIRTUAL_CLIENTS))

# ----------------------------
# Output directory
# ----------------------------
BASE_OUT_DIR="/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes/scalability_launcher_logs/clients_${TOTAL_EXPERIMENT_CLIENTS}/client1_ton_iot"

mkdir -p "$BASE_OUT_DIR"

# ----------------------------
# Validate Python script
# ----------------------------
if [[ ! -f "$CLIENT_SCRIPT" ]]; then
    echo "[ERROR] Client script not found:"
    echo "$CLIENT_SCRIPT"
    exit 1
fi

# ----------------------------
# Validate client count
# ----------------------------
case "$NUM_VIRTUAL_CLIENTS" in
    1|5|9|15|20)
        ;;
    *)
        echo "[ERROR] NUM_VIRTUAL_CLIENTS must be one of:"
        echo "1, 5, 9, 15, or 20"
        exit 1
        ;;
esac

echo "============================================================"
echo "Launching Client 1 CIC-ToN-IoT virtual clients"
echo "Virtual clients for Client 1 : $NUM_VIRTUAL_CLIENTS"
echo "Expected total federation    : $TOTAL_EXPERIMENT_CLIENTS"
echo "Python client                : $CLIENT_SCRIPT"
echo "Launcher logs                : $BASE_OUT_DIR"
echo "============================================================"

PIDS=()
CLIENT_IDS=()

# ----------------------------
# Launch Client 1 virtual clients
# ----------------------------
for ((i=0; i<NUM_VIRTUAL_CLIENTS; i++))
do
    VIRTUAL_NUMBER=$((i + 1))
    CLIENT_ID="client1_ton_iot_v${VIRTUAL_NUMBER}"

    STDOUT_FILE="${BASE_OUT_DIR}/${CLIENT_ID}.out"

    echo "Starting ${CLIENT_ID} | index=${i}"

    python3.10 "$CLIENT_SCRIPT" "$i" \
        > "$STDOUT_FILE" 2>&1 &

    PIDS+=("$!")
    CLIENT_IDS+=("$CLIENT_ID")

    # Small delay to reduce simultaneous NFS reads.
    sleep 0.1
done

echo
echo "Started ${#PIDS[@]} Client 1 processes."
echo "Waiting for them to finish..."

# ----------------------------
# Wait and detect failures
# ----------------------------
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

echo

if [[ "$FAILED" -gt 0 ]]; then
    echo "${FAILED} Client 1 process(es) failed."
    exit 1
fi

echo "All ${NUM_VIRTUAL_CLIENTS} Client 1 virtual clients finished."
echo "Expected full federation size: ${TOTAL_EXPERIMENT_CLIENTS}"
echo "Launcher output logs: ${BASE_OUT_DIR}"