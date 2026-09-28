#!/bin/bash

# ============================================================
# Launch all six full-size non-IID PFTL virtual-client groups
# concurrently.
#
# IMPORTANT:
# NUM_VIRTUAL_CLIENTS must match the value inside all six
# Python client files.
#
#   1  -> 6 total clients
#   5  -> 30 total clients
#   9  -> 54 total clients
#   15 -> 90 total clients
#   20 -> 120 total clients
#
# Each virtual client uses:
# - Full training-pool size
# - Dirichlet non-IID class distribution
# - NONIID_ALPHA = 0.3
# - Minimum 5 samples per class
# ============================================================

set -u

ROOT_DIR="/nfs/aalqahtani/PFTL_Multi_class/M_PFTL_codes"

NUM_VIRTUAL_CLIENTS=1
TOTAL_CLIENTS=$((6 * NUM_VIRTUAL_CLIENTS))

# ============================================================
# Full-size non-IID client files
# ============================================================

CLIENT1_SCRIPT="${ROOT_DIR}/client1_CIC_ToN_IoT_PFTL_fullsize_iid.py"
CLIENT2_SCRIPT="${ROOT_DIR}/client2_CIC_IOT2023_PFTL_fullsize_iid.py"
CLIENT3_SCRIPT="${ROOT_DIR}/client3_UNSWNB15_PFTL_fullsize_iid.py"
CLIENT4_SCRIPT="${ROOT_DIR}/client4_CICIDS2017_PFTL_fullsize_iid.py"
CLIENT5_SCRIPT="${ROOT_DIR}/client5_BCCC_NRC_2024_PFTL_fullsize_iid.py"
CLIENT6_SCRIPT="${ROOT_DIR}/client6_IDAD_2024_PFTL_fullsize_iid.py"

# Launcher stdout/stderr logs
LAUNCH_LOG_DIR="${ROOT_DIR}/scalability_launcher_logs_fullsize_iid/clients_${TOTAL_CLIENTS}"

mkdir -p "$LAUNCH_LOG_DIR"

# ============================================================
# Validate client-count setting
# ============================================================

case "$NUM_VIRTUAL_CLIENTS" in
    1|5|10|15|20)
        ;;
    *)
        echo "[ERROR] NUM_VIRTUAL_CLIENTS must be 1, 5, 9, 15, or 20."
        exit 1
        ;;
esac

SCRIPTS=(
    "$CLIENT1_SCRIPT"
    "$CLIENT2_SCRIPT"
    "$CLIENT3_SCRIPT"
    "$CLIENT4_SCRIPT"
    "$CLIENT5_SCRIPT"
    "$CLIENT6_SCRIPT"
)

BASE_IDS=(
    "client1_ton_iot"
    "client2_cic_iot_2023"
    "client3_unsw_nb15"
    "client4_cicids2017"
    "client5_bccc_nrc_2024"
    "client6_idad_2024"
)

# ============================================================
# Verify all client files before launching
# ============================================================

for script in "${SCRIPTS[@]}"
do
    if [[ ! -f "$script" ]]; then
        echo "[ERROR] Missing client file:"
        echo "$script"
        exit 1
    fi
done

echo "================================================================"
echo "PFTL full-size non-IID scalability experiment"
echo "Virtual clients per dataset : $NUM_VIRTUAL_CLIENTS"
echo "Total federation clients    : $TOTAL_CLIENTS"
echo "Partition method            : full_size_dirichlet_resampling"
echo "Dirichlet alpha             : 0.3"
echo "Launcher log directory      : $LAUNCH_LOG_DIR"
echo "================================================================"

PIDS=()
CLIENT_IDS=()

# ============================================================
# Launch one original-dataset group
# ============================================================

launch_group() {
    local script="$1"
    local base_id="$2"

    for ((i=0; i<NUM_VIRTUAL_CLIENTS; i++))
    do
        local virtual_number=$((i + 1))
        local client_id="${base_id}_v${virtual_number}"
        local output_file="${LAUNCH_LOG_DIR}/${client_id}.out"

        echo "Launching ${client_id}"

        python "$script" "$i" \
            > "$output_file" 2>&1 &

        PIDS+=("$!")
        CLIENT_IDS+=("$client_id")

        # Reduce simultaneous NFS and dataset-opening pressure.
        sleep 0.1
    done
}

# ============================================================
# Launch all six groups before waiting
# ============================================================

for index in "${!SCRIPTS[@]}"
do
    launch_group \
        "${SCRIPTS[$index]}" \
        "${BASE_IDS[$index]}"
done

echo
echo "Launched ${#PIDS[@]} client processes."
echo "Waiting for the complete federation..."

FAILED=0

# ============================================================
# Wait for every virtual client
# ============================================================

for index in "${!PIDS[@]}"
do
    pid="${PIDS[$index]}"
    client_id="${CLIENT_IDS[$index]}"

    if wait "$pid"; then
        echo "[OK] ${client_id}"
    else
        exit_code=$?
        echo "[ERROR] ${client_id} exited with code ${exit_code}"
        FAILED=$((FAILED + 1))
    fi
done

echo

if [[ "$FAILED" -gt 0 ]]; then
    echo "${FAILED} client process(es) failed."
    exit 1
fi

echo "All ${TOTAL_CLIENTS} full-size non-IID virtual clients finished successfully."