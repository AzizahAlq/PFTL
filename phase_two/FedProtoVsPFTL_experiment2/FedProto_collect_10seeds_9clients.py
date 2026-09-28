from pathlib import Path
import pandas as pd

BASE_DIR = Path(
    "/nfs/aalqahtani/Proto_PFTL_Multi_class/"
    "FedProtoVsPFTL/fedproto_logs"
)

SEEDS = [32, 123, 12, 999, 308, 5, 77, 45, 101, 190]

all_clients = []

for client_num in range(1, 10):

    csv_path = (
        BASE_DIR
        / f"client{client_num}_ton_iot_aligned_20k"
        / f"client{client_num}_final_test_results.csv"
    )

    if not csv_path.exists():
        print(f"Client {client_num}: FILE NOT FOUND")
        continue

    df = pd.read_csv(csv_path)

    # Keep only requested seeds
    df = df[df["seed"].isin(SEEDS)].copy()

    # Keep requested seed order
    df["seed"] = pd.Categorical(
        df["seed"],
        categories=SEEDS,
        ordered=True
    )
    df = df.sort_values("seed")

    df.insert(0, "client", f"client{client_num}")

    print(f"\n===== CLIENT {client_num} =====")
    print(f"Seeds found: {len(df)}/10")

    print(
        df[
            [
                "seed",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1"
            ]
        ].to_string(index=False)
    )

    missing = set(SEEDS) - set(df["seed"].astype(int))

    if missing:
        print("Missing seeds:", sorted(missing))

    all_clients.append(df)

# Combine all clients
combined = pd.concat(all_clients, ignore_index=True)

output = BASE_DIR / "fedproto_9clients_10seeds.csv"
combined.to_csv(output, index=False)

print("\n======================================")
print("Saved combined results:")
print(output)
print("======================================")