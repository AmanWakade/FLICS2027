from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "MODEL_DATASET_LEAKAGE_FREE_WITH_CENTER.csv"
)

OUTPUT_DIR = Path("federated_data")

CENTER_COL = "CTR_CODE"
EVENT_COL = "EVENT"
ID_COL = "TRR_ID_CODE"

NUM_CLIENTS = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("CREATING FEDERATED CLIENT DATASETS")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    dtype={CENTER_COL: "string"},
)

required_columns = {
    CENTER_COL,
    EVENT_COL,
    ID_COL,
}

missing_columns = required_columns.difference(
    df.columns
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )

df[CENTER_COL] = (
    df[CENTER_COL]
    .astype("string")
    .str.strip()
    .str.zfill(5)
)

df[EVENT_COL] = pd.to_numeric(
    df[EVENT_COL],
    errors="coerce",
)

df = df[
    df[CENTER_COL].notna()
    & df[EVENT_COL].isin([0, 1])
].copy()

df[EVENT_COL] = df[EVENT_COL].astype(int)

print("\nComplete dataset shape:", df.shape)
print(
    "Unique transplant centres:",
    df[CENTER_COL].nunique()
)


# ============================================================
# Select five largest centres
# ============================================================

center_counts = (
    df[CENTER_COL]
    .value_counts()
    .sort_values(ascending=False)
)

top_centers = (
    center_counts
    .head(NUM_CLIENTS)
    .index
    .tolist()
)

print("\nSelected centres")
print("----------------")

for client_number, center_code in enumerate(
    top_centers,
    start=1,
):
    center_data = df[
        df[CENTER_COL] == center_code
    ]

    print(
        f"Client {client_number}: "
        f"CTR_CODE={center_code}, "
        f"patients={len(center_data):,}, "
        f"events={int(center_data[EVENT_COL].sum()):,}, "
        f"event rate="
        f"{center_data[EVENT_COL].mean() * 100:.2f}%"
    )


# ============================================================
# Keep only selected centres
# ============================================================

selected_df = df[
    df[CENTER_COL].isin(top_centers)
].copy()

print(
    "\nPatients across selected centres:",
    f"{len(selected_df):,}"
)

print(
    "Events across selected centres:",
    f"{int(selected_df[EVENT_COL].sum()):,}"
)


# ============================================================
# Split each centre into local training and global testing
# ============================================================

global_test_parts = []
summary_records = []

for client_number, center_code in enumerate(
    top_centers,
    start=1,
):
    center_df = (
        selected_df[
            selected_df[CENTER_COL] == center_code
        ]
        .copy()
        .reset_index(drop=True)
    )

    event_counts = center_df[
        EVENT_COL
    ].value_counts()

    # Stratification requires at least two observations
    # in each outcome category.
    can_stratify = (
        len(event_counts) == 2
        and event_counts.min() >= 2
    )

    stratify_values = (
        center_df[EVENT_COL]
        if can_stratify
        else None
    )

    client_train, client_test = train_test_split(
        center_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_values,
    )

    client_train = client_train.reset_index(
        drop=True
    )

    client_test = client_test.reset_index(
        drop=True
    )

    # Save complete client dataset for auditing.
    center_df.to_csv(
        OUTPUT_DIR
        / f"client_{client_number}_full.csv",
        index=False,
    )

    # Local training dataset used by Flower.
    client_train.to_csv(
        OUTPUT_DIR
        / f"client_{client_number}_train.csv",
        index=False,
    )

    # Centre-specific test dataset.
    client_test.to_csv(
        OUTPUT_DIR
        / f"client_{client_number}_test.csv",
        index=False,
    )

    client_test = client_test.copy()
    client_test["CLIENT_ID"] = client_number

    global_test_parts.append(client_test)

    summary_records.append(
        {
            "CLIENT_ID": client_number,
            "CTR_CODE": center_code,
            "TOTAL_PATIENTS": len(center_df),
            "TOTAL_EVENTS": int(
                center_df[EVENT_COL].sum()
            ),
            "TOTAL_EVENT_RATE": float(
                center_df[EVENT_COL].mean()
            ),
            "TRAIN_PATIENTS": len(client_train),
            "TRAIN_EVENTS": int(
                client_train[EVENT_COL].sum()
            ),
            "TRAIN_EVENT_RATE": float(
                client_train[EVENT_COL].mean()
            ),
            "TEST_PATIENTS": len(client_test),
            "TEST_EVENTS": int(
                client_test[EVENT_COL].sum()
            ),
            "TEST_EVENT_RATE": float(
                client_test[EVENT_COL].mean()
            ),
        }
    )

    print(
        f"\nClient {client_number} created"
        f"\n  Center       : {center_code}"
        f"\n  Training rows: {len(client_train):,}"
        f"\n  Training events: "
        f"{int(client_train[EVENT_COL].sum()):,}"
        f"\n  Testing rows : {len(client_test):,}"
        f"\n  Testing events: "
        f"{int(client_test[EVENT_COL].sum()):,}"
    )


# ============================================================
# Create common global test dataset
# ============================================================

global_test_df = pd.concat(
    global_test_parts,
    ignore_index=True,
)

global_test_df = global_test_df.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

global_test_df.to_csv(
    OUTPUT_DIR / "global_test.csv",
    index=False,
)


# ============================================================
# Save selected-centre dataset
# ============================================================

selected_df.to_csv(
    OUTPUT_DIR / "top_5_centers_full.csv",
    index=False,
)


# ============================================================
# Save summary
# ============================================================

summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    OUTPUT_DIR
    / "client_distribution_summary.csv",
    index=False,
)


# ============================================================
# Validation
# ============================================================

train_ids = set()

for client_number in range(
    1,
    NUM_CLIENTS + 1,
):
    client_train = pd.read_csv(
        OUTPUT_DIR
        / f"client_{client_number}_train.csv",
        usecols=[ID_COL],
        dtype={ID_COL: "string"},
    )

    current_ids = set(
        client_train[ID_COL]
    )

    overlap = train_ids.intersection(
        current_ids
    )

    if overlap:
        raise ValueError(
            f"Patient overlap detected between clients: "
            f"{len(overlap)} IDs"
        )

    train_ids.update(current_ids)

global_test_ids = set(
    global_test_df[ID_COL].astype("string")
)

train_test_overlap = train_ids.intersection(
    global_test_ids
)

if train_test_overlap:
    raise ValueError(
        "Patient leakage detected between client "
        "training data and global test data."
    )


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("FEDERATED DATASET SUMMARY")
print("=" * 70)

print(
    f"Number of clients       : {NUM_CLIENTS}"
)

print(
    f"Total selected patients : "
    f"{len(selected_df):,}"
)

print(
    f"Total training patients : "
    f"{len(train_ids):,}"
)

print(
    f"Global test patients    : "
    f"{len(global_test_df):,}"
)

print(
    f"Global test events      : "
    f"{int(global_test_df[EVENT_COL].sum()):,}"
)

print(
    f"Global test event rate  : "
    f"{global_test_df[EVENT_COL].mean() * 100:.2f}%"
)

print("\nValidation passed:")
print("- No patient overlap between clients")
print("- No training/test patient leakage")

print("\nSaved in federated_data/:")
print("- client_1_train.csv ... client_5_train.csv")
print("- client_1_test.csv ... client_5_test.csv")
print("- client_1_full.csv ... client_5_full.csv")
print("- global_test.csv")
print("- top_5_centers_full.csv")
print("- client_distribution_summary.csv")