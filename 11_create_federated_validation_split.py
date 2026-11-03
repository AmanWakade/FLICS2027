from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("federated_data")
OUTPUT_DIR = INPUT_DIR / "validation_split"

NUM_CLIENTS = 5

EVENT_COL = "EVENT"
ID_COL = "TRR_ID_CODE"

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Setup
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 72)
print("CREATING FEDERATED TRAIN / VALIDATION SPLITS")
print("=" * 72)


# ============================================================
# Process each client
# ============================================================

summary_rows = []

all_train_ids = set()
all_validation_ids = set()

for client_id in range(1, NUM_CLIENTS + 1):

    input_file = (
        INPUT_DIR
        / f"client_{client_id}_train.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing: {input_file}"
        )

    df = pd.read_csv(
        input_file,
        low_memory=False,
        dtype={
            ID_COL: "string",
            "CTR_CODE": "string",
        },
    )

    df[EVENT_COL] = pd.to_numeric(
        df[EVENT_COL],
        errors="coerce",
    )

    if not df[EVENT_COL].isin([0, 1]).all():
        raise ValueError(
            f"Invalid EVENT values in client {client_id}"
        )

    event_counts = df[
        EVENT_COL
    ].value_counts()

    can_stratify = (
        len(event_counts) == 2
        and event_counts.min() >= 2
    )

    stratify_values = (
        df[EVENT_COL]
        if can_stratify
        else None
    )

    train_df, validation_df = (
        train_test_split(
            df,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_values,
        )
    )

    train_df = train_df.reset_index(
        drop=True
    )

    validation_df = (
        validation_df.reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_file = (
        OUTPUT_DIR
        / f"client_{client_id}_train_final.csv"
    )

    validation_file = (
        OUTPUT_DIR
        / f"client_{client_id}_validation.csv"
    )

    train_df.to_csv(
        train_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_ids = set(
        train_df[ID_COL].astype("string")
    )

    validation_ids = set(
        validation_df[
            ID_COL
        ].astype("string")
    )

    overlap = train_ids.intersection(
        validation_ids
    )

    if overlap:
        raise ValueError(
            f"Train-validation leakage "
            f"in client {client_id}"
        )

    all_train_ids.update(
        train_ids
    )

    all_validation_ids.update(
        validation_ids
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "CLIENT_ID": client_id,

            "ORIGINAL_TRAIN_ROWS": (
                len(df)
            ),

            "ORIGINAL_EVENTS": int(
                df[EVENT_COL].sum()
            ),

            "FINAL_TRAIN_ROWS": (
                len(train_df)
            ),

            "FINAL_TRAIN_EVENTS": int(
                train_df[
                    EVENT_COL
                ].sum()
            ),

            "VALIDATION_ROWS": (
                len(validation_df)
            ),

            "VALIDATION_EVENTS": int(
                validation_df[
                    EVENT_COL
                ].sum()
            ),

            "VALIDATION_EVENT_RATE": float(
                validation_df[
                    EVENT_COL
                ].mean()
            ),
        }
    )

    print(
        f"\nClient {client_id}"
    )

    print(
        f"Original training : "
        f"{len(df):,}"
    )

    print(
        f"Final training    : "
        f"{len(train_df):,}"
    )

    print(
        f"Training events   : "
        f"{int(train_df[EVENT_COL].sum()):,}"
    )

    print(
        f"Validation rows   : "
        f"{len(validation_df):,}"
    )

    print(
        f"Validation events : "
        f"{int(validation_df[EVENT_COL].sum()):,}"
    )


# ============================================================
# Cross-client leakage check
# ============================================================

cross_overlap = (
    all_train_ids.intersection(
        all_validation_ids
    )
)

if cross_overlap:
    raise ValueError(
        "Patient overlap detected between "
        "training and validation sets."
    )


# ============================================================
# Save summary
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_DIR
    / "validation_split_summary.csv",
    index=False,
)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 72)
print("VALIDATION SPLIT COMPLETE")
print("=" * 72)

print(
    f"Total final training patients : "
    f"{sum(summary_df['FINAL_TRAIN_ROWS']):,}"
)

print(
    f"Total validation patients     : "
    f"{sum(summary_df['VALIDATION_ROWS']):,}"
)

print(
    f"Total validation events       : "
    f"{sum(summary_df['VALIDATION_EVENTS']):,}"
)

print("\nValidation passed:")
print("- No train/validation overlap")
print("- Existing client test sets untouched")
print("- Existing global test set untouched")

print("\nSaved in:")
print(
    "federated_data/validation_split/"
)