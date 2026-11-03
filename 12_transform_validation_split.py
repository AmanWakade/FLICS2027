from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from scipy import sparse


# ============================================================
# Configuration
# ============================================================

FEDERATED_DIR = Path("federated_data")

INPUT_DIR = (
    FEDERATED_DIR
    / "validation_split"
)

OLD_PROCESSED_DIR = (
    FEDERATED_DIR
    / "processed"
)

OUTPUT_DIR = (
    FEDERATED_DIR
    / "processed_validation"
)

PREPROCESSOR_FILE = (
    OLD_PROCESSED_DIR
    / "federated_preprocessor.pkl"
)

NUM_CLIENTS = 5

EVENT_COL = "EVENT"
TIME_COL = "TIME_YEARS"

METADATA_FILE = (
    OUTPUT_DIR
    / "validation_processing_metadata.json"
)


# ============================================================
# Missing-value cleaning
# ============================================================

MISSING_VALUES = {
    ".": np.nan,
    "": np.nan,
    " ": np.nan,
    "nan": np.nan,
    "NaN": np.nan,
    "None": np.nan,
    "<NA>": np.nan,
    "NULL": np.nan,
    "null": np.nan,
}


def clean_numeric_column(series):
    cleaned = (
        series
        .astype(object)
        .replace(MISSING_VALUES)
        .astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .replace(MISSING_VALUES)
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    ).astype(np.float32)


def clean_categorical_column(series):
    cleaned = series.astype(object)

    cleaned = cleaned.map(
        lambda value: (
            np.nan
            if value is pd.NA
            else value
        )
    )

    cleaned = cleaned.replace(
        MISSING_VALUES
    )

    cleaned = cleaned.map(
        lambda value: (
            str(value).strip()
            if pd.notna(value)
            else np.nan
        )
    )

    return (
        cleaned
        .replace(MISSING_VALUES)
        .astype(object)
    )


def clean_dataset(
    df,
    numeric_columns,
    categorical_columns,
):
    df = df.copy()

    for column in numeric_columns:
        if column in df.columns:
            df[column] = (
                clean_numeric_column(
                    df[column]
                )
            )

    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                clean_categorical_column(
                    df[column]
                )
            )

    return df


# ============================================================
# Label validation
# ============================================================

def validate_labels(
    df,
    dataset_name,
):
    if EVENT_COL not in df.columns:
        raise ValueError(
            f"{EVENT_COL} missing from "
            f"{dataset_name}"
        )

    if TIME_COL not in df.columns:
        raise ValueError(
            f"{TIME_COL} missing from "
            f"{dataset_name}"
        )

    df[EVENT_COL] = pd.to_numeric(
        df[EVENT_COL],
        errors="coerce",
    )

    df[TIME_COL] = pd.to_numeric(
        df[TIME_COL],
        errors="coerce",
    )

    invalid_event = (
        ~df[EVENT_COL].isin([0, 1])
    )

    invalid_time = (
        df[TIME_COL].isna()
        | ~np.isfinite(
            df[TIME_COL]
        )
        | (
            df[TIME_COL] <= 0
        )
    )

    if invalid_event.any():
        raise ValueError(
            f"{dataset_name}: "
            f"{int(invalid_event.sum())} "
            f"invalid EVENT values"
        )

    if invalid_time.any():
        raise ValueError(
            f"{dataset_name}: "
            f"{int(invalid_time.sum())} "
            f"invalid TIME_YEARS values"
        )


# ============================================================
# Save transformed dataset
# ============================================================

def save_processed(
    df,
    X_processed,
    output_prefix,
):
    X_processed = sparse.csr_matrix(
        X_processed,
        dtype=np.float32,
    )

    sparse.save_npz(
        f"{output_prefix}_X.npz",
        X_processed,
    )

    y = pd.DataFrame(
        {
            EVENT_COL: (
                df[EVENT_COL]
                .astype(int)
            ),
            TIME_COL: (
                df[TIME_COL]
                .astype(float)
            ),
        }
    )

    y.to_csv(
        f"{output_prefix}_y.csv",
        index=False,
    )

    return {
        "rows": int(
            len(df)
        ),
        "events": int(
            df[EVENT_COL].sum()
        ),
        "event_rate": float(
            df[EVENT_COL].mean()
        ),
        "features": int(
            X_processed.shape[1]
        ),
    }


# ============================================================
# Start
# ============================================================

print("=" * 72)
print("TRANSFORMING FEDERATED VALIDATION SPLIT")
print("=" * 72)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not PREPROCESSOR_FILE.exists():
    raise FileNotFoundError(
        f"Missing preprocessor: "
        f"{PREPROCESSOR_FILE}"
    )


# ============================================================
# Load existing fitted preprocessor
# ============================================================

print(
    "\nLoading existing federated preprocessor..."
)

with PREPROCESSOR_FILE.open(
    "rb"
) as file:
    preprocessing_data = (
        pickle.load(file)
    )

preprocessor = (
    preprocessing_data[
        "preprocessor"
    ]
)

feature_columns = (
    preprocessing_data[
        "feature_columns"
    ]
)

numeric_columns = (
    preprocessing_data[
        "numeric_columns"
    ]
)

categorical_columns = (
    preprocessing_data[
        "categorical_columns"
    ]
)

print(
    "Original predictor count:",
    len(feature_columns),
)

expected_feature_count = len(
    preprocessor.get_feature_names_out()
)

print(
    "Expected processed features:",
    expected_feature_count,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "preprocessor_source": str(
        PREPROCESSOR_FILE
    ),
    "processed_features": int(
        expected_feature_count
    ),
    "clients": {},
}


# ============================================================
# Transform all clients
# ============================================================

for client_id in range(
    1,
    NUM_CLIENTS + 1,
):
    print(
        f"\nClient {client_id}"
    )

    print("-" * 50)

    train_file = (
        INPUT_DIR
        / f"client_{client_id}_train_final.csv"
    )

    validation_file = (
        INPUT_DIR
        / f"client_{client_id}_validation.csv"
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Missing: {train_file}"
        )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Missing: {validation_file}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df = pd.read_csv(
        train_file,
        low_memory=False,
        dtype={
            "TRR_ID_CODE": "string",
            "CTR_CODE": "string",
        },
    )

    validation_df = pd.read_csv(
        validation_file,
        low_memory=False,
        dtype={
            "TRR_ID_CODE": "string",
            "CTR_CODE": "string",
        },
    )

    validate_labels(
        train_df,
        f"Client {client_id} train"
    )

    validate_labels(
        validation_df,
        f"Client {client_id} validation"
    )

    # --------------------------------------------------------
    # Check feature columns exist
    # --------------------------------------------------------

    missing_train_features = [
        column
        for column in feature_columns
        if column not in train_df.columns
    ]

    missing_validation_features = [
        column
        for column in feature_columns
        if column not in validation_df.columns
    ]

    if missing_train_features:
        raise ValueError(
            f"Client {client_id} training "
            f"missing features: "
            f"{missing_train_features}"
        )

    if missing_validation_features:
        raise ValueError(
            f"Client {client_id} validation "
            f"missing features: "
            f"{missing_validation_features}"
        )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    train_df = clean_dataset(
        train_df,
        numeric_columns,
        categorical_columns,
    )

    validation_df = clean_dataset(
        validation_df,
        numeric_columns,
        categorical_columns,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Transform only.
    # DO NOT fit or refit here.
    # --------------------------------------------------------

    X_train = (
        preprocessor.transform(
            train_df[
                feature_columns
            ]
        )
    )

    X_validation = (
        preprocessor.transform(
            validation_df[
                feature_columns
            ]
        )
    )

    # --------------------------------------------------------
    # Feature dimension check
    # --------------------------------------------------------

    if (
        X_train.shape[1]
        != expected_feature_count
    ):
        raise ValueError(
            f"Client {client_id} train "
            f"has {X_train.shape[1]} features; "
            f"expected "
            f"{expected_feature_count}"
        )

    if (
        X_validation.shape[1]
        != expected_feature_count
    ):
        raise ValueError(
            f"Client {client_id} validation "
            f"has {X_validation.shape[1]} features; "
            f"expected "
            f"{expected_feature_count}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_summary = save_processed(
        train_df,
        X_train,
        OUTPUT_DIR
        / f"client_{client_id}_train",
    )

    validation_summary = save_processed(
        validation_df,
        X_validation,
        OUTPUT_DIR
        / f"client_{client_id}_validation",
    )

    metadata[
        "clients"
    ][
        f"client_{client_id}"
    ] = {
        "train": train_summary,
        "validation": (
            validation_summary
        ),
    }

    print(
        f"Train      : "
        f"{X_train.shape}, "
        f"events="
        f"{train_summary['events']}"
    )

    print(
        f"Validation : "
        f"{X_validation.shape}, "
        f"events="
        f"{validation_summary['events']}"
    )


# ============================================================
# Copy/use existing test datasets
# ============================================================

print(
    "\nChecking existing client/global test data..."
)

for client_id in range(
    1,
    NUM_CLIENTS + 1,
):
    test_x = (
        OLD_PROCESSED_DIR
        / f"client_{client_id}_test_X.npz"
    )

    test_y = (
        OLD_PROCESSED_DIR
        / f"client_{client_id}_test_y.csv"
    )

    if not test_x.exists():
        raise FileNotFoundError(
            f"Missing existing test file: "
            f"{test_x}"
        )

    if not test_y.exists():
        raise FileNotFoundError(
            f"Missing existing test labels: "
            f"{test_y}"
        )

    X_test = sparse.load_npz(
        test_x
    )

    if (
        X_test.shape[1]
        != expected_feature_count
    ):
        raise ValueError(
            f"Client {client_id} test "
            f"feature count mismatch"
        )


global_x = (
    OLD_PROCESSED_DIR
    / "global_test_X.npz"
)

global_y = (
    OLD_PROCESSED_DIR
    / "global_test_y.csv"
)

if not global_x.exists():
    raise FileNotFoundError(
        f"Missing: {global_x}"
    )

if not global_y.exists():
    raise FileNotFoundError(
        f"Missing: {global_y}"
    )

X_global = sparse.load_npz(
    global_x
)

if (
    X_global.shape[1]
    != expected_feature_count
):
    raise ValueError(
        "Global test feature count mismatch"
    )


# ============================================================
# Save metadata
# ============================================================

with METADATA_FILE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


# ============================================================
# Final summary
# ============================================================

total_train = sum(
    metadata["clients"][
        f"client_{client_id}"
    ]["train"]["rows"]
    for client_id in range(
        1,
        NUM_CLIENTS + 1
    )
)

total_validation = sum(
    metadata["clients"][
        f"client_{client_id}"
    ]["validation"]["rows"]
    for client_id in range(
        1,
        NUM_CLIENTS + 1
    )
)

total_validation_events = sum(
    metadata["clients"][
        f"client_{client_id}"
    ]["validation"]["events"]
    for client_id in range(
        1,
        NUM_CLIENTS + 1
    )
)


print("\n" + "=" * 72)
print("VALIDATION PREPROCESSING COMPLETE")
print("=" * 72)

print(
    f"Processed features        : "
    f"{expected_feature_count}"
)

print(
    f"Final training patients   : "
    f"{total_train:,}"
)

print(
    f"Validation patients       : "
    f"{total_validation:,}"
)

print(
    f"Validation events         : "
    f"{total_validation_events:,}"
)

print("\nValidation passed:")
print(
    "- Existing preprocessor reused"
)
print(
    "- Preprocessor was NOT refitted"
)
print(
    "- All client dimensions identical"
)
print(
    "- Existing test sets unchanged"
)
print(
    "- Global test set unchanged"
)

print("\nSaved in:")
print(
    "federated_data/processed_validation/"
)