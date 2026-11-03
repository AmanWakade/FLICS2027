from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("federated_data")
OUTPUT_DIR = INPUT_DIR / "processed"

NUM_CLIENTS = 5

EVENT_COL = "EVENT"
TIME_COL = "TIME_YEARS"

EXCLUDED_COLUMNS = [
    "TRR_ID_CODE",
    "CTR_CODE",
    "CLIENT_ID",
    "TX_DATE",
    EVENT_COL,
    TIME_COL,
]

PREPROCESSOR_FILE = OUTPUT_DIR / "federated_preprocessor.pkl"
METADATA_FILE = OUTPUT_DIR / "federated_preprocessing_metadata.json"
FEATURE_NAMES_FILE = OUTPUT_DIR / "processed_feature_names.csv"


# ============================================================
# Numeric predictors
# ============================================================

NUMERIC_COLUMNS = [
    "AGE",
    "BMI_CALC",
    "CURRENT_PRA",
    "PEAK_PRA",
    "INIT_CPRA",
    "END_CPRA",
    "AMIS",
    "BMIS",
    "DRMIS",
    "HLAMIS",
    "AGE_DON",
    "KDPI",
    "KDRI_MED",
    "BMI_DON_CALC",
    "CREAT_DON",
    "BUN_DON",
    "COLD_ISCH_KI",
    "WARM_ISCH_TM_DON",
    "DISTANCE",
]


# ============================================================
# Utility functions
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


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert OPTN numeric values to float32."""

    cleaned = (
        series.astype(object)
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


def clean_categorical_column(series: pd.Series) -> pd.Series:
    """Convert categorical values to object dtype with np.nan."""

    cleaned = series.astype(object)

    cleaned = cleaned.map(
        lambda value: np.nan
        if value is pd.NA
        else value
    )

    cleaned = cleaned.replace(MISSING_VALUES)

    cleaned = cleaned.map(
        lambda value: (
            str(value).strip()
            if pd.notna(value)
            else np.nan
        )
    )

    return cleaned.replace(MISSING_VALUES).astype(object)


def clean_dataset(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Clean all predictor columns consistently."""

    dataframe = dataframe.copy()

    for column in numeric_columns:
        dataframe[column] = clean_numeric_column(
            dataframe[column]
        )

    for column in categorical_columns:
        dataframe[column] = clean_categorical_column(
            dataframe[column]
        )

    return dataframe


def validate_labels(dataframe: pd.DataFrame, name: str) -> None:
    """Validate EVENT and TIME_YEARS."""

    if EVENT_COL not in dataframe.columns:
        raise ValueError(
            f"{EVENT_COL} missing from {name}"
        )

    if TIME_COL not in dataframe.columns:
        raise ValueError(
            f"{TIME_COL} missing from {name}"
        )

    dataframe[EVENT_COL] = pd.to_numeric(
        dataframe[EVENT_COL],
        errors="coerce",
    )

    dataframe[TIME_COL] = pd.to_numeric(
        dataframe[TIME_COL],
        errors="coerce",
    )

    invalid_event = ~dataframe[EVENT_COL].isin([0, 1])

    invalid_time = (
        dataframe[TIME_COL].isna()
        | ~np.isfinite(dataframe[TIME_COL])
        | (dataframe[TIME_COL] <= 0)
    )

    if invalid_event.any():
        raise ValueError(
            f"{name} contains "
            f"{int(invalid_event.sum())} invalid EVENT values."
        )

    if invalid_time.any():
        raise ValueError(
            f"{name} contains "
            f"{int(invalid_time.sum())} invalid TIME_YEARS values."
        )


def save_processed_dataset(
    dataframe: pd.DataFrame,
    processed_features,
    output_prefix: Path,
) -> dict:
    """Save processed predictors and survival labels."""

    processed_features = sparse.csr_matrix(
        processed_features,
        dtype=np.float32,
    )

    sparse.save_npz(
        f"{output_prefix}_X.npz",
        processed_features,
    )

    labels = pd.DataFrame(
        {
            EVENT_COL: dataframe[EVENT_COL].astype(int),
            TIME_COL: dataframe[TIME_COL].astype(float),
        }
    )

    labels.to_csv(
        f"{output_prefix}_y.csv",
        index=False,
    )

    return {
        "rows": int(len(dataframe)),
        "events": int(dataframe[EVENT_COL].sum()),
        "event_rate": float(
            dataframe[EVENT_COL].mean()
        ),
        "processed_features": int(
            processed_features.shape[1]
        ),
    }


# ============================================================
# Initial validation
# ============================================================

print("=" * 72)
print("PREPARING FEDERATED PREPROCESSING")
print("=" * 72)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

client_files = [
    INPUT_DIR / f"client_{client_id}_train.csv"
    for client_id in range(1, NUM_CLIENTS + 1)
]

global_test_file = INPUT_DIR / "global_test.csv"

missing_files = [
    str(file)
    for file in client_files + [global_test_file]
    if not file.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Missing files:\n" + "\n".join(missing_files)
    )


# ============================================================
# Load client training datasets
# ============================================================

client_dataframes = []

for client_id, file in enumerate(
    client_files,
    start=1,
):
    client_df = pd.read_csv(
        file,
        low_memory=False,
        dtype={
            "TRR_ID_CODE": "string",
            "CTR_CODE": "string",
        },
    )

    validate_labels(
        client_df,
        f"client_{client_id}_train.csv",
    )

    client_dataframes.append(client_df)

    print(
        f"Client {client_id}: "
        f"rows={len(client_df):,}, "
        f"events={int(client_df[EVENT_COL].sum()):,}"
    )


# ============================================================
# Create pooled training data for shared preprocessing
# ============================================================

pooled_train_df = pd.concat(
    client_dataframes,
    ignore_index=True,
)

print(
    "\nPooled training shape:",
    pooled_train_df.shape,
)


# ============================================================
# Define predictor columns
# ============================================================

feature_columns = [
    column
    for column in pooled_train_df.columns
    if column not in EXCLUDED_COLUMNS
]

numeric_columns = [
    column
    for column in NUMERIC_COLUMNS
    if column in feature_columns
]

categorical_columns = [
    column
    for column in feature_columns
    if column not in numeric_columns
]

print("\nFeature configuration")
print("---------------------")
print("Total predictors      :", len(feature_columns))
print("Numeric predictors    :", len(numeric_columns))
print("Categorical predictors:", len(categorical_columns))


# ============================================================
# Clean pooled training predictors
# ============================================================

pooled_train_df = clean_dataset(
    pooled_train_df,
    numeric_columns,
    categorical_columns,
)


# ============================================================
# Shared preprocessing pipeline
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="MISSING",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=True,
                dtype=np.float32,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ],
    remainder="drop",
    sparse_threshold=1.0,
    n_jobs=1,
    verbose_feature_names_out=True,
)


# ============================================================
# Fit shared preprocessing pipeline
# ============================================================

print("\nFitting shared preprocessing pipeline...")

preprocessor.fit(
    pooled_train_df[feature_columns]
)

feature_names = preprocessor.get_feature_names_out()

print(
    "Processed feature count:",
    len(feature_names),
)


# ============================================================
# Save shared preprocessor
# ============================================================

with PREPROCESSOR_FILE.open("wb") as file:
    pickle.dump(
        {
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "excluded_columns": EXCLUDED_COLUMNS,
        },
        file,
    )

pd.DataFrame(
    {"FEATURE_NAME": feature_names}
).to_csv(
    FEATURE_NAMES_FILE,
    index=False,
)


# ============================================================
# Transform each federated client
# ============================================================

summary = {
    "number_of_clients": NUM_CLIENTS,
    "original_predictors": len(feature_columns),
    "processed_predictors": len(feature_names),
    "numeric_predictors": numeric_columns,
    "categorical_predictors": categorical_columns,
    "clients": {},
}

print("\nTransforming client datasets...")

for client_id, client_df in enumerate(
    client_dataframes,
    start=1,
):
    cleaned_client_df = clean_dataset(
        client_df,
        numeric_columns,
        categorical_columns,
    )

    client_features = preprocessor.transform(
        cleaned_client_df[feature_columns]
    )

    client_summary = save_processed_dataset(
        dataframe=cleaned_client_df,
        processed_features=client_features,
        output_prefix=(
            OUTPUT_DIR
            / f"client_{client_id}_train"
        ),
    )

    summary["clients"][
        f"client_{client_id}"
    ] = client_summary

    print(
        f"Client {client_id}: "
        f"X={client_features.shape}, "
        f"events={client_summary['events']}"
    )


# ============================================================
# Transform centre-specific test datasets
# ============================================================

print("\nTransforming local client test datasets...")

for client_id in range(1, NUM_CLIENTS + 1):
    client_test_file = (
        INPUT_DIR
        / f"client_{client_id}_test.csv"
    )

    client_test_df = pd.read_csv(
        client_test_file,
        low_memory=False,
        dtype={
            "TRR_ID_CODE": "string",
            "CTR_CODE": "string",
        },
    )

    validate_labels(
        client_test_df,
        f"client_{client_id}_test.csv",
    )

    client_test_df = clean_dataset(
        client_test_df,
        numeric_columns,
        categorical_columns,
    )

    client_test_features = preprocessor.transform(
        client_test_df[feature_columns]
    )

    save_processed_dataset(
        dataframe=client_test_df,
        processed_features=client_test_features,
        output_prefix=(
            OUTPUT_DIR
            / f"client_{client_id}_test"
        ),
    )


# ============================================================
# Transform global test dataset
# ============================================================

print("\nTransforming global test dataset...")

global_test_df = pd.read_csv(
    global_test_file,
    low_memory=False,
    dtype={
        "TRR_ID_CODE": "string",
        "CTR_CODE": "string",
    },
)

validate_labels(
    global_test_df,
    "global_test.csv",
)

global_test_df = clean_dataset(
    global_test_df,
    numeric_columns,
    categorical_columns,
)

global_test_features = preprocessor.transform(
    global_test_df[feature_columns]
)

global_test_summary = save_processed_dataset(
    dataframe=global_test_df,
    processed_features=global_test_features,
    output_prefix=(
        OUTPUT_DIR / "global_test"
    ),
)

summary["global_test"] = global_test_summary


# ============================================================
# Final validation
# ============================================================

expected_feature_count = len(feature_names)

for client_id in range(1, NUM_CLIENTS + 1):
    client_matrix = sparse.load_npz(
        OUTPUT_DIR
        / f"client_{client_id}_train_X.npz"
    )

    if client_matrix.shape[1] != expected_feature_count:
        raise ValueError(
            f"Client {client_id} has "
            f"{client_matrix.shape[1]} features, "
            f"expected {expected_feature_count}."
        )

global_matrix = sparse.load_npz(
    OUTPUT_DIR / "global_test_X.npz"
)

if global_matrix.shape[1] != expected_feature_count:
    raise ValueError(
        "Global test feature dimension does not "
        "match client feature dimensions."
    )


# ============================================================
# Save metadata
# ============================================================

with METADATA_FILE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 72)
print("FEDERATED PREPROCESSING COMPLETE")
print("=" * 72)

print(
    f"Original predictor count : "
    f"{len(feature_columns)}"
)

print(
    f"Processed feature count  : "
    f"{expected_feature_count}"
)

print(
    f"Global test rows         : "
    f"{global_test_summary['rows']:,}"
)

print(
    f"Global test events       : "
    f"{global_test_summary['events']:,}"
)

print("\nValidation passed:")
print("- Every client has identical feature dimensions")
print("- Global test uses the same preprocessing")
print("- EVENT and TIME_YEARS remain separate labels")
print("- IDs and center codes are excluded from predictors")

print("\nSaved in federated_data/processed/:")
print("- federated_preprocessor.pkl")
print("- federated_preprocessing_metadata.json")
print("- processed_feature_names.csv")
print("- client_1_train_X.npz ... client_5_train_X.npz")
print("- client_1_train_y.csv ... client_5_train_y.csv")
print("- client_1_test_X.npz ... client_5_test_X.npz")
print("- client_1_test_y.csv ... client_5_test_y.csv")
print("- global_test_X.npz")
print("- global_test_y.csv")