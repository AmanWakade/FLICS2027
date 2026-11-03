import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path("MODEL_DATASET_LEAKAGE_FREE.csv")

MODEL_FILE = Path("rsf_survival_model.pkl")
RESULTS_FILE = Path("rsf_survival_results.json")
PREDICTIONS_FILE = Path("rsf_test_predictions.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# RSF is substantially more memory-intensive than XGBoost.
# Keep every event and sample censored observations.
MAX_ROWS = None
N_ESTIMATORS = 50
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 100
MIN_SAMPLES_LEAF = 50
MAX_SAMPLES = 0.7


# ============================================================
# Helper functions
# ============================================================

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Convert OPTN numeric columns to float32.

    Handles:
    - "."
    - empty strings
    - textual missing values
    - percentage symbols
    - pandas NA
    """
    cleaned = (
        series.astype(object)
        .replace(
            {
                pd.NA: np.nan,
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
        )
    )

    cleaned = (
        cleaned.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .replace(
            {
                "nan": np.nan,
                "NaN": np.nan,
                "None": np.nan,
                "<NA>": np.nan,
            }
        )
    )

    return pd.to_numeric(cleaned, errors="coerce").astype(np.float32)


def clean_categorical_column(series: pd.Series) -> pd.Series:
    """
    Convert categorical columns to ordinary Python objects.

    Using object dtype and np.nan prevents the pandas pd.NA
    ambiguity error inside SimpleImputer.
    """
    cleaned = series.astype(object)

    cleaned = cleaned.replace(
        {
            pd.NA: np.nan,
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
    )

    non_missing = cleaned.notna()

    cleaned.loc[non_missing] = (
        cleaned.loc[non_missing]
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned.replace(
        {
            "": np.nan,
            ".": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "<NA>": np.nan,
        }
    )

    return cleaned.astype(object)


# ============================================================
# Load dataset
# ============================================================

print("=" * 65)
print("RANDOM SURVIVAL FOREST")
print("=" * 65)
print("\nLoading dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE, low_memory=False)

print("Original dataset shape:", df.shape)

duration_col = "TIME_YEARS"
event_col = "EVENT"

required_columns = {duration_col, event_col}
missing_required = required_columns.difference(df.columns)

if missing_required:
    raise ValueError(
        f"Required columns missing: {sorted(missing_required)}"
    )

# ID and date are not predictors.
df = df.drop(
    columns=["TRR_ID_CODE", "TX_DATE"],
    errors="ignore"
)


# ============================================================
# Clean survival outcomes
# ============================================================

df[event_col] = pd.to_numeric(
    df[event_col],
    errors="coerce"
)

df[duration_col] = pd.to_numeric(
    df[duration_col],
    errors="coerce"
)

df = df[
    df[event_col].isin([0, 1])
    & df[duration_col].notna()
    & np.isfinite(df[duration_col])
    & (df[duration_col] > 0)
].copy()

df[event_col] = df[event_col].astype(np.uint8)
df[duration_col] = df[duration_col].astype(np.float32)

print("Valid dataset shape:", df.shape)

print("\nEvent distribution:")
print(df[event_col].value_counts().sort_index())

print(
    f"Event proportion: "
    f"{df[event_col].mean() * 100:.2f}%"
)


# ============================================================
# Event-preserving sampling
# ============================================================

if MAX_ROWS is not None and len(df) > MAX_ROWS:
    event_df = df[df[event_col] == 1].copy()
    censored_df = df[df[event_col] == 0].copy()

    number_of_censored_needed = MAX_ROWS - len(event_df)

    if number_of_censored_needed <= 0:
        raise ValueError(
            "MAX_ROWS must exceed the number of event records."
        )

    censored_sample = censored_df.sample(
        n=min(number_of_censored_needed, len(censored_df)),
        random_state=RANDOM_STATE,
        replace=False,
    )

    df = pd.concat(
        [event_df, censored_sample],
        axis=0,
        ignore_index=True,
    )

    df = df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    del event_df
    del censored_df
    del censored_sample
    gc.collect()

print(
    "\nDataset after memory-controlled sampling:",
    df.shape
)
print("Events retained:", int(df[event_col].sum()))
print(
    f"Sample event proportion: "
    f"{df[event_col].mean() * 100:.2f}%"
)


# ============================================================
# Define feature types
# ============================================================

numeric_cols = [
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

numeric_cols = [
    column
    for column in numeric_cols
    if column in df.columns
]

feature_cols = [
    column
    for column in df.columns
    if column not in [duration_col, event_col]
]

categorical_cols = [
    column
    for column in feature_cols
    if column not in numeric_cols
]

print("\nFeature information")
print("-------------------")
print("Total predictor columns:", len(feature_cols))
print("Numeric predictors     :", len(numeric_cols))
print("Categorical predictors :", len(categorical_cols))


# ============================================================
# Clean predictor columns
# ============================================================

print("\nCleaning numeric columns...")

for column in numeric_cols:
    df[column] = clean_numeric_column(df[column])

print("Cleaning categorical columns...")

for column in categorical_cols:
    df[column] = clean_categorical_column(df[column])

# Critical protection against pd.NA reaching scikit-learn.
df = df.astype(
    {
        column: object
        for column in categorical_cols
    }
)

df[categorical_cols] = df[categorical_cols].replace(
    {pd.NA: np.nan}
)

# Verify that no pandas nullable values remain.
pd_na_columns = []

for column in categorical_cols:
    contains_pd_na = df[column].map(
        lambda value: value is pd.NA
    ).any()

    if contains_pd_na:
        pd_na_columns.append(column)

if pd_na_columns:
    raise ValueError(
        "pd.NA values remain in these columns: "
        f"{pd_na_columns}"
    )

print("Categorical pd.NA check: passed")


# ============================================================
# Construct features and survival target
# ============================================================

X = df[feature_cols].copy()

event_values = df[event_col].astype(bool).to_numpy()
time_values = df[duration_col].astype(float).to_numpy()

y = Surv.from_arrays(
    event=event_values,
    time=time_values,
)

indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=event_values,
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y[train_indices]
y_test = y[test_indices]

del X
del df
del event_values
del time_values
gc.collect()

print("\nData split")
print("----------")
print("Training rows  :", len(X_train))
print("Testing rows   :", len(X_test))
print("Training events:", int(y_train["event"].sum()))
print("Testing events :", int(y_test["event"].sum()))


# ============================================================
# Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                missing_values=np.nan,
                strategy="median",
                keep_empty_features=True,
            ),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                missing_values=np.nan,
                strategy="constant",
                fill_value="MISSING",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
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
            numeric_cols,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols,
        ),
    ],
    remainder="drop",
    sparse_threshold=0.0,
    verbose_feature_names_out=False,
    n_jobs=1,
)

print("\nPreprocessing training data...")

preprocessing_start = time.perf_counter()

X_train_processed = preprocessor.fit_transform(X_train)

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32,
    order="C",
)

training_preprocessing_seconds = (
    time.perf_counter() - preprocessing_start
)

print(
    "Training matrix shape:",
    X_train_processed.shape
)
print(
    "Training matrix memory:",
    f"{X_train_processed.nbytes / (1024 ** 2):.2f} MB"
)

del X_train
gc.collect()

print("\nPreprocessing testing data...")

test_preprocessing_start = time.perf_counter()

X_test_processed = preprocessor.transform(X_test)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32,
    order="C",
)

testing_preprocessing_seconds = (
    time.perf_counter() - test_preprocessing_start
)

print(
    "Testing matrix shape:",
    X_test_processed.shape
)
print(
    "Testing matrix memory:",
    f"{X_test_processed.nbytes / (1024 ** 2):.2f} MB"
)

del X_test
gc.collect()

if np.isnan(X_train_processed).any():
    raise ValueError(
        "NaN values remain in the processed training matrix."
    )

if np.isnan(X_test_processed).any():
    raise ValueError(
        "NaN values remain in the processed testing matrix."
    )

if not np.isfinite(X_train_processed).all():
    raise ValueError(
        "Infinite values remain in the training matrix."
    )

if not np.isfinite(X_test_processed).all():
    raise ValueError(
        "Infinite values remain in the testing matrix."
    )

print("Processed matrix validation: passed")


# ============================================================
# Random Survival Forest model
# ============================================================

rsf = RandomSurvivalForest(
    n_estimators=N_ESTIMATORS,

    # Restrict tree complexity and memory consumption.
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,

    # Test only a subset of features at each node.
    max_features="sqrt",

    # Bootstrap subset used for each tree.
    bootstrap=True,
    max_samples=MAX_SAMPLES,

    # One process prevents parallel copies of the dataset.
    n_jobs=1,

    # Predict risk scores using reduced memory.
    low_memory=True,

    random_state=RANDOM_STATE,
    verbose=1,
)


# ============================================================
# Train model
# ============================================================

print("\nTraining Random Survival Forest...")
print(
    f"Configuration: {N_ESTIMATORS} trees, "
    f"maximum depth {MAX_DEPTH}"
)

training_start = time.perf_counter()

rsf.fit(
    X_train_processed,
    y_train,
)

training_seconds = time.perf_counter() - training_start


# ============================================================
# Predict and evaluate
# ============================================================

print("\nGenerating test risk scores...")

prediction_start = time.perf_counter()

risk_scores = rsf.predict(
    X_test_processed
)

inference_seconds = time.perf_counter() - prediction_start

c_index_result = concordance_index_censored(
    event_indicator=y_test["event"],
    event_time=y_test["time"],
    estimate=risk_scores,
)

c_index = float(c_index_result[0])
concordant_pairs = int(c_index_result[1])
discordant_pairs = int(c_index_result[2])
tied_risk_pairs = int(c_index_result[3])
tied_time_pairs = int(c_index_result[4])


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 60)
print("RANDOM SURVIVAL FOREST RESULTS")
print("=" * 60)
print(f"Sample rows           : {len(y):,}")
print(f"Training rows         : {len(y_train):,}")
print(f"Testing rows          : {len(y_test):,}")
print(f"Training events       : {int(y_train['event'].sum()):,}")
print(f"Testing events        : {int(y_test['event'].sum()):,}")
print(f"Processed features    : {X_train_processed.shape[1]}")
print(f"Trees                 : {N_ESTIMATORS}")
print(f"Maximum depth         : {MAX_DEPTH}")
print(f"Minimum leaf size     : {MIN_SAMPLES_LEAF}")
print(f"Test C-index          : {c_index:.4f}")
print(
    f"Preprocessing time    : "
    f"{training_preprocessing_seconds:.2f} seconds"
)
print(f"Training time         : {training_seconds:.2f} seconds")
print(f"Inference time        : {inference_seconds:.2f} seconds")


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame(
    {
        "EVENT": y_test["event"].astype(int),
        "TIME_YEARS": y_test["time"],
        "RSF_RISK_SCORE": risk_scores,
    }
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# Save model
# ============================================================

saved_model = {
    "preprocessor": preprocessor,
    "model": rsf,
    "feature_columns": feature_cols,
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "sampling_configuration": {
        "maximum_rows": MAX_ROWS,
        "all_events_retained": True,
        "random_state": RANDOM_STATE,
    },
}

print("\nSaving model...")

joblib.dump(
    saved_model,
    MODEL_FILE,
    compress=3,
)


# ============================================================
# Save results
# ============================================================

results = {
    "model": "Random Survival Forest",
    "data_file": str(DATA_FILE),
    "sampling_method": (
        "All events retained with random sampling of censored records"
    ),
    "maximum_sample_rows": MAX_ROWS,
    "sample_rows": int(len(y)),
    "training_rows": int(len(y_train)),
    "testing_rows": int(len(y_test)),
    "training_events": int(y_train["event"].sum()),
    "testing_events": int(y_test["event"].sum()),
    "numeric_features": len(numeric_cols),
    "categorical_features": len(categorical_cols),
    "processed_features": int(X_train_processed.shape[1]),
    "parameters": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": "sqrt",
        "max_samples": MAX_SAMPLES,
        "bootstrap": True,
        "low_memory": True,
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
    },
    "test_c_index": c_index,
    "concordant_pairs": concordant_pairs,
    "discordant_pairs": discordant_pairs,
    "tied_risk_pairs": tied_risk_pairs,
    "tied_time_pairs": tied_time_pairs,
    "training_preprocessing_seconds": (
        training_preprocessing_seconds
    ),
    "testing_preprocessing_seconds": (
        testing_preprocessing_seconds
    ),
    "training_seconds": training_seconds,
    "inference_seconds": inference_seconds,
}

with RESULTS_FILE.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=4,
    )

print("\nSaved:")
print(f"- {MODEL_FILE}")
print(f"- {RESULTS_FILE}")
print(f"- {PREDICTIONS_FILE}")