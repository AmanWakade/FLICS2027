import pandas as pd
import numpy as np

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "MODEL_DATASET_LEAKAGE_FREE.csv"

DURATION_COL = "TIME_YEARS"
EVENT_COL = "EVENT"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_CATEGORIES = 50


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("COX PROPORTIONAL HAZARDS BASELINE")
print("=" * 60)

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

print("\nOriginal shape:", df.shape)


# ============================================================
# Remove non-predictor columns
# ============================================================

drop_cols = [
    "TRR_ID_CODE",
    "TX_DATE",
]

df = df.drop(
    columns=[
        column
        for column in drop_cols
        if column in df.columns
    ]
)


# ============================================================
# Validate survival outcomes
# ============================================================

required_columns = {
    DURATION_COL,
    EVENT_COL,
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Required columns missing: {sorted(missing_columns)}"
    )

df[DURATION_COL] = pd.to_numeric(
    df[DURATION_COL],
    errors="coerce",
)

df[EVENT_COL] = pd.to_numeric(
    df[EVENT_COL],
    errors="coerce",
)

valid_mask = (
    df[EVENT_COL].isin([0, 1])
    & df[DURATION_COL].notna()
    & np.isfinite(df[DURATION_COL])
    & (df[DURATION_COL] > 0)
)

df = df.loc[valid_mask].copy()

df[EVENT_COL] = df[EVENT_COL].astype(np.uint8)
df[DURATION_COL] = df[DURATION_COL].astype(np.float64)

print("Valid shape:", df.shape)

print("\nEvent distribution:")
print(df[EVENT_COL].value_counts().sort_index())


# ============================================================
# Define numeric and categorical predictors
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
    if column not in [
        DURATION_COL,
        EVENT_COL,
    ]
]

categorical_cols = [
    column
    for column in feature_cols
    if column not in numeric_cols
]

print("\nFeature information")
print("-------------------")
print("Numeric predictors     :", len(numeric_cols))
print("Categorical predictors :", len(categorical_cols))


# ============================================================
# Clean numeric predictors
# ============================================================

missing_tokens = {
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

for column in numeric_cols:
    cleaned = (
        df[column]
        .astype(object)
        .replace(missing_tokens)
        .astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .replace(missing_tokens)
    )

    df[column] = pd.to_numeric(
        cleaned,
        errors="coerce",
    )


# ============================================================
# Clean categorical predictors
# ============================================================

for column in categorical_cols:
    cleaned = df[column].astype(object)

    cleaned = cleaned.map(
        lambda value: (
            np.nan
            if value is pd.NA
            else value
        )
    )

    cleaned = cleaned.replace(missing_tokens)

    cleaned = cleaned.map(
        lambda value: (
            str(value).strip()
            if pd.notna(value)
            else np.nan
        )
    )

    cleaned = cleaned.replace(missing_tokens)

    df[column] = cleaned.astype(object)


# ============================================================
# Train/test split
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[EVENT_COL],
)

train_df = train_df.copy()
test_df = test_df.copy()

print("\nData split")
print("----------")
print("Training rows :", len(train_df))
print("Testing rows  :", len(test_df))
print("Training events:", int(train_df[EVENT_COL].sum()))
print("Testing events :", int(test_df[EVENT_COL].sum()))


# ============================================================
# Reduce categorical cardinality using training data only
# ============================================================

for column in categorical_cols:
    top_categories = (
        train_df[column]
        .value_counts(dropna=True)
        .head(TOP_CATEGORIES)
        .index
    )

    train_df[column] = train_df[column].where(
        train_df[column].isin(top_categories),
        "OTHER",
    )

    test_df[column] = test_df[column].where(
        test_df[column].isin(top_categories),
        "OTHER",
    )


# ============================================================
# Numeric imputation
# ============================================================

if numeric_cols:
    numeric_imputer = SimpleImputer(
        strategy="median",
    )

    train_df.loc[:, numeric_cols] = (
        numeric_imputer.fit_transform(
            train_df[numeric_cols]
        )
    )

    test_df.loc[:, numeric_cols] = (
        numeric_imputer.transform(
            test_df[numeric_cols]
        )
    )


# ============================================================
# Categorical imputation
# ============================================================

for column in categorical_cols:
    mode_values = train_df[column].mode(
        dropna=True
    )

    fill_value = (
        mode_values.iloc[0]
        if not mode_values.empty
        else "UNKNOWN"
    )

    train_df[column] = train_df[column].fillna(
        fill_value
    )

    test_df[column] = test_df[column].fillna(
        fill_value
    )


# ============================================================
# One-hot encoding
# ============================================================

train_df = pd.get_dummies(
    train_df,
    columns=categorical_cols,
    drop_first=True,
    dtype=np.float64,
)

test_df = pd.get_dummies(
    test_df,
    columns=categorical_cols,
    drop_first=True,
    dtype=np.float64,
)

# Ensure identical feature layout.
test_df = test_df.reindex(
    columns=train_df.columns,
    fill_value=0,
)

print("\nEncoded matrices")
print("----------------")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)


# ============================================================
# Remove constant predictor columns
# ============================================================

predictor_columns = [
    column
    for column in train_df.columns
    if column not in [
        DURATION_COL,
        EVENT_COL,
    ]
]

constant_columns = [
    column
    for column in predictor_columns
    if train_df[column].nunique(dropna=False) <= 1
]

if constant_columns:
    train_df = train_df.drop(
        columns=constant_columns
    )

    test_df = test_df.drop(
        columns=constant_columns
    )

print(
    "Constant predictors removed:",
    len(constant_columns),
)


# ============================================================
# Final validation
# ============================================================

if train_df.isna().any().any():
    raise ValueError(
        "Training data still contains missing values."
    )

if test_df.isna().any().any():
    raise ValueError(
        "Testing data still contains missing values."
    )

if not np.isfinite(
    train_df.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "Training data contains infinite values."
    )

if not np.isfinite(
    test_df.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "Testing data contains infinite values."
    )


# ============================================================
# Train CoxPH
# ============================================================

print("\nTraining CoxPH model...")

cph = CoxPHFitter(
    penalizer=0.1
)

cph.fit(
    train_df,
    duration_col=DURATION_COL,
    event_col=EVENT_COL,
    show_progress=True,
)


# ============================================================
# Evaluate on test set
# ============================================================

test_features = test_df.drop(
    columns=[
        DURATION_COL,
        EVENT_COL,
    ]
)

risk_scores = cph.predict_partial_hazard(
    test_features
)

c_index = concordance_index(
    event_times=test_df[DURATION_COL],
    predicted_scores=-risk_scores.to_numpy().ravel(),
    event_observed=test_df[EVENT_COL],
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("COXPH BASELINE RESULTS")
print("=" * 60)
print(f"Training records : {len(train_df):,}")
print(f"Testing records  : {len(test_df):,}")
print(f"Predictors       : {len(test_features.columns):,}")
print(f"Test C-index     : {c_index:.4f}")


# ============================================================
# Save outputs
# ============================================================

cph.summary.to_csv(
    "coxph_coefficients_leakage_free.csv"
)

prediction_df = pd.DataFrame(
    {
        "EVENT": test_df[EVENT_COL].to_numpy(),
        "TIME_YEARS": test_df[DURATION_COL].to_numpy(),
        "COXPH_RISK_SCORE": risk_scores.to_numpy().ravel(),
    }
)

prediction_df.to_csv(
    "coxph_test_predictions_leakage_free.csv",
    index=False,
)

print("\nSaved:")
print("- coxph_coefficients_leakage_free.csv")
print("- coxph_test_predictions_leakage_free.csv")