import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored


# =========================================================
# Configuration
# =========================================================
DATA_FILE = Path("MODEL_DATASET_LEAKAGE_FREE.csv")
MODEL_FILE = Path("xgboost_survival_cox.json")
PREPROCESSOR_FILE = Path("xgboost_preprocessor.pkl")
RESULTS_FILE = Path("xgboost_survival_results.json")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# Load data
# =========================================================
df = pd.read_csv(DATA_FILE, low_memory=False)

required_columns = {"EVENT", "TIME_YEARS"}
missing_required = required_columns.difference(df.columns)

if missing_required:
    raise ValueError(f"Missing required columns: {sorted(missing_required)}")

# Identifier and transplant date are not model predictors.
df = df.drop(columns=["TRR_ID_CODE", "TX_DATE"], errors="ignore")

# Remove invalid target rows.
df["EVENT"] = pd.to_numeric(df["EVENT"], errors="coerce")
df["TIME_YEARS"] = pd.to_numeric(df["TIME_YEARS"], errors="coerce")

df = df[
    df["EVENT"].isin([0, 1])
    & df["TIME_YEARS"].notna()
    & (df["TIME_YEARS"] > 0)
].copy()

print("Dataset shape:", df.shape)
print("\nEvent distribution:")
print(df["EVENT"].value_counts())


# =========================================================
# Define numeric variables
# =========================================================
numeric_columns = [
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

numeric_columns = [column for column in numeric_columns if column in df.columns]

target_columns = ["EVENT", "TIME_YEARS"]
feature_columns = [
    column for column in df.columns
    if column not in target_columns
]

categorical_columns = [
    column for column in feature_columns
    if column not in numeric_columns
]


# =========================================================
# Clean OPTN values
# =========================================================
for column in numeric_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .replace(
            {
                ".": np.nan,
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
            }
        )
    )

    df[column] = pd.to_numeric(df[column], errors="coerce")

for column in categorical_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .replace(
            {
                ".": np.nan,
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
            }
        )
    )


# =========================================================
# Split data
# =========================================================
X = df[feature_columns].copy()
event = df["EVENT"].astype(int).to_numpy()
time_years = df["TIME_YEARS"].astype(float).to_numpy()

indices = np.arange(len(df))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=event,
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

event_train = event[train_indices]
event_test = event[test_indices]

time_train = time_years[train_indices]
time_test = time_years[test_indices]

# XGBoost survival:cox label convention:
# event observed  -> positive time
# right censored  -> negative time
cox_label_train = np.where(
    event_train == 1,
    time_train,
    -time_train,
)

cox_label_test = np.where(
    event_test == 1,
    time_test,
    -time_test,
)


# =========================================================
# Preprocessing
# =========================================================
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                min_frequency=100,
                dtype=np.float32,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    sparse_threshold=1.0,
)

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Ensure memory-efficient CSR float32 matrices.
X_train_processed = sparse.csr_matrix(
    X_train_processed,
    dtype=np.float32,
)

X_test_processed = sparse.csr_matrix(
    X_test_processed,
    dtype=np.float32,
)

print("Training matrix:", X_train_processed.shape)
print("Testing matrix :", X_test_processed.shape)


# =========================================================
# XGBoost matrices
# =========================================================
dtrain = xgb.DMatrix(
    X_train_processed,
    label=cox_label_train,
)

dtest = xgb.DMatrix(
    X_test_processed,
    label=cox_label_test,
)


# =========================================================
# Train XGBoost Survival-Cox
# =========================================================
parameters = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",

    # Tree configuration
    "tree_method": "hist",
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.80,
    "colsample_bytree": 0.80,

    # Learning configuration
    "eta": 0.03,
    "lambda": 1.0,
    "alpha": 0.1,

    # Reproducibility and memory
    "seed": RANDOM_STATE,
    "nthread": 4,
}

evaluation_sets = [
    (dtrain, "train"),
    (dtest, "validation"),
]

print("\nTraining XGBoost Survival-Cox...")

start_time = time.perf_counter()

model = xgb.train(
    params=parameters,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evaluation_sets,
    early_stopping_rounds=50,
    verbose_eval=25,
)

training_seconds = time.perf_counter() - start_time


# =========================================================
# Evaluate
# =========================================================
prediction_start = time.perf_counter()

risk_scores = model.predict(
    dtest,
    iteration_range=(0, model.best_iteration + 1),
)

inference_seconds = time.perf_counter() - prediction_start

# Higher XGBoost prediction = higher hazard/risk.
c_index_result = concordance_index_censored(
    event_indicator=event_test.astype(bool),
    event_time=time_test,
    estimate=risk_scores,
)

c_index = float(c_index_result[0])

print("\n" + "=" * 55)
print("XGBOOST SURVIVAL-COX RESULTS")
print("=" * 55)
print(f"Best iteration : {model.best_iteration}")
print(f"Best score     : {model.best_score:.6f}")
print(f"Test C-index   : {c_index:.4f}")
print(f"Training time  : {training_seconds:.2f} seconds")
print(f"Inference time : {inference_seconds:.2f} seconds")


# =========================================================
# Save model, preprocessing and results
# =========================================================
model.save_model(MODEL_FILE)
joblib.dump(preprocessor, PREPROCESSOR_FILE)

results = {
    "model": "XGBoost Survival-Cox",
    "rows_total": int(len(df)),
    "rows_train": int(len(train_indices)),
    "rows_test": int(len(test_indices)),
    "events_train": int(event_train.sum()),
    "events_test": int(event_test.sum()),
    "processed_features": int(X_train_processed.shape[1]),
    "best_iteration": int(model.best_iteration),
    "best_validation_cox_nloglik": float(model.best_score),
    "test_c_index": c_index,
    "training_seconds": training_seconds,
    "inference_seconds": inference_seconds,
    "parameters": parameters,
}

with RESULTS_FILE.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nSaved:")
print(f"- {MODEL_FILE}")
print(f"- {PREPROCESSOR_FILE}")
print(f"- {RESULTS_FILE}")