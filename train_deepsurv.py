import copy
import json
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lifelines.utils import concordance_index
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "MODEL_DATASET_LEAKAGE_FREE.csv"

DURATION_COL = "TIME_YEARS"
EVENT_COL = "EVENT"

DROP_COLUMNS = [
    "TRR_ID_CODE",
    "TX_DATE",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 300
PATIENCE = 30

HIDDEN_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.30

TOP_CATEGORIES = 50

MODEL_FILE = "deepsurv_model.pt"
PREPROCESSOR_FILE = "deepsurv_preprocessor.pkl"
RESULTS_FILE = "deepsurv_results.json"
PREDICTIONS_FILE = "deepsurv_test_predictions.csv"


# ============================================================
# Reproducibility
# ============================================================

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(RANDOM_STATE)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("DEEPSURV")
print("=" * 65)
print("Device:", device)


# ============================================================
# Load dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

print("Original dataset shape:", df.shape)


# ============================================================
# Remove identifier/date columns
# ============================================================

df = df.drop(
    columns=[
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ]
)


# ============================================================
# Validate outcomes
# ============================================================

required_columns = {
    DURATION_COL,
    EVENT_COL,
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
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
    df[DURATION_COL].notna()
    & np.isfinite(df[DURATION_COL])
    & (df[DURATION_COL] > 0)
    & df[EVENT_COL].isin([0, 1])
)

df = df.loc[valid_mask].copy()

df[DURATION_COL] = df[DURATION_COL].astype(np.float32)
df[EVENT_COL] = df[EVENT_COL].astype(np.int64)

print("Valid dataset shape:", df.shape)

print("\nEvent distribution:")
print(df[EVENT_COL].value_counts().sort_index())

print(
    f"Event proportion: "
    f"{100 * df[EVENT_COL].mean():.2f}%"
)


# ============================================================
# Numeric columns
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
print("Total predictors      :", len(feature_cols))
print("Numeric predictors    :", len(numeric_cols))
print("Categorical predictors:", len(categorical_cols))


# ============================================================
# Clean numeric data
# ============================================================

missing_tokens = [
    ".",
    "",
    " ",
    "nan",
    "NaN",
    "None",
    "NULL",
    "null",
    "<NA>",
]

for column in numeric_cols:
    cleaned = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .replace(missing_tokens, np.nan)
    )

    df[column] = pd.to_numeric(
        cleaned,
        errors="coerce",
    )


# ============================================================
# Clean categorical data
# ============================================================

for column in categorical_cols:
    cleaned = (
        df[column]
        .astype("string")
        .str.strip()
        .replace(missing_tokens, pd.NA)
    )

    df[column] = cleaned


# ============================================================
# Train/test split
# ============================================================

train_full_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[EVENT_COL],
)

train_df, validation_df = train_test_split(
    train_full_df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_full_df[EVENT_COL],
)

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

print("\nData split")
print("----------")
print("Training rows   :", len(train_df))
print("Validation rows :", len(validation_df))
print("Testing rows    :", len(test_df))

print("Training events   :", int(train_df[EVENT_COL].sum()))
print("Validation events :", int(validation_df[EVENT_COL].sum()))
print("Testing events    :", int(test_df[EVENT_COL].sum()))


# ============================================================
# Limit categories using training data only
# ============================================================

for column in categorical_cols:
    top_values = (
        train_df[column]
        .value_counts(dropna=True)
        .head(TOP_CATEGORIES)
        .index
    )

    train_df[column] = train_df[column].where(
        train_df[column].isin(top_values),
        "OTHER",
    )

    validation_df[column] = validation_df[column].where(
        validation_df[column].isin(top_values),
        "OTHER",
    )

    test_df[column] = test_df[column].where(
        test_df[column].isin(top_values),
        "OTHER",
    )


# ============================================================
# Preprocessing pipeline
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
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
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
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
)

print("\nFitting preprocessing pipeline...")

X_train = preprocessor.fit_transform(
    train_df[feature_cols]
)

X_validation = preprocessor.transform(
    validation_df[feature_cols]
)

X_test = preprocessor.transform(
    test_df[feature_cols]
)

X_train = np.asarray(
    X_train,
    dtype=np.float32,
)

X_validation = np.asarray(
    X_validation,
    dtype=np.float32,
)

X_test = np.asarray(
    X_test,
    dtype=np.float32,
)

print("Training matrix  :", X_train.shape)
print("Validation matrix:", X_validation.shape)
print("Testing matrix   :", X_test.shape)


# ============================================================
# Survival outcomes
# ============================================================

time_train = train_df[DURATION_COL].to_numpy(
    dtype=np.float32
)

event_train = train_df[EVENT_COL].to_numpy(
    dtype=np.float32
)

time_validation = validation_df[
    DURATION_COL
].to_numpy(dtype=np.float32)

event_validation = validation_df[
    EVENT_COL
].to_numpy(dtype=np.float32)

time_test = test_df[DURATION_COL].to_numpy(
    dtype=np.float32
)

event_test = test_df[EVENT_COL].to_numpy(
    dtype=np.float32
)


# ============================================================
# Convert to tensors
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32,
    device=device,
)

time_train_tensor = torch.tensor(
    time_train,
    dtype=torch.float32,
    device=device,
)

event_train_tensor = torch.tensor(
    event_train,
    dtype=torch.float32,
    device=device,
)

X_validation_tensor = torch.tensor(
    X_validation,
    dtype=torch.float32,
    device=device,
)

time_validation_tensor = torch.tensor(
    time_validation,
    dtype=torch.float32,
    device=device,
)

event_validation_tensor = torch.tensor(
    event_validation,
    dtype=torch.float32,
    device=device,
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
    device=device,
)


# ============================================================
# DeepSurv neural network
# ============================================================

class DeepSurv(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_units,
        dropout_rate,
    ):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_units:
            layers.extend(
                [
                    nn.Linear(
                        previous_size,
                        hidden_size,
                    ),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

            previous_size = hidden_size

        layers.append(
            nn.Linear(previous_size, 1)
        )

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================
# Negative Cox partial log-likelihood
# ============================================================

def negative_cox_log_likelihood(
    risk_scores,
    durations,
    events,
):
    order = torch.argsort(
        durations,
        descending=True,
    )

    sorted_risk = risk_scores[order]
    sorted_events = events[order]

    log_cumulative_hazard = torch.logcumsumexp(
        sorted_risk,
        dim=0,
    )

    partial_likelihood = (
        sorted_risk - log_cumulative_hazard
    )

    observed_likelihood = (
        partial_likelihood * sorted_events
    )

    number_of_events = torch.clamp(
        sorted_events.sum(),
        min=1.0,
    )

    return -observed_likelihood.sum() / number_of_events


# ============================================================
# C-index function
# ============================================================

def calculate_c_index(
    model,
    features,
    durations,
    events,
):
    model.eval()

    with torch.no_grad():
        risk_scores = model(
            features
        ).detach().cpu().numpy()

    return concordance_index(
        event_times=durations,
        predicted_scores=-risk_scores,
        event_observed=events,
    )


# ============================================================
# Create model
# ============================================================

model = DeepSurv(
    input_size=X_train.shape[1],
    hidden_units=HIDDEN_UNITS,
    dropout_rate=DROPOUT_RATE,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
)


# ============================================================
# Train model
# ============================================================

print("\nTraining DeepSurv...")
print(
    f"Architecture: "
    f"{X_train.shape[1]} -> "
    f"{' -> '.join(map(str, HIDDEN_UNITS))} -> 1"
)

training_start = time.time()

best_validation_loss = float("inf")
best_model_state = None
best_epoch = 0
epochs_without_improvement = 0

training_history = []

for epoch in range(1, MAX_EPOCHS + 1):

    # Training
    model.train()

    optimizer.zero_grad()

    train_risk_scores = model(
        X_train_tensor
    )

    train_loss = negative_cox_log_likelihood(
        train_risk_scores,
        time_train_tensor,
        event_train_tensor,
    )

    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0,
    )

    optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():
        validation_risk_scores = model(
            X_validation_tensor
        )

        validation_loss = (
            negative_cox_log_likelihood(
                validation_risk_scores,
                time_validation_tensor,
                event_validation_tensor,
            )
        )

    train_loss_value = train_loss.item()
    validation_loss_value = validation_loss.item()

    scheduler.step(validation_loss_value)

    validation_c_index = calculate_c_index(
        model,
        X_validation_tensor,
        time_validation,
        event_validation,
    )

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    training_history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss_value,
            "validation_loss": validation_loss_value,
            "validation_c_index": validation_c_index,
            "learning_rate": current_learning_rate,
        }
    )

    if validation_loss_value < (
        best_validation_loss - 1e-5
    ):
        best_validation_loss = validation_loss_value
        best_epoch = epoch

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss_value:.5f} | "
            f"Validation loss: "
            f"{validation_loss_value:.5f} | "
            f"Validation C-index: "
            f"{validation_c_index:.4f} | "
            f"LR: {current_learning_rate:.6f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}."
        )
        break


training_time = time.time() - training_start


# ============================================================
# Restore best model
# ============================================================

if best_model_state is None:
    raise RuntimeError(
        "Training failed to produce a valid model."
    )

model.load_state_dict(best_model_state)


# ============================================================
# Evaluate test set
# ============================================================

inference_start = time.time()

model.eval()

with torch.no_grad():
    test_risk_scores = (
        model(X_test_tensor)
        .detach()
        .cpu()
        .numpy()
        .ravel()
    )

inference_time = time.time() - inference_start

test_c_index = concordance_index(
    event_times=time_test,
    predicted_scores=-test_risk_scores,
    event_observed=event_test,
)

validation_c_index = calculate_c_index(
    model,
    X_validation_tensor,
    time_validation,
    event_validation,
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 65)
print("DEEPSURV RESULTS")
print("=" * 65)

print(f"Best epoch          : {best_epoch}")
print(
    f"Best validation loss: "
    f"{best_validation_loss:.6f}"
)
print(
    f"Validation C-index  : "
    f"{validation_c_index:.4f}"
)
print(f"Test C-index        : {test_c_index:.4f}")
print(f"Training time       : {training_time:.2f} seconds")
print(f"Inference time      : {inference_time:.4f} seconds")


# ============================================================
# Save model
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": X_train.shape[1],
        "hidden_units": HIDDEN_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "best_epoch": best_epoch,
        "test_c_index": test_c_index,
    },
    MODEL_FILE,
)

with open(
    PREPROCESSOR_FILE,
    "wb",
) as file:
    pickle.dump(
        {
            "preprocessor": preprocessor,
            "feature_columns": feature_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        },
        file,
    )


# ============================================================
# Save results
# ============================================================

results = {
    "dataset": DATA_FILE,
    "random_state": RANDOM_STATE,
    "training_rows": len(train_df),
    "validation_rows": len(validation_df),
    "testing_rows": len(test_df),
    "training_events": int(event_train.sum()),
    "validation_events": int(event_validation.sum()),
    "testing_events": int(event_test.sum()),
    "processed_features": int(X_train.shape[1]),
    "hidden_units": HIDDEN_UNITS,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "best_epoch": best_epoch,
    "best_validation_loss": float(
        best_validation_loss
    ),
    "validation_c_index": float(
        validation_c_index
    ),
    "test_c_index": float(test_c_index),
    "training_time_seconds": float(
        training_time
    ),
    "inference_time_seconds": float(
        inference_time
    ),
}

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=4,
    )


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame(
    {
        "TIME_YEARS": time_test,
        "EVENT": event_test.astype(int),
        "DEEPSURV_RISK_SCORE": test_risk_scores,
    }
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# Save training history
# ============================================================

pd.DataFrame(training_history).to_csv(
    "deepsurv_training_history.csv",
    index=False,
)

print("\nSaved:")
print(f"- {MODEL_FILE}")
print(f"- {PREPROCESSOR_FILE}")
print(f"- {RESULTS_FILE}")
print(f"- {PREDICTIONS_FILE}")
print("- deepsurv_training_history.csv")