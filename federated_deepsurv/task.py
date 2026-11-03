from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy import sparse
from torch.utils.data import TensorDataset, DataLoader

from lifelines.utils import concordance_index

from .model import (
    create_model,
    negative_cox_log_likelihood,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Original processed directory:
# contains untouched client test sets + global test set
PROCESSED_DIR = (
    BASE_DIR
    / "federated_data"
    / "processed"
)

# New processed directory:
# contains final training + validation datasets
PROCESSED_VALIDATION_DIR = (
    BASE_DIR
    / "federated_data"
    / "processed_validation"
)

INPUT_DIM = 108

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

LOCAL_EPOCHS = 10
BATCH_SIZE = 128

RANDOM_STATE = 42

NUM_CLIENTS = 5


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Helper: load X and y
# ============================================================

def _load_processed_dataset(
    x_file: Path,
    y_file: Path,
    dataset_name: str,
):
    """
    Load a processed feature matrix and survival labels.
    """

    if not x_file.exists():
        raise FileNotFoundError(
            f"Missing file: {x_file}"
        )

    if not y_file.exists():
        raise FileNotFoundError(
            f"Missing file: {y_file}"
        )

    X = sparse.load_npz(
        x_file
    ).astype(
        np.float32
    )

    X = X.toarray()

    y = pd.read_csv(
        y_file
    )

    if "EVENT" not in y.columns:
        raise ValueError(
            f"EVENT missing from {dataset_name}"
        )

    if "TIME_YEARS" not in y.columns:
        raise ValueError(
            f"TIME_YEARS missing from {dataset_name}"
        )

    events = (
        y["EVENT"]
        .to_numpy(
            dtype=np.float32
        )
    )

    times = (
        y["TIME_YEARS"]
        .to_numpy(
            dtype=np.float32
        )
    )

    if X.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{dataset_name} feature dimension "
            f"{X.shape[1]} != {INPUT_DIM}"
        )

    if len(X) != len(events):
        raise ValueError(
            f"{dataset_name}: "
            f"X rows ({len(X)}) != "
            f"label rows ({len(events)})"
        )

    if not np.isin(
        events,
        [0, 1]
    ).all():
        raise ValueError(
            f"{dataset_name} contains "
            f"invalid EVENT values."
        )

    if (
        np.isnan(times).any()
        or np.isinf(times).any()
        or (times <= 0).any()
    ):
        raise ValueError(
            f"{dataset_name} contains "
            f"invalid TIME_YEARS values."
        )

    return (
        X,
        events,
        times,
    )


# ============================================================
# Load client TRAIN + TEST
# ============================================================

def load_client_data(
    client_id: int,
):
    """
    Load:
        - FINAL training dataset
        - original untouched client test dataset

    This function keeps compatibility with existing
    FedAvg, FedProx and FedAdam clients.
    """

    if client_id < 1 or client_id > NUM_CLIENTS:
        raise ValueError(
            f"Invalid client_id: {client_id}"
        )

    # --------------------------------------------------------
    # FINAL TRAINING DATA
    # --------------------------------------------------------

    train_x_file = (
        PROCESSED_VALIDATION_DIR
        / f"client_{client_id}_train_X.npz"
    )

    train_y_file = (
        PROCESSED_VALIDATION_DIR
        / f"client_{client_id}_train_y.csv"
    )

    # --------------------------------------------------------
    # ORIGINAL UNTOUCHED LOCAL TEST DATA
    # --------------------------------------------------------

    test_x_file = (
        PROCESSED_DIR
        / f"client_{client_id}_test_X.npz"
    )

    test_y_file = (
        PROCESSED_DIR
        / f"client_{client_id}_test_y.csv"
    )

    (
        X_train,
        event_train,
        time_train,
    ) = _load_processed_dataset(
        train_x_file,
        train_y_file,
        f"Client {client_id} final training",
    )

    (
        X_test,
        event_test,
        time_test,
    ) = _load_processed_dataset(
        test_x_file,
        test_y_file,
        f"Client {client_id} test",
    )

    return (
        X_train,
        event_train,
        time_train,
        X_test,
        event_test,
        time_test,
    )


# ============================================================
# Load client VALIDATION data
# ============================================================

def load_client_validation_data(
    client_id: int,
):
    """
    Load one client's validation dataset.
    """

    if client_id < 1 or client_id > NUM_CLIENTS:
        raise ValueError(
            f"Invalid client_id: {client_id}"
        )

    validation_x_file = (
        PROCESSED_VALIDATION_DIR
        / f"client_{client_id}_validation_X.npz"
    )

    validation_y_file = (
        PROCESSED_VALIDATION_DIR
        / f"client_{client_id}_validation_y.csv"
    )

    return _load_processed_dataset(
        validation_x_file,
        validation_y_file,
        f"Client {client_id} validation",
    )


# ============================================================
# Load pooled GLOBAL VALIDATION data
# ============================================================

def load_global_validation_data():
    """
    Pool validation sets from all five centres.

    Used for:
        - communication-round selection
        - best-model checkpoint selection

    This must NOT be used as the final test set.
    """

    X_parts = []
    event_parts = []
    time_parts = []

    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):
        (
            X_client,
            event_client,
            time_client,
        ) = load_client_validation_data(
            client_id
        )

        X_parts.append(
            X_client
        )

        event_parts.append(
            event_client
        )

        time_parts.append(
            time_client
        )

    X_validation = np.concatenate(
        X_parts,
        axis=0,
    )

    event_validation = np.concatenate(
        event_parts,
        axis=0,
    )

    time_validation = np.concatenate(
        time_parts,
        axis=0,
    )

    return (
        X_validation,
        event_validation,
        time_validation,
    )


# ============================================================
# Load GLOBAL TEST data
# ============================================================

def load_global_test_data():
    """
    Load the fixed global test dataset.

    IMPORTANT:
    This dataset must only be used for final evaluation.
    """

    x_file = (
        PROCESSED_DIR
        / "global_test_X.npz"
    )

    y_file = (
        PROCESSED_DIR
        / "global_test_y.csv"
    )

    return _load_processed_dataset(
        x_file,
        y_file,
        "Global test",
    )


# ============================================================
# Convert arrays to tensors
# ============================================================

def to_tensors(
    X,
    events,
    times,
):
    """
    Convert NumPy arrays to PyTorch tensors.
    """

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    event_tensor = torch.tensor(
        events,
        dtype=torch.float32,
        device=DEVICE,
    )

    time_tensor = torch.tensor(
        times,
        dtype=torch.float32,
        device=DEVICE,
    )

    return (
        X_tensor,
        event_tensor,
        time_tensor,
    )


# ============================================================
# Standard local DeepSurv training
# ============================================================

def train_local_model(
    model,
    X_train,
    event_train,
    time_train,
    local_epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """
    Train DeepSurv locally.

    Full-batch optimization is used because Cox partial
    likelihood depends on the local risk set.
    """

    model = model.to(
        DEVICE
    )

    (
        X_tensor,
        event_tensor,
        time_tensor,
    ) = to_tensors(
        X_train,
        event_train,
        time_train,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []

    for epoch in range(
        1,
        local_epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        risk_scores = model(
            X_tensor
        )

        loss = (
            negative_cox_log_likelihood(
                risk_scores,
                time_tensor,
                event_tensor,
            )
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite Cox loss detected "
                f"during local training: "
                f"{loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        history.append(
            float(
                loss.item()
            )
        )

    final_loss = (
        history[-1]
    )

    return (
        model,
        final_loss,
    )


# ============================================================
# FedProx local training
# ============================================================

def train_local_model_fedprox(
    model,
    X_train,
    event_train,
    time_train,
    local_epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    proximal_mu=0.1,
):
    """
    Train DeepSurv using FedProx.

    Objective:

        Cox loss
        +
        (mu / 2) *
        || local_parameters - global_parameters ||^2

    IMPORTANT:
    The model must already contain the global weights
    received from the Flower server.
    """

    model = model.to(
        DEVICE
    )

    (
        X_tensor,
        event_tensor,
        time_tensor,
    ) = to_tensors(
        X_train,
        event_train,
        time_train,
    )

    # --------------------------------------------------------
    # Frozen copy of the global TRAINABLE parameters
    # --------------------------------------------------------

    global_parameters = [
        parameter
        .detach()
        .clone()
        for parameter
        in model.parameters()
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    final_total_loss = None
    final_cox_loss = None
    final_prox_loss = None

    for epoch in range(
        1,
        local_epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        risk_scores = model(
            X_tensor
        )

        cox_loss = (
            negative_cox_log_likelihood(
                risk_scores,
                time_tensor,
                event_tensor,
            )
        )

        proximal_term = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=DEVICE,
        )

        for (
            local_parameter,
            global_parameter,
        ) in zip(
            model.parameters(),
            global_parameters,
        ):
            proximal_term = (
                proximal_term
                + torch.sum(
                    (
                        local_parameter
                        - global_parameter
                    )
                    ** 2
                )
            )

        prox_loss = (
            proximal_mu
            / 2.0
        ) * proximal_term

        total_loss = (
            cox_loss
            + prox_loss
        )

        if not torch.isfinite(
            total_loss
        ):
            raise RuntimeError(
                "Non-finite FedProx "
                "loss detected."
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        final_total_loss = float(
            total_loss.item()
        )

        final_cox_loss = float(
            cox_loss.item()
        )

        final_prox_loss = float(
            prox_loss.item()
        )

    return (
        model,
        final_total_loss,
        final_cox_loss,
        final_prox_loss,
    )


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X,
    events,
    times,
):
    """
    Evaluate using:
        - Cox loss
        - C-index
    """

    model = model.to(
        DEVICE
    )

    model.eval()

    (
        X_tensor,
        event_tensor,
        time_tensor,
    ) = to_tensors(
        X,
        events,
        times,
    )

    with torch.no_grad():

        risk_scores = model(
            X_tensor
        )

        loss = (
            negative_cox_log_likelihood(
                risk_scores,
                time_tensor,
                event_tensor,
            )
        )

        risk_scores_np = (
            risk_scores
            .detach()
            .cpu()
            .numpy()
            .ravel()
        )

    # DeepSurv:
    # higher output = higher risk.
    #
    # lifelines concordance_index:
    # higher prediction = longer survival.
    #
    # Therefore negate risk scores.
    c_index = concordance_index(
        event_times=times,
        predicted_scores=-risk_scores_np,
        event_observed=events,
    )

    return (
        float(
            loss.item()
        ),
        float(
            c_index
        ),
    )


# ============================================================
# Predict risk scores
# ============================================================

def predict_risk(
    model,
    X,
):
    """
    Return DeepSurv risk scores.

    Higher score = higher predicted hazard/risk.
    """

    model = model.to(
        DEVICE
    )

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():

        risk_scores = model(
            X_tensor
        )

    return (
        risk_scores
        .detach()
        .cpu()
        .numpy()
        .ravel()
    )


# ============================================================
# Create DeepSurv model
# ============================================================

def initialize_model():
    """
    Create a new DeepSurv model.
    """

    model = create_model(
        input_dim=INPUT_DIM
    )

    return model.to(
        DEVICE
    )


# ============================================================
# Optional mini-batch loader
# ============================================================

def create_dataloader(
    X,
    events,
    times,
    batch_size=BATCH_SIZE,
    shuffle=True,
):
    """
    Optional loader for future experiments.

    Current survival training uses full-batch Cox loss.
    """

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    event_tensor = torch.tensor(
        events,
        dtype=torch.float32,
    )

    time_tensor = torch.tensor(
        times,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        event_tensor,
        time_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ============================================================
# Test task.py
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING UPDATED FEDERATED DEEPSURV TASK")
    print("=" * 70)

    print(
        "\nDevice:",
        DEVICE
    )

    # ========================================================
    # Client 1 training + test
    # ========================================================

    (
        X_train,
        event_train,
        time_train,
        X_test,
        event_test,
        time_test,
    ) = load_client_data(
        client_id=1
    )

    print("\nClient 1")
    print("-" * 40)

    print(
        "Final training X:",
        X_train.shape
    )

    print(
        "Training events:",
        int(
            event_train.sum()
        )
    )

    # ========================================================
    # Client 1 validation
    # ========================================================

    (
        X_validation,
        event_validation,
        time_validation,
    ) = load_client_validation_data(
        client_id=1
    )

    print(
        "Validation X:",
        X_validation.shape
    )

    print(
        "Validation events:",
        int(
            event_validation.sum()
        )
    )

    print(
        "Testing X:",
        X_test.shape
    )

    print(
        "Testing events:",
        int(
            event_test.sum()
        )
    )

    # ========================================================
    # Pooled validation
    # ========================================================

    (
        X_global_validation,
        event_global_validation,
        time_global_validation,
    ) = load_global_validation_data()

    print("\nGlobal validation")
    print("-" * 40)

    print(
        "Patients:",
        len(
            X_global_validation
        )
    )

    print(
        "Events:",
        int(
            event_global_validation.sum()
        )
    )

    print(
        "Shape:",
        X_global_validation.shape
    )

    # ========================================================
    # Global test
    # ========================================================

    (
        X_global_test,
        event_global_test,
        time_global_test,
    ) = load_global_test_data()

    print("\nGlobal test")
    print("-" * 40)

    print(
        "Patients:",
        len(
            X_global_test
        )
    )

    print(
        "Events:",
        int(
            event_global_test.sum()
        )
    )

    print(
        "Shape:",
        X_global_test.shape
    )

    # ========================================================
    # Small functionality test
    # ========================================================

    model = initialize_model()

    initial_validation_loss, (
        initial_validation_cindex
    ) = evaluate_model(
        model,
        X_global_validation,
        event_global_validation,
        time_global_validation,
    )

    print("\nInitial model validation")
    print("-" * 40)

    print(
        f"Loss    : "
        f"{initial_validation_loss:.5f}"
    )

    print(
        f"C-index : "
        f"{initial_validation_cindex:.4f}"
    )

    model, train_loss = (
        train_local_model(
            model=model,
            X_train=X_train,
            event_train=event_train,
            time_train=time_train,
            local_epochs=2,
        )
    )

    validation_loss, (
        validation_cindex
    ) = evaluate_model(
        model,
        X_global_validation,
        event_global_validation,
        time_global_validation,
    )

    print("\nAfter 2 local epochs on Client 1")
    print("-" * 40)

    print(
        f"Train loss      : "
        f"{train_loss:.5f}"
    )

    print(
        f"Validation loss : "
        f"{validation_loss:.5f}"
    )

    print(
        f"Validation C-index: "
        f"{validation_cindex:.4f}"
    )

    print("\n" + "=" * 70)
    print("TASK VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "Expected:"
    )

    print(
        "- Client 1 training: "
        "818 patients"
    )

    print(
        "- Client 1 validation: "
        "205 patients"
    )

    print(
        "- Client 1 test: "
        "256 patients"
    )

    print(
        "- Global validation: "
        "674 patients"
    )

    print(
        "- Global validation events: "
        "100"
    )

    print(
        "- Global test: "
        "842 patients"
    )

    print(
        "- Input features: "
        "108"
    )