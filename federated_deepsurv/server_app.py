import json
from pathlib import Path

import flwr as fl
import pandas as pd
import torch

from flwr.common import ndarrays_to_parameters

from federated_deepsurv.model import (
    get_model_parameters,
    set_model_parameters,
)

from federated_deepsurv.task import (
    initialize_model,
    load_global_validation_data,
    load_global_test_data,
    evaluate_model,
)


# ============================================================
# Configuration
# ============================================================

NUM_CLIENTS = 5
NUM_ROUNDS = 30

LOCAL_EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

SERVER_ADDRESS = "0.0.0.0:8080"


# ============================================================
# Result paths
# ============================================================

RESULT_DIR = Path("federated_results")
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VALIDATION_HISTORY_FILE = (
    RESULT_DIR
    / "fedavg_validation_history.csv"
)

BEST_MODEL_FILE = (
    RESULT_DIR
    / "fedavg_best_model.pt"
)

BEST_MODEL_INFO_FILE = (
    RESULT_DIR
    / "fedavg_best_model_info.json"
)

FINAL_TEST_RESULTS_FILE = (
    RESULT_DIR
    / "fedavg_final_test_results.json"
)


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)


# ============================================================
# Load GLOBAL VALIDATION data
#
# This is the ONLY server-side dataset used during training.
# ============================================================

print("=" * 70)
print("FEDERATED DEEPSURV - FEDAVG")
print("VALIDATION-BASED CHECKPOINTING")
print("=" * 70)

(
    X_validation,
    event_validation,
    time_validation,
) = load_global_validation_data()

print(
    f"Global validation patients : "
    f"{len(X_validation):,}"
)

print(
    f"Global validation events   : "
    f"{int(event_validation.sum()):,}"
)


# ============================================================
# Initial global model
# ============================================================

initial_model = initialize_model()

initial_parameters = ndarrays_to_parameters(
    get_model_parameters(
        initial_model
    )
)


# ============================================================
# Training configuration sent to clients
# ============================================================

def fit_config(
    server_round: int,
):
    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }


# ============================================================
# Best-checkpoint tracking
# ============================================================

validation_history = []

best_validation_cindex = -float("inf")
best_validation_loss = float("inf")
best_round = -1


# ============================================================
# Validation callback
# ============================================================

def evaluate_validation(
    server_round,
    parameters,
    config,
):
    """
    Evaluate aggregated global model on the pooled
    validation cohort.

    IMPORTANT:
    In Flower legacy FedAvg evaluate_fn,
    `parameters` is already a list of NumPy arrays.
    """

    global best_validation_cindex, best_validation_loss, best_round

    model = initialize_model()

    set_model_parameters(
        model,
        parameters,
    )

    validation_loss, validation_cindex = (
        evaluate_model(
            model=model,
            X=X_validation,
            events=event_validation,
            times=time_validation,
        )
    )

    print("\n" + "=" * 65)
    print(
        f"GLOBAL VALIDATION - ROUND "
        f"{server_round}"
    )
    print("=" * 65)

    print(
        f"Patients : "
        f"{len(X_validation):,}"
    )

    print(
        f"Events   : "
        f"{int(event_validation.sum()):,}"
    )

    print(
        f"Loss     : "
        f"{validation_loss:.5f}"
    )

    print(
        f"C-index  : "
        f"{validation_cindex:.4f}"
    )

    # --------------------------------------------------------
    # Save validation history
    # --------------------------------------------------------

    validation_history.append(
        {
            "ROUND": int(
                server_round
            ),
            "VALIDATION_LOSS": float(
                validation_loss
            ),
            "VALIDATION_C_INDEX": float(
                validation_cindex
            ),
            "VALIDATION_PATIENTS": int(
                len(X_validation)
            ),
            "VALIDATION_EVENTS": int(
                event_validation.sum()
            ),
        }
    )

    pd.DataFrame(
        validation_history
    ).to_csv(
        VALIDATION_HISTORY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save BEST model according to validation C-index
    # --------------------------------------------------------

    if (
        validation_cindex
        > best_validation_cindex
    ):

        best_validation_cindex = float(
            validation_cindex
        )

        best_validation_loss = float(
            validation_loss
        )

        best_round = int(
            server_round
        )

        checkpoint = {
            "round": best_round,
            "validation_c_index": (
                best_validation_cindex
            ),
            "validation_loss": (
                best_validation_loss
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "input_dim": 108,
            "local_epochs": (
                LOCAL_EPOCHS
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
        }

        torch.save(
            checkpoint,
            BEST_MODEL_FILE,
        )

        with open(
            BEST_MODEL_INFO_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "best_round": (
                        best_round
                    ),
                    "best_validation_c_index": (
                        best_validation_cindex
                    ),
                    "best_validation_loss": (
                        best_validation_loss
                    ),
                    "local_epochs": (
                        LOCAL_EPOCHS
                    ),
                    "learning_rate": (
                        LEARNING_RATE
                    ),
                    "weight_decay": (
                        WEIGHT_DECAY
                    ),
                },
                file,
                indent=4,
            )

        print(
            "\n*** NEW BEST VALIDATION "
            "CHECKPOINT ***"
        )

        print(
            f"Round   : {best_round}"
        )

        print(
            f"C-index : "
            f"{best_validation_cindex:.4f}"
        )

        print(
            f"Saved   : "
            f"{BEST_MODEL_FILE}"
        )

    else:

        print(
            f"Best remains Round "
            f"{best_round} "
            f"(C-index="
            f"{best_validation_cindex:.4f})"
        )

    return (
        float(
            validation_loss
        ),
        {
            "validation_c_index": float(
                validation_cindex
            )
        },
    )


# ============================================================
# Weighted training metric aggregation
# ============================================================

def weighted_average(
    metrics,
):
    """
    Weighted aggregation of client training metrics.
    """

    if not metrics:
        return {}

    aggregated = {}

    metric_names = [
        "train_loss",
    ]

    for metric_name in metric_names:

        values = []

        for (
            num_examples,
            metric_dict,
        ) in metrics:

            if metric_name in metric_dict:

                values.append(
                    (
                        num_examples,
                        float(
                            metric_dict[
                                metric_name
                            ]
                        ),
                    )
                )

        if values:

            denominator = sum(
                n
                for n, _
                in values
            )

            aggregated[
                metric_name
            ] = (
                sum(
                    n * value
                    for n, value
                    in values
                )
                / denominator
            )

    return aggregated


# ============================================================
# FedAvg strategy
# ============================================================

strategy = fl.server.strategy.FedAvg(

    # Use every centre for local training
    fraction_fit=1.0,

    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,

    initial_parameters=initial_parameters,

    on_fit_config_fn=fit_config,

    # --------------------------------------------------------
    # SERVER-SIDE validation
    # --------------------------------------------------------

    evaluate_fn=evaluate_validation,

    fit_metrics_aggregation_fn=(
        weighted_average
    ),

    # --------------------------------------------------------
    # IMPORTANT:
    # Disable client test-set evaluation during training.
    #
    # Client test sets should remain untouched until
    # final evaluation.
    # --------------------------------------------------------

    fraction_evaluate=0.0,
    min_evaluate_clients=0,
)


# ============================================================
# Server information
# ============================================================

print("\nServer configuration")
print("--------------------")

print(
    f"Address          : "
    f"{SERVER_ADDRESS}"
)

print(
    f"Clients          : "
    f"{NUM_CLIENTS}"
)

print(
    f"Rounds           : "
    f"{NUM_ROUNDS}"
)

print(
    f"Local epochs     : "
    f"{LOCAL_EPOCHS}"
)

print(
    f"Learning rate    : "
    f"{LEARNING_RATE}"
)

print(
    f"Weight decay     : "
    f"{WEIGHT_DECAY}"
)

print(
    f"Validation rows  : "
    f"{len(X_validation)}"
)

print(
    "\nGlobal test set will NOT "
    "be evaluated during training."
)

print(
    "\nWaiting for 5 clients...\n"
)


# ============================================================
# Start federated training
# ============================================================

history = fl.server.start_server(

    server_address=SERVER_ADDRESS,

    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    ),

    strategy=strategy,
)


# ============================================================
# Training finished
# ============================================================

print("\n" + "=" * 70)
print("FEDAVG TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation round   : "
    f"{best_round}"
)

print(
    f"Best validation C-index : "
    f"{best_validation_cindex:.4f}"
)

print(
    f"Best validation loss    : "
    f"{best_validation_loss:.5f}"
)


# ============================================================
# FINAL TEST EVALUATION
#
# Global test is loaded ONLY NOW.
# ============================================================

print("\n" + "=" * 70)
print("FINAL GLOBAL TEST EVALUATION")
print("=" * 70)

if not BEST_MODEL_FILE.exists():
    raise FileNotFoundError(
        "Best FedAvg checkpoint was not saved."
    )


# ------------------------------------------------------------
# Load untouched global test set
# ------------------------------------------------------------

(
    X_test,
    event_test,
    time_test,
) = load_global_test_data()

print(
    f"Global test patients : "
    f"{len(X_test):,}"
)

print(
    f"Global test events   : "
    f"{int(event_test.sum()):,}"
)


# ------------------------------------------------------------
# Restore best-validation model
# ------------------------------------------------------------

checkpoint = torch.load(
    BEST_MODEL_FILE,
    map_location="cpu",
    weights_only=False,
)

best_model = initialize_model()

best_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ],
    strict=True,
)


# ------------------------------------------------------------
# Evaluate ONCE on global test
# ------------------------------------------------------------

test_loss, test_cindex = (
    evaluate_model(
        model=best_model,
        X=X_test,
        events=event_test,
        times=time_test,
    )
)

print(
    f"\nSelected checkpoint round : "
    f"{best_round}"
)

print(
    f"Validation C-index        : "
    f"{best_validation_cindex:.4f}"
)

print(
    f"Final test loss           : "
    f"{test_loss:.5f}"
)

print(
    f"Final test C-index        : "
    f"{test_cindex:.4f}"
)


# ============================================================
# Save final test results
# ============================================================

final_results = {

    "method": "FedAvg",

    "num_clients": (
        NUM_CLIENTS
    ),

    "num_rounds": (
        NUM_ROUNDS
    ),

    "local_epochs": (
        LOCAL_EPOCHS
    ),

    "learning_rate": (
        LEARNING_RATE
    ),

    "weight_decay": (
        WEIGHT_DECAY
    ),

    "best_round": (
        best_round
    ),

    "best_validation_c_index": (
        best_validation_cindex
    ),

    "best_validation_loss": (
        best_validation_loss
    ),

    "global_test_patients": int(
        len(X_test)
    ),

    "global_test_events": int(
        event_test.sum()
    ),

    "final_test_loss": float(
        test_loss
    ),

    "final_test_c_index": float(
        test_cindex
    ),
}


with open(
    FINAL_TEST_RESULTS_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_results,
        file,
        indent=4,
    )


# ============================================================
# Final summary
# ============================================================

print("\nSaved:")
print(
    f"- {VALIDATION_HISTORY_FILE}"
)

print(
    f"- {BEST_MODEL_FILE}"
)

print(
    f"- {BEST_MODEL_INFO_FILE}"
)

print(
    f"- {FINAL_TEST_RESULTS_FILE}"
)

print("\n" + "=" * 70)
print("FEDAVG EXPERIMENT FINISHED")
print("=" * 70)