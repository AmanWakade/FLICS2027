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

PROXIMAL_MU = 0.1

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
    / "fedprox_validation_history.csv"
)

BEST_MODEL_FILE = (
    RESULT_DIR
    / "fedprox_best_model.pt"
)

BEST_MODEL_INFO_FILE = (
    RESULT_DIR
    / "fedprox_best_model_info.json"
)

FINAL_TEST_RESULTS_FILE = (
    RESULT_DIR
    / "fedprox_final_test_results.json"
)


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)


# ============================================================
# Load validation data only
# ============================================================

print("=" * 70)
print("FEDERATED DEEPSURV - FEDPROX")
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
# Initial model
# ============================================================

initial_model = initialize_model()

initial_parameters = (
    ndarrays_to_parameters(
        get_model_parameters(
            initial_model
        )
    )
)


# ============================================================
# Client configuration
# ============================================================

def fit_config(
    server_round: int,
):
    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "proximal_mu": PROXIMAL_MU,
    }


# ============================================================
# Checkpoint state
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
    global best_validation_cindex, best_validation_loss, best_round

    model = initialize_model()

    set_model_parameters(
        model,
        parameters,
    )

    (
        validation_loss,
        validation_cindex,
    ) = evaluate_model(
        model=model,
        X=X_validation,
        events=event_validation,
        times=time_validation,
    )

    print("\n" + "=" * 65)
    print(
        f"FEDPROX GLOBAL VALIDATION "
        f"- ROUND {server_round}"
    )
    print("=" * 65)

    print(
        f"Patients : {len(X_validation):,}"
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
    # Save history
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
    # Best checkpoint
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
            "method": "FedProx",
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
            "proximal_mu": (
                PROXIMAL_MU
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
                    "method": "FedProx",
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
                    "proximal_mu": (
                        PROXIMAL_MU
                    ),
                },
                file,
                indent=4,
            )

        print(
            "\n*** NEW BEST FEDPROX "
            "CHECKPOINT ***"
        )

        print(
            f"Round   : {best_round}"
        )

        print(
            f"C-index : "
            f"{best_validation_cindex:.4f}"
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
# Metric aggregation
# ============================================================

def weighted_average(
    metrics,
):
    if not metrics:
        return {}

    output = {}

    metric_names = [
        "train_loss",
        "cox_loss",
        "prox_loss",
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

            output[
                metric_name
            ] = (
                sum(
                    n * value
                    for n, value
                    in values
                )
                / denominator
            )

    return output


# ============================================================
# FedProx strategy
#
# Server aggregation is weighted averaging.
# FedProx difference is in the client-side objective.
# ============================================================

strategy = fl.server.strategy.FedAvg(

    fraction_fit=1.0,

    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,

    initial_parameters=initial_parameters,

    on_fit_config_fn=fit_config,

    evaluate_fn=evaluate_validation,

    fit_metrics_aggregation_fn=(
        weighted_average
    ),

    # Do not use local test sets during training
    fraction_evaluate=0.0,
    min_evaluate_clients=0,
)


# ============================================================
# Start server
# ============================================================

print("\nServer configuration")
print("--------------------")

print(
    f"Clients       : {NUM_CLIENTS}"
)

print(
    f"Rounds        : {NUM_ROUNDS}"
)

print(
    f"Local epochs  : {LOCAL_EPOCHS}"
)

print(
    f"Learning rate : {LEARNING_RATE}"
)

print(
    f"Weight decay  : {WEIGHT_DECAY}"
)

print(
    f"Proximal mu   : {PROXIMAL_MU}"
)

print(
    "\nGlobal test will NOT "
    "be evaluated during training."
)

print(
    "\nWaiting for 5 clients...\n"
)


fl.server.start_server(

    server_address=SERVER_ADDRESS,

    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS
    ),

    strategy=strategy,
)


# ============================================================
# Training complete
# ============================================================

print("\n" + "=" * 70)
print("FEDPROX TRAINING COMPLETE")
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
# Final global test
# ============================================================

print("\n" + "=" * 70)
print("FINAL GLOBAL TEST EVALUATION")
print("=" * 70)

if not BEST_MODEL_FILE.exists():
    raise FileNotFoundError(
        "Best FedProx checkpoint "
        "was not saved."
    )


(
    X_test,
    event_test,
    time_test,
) = load_global_test_data()


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


test_loss, test_cindex = (
    evaluate_model(
        model=best_model,
        X=X_test,
        events=event_test,
        times=time_test,
    )
)


print(
    f"Global test patients : "
    f"{len(X_test):,}"
)

print(
    f"Global test events   : "
    f"{int(event_test.sum()):,}"
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
# Save final results
# ============================================================

final_results = {
    "method": "FedProx",
    "num_clients": NUM_CLIENTS,
    "num_rounds": NUM_ROUNDS,
    "local_epochs": LOCAL_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "proximal_mu": PROXIMAL_MU,
    "best_round": best_round,
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