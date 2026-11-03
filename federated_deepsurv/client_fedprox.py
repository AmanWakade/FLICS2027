import argparse

import flwr as fl

from federated_deepsurv.model import (
    get_model_parameters,
    set_model_parameters,
)

from federated_deepsurv.task import (
    initialize_model,
    load_client_data,
    train_local_model_fedprox,
    evaluate_model,
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--client-id",
    type=int,
    required=True,
    choices=[1, 2, 3, 4, 5],
)

args = parser.parse_args()

CLIENT_ID = args.client_id


# ============================================================
# Centre mapping
# ============================================================

CENTER_MAPPING = {
    1: "07905",
    2: "16523",
    3: "07347",
    4: "25110",
    5: "18755",
}

CENTER_CODE = CENTER_MAPPING[
    CLIENT_ID
]


# ============================================================
# Load local data
# ============================================================

(
    X_train,
    event_train,
    time_train,
    X_test,
    event_test,
    time_test,
) = load_client_data(
    client_id=CLIENT_ID
)


# ============================================================
# Model
# ============================================================

model = initialize_model()


# ============================================================
# Flower Client
# ============================================================

class FedProxDeepSurvClient(
    fl.client.NumPyClient
):

    def get_parameters(
        self,
        config,
    ):
        return get_model_parameters(
            model
        )

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

    def fit(
        self,
        parameters,
        config,
    ):
        # Store global model parameters
        # global_parameters = [
        #     p.copy()
        #     for p in parameters
        # ]

        # Load global model
        set_model_parameters(
            model,
            parameters,
        )

        server_round = int(
            config.get(
                "server_round",
                0,
            )
        )

        local_epochs = int(
            config.get(
                "local_epochs",
                10,
            )
        )

        learning_rate = float(
            config.get(
                "learning_rate",
                0.001,
            )
        )

        weight_decay = float(
            config.get(
                "weight_decay",
                0.0001,
            )
        )

        proximal_mu = float(
            config.get(
                "proximal_mu",
                0.01,
            )
        )

        print(
            f"\n[Client {CLIENT_ID}] "
            f"Centre={CENTER_CODE}"
        )

        print(
            f"Round {server_round} | "
            f"FedProx training | "
            f"mu={proximal_mu}"
        )

        (
    _,
    total_loss,
    cox_loss,
    prox_loss,
) = train_local_model_fedprox(
    model=model,
    X_train=X_train,
    event_train=event_train,
    time_train=time_train,
    local_epochs=local_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    proximal_mu=proximal_mu,
)

        print(
            f"[Client {CLIENT_ID}] "
            f"Patients={len(X_train)} | "
            f"Events={int(event_train.sum())} | "
            f"Cox Loss={cox_loss:.5f} | "
            f"Prox Loss={prox_loss:.5f} | "
            f"Total Loss={total_loss:.5f}"
        )

        return (
            get_model_parameters(
                model
            ),
            len(X_train),
            {
                "train_loss": float(
                    total_loss
                ),
                "cox_loss": float(
                    cox_loss
                ),
                "prox_loss": float(
                    prox_loss
                ),
                "client_id": CLIENT_ID,
            },
        )

    # --------------------------------------------------------
    # Local evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        parameters,
        config,
    ):
        set_model_parameters(
            model,
            parameters,
        )

        loss, c_index = evaluate_model(
            model=model,
            X=X_test,
            events=event_test,
            times=time_test,
        )

        print(
            f"[Client {CLIENT_ID}] "
            f"Evaluation | "
            f"C-index={c_index:.4f}"
        )

        return (
            float(loss),
            len(X_test),
            {
                "c_index": float(
                    c_index
                ),
                "client_id": CLIENT_ID,
            },
        )


# ============================================================
# Start client
# ============================================================

print("=" * 60)
print(
    f"FEDPROX DEEPSURV CLIENT {CLIENT_ID}"
)
print("=" * 60)

print(
    f"Centre          : {CENTER_CODE}"
)

print(
    f"Training rows   : {len(X_train)}"
)

print(
    f"Training events : {int(event_train.sum())}"
)

print(
    f"Testing rows    : {len(X_test)}"
)

print(
    f"Testing events  : {int(event_test.sum())}"
)

print("\nConnecting to server...")


client = FedProxDeepSurvClient()

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=client.to_client(),
    insecure=True,
)