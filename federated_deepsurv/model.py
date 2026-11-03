import torch
import torch.nn as nn


# ============================================================
# DeepSurv Model
# ============================================================

class DeepSurv(nn.Module):
    def __init__(
        self,
        input_dim=108,
        hidden_dims=(128, 64, 32),
        dropout=0.30,
    ):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(previous_dim, 1)
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
# Cox Partial Log-Likelihood Loss
# ============================================================

def negative_cox_log_likelihood(
    risk_scores,
    durations,
    events,
):
    """
    Negative Cox partial log-likelihood.

    risk_scores:
        DeepSurv output / log-risk.

    durations:
        TIME_YEARS.

    events:
        EVENT:
            1 = malignancy
            0 = censored
    """

    # Sort from longest survival time to shortest
    order = torch.argsort(
        durations,
        descending=True,
    )

    sorted_risk = risk_scores[order]
    sorted_events = events[order]

    # Log cumulative hazard for Cox risk sets
    log_cumulative_hazard = torch.logcumsumexp(
        sorted_risk,
        dim=0,
    )

    partial_likelihood = (
        sorted_risk
        - log_cumulative_hazard
    )

    observed_likelihood = (
        partial_likelihood
        * sorted_events
    )

    number_of_events = torch.clamp(
        sorted_events.sum(),
        min=1.0,
    )

    loss = (
        -observed_likelihood.sum()
        / number_of_events
    )

    return loss


# ============================================================
# Parameter Helpers for Flower
# ============================================================

def get_model_parameters(model):
    """
    Convert PyTorch parameters to NumPy arrays
    for Flower aggregation.
    """

    return [
        value.detach().cpu().numpy()
        for _, value in model.state_dict().items()
    ]


def set_model_parameters(
    model,
    parameters,
):
    """
    Load Flower aggregated parameters
    back into the PyTorch DeepSurv model.
    """

    state_dict = model.state_dict()

    new_state_dict = {}

    for (
        key,
        old_value,
    ), new_value in zip(
        state_dict.items(),
        parameters,
    ):
        tensor = torch.tensor(
            new_value,
            dtype=old_value.dtype,
        )

        new_state_dict[key] = tensor

    model.load_state_dict(
        new_state_dict,
        strict=True,
    )


# ============================================================
# Utility
# ============================================================

def create_model(
    input_dim=108,
):
    return DeepSurv(
        input_dim=input_dim,
        hidden_dims=(128, 64, 32),
        dropout=0.30,
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    model = create_model(
        input_dim=108
    )

    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{total_parameters:,}"
    )