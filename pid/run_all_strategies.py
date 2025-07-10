import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg, Krum, Bulyan
from pid.server_app import FedPIDAvg
from pid.server_app import evaluate_fn
from pid.client_app import client_fn
from pid.task import Net, get_weights
import pandas as pd
import os

# Number of clients and tolerated malicious clients
NUM_CLIENTS = 30
NUM_ROUNDS = 30
NUM_MALCIOUS_CLIENTS = 0

# Define strategies to compare
STRATEGIES = {
    "FedAvg": lambda: FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(get_weights(Net())),
    ),
    #  "FedPIDAvg_default": lambda: FedPIDAvg(
    #     Kp=0.1, Ki=0.01, Kd=0.05,
    #     fraction_fit=1.0,
    #     fraction_evaluate=1.0,
    #     min_available_clients=2,
    #     evaluate_fn=evaluate_fn,
    #     initial_parameters=ndarrays_to_parameters(get_weights(Net())),
    # ),
    "FedPIDAvg_tuned": None,  # Will be set below if best_pid.txt exists
    "Krum": lambda: Krum(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        num_malicious_clients=NUM_MALCIOUS_CLIENTS,
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(get_weights(Net())),
    ),
    "Bulyan": lambda: Bulyan(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters(get_weights(Net())),
        num_malicious_clients=NUM_MALCIOUS_CLIENTS,
        evaluate_fn=evaluate_fn,
        to_keep=0,
    ),
}

# Try to load tuned PID params
if os.path.exists("best_pid.txt"):
    pid_params = {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05}
    with open("best_pid.txt") as f:
        lines = f.readlines()
        for line in lines[1:]:
            k, v = line.strip().split(" = ")
            pid_params[k] = float(v)
    STRATEGIES["FedPIDAvg_tuned"] = lambda: FedPIDAvg(
        Kp=pid_params["Kp"], Ki=pid_params["Ki"], Kd=pid_params["Kd"],
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters(get_weights(Net())),
    )
else:
    # Remove tuned strategy if no params found
    STRATEGIES.pop("FedPIDAvg_tuned", None)

# Run each strategy
for name, strategy_fn in STRATEGIES.items():
    print(f"\nRunning strategy: {name}")
    strategy = strategy_fn()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # Save loss, accuracy, recall, and precision per round
    rounds = [r for r, _ in history.losses_distributed]
    losses = [l for _, l in history.losses_distributed]

    # Extract metrics from metrics_centralized if available (preferred)
    metrics = {"accuracy": {}, "recall": {}, "precision": {}}
    if hasattr(history, "metrics_centralized") and history.metrics_centralized:
        for metric_name in ["accuracy", "recall", "precision"]:
            metric_list = history.metrics_centralized.get(metric_name, [])
            for r, v in metric_list:
                metrics[metric_name][r] = v
    # Build lists for each metric, aligned with rounds
    accuracies = [metrics["accuracy"].get(r, None) for r in rounds]
    recalls = [metrics["recall"].get(r, None) for r in rounds]
    precisions = [metrics["precision"].get(r, None) for r in rounds]

    df = pd.DataFrame({
        "round": rounds,
        "loss": losses,
        "accuracy": accuracies,
        "recall": recalls,
        "precision": precisions,
    })
    df.to_csv(f"{name}_log.csv", index=False)
    print(f"Saved {name}_log.csv")

print("\nAll strategies complete. You can now run plot_comparison.py to visualize results.")
