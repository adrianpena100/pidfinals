import optuna
import flwr as fl

from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters
from pid.client_app import client_fn
from pid.task import Net, get_weights
from pid.server_app import FedPIDAvg  # Use your custom PID strategy


def objective(trial):
    # Suggest PID hyperparameters
    Kp = trial.suggest_float("Kp", 0.01, 1.0)
    Ki = trial.suggest_float("Ki", 0.0, 0.5)
    Kd = trial.suggest_float("Kd", 0.0, 0.5)

    # Initial model parameters
    initial_parameters = ndarrays_to_parameters(get_weights(Net()))

    # Create the strategy with suggested PID values
    strategy = FedPIDAvg(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        fraction_fit=0.5,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )

    # Run a short federated simulation with the suggested PID params
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=ServerConfig(num_rounds=2),
        strategy=strategy,
    )

    # Return the negative final loss as the metric to maximize
    return -history.losses_distributed[-1][1]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params

    # Save to file
    with open("best_pid.txt", "w") as f:
        f.write("Best PID parameters:\n")
        for k, v in best_params.items():
            f.write(f"{k} = {v}\n")

    print("\n✅ Best PID parameters saved to best_pid.txt")

