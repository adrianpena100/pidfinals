import optuna  # Hyperparameter optimization framework
import flwr as fl  # Flower federated learning framework
import numpy as np  # Numerical operations for loss calculations
from flwr.server import ServerConfig  # Configure simulation server parameters
from flwr.common import ndarrays_to_parameters  # Convert numpy arrays to Flower Parameters
from pid.client_app import client_fn  # Client-side training function
from pid.task import Net, get_weights  # Model definition and weight extraction
from pid.server_app import FedPIDAvg  # Custom PID-based aggregation strategy


def objective(trial):
    # Suggest PID hyperparameters for this trial
    Kp = trial.suggest_float("Kp", 0.0, 0.1)  # Proportional gain
    Ki = trial.suggest_float("Ki", 0.0, 0.005)   # Integral gain
    Kd = trial.suggest_float("Kd", 0.0, 0.0)   # Derivative gain

    # Obtain initial global model parameters from a fresh network
    initial_parameters = ndarrays_to_parameters(get_weights(Net()))

    # Initialize the federated strategy with current PID parameters
    strategy = FedPIDAvg(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        fraction_fit=1.0,              # Fraction of clients used for training each round
        fraction_evaluate=1.0,         # Fraction of clients used for evaluation each round
        min_available_clients=2,       # Minimum clients required to start a round
        initial_parameters=initial_parameters  # Starting model parameters
    )

    # Run a federated simulation:
    # - client_fn: function to create client instances
    # - num_clients: total simulated clients
    # - num_rounds: total federated rounds to execute
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=30,
        config=ServerConfig(num_rounds=30),
        strategy=strategy,
    )

    # Extract the final loss from the simulation history
    #final_loss = history.losses_distributed[-1][1]  # Loss of global model at last round
    # Return negative loss so Optuna will maximize it (i.e., minimize actual loss)
    #return -final_loss
    avg_loss = np.mean([l for _, l in history.losses_distributed[-10:]])
    return -avg_loss


if __name__ == "__main__":
    # Create an Optuna study configured to maximize the objective
    study = optuna.create_study(direction="maximize")
    # Run hyperparameter tuning for a fixed number of trials
    study.optimize(objective, n_trials=20)

    # Retrieve the best parameters found
    best_params = study.best_params

    # Save best PID parameters to a text file for use in the federated simulation
    with open("best_pid.txt", "w") as f:
        f.write("Best PID parameters:\n")
        for name, value in best_params.items():
            f.write(f"{name} = {value}\n")

    # Confirm that the parameters were saved
    print("\n✅ Best PID parameters saved to best_pid.txt")
