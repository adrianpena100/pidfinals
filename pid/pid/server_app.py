from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays  # Flower types for server context and parameter conversions
from flwr.server import ServerApp, ServerAppComponents, ServerConfig      # Flower server application and configuration
from flwr.server.strategy import FedAvg                                    # Base FedAvg strategy to extend
from flwr.server.strategy.aggregate import aggregate                        # Helper to aggregate weighted updates
from pid.task import Net, get_weights                                       # User-defined model and weight utilities
import numpy as np                                                           # Numerical operations (e.g., mean)
import csv                                                                   # CSV logging for PID outputs and malicious rounds
import os                                                                    # OS utilities (file existence checks)

# Filename for logging rounds detected as malicious
MALICIOUS_ROUNDS_LOG = "malicious_rounds.txt"

class FedPIDAvg(FedAvg):
    """
    Custom federated strategy that applies a PID controller to adjust client update weights.
    """
    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.05, **kwargs):
        # Initialize FedAvg with same kwargs (e.g., fraction_fit, initial_parameters)
        super().__init__(**kwargs)
        # PID gain coefficients
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        # State for integral and derivative calculations
        self.integral = 0.0      # Cumulative sum of past errors
        self.prev_error = 0.0    # Previous error for derivative term

        # Create/overwrite PID log file with header row
        with open("pid_log.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "avg_loss", "P", "I", "D", "pid_output"])
        # Initialize malicious rounds log with header
        with open(MALICIOUS_ROUNDS_LOG, "w") as f:
            f.write("round\n")

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate client updates with PID-weighting and detect malicious rounds.

        Args:
            server_round: Current federated learning round index
            results: List of (client_id, FitRes) tuples containing updates and metrics
            failures: List of failures (ignored here)

        Returns:
            A tuple (Parameters, Dict) for new global model and empty metrics dict
        """
        # Return early if no results
        if not results:
            return None

        # Extract client-reported training losses
        losses = [res.metrics["train_loss"] for _, res in results]
        # Compute average loss across clients
        avg_loss = np.mean(losses)

        # PID error terms calculation
        error = avg_loss                              # Current error = average loss
        derivative = error - self.prev_error          # Change in error since last round
        self.integral += error                        # Accumulate error over time
        self.prev_error = error                       # Update prev_error for next round

        # Compute PID components
        P = self.Kp * error                           # Proportional term
        I = self.Ki * self.integral                   # Integral term
        D = self.Kd * derivative                      # Derivative term
        pid_output = P + I + D                        # Combined PID output as weight

        # Log PID data for this round
        with open("pid_log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, avg_loss, P, I, D, pid_output])

        weighted_results = []    # List to store (ndarrays, weight) for aggregation
        attack_detected = False  # Flag to record if any malicious client update

        # Process each client's update
        for cid, res in results:
            # Use PID output as the weight for this client's update
            weight = pid_output
            # Convert Parameters object to numpy arrays
            res_ndarrays = parameters_to_ndarrays(res.parameters)

            #####################
            ####################################################
            #### COMMENT THIS OUT FOR NON MALICIOUS ############
            if res.metrics.get("malicious"):            # If client reported as malicious
                # Invert and amplify malicious updates to penalize them
                res_ndarrays = [w * -5.0 for w in res_ndarrays]
                attack_detected = True
            ####################################################

            # Append weighted update for aggregation
            weighted_results.append((res_ndarrays, weight))

        # If any attack was detected this round, log the round number
        if attack_detected:
            with open(MALICIOUS_ROUNDS_LOG, "a") as f:
                f.write(f"{server_round}\n")

        # Aggregate all weighted client updates into final global model parameters
        aggregated = aggregate(weighted_results)
        # Convert back to Flower Parameters and return
        return ndarrays_to_parameters(aggregated), {}

# Server-side function to configure and return Flower server components

def server_fn(context: Context):
    # Load best PID parameters from file if available, otherwise fallback defaults
    pid_params = {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05}  # default fallback values
    if os.path.exists("best_pid.txt"):
        with open("best_pid.txt", "r") as f:
            lines = f.readlines()
            # Parse lines of format "Kp = value"
            for line in lines[1:]:
                k, v = line.strip().split(" = ")
                pid_params[k] = float(v)

    # Extract server run configuration from context
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    # Initialize model weights for first round
    init_params = ndarrays_to_parameters(get_weights(Net()))

    # on_fit_config_fn allows passing metadata (e.g. round index) to clients
    def on_fit_config_fn(rnd):
        return {"server_round": rnd}

    # Instantiate the custom PID federated strategy
    strategy = FedPIDAvg(
        Kp=pid_params["Kp"],             # Loaded or default proportional gain
        Ki=pid_params["Ki"],             # Loaded or default integral gain
        Kd=pid_params["Kd"],             # Loaded or default derivative gain
        fraction_fit=fraction_fit,         # Fraction of clients for training
        fraction_evaluate=1.0,             # Fraction of clients for evaluation
        min_available_clients=2,           # Minimum clients to start a round
        initial_parameters=init_params,    # Starting global model weights
        on_fit_config_fn=on_fit_config_fn, # Config function for client runs
    )

    # Configure the number of server rounds to run
    config = ServerConfig(num_rounds=num_rounds)
    # Return the strategy and config wrapped in ServerAppComponents
    return ServerAppComponents(strategy=strategy, config=config)

# Launch the Flower server application using the defined server_fn
app = ServerApp(server_fn=server_fn)
