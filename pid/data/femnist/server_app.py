from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays  # Flower types for server context and parameter conversions
from flwr.server import ServerApp, ServerAppComponents, ServerConfig      # Flower server application and configuration
from flwr.server.strategy import FedAvg                                    # Base FedAvg strategy to extend
from flwr.server.strategy.aggregate import aggregate                        # Helper to aggregate weighted updates
from pid.task import Net, get_weights, load_data                           # User-defined model and weight utilities
import numpy as np                                                           # Numerical operations (e.g., mean)
import csv                                                                   # CSV logging for PID outputs and malicious rounds
import os                                                                    # OS utilities (file existence checks)
from sklearn.metrics import accuracy_score, recall_score, precision_score
import torch

# Filename for logging rounds detected as malicious
MALICIOUS_ROUNDS_LOG = "malicious_rounds.txt"
NUM_MALICIOUS_CLIENTS = 0  # Number of clients to simulate as malicious

class FedPIDAvg(FedAvg):
    """
    Custom federated strategy that applies a PID controller to adjust client update weights.
    """
    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.05, integral_max=10.0, **kwargs):
        # Initialize FedAvg with same kwargs (e.g., fraction_fit, initial_parameters)
        super().__init__(**kwargs)
        # PID gain coefficients
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.integral_max = integral_max  # Anti-windup clamp limit
        # State for PID calculations
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_loss = None

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
        Now includes exclusion (outlier filtering) to remove the most suspicious client update.
        """
        # Return early if no results
        if not results:
            return None

        # Extract client-reported training losses
        losses = [res.metrics["train_loss"] for _, res in results]
        # Compute average loss across clients
        avg_loss = np.mean(losses)

        # Initialize prev_loss on first round
        if self.prev_loss is None:
            self.prev_loss = avg_loss
        # PID error calculation (positive if loss decreased)
        error = self.prev_loss - avg_loss
        # Integral with anti-windup
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral + error))
        # Derivative term
        derivative = error - self.prev_error
        # PID components
        P = self.Kp * error
        I = self.Ki * self.integral
        D = self.Kd * derivative
        # Compute PID term
        pid_term = P + I + D
        # Allow scaling up and down for better performance
        scaling = max(0.2, min(2.0, 1.0 + pid_term))  # 20%-200% scaling range
        # Update state
        self.prev_error = error
        self.prev_loss = avg_loss

        # Log PID data for this round
        with open("pid_log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, avg_loss, P, I, D, scaling])

        weighted_results = []    # List to store (ndarrays, weight) for aggregation
        attack_detected = False  # Flag to record if any malicious client update

        for idx, (cid, res) in enumerate(results):
            # Combine trust weight with PID output for adaptive scaling
            weight = scaling
            res_ndarrays = parameters_to_ndarrays(res.parameters)
            # Existing malicious client logic (for logging/experiments)
            #####################
            ####################################################
            #### COMMENT THIS OUT FOR NON MALICIOUS ############
            if res.metrics.get("malicious"):            # If client reported as malicious
                res_ndarrays = [w * -5.0 for w in res_ndarrays]
                attack_detected = True
            ####################################################
            weighted_results.append((res_ndarrays, weight))

        # If any attack was detected this round, log the round number
        if attack_detected:
            with open(MALICIOUS_ROUNDS_LOG, "a") as f:
                f.write(f"{server_round}\n")

        # --- Trimmed mean + trust weighting aggregation for robustness ---
        # Collect all client updates as flat arrays
        client_updates = [parameters_to_ndarrays(res.parameters) for _, res in results]
        flat_updates = [np.concatenate([w.flatten() for w in upd]) for upd in client_updates]
        updates_matrix = np.stack(flat_updates)
        # Compute distances to the mean
        mean_update = np.mean(updates_matrix, axis=0)
        distances = np.linalg.norm(updates_matrix - mean_update, axis=1)
        # Set k for trimmed mean (number of outliers to trim from each end)
        k = NUM_MALICIOUS_CLIENTS  # Adjust based on number of malicious clients
        # Sort indices by distance
        sorted_indices = np.argsort(distances)
        # Keep only the middle updates (trim k from each end)
        if k == 0:
            # No trimming needed - use all clients
            trimmed_indices = sorted_indices
        else:
            # Trim k clients from each end
            trimmed_indices = sorted_indices[k:-k]
        trimmed_updates = updates_matrix[trimmed_indices]
        trimmed_distances = distances[trimmed_indices]
        
        # Always use advanced trust weighting (superior for precision/recall)
        # This works for both malicious and non-malicious cases
        epsilon = 1e-6
        trust_weights = 1 / (trimmed_distances**2 + epsilon)
        trust_weights = trust_weights / np.sum(trust_weights)  # Normalize
        weighted_mean_update = np.average(trimmed_updates, axis=0, weights=trust_weights)
        # Apply PID scaling to the trust-weighted trimmed mean
        scaled_update = scaling * weighted_mean_update
        # Convert back to original shape (list of arrays per layer)
        ref_shapes = [w.shape for w in client_updates[0]]
        split_indices = np.cumsum([np.prod(s) for s in ref_shapes])[:-1]
        # Ensure indices are Python ints for numpy.split
        split_indices = [int(idx) for idx in split_indices]
        split_arrays = np.split(scaled_update, split_indices)
        reshaped_arrays = [arr.reshape(shape) for arr, shape in zip(split_arrays, ref_shapes)]
        # Return the scaled, robust aggregated update
        return ndarrays_to_parameters(reshaped_arrays), {}

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

def evaluate_fn(server_round, parameters, config):
    """
    Custom evaluation function for Flower strategies.
    Evaluates the global model on a validation set and returns accuracy, recall, and precision.
    Handles empty validation sets gracefully.
    Accepts both Flower Parameters objects and lists of ndarrays.
    Uses set_weights to avoid state_dict mismatches.
    Handles batch dicts and ensures only Tensor is passed to model.
    """
    model = Net()
    # Accept both Parameters objects and list of ndarrays
    if isinstance(parameters, list):
        weights = parameters
    else:
        weights = parameters_to_ndarrays(parameters)
    # Use set_weights to load weights robustly
    from pid.task import set_weights
    set_weights(model, weights)
    # Load validation data
    _, valloader = load_data(0, 1)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in valloader:
            # Handle both dict and tuple batch
            if isinstance(batch, dict):
                x = batch["image"]
                y = batch["character"]
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            # Ensure data is on the same device as the model (if needed)
            if hasattr(model, 'device'):
                x = x.to(model.device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    if len(y_true) == 0:
        # Avoid division by zero if validation set is empty
        return 0.0, {"accuracy": 0.0, "recall": 0.0, "precision": 0.0}
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    # Compute confusion matrix for FPR and TNR
    from sklearn.metrics import confusion_matrix
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn_list, fp_list = [], []
    total = cm.sum()
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp_i = cm[:, i].sum() - tp
        tn_i = total - (tp + fn + fp_i)
        tn_list.append(tn_i)
        fp_list.append(fp_i)
    # Macro-average TNR and FPR
    tnr = float(sum(tn_i / (tn_i + fp_i + 1e-10) for tn_i, fp_i in zip(tn_list, fp_list)) / len(labels))
    fpr = float(sum(fp_i / (fp_i + tn_i + 1e-10) for tn_i, fp_i in zip(tn_list, fp_list)) / len(labels))
    # Return metrics including FPR and TNR
    return 0.0, {"accuracy": acc, "recall": rec, "precision": prec, "fpr": fpr, "tnr": tnr}