"""server_app.py — Federated PID strategy with benign/malicious tally
------------------------------------------------------------------
"""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from pid.task import Net, get_weights, load_data

import numpy as np
import csv
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score
import torch

# ─────────────────────────── Config ────────────────────────────────
MALICIOUS_ROUNDS_LOG = "malicious_rounds.txt"
NUM_MALICIOUS_CLIENTS = 0  # Number of simulated malicious clients (for trimming)


# ────────────────────── Custom Strategy ───────────────────────────
class FedPIDAvg(FedAvg):
    """FedAvg + PID with robustness & tally of benign/malicious detections."""

    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.05, integral_max=10.0, **kwargs):
        super().__init__(**kwargs)
        # PID coefficients
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral_max = integral_max
        # PID state
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_loss = None

        # ── Tally counters ──
        self.total_benign = 0
        self.total_malicious = 0
        # Prepare per‑round CSV log
        with open("client_tally.csv", "w", newline="") as f:
            csv.writer(f).writerow(["round", "benign_kept", "malicious_pruned"])

        # Files for PID and malicious‑round logs
        with open("pid_log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["round", "avg_loss", "P", "I", "D", "pid_output"])
        with open(MALICIOUS_ROUNDS_LOG, "w") as f:
            f.write("round\n")

    # ───────────────────────── aggregate_fit ─────────────────────────
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None

        # ------- PID compute on training loss -------
        losses = [res.metrics["train_loss"] for _, res in results]
        avg_loss = float(np.mean(losses))
        if self.prev_loss is None:
            self.prev_loss = avg_loss
        error = self.prev_loss - avg_loss
        self.integral = np.clip(self.integral + error, -self.integral_max, self.integral_max)
        derivative = error - self.prev_error
        P, I, D = self.Kp * error, self.Ki * self.integral, self.Kd * derivative
        pid_term = P + I + D
        scaling = np.clip(1.0 + pid_term, 0.2, 2.0)
        self.prev_error, self.prev_loss = error, avg_loss

        with open("pid_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([server_round, avg_loss, P, I, D, scaling])

        # --------- (Optional) label‑flip sign‑flip handling ---------
        weighted_results = []
        attack_detected_flag = False  # for malicious_rounds.txt
        for cid, res in results:
            weight = scaling  # simple uniform scaling per client
            res_ndarrays = parameters_to_ndarrays(res.parameters)

            # COMMENT/UNCOMMENT to enable sign‑flip simulation detection
            # if res.metrics.get("malicious"):
            #     res_ndarrays = [w * -5.0 for w in res_ndarrays]
            #     attack_detected_flag = True
            weighted_results.append((res_ndarrays, weight))

        if attack_detected_flag:
            with open(MALICIOUS_ROUNDS_LOG, "a") as f:
                f.write(f"{server_round}\n")

        # -------------------- Robust aggregation --------------------
        client_updates = [parameters_to_ndarrays(res.parameters) for _, res in results]
        flat_updates = [np.concatenate([w.flatten() for w in upd]) for upd in client_updates]
        updates_matrix = np.stack(flat_updates)

        mean_update = updates_matrix.mean(axis=0)
        distances = np.linalg.norm(updates_matrix - mean_update, axis=1)
        k = NUM_MALICIOUS_CLIENTS
        sorted_idx = np.argsort(distances)
        trimmed_idx = sorted_idx[k:-k] if k else sorted_idx

        # ── Benign/Malicious count for this round ──
        benign_this_round = len(trimmed_idx)
        malicious_this_round = len(results) - benign_this_round
        self.total_benign += benign_this_round
        self.total_malicious += malicious_this_round
        with open("client_tally.csv", "a", newline="") as f:
            csv.writer(f).writerow([server_round, benign_this_round, malicious_this_round])

        trimmed_updates = updates_matrix[trimmed_idx]
        trimmed_distances = distances[trimmed_idx]
        epsilon = 1e-6
        trust_weights = 1 / (trimmed_distances ** 2 + epsilon)
        trust_weights /= trust_weights.sum()
        weighted_mean = np.average(trimmed_updates, axis=0, weights=trust_weights)
        scaled_update = scaling * weighted_mean

        # Re‑pack to original tensor shapes
        ref_shapes = [w.shape for w in client_updates[0]]
        split_points = np.cumsum([np.prod(s) for s in ref_shapes])[:-1].astype(int)
        split_arrays = np.split(scaled_update, split_points)
        reshaped = [arr.reshape(shape) for arr, shape in zip(split_arrays, ref_shapes)]
        return ndarrays_to_parameters(reshaped), {}


# ────────────── Server factory & Flower wiring ─────────────────────

def server_fn(context: Context):
    # Optionally load tuned PID gains
    pid_params = {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05}
    if os.path.exists("best_pid.txt"):
        with open("best_pid.txt", "r") as f:
            for line in f.readlines()[1:]:
                k, v = line.strip().split(" = ")
                pid_params[k] = float(v)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    init_params = ndarrays_to_parameters(get_weights(Net()))

    def on_fit_config_fn(rnd):
        return {"server_round": rnd}

    strategy = FedPIDAvg(
        Kp=pid_params["Kp"], Ki=pid_params["Ki"], Kd=pid_params["Kd"],
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=init_params,
        on_fit_config_fn=on_fit_config_fn,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)


# ─────────────────── Optional global evaluation ────────────────────

def evaluate_fn(server_round, parameters, config):
    """Evaluate the aggregated model on a hold‑out validation set."""
    model = Net()
    weights = parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
    from pid.task import set_weights as _set_weights
    _set_weights(model, weights)

    _, valloader = load_data(0, 1)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in valloader:
            if isinstance(batch, dict):
                x, y = batch["image"], batch["character"]
            else:
                x, y = batch
            preds = model(x).argmax(dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())
    if not y_true:
        return 0.0, {"accuracy": 0.0, "recall": 0.0, "precision": 0.0}
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    return 0.0, {"accuracy": acc, "recall": rec, "precision": prec}
