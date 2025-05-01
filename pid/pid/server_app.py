from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from pid.task import Net, get_weights
import numpy as np
import csv
import os

MALICIOUS_ROUNDS_LOG = "malicious_rounds.txt"

class FedPIDAvg(FedAvg):
    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.05, **kwargs):
        super().__init__(**kwargs)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

        with open("pid_log.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "avg_loss", "P", "I", "D", "pid_output"])
        with open(MALICIOUS_ROUNDS_LOG, "w") as f:
            f.write("round\n")

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None

        losses = [res.metrics["train_loss"] for _, res in results]
        avg_loss = np.mean(losses)

        error = avg_loss
        derivative = error - self.prev_error
        self.integral += error
        self.prev_error = error

        P = self.Kp * error
        I = self.Ki * self.integral
        D = self.Kd * derivative
        pid_output = P + I + D

        with open("pid_log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, avg_loss, P, I, D, pid_output])

        weighted_results = []
        attack_detected = False

        for cid, res in results:
            weight = pid_output
            res_ndarrays = parameters_to_ndarrays(res.parameters)
            #####################
            ####################################################
            #### COMMENT THIS OUT FOR NON MALICIOUS ############
            if res.metrics.get("malicious"):
                res_ndarrays = [w * -5.0 for w in res_ndarrays]
                attack_detected = True
            ####################################################
            weighted_results.append((res_ndarrays, weight))

        if attack_detected:
            with open(MALICIOUS_ROUNDS_LOG, "a") as f:
                f.write(f"{server_round}\n")

        aggregated = aggregate(weighted_results)
        return ndarrays_to_parameters(aggregated), {}

def server_fn(context: Context):
    # Load best PID params from file
    pid_params = {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05}  # default fallback values
    if os.path.exists("best_pid.txt"):
        with open("best_pid.txt", "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                k, v = line.strip().split(" = ")
                pid_params[k] = float(v)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    init_params = ndarrays_to_parameters(get_weights(Net()))

    def on_fit_config_fn(rnd): return {"server_round": rnd}

    strategy = FedPIDAvg(
        Kp=pid_params["Kp"],
        Ki=pid_params["Ki"],
        Kd=pid_params["Kd"],
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=init_params,
        on_fit_config_fn=on_fit_config_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
