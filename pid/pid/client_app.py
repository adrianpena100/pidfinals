import torch  # PyTorch for tensor operations and model training
import random  # Random number generation for client selection
import numpy as np  # Numerical operations for confusion‑matrix metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score  # Evaluation metrics

from flwr.client import ClientApp, NumPyClient  # Flower client interfaces
from flwr.common import Context  # Flower context for client configuration
from pid.task import Net, get_weights, load_data, set_weights, test, train  # Model and data utilities

random.seed(42)  # For reproducibility across runs

# ───── Federated‑learning population setup ─────────────────────────────
NUM_CLIENTS = 30            # Total number of logical clients
NUM_MALICIOUS_CLIENTS = 0   # Number of clients acting adversarially
MALICIOUS_CLIENTS = random.sample(range(NUM_CLIENTS), NUM_MALICIOUS_CLIENTS)


class FlowerClient(NumPyClient):
    """Custom Flower NumPyClient that trains and evaluates a PyTorch model."""

    def __init__(self, net, trainloader, valloader, local_epochs, is_malicious=False):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.is_malicious = is_malicious

        # GPU if available, CPU otherwise
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    # ────────────────────────── Fit ────────────────────────────────────
    def fit(self, parameters, config):
        """Perform local training and return model updates."""
        # Synchronize with the latest global weights
        set_weights(self.net, parameters)

        # Label‑flipping attack (only for malicious clients)
        if self.is_malicious:
            for batch in self.trainloader:
                # FEMNIST has 62 classes; rotate label by 5 positions
                batch["character"] = torch.remainder(batch["character"] + 5, 62)

        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        new_weights = get_weights(self.net)

        # Gradient‑sign‑flip attack
        if self.is_malicious:
            new_weights = [w * -5.0 for w in new_weights]

        return (
            new_weights,
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "malicious": self.is_malicious},
        )

    # ─────────────────────── Evaluate ──────────────────────────────────
    def evaluate(self, parameters, config):
        """Evaluate on the validation set and report rich metrics.

        Adds macro‑averaged *false‑positive rate* (FPR) and *true negatives* (TN)
        so the server can track them round‑by‑round without extra plumbing.
        """
        # Update model weights to global parameters before evaluation
        set_weights(self.net, parameters)

        # Standard loss and top‑1 accuracy
        loss, accuracy = test(self.net, self.valloader, self.device)

        # ── Gather predictions and labels for confusion‑matrix stats ──
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in self.valloader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.net(x).argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)           # Shape (C, C)
        fp = cm.sum(axis=0) - np.diag(cm)               # False positives per class
        fn = cm.sum(axis=1) - np.diag(cm)               # False negatives (unused but shown)
        tp = np.diag(cm)                                # True positives
        tn = cm.sum() - (fp + fn + tp)                  # True negatives per class

        # Macro averages (multi‑class friendly)
        true_neg_macro = tn.mean().item()
        fpr_macro = np.where(fp + tn == 0, 0, fp / (fp + tn)).mean().item()

        metrics = {
            "accuracy": accuracy,
            "false_pos_rate": fpr_macro,
            "true_neg": true_neg_macro,
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

        return loss, len(self.valloader.dataset), metrics


# ────────────────────────── Factory ────────────────────────────────────

def client_fn(context: Context):
    """Create a FlowerClient instance for the given runtime context."""
    net = Net()

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    local_epochs = int(context.run_config.get("local-epochs", 1))

    is_malicious = partition_id in MALICIOUS_CLIENTS

    trainloader, valloader = load_data(partition_id, num_partitions)
    return FlowerClient(net, trainloader, valloader, local_epochs, is_malicious).to_client()


# Expose the client application to Flower
app = ClientApp(client_fn)
