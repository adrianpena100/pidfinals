import torch  # PyTorch for tensor operations and model training
from flwr.client import ClientApp, NumPyClient  # Flower client interfaces
from flwr.common import Context  # Flower context for client configuration
from pid.task import Net, get_weights, load_data, set_weights, test, train  # Model and data utilities

NUM_MALICIOUS_CLIENTS = 6  # Number of clients simulating malicious behavior

class FlowerClient(NumPyClient):
    """
    Custom Flower NumPyClient that trains and evaluates a PyTorch model.
    """
    def __init__(self, net, trainloader, valloader, local_epochs, is_malicious=False):
        # Initialize client with model, data loaders, training epochs, and malicious flag
        self.net = net  # PyTorch model instance
        self.trainloader = trainloader  # DataLoader for local training data
        self.valloader = valloader  # DataLoader for local validation data
        self.local_epochs = local_epochs  # Number of epochs to train locally
        self.is_malicious = is_malicious  # Whether client simulates malicious behavior
        # Select device: GPU if available, else CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Move model to selected device
        self.net.to(self.device)

    def fit(self, parameters, config):
        """
        Train the local model on client's data and return updated weights and metrics.
        """
        # Update model weights to global parameters received from server
        set_weights(self.net, parameters)
        # If malicious, alter labels to simulate adversarial behavior in training
        if self.is_malicious:
            for batch in self.trainloader:
                batch["label"] = torch.remainder(batch["label"] + 5, 10)

        # Perform local training and retrieve training loss
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        # Extract updated weights
        new_weights = get_weights(self.net)
        # POISON the weights if this client is malicious
        if self.is_malicious:
            new_weights = [w * -5.0 for w in new_weights]
        # Return (possibly poisoned) weights, dataset size, and metrics
        return (
            new_weights,
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "malicious": self.is_malicious},
        )

    def evaluate(self, parameters, config):
        """
        Evaluate the local model on client's validation data.
        """
        # Update model weights to global parameters before evaluation
        set_weights(self.net, parameters)
        # Compute loss and accuracy on validation set
        loss, accuracy = test(self.net, self.valloader, self.device)
        # Return evaluation loss, dataset size, and accuracy metric
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Instantiate model
    net = Net()
    # Determine this client's data partition ID and total partitions
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    # Get number of local epochs from run configuration (default to 1)
    local_epochs = int(context.run_config.get("local-epochs", 1))

    ###################################################################
    #### CHANGE TO FALSE IF YOU WANT TO TURN OFF MALICIOUS CLIENTS ####
    #is_malicious = False
    is_malicious = partition_id < NUM_MALICIOUS_CLIENTS # First two clients simulate malicious behavior
    ####################################################################

    # Load the partitioned train and validation data for this client
    trainloader, valloader = load_data(partition_id, num_partitions)
    # Return a FlowerClient wrapped as a Flower client application
    return FlowerClient(net, trainloader, valloader, local_epochs, is_malicious).to_client()

# Create a Flower client application using client_fn
app = ClientApp(client_fn)