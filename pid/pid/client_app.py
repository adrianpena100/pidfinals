import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pid.task import Net, get_weights, load_data, set_weights, test, train

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, is_malicious=False):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.is_malicious = is_malicious
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        if self.is_malicious:
            for batch in self.trainloader:
                batch["label"] = torch.remainder(batch["label"] + 5, 10)

        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "malicious": self.is_malicious
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    net = Net()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    local_epochs = int(context.run_config.get("local-epochs", 1))

    ###################################################################
    #### CHANGE TO FALSE IF YOU WANT TO TURN OFF MALICIOUS CLIENTS ####
    #is_malicious = False
    is_malicious = partition_id < 2  # First two clients are malicious
    ####################################################################

    trainloader, valloader = load_data(partition_id, num_partitions)
    return FlowerClient(net, trainloader, valloader, local_epochs, is_malicious).to_client()

app = ClientApp(client_fn)
