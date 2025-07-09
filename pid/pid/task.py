"""pid: A Flower / PyTorch app."""

# Standard library
from collections import OrderedDict  # Preserve parameter order when reconstructing state dict

# PyTorch imports for model definition and training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Federated datasets and partitioning utilities
from flwr_datasets import FederatedDataset  # Load federated CIFAR-10 dataset
from flwr_datasets.partitioner import IidPartitioner  # IID partitioner for data splitting

# Image transformation utilities
from torchvision.transforms import Compose, Normalize, ToTensor

# Global cache for the federated dataset to avoid repeated downloads
fds = None  # Cache FederatedDataset instance across calls


class Net(nn.Module):
    """
    Simple CNN model adapted from 'PyTorch: A 60 Minute Blitz'.
    Architecture:
      - Conv2d -> ReLU -> MaxPool
      - Conv2d -> ReLU -> MaxPool
      - Flatten -> FC -> ReLU -> FC -> ReLU -> FC
    Output for 10 classes.
    """
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: in_channels=3 (RGB), out_channels=6, kernel_size=5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max-pooling layer with 2x2 window
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: in_channels=6, out_channels=16, kernel_size=5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers
        # Input features = 16 * 5 * 5 after two poolings on 32x32 input
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # Final output layer for 10 classes
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Pass input through conv => ReLU => pool
        x = self.pool(F.relu(self.conv1(x)))
        # Pass through second conv => ReLU => pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten tensor to (batch_size, features)
        x = x.view(-1, 16 * 5 * 5)
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Return raw logits for final classification
        return self.fc3(x)


def load_data(partition_id: int, num_partitions: int):
    """
    Load and partition CIFAR-10 data for a specific client.

    Steps:
      1. Initialize FederatedDataset once (global cache).
      2. Partition dataset IID across `num_partitions` clients.
      3. Split each partition into train (80%) and test (20%).
      4. Apply PyTorch transforms (ToTensor & Normalize).
      5. Return DataLoader for train and test splits.
    """
    global fds
    # Initialize dataset and partitioner on first call
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    # Load specific client partition
    partition = fds.load_partition(partition_id)
    # Split into train/test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Define transformations: convert to tensor and normalize to [-1,1]
    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def apply_transforms(batch):
        """
        Apply PyTorch transforms to each image in the batch.
        """
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Attach transforms to data
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    # Create DataLoaders
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=32, shuffle=True
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=32
    )
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """
    Train the model for a given number of epochs.

    - Moves model to the specified device (CPU/GPU).
    - Uses Adam optimizer and CrossEntropyLoss.
    - Returns average training loss per batch.
    """
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    # Compute average loss over all batches
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """
    Evaluate the model on the test set.

    - Computes average loss and accuracy.
    - Returns (loss, accuracy).
    """
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            # Count correct predictions
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    """
    Extract model weights as a list of NumPy arrays.
    Used to send parameters to the Flower server.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """
    Load a list of NumPy arrays into the model's state_dict.
    Used to update client model before training/evaluation.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in params_dict
    })
    net.load_state_dict(state_dict, strict=True)
