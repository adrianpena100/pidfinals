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
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip

# Global cache for the federated dataset to avoid repeated downloads
fds = None  # Cache FederatedDataset instance across calls


class Net(nn.Module):
    """
    Stronger CNN model for CIFAR-10.
    Architecture:
      - Conv2d -> BatchNorm -> ReLU -> MaxPool
      - Conv2d -> BatchNorm -> ReLU -> MaxPool
      - Conv2d -> BatchNorm -> ReLU -> MaxPool
      - Flatten -> Dropout -> FC -> ReLU -> FC
    Output for 10 classes.
    """
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional block: 3 input channels to 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional block: 32 to 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Third convolutional block: 64 to 128 output channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Max-pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Pass input through first conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten tensor
        x = x.view(-1, 128 * 4 * 4)
        # Dropout and FC layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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

    # Define transformations: random crop, flip, convert to tensor, and normalize to [-1,1]
    pytorch_transforms = Compose([
        RandomCrop(32, padding=4),              # Randomly crop image with padding
        RandomHorizontalFlip(),                 # Flip image horizontally 50% of the time
        ToTensor(),                             # Convert image to PyTorch tensor
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize RGB channels to [-1, 1]
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