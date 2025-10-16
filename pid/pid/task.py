"""pid: A Flower / PyTorch app."""

# Standard library
from collections import OrderedDict  # Preserve parameter order when reconstructing state dict

# PyTorch imports for model definition and training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Federated datasets and partitioning utilities
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

# Image transformation utilities
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, v2

# Global cache for the federated dataset to avoid repeated downloads
fds = None  # Cache FederatedDataset instance across calls


def batch_transform(batch):
    """Apply transformations to a batch of FEMNIST data."""
    # Define the image transformation pipeline using torchvision.transforms.v2
    transform = v2.Compose(
        [
            v2.ToImage(),  # Convert PIL Image to PyTorch tensor
            v2.ToDtype(torch.float32, scale=True),  # Scale to [0.0, 1.0]
            v2.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1.0, 1.0]
        ]
    )
    # Apply the transform to each image in the 'image' list of the batch
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


class Net(nn.Module):
    """
    Stronger CNN model for FEMNIST.
    Architecture:
      - Conv2d -> BatchNorm -> ReLU -> MaxPool
      - Conv2d -> BatchNorm -> ReLU -> MaxPool
      - Conv2d -> BatchNorm -> ReLU -> MaxPool
      - Flatten -> Dropout -> FC -> ReLU -> FC
    Output for 62 classes (0-9, A-Z, a-z).
    """
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional block: 3 input channels to 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
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
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 62)  # 62 classes for FEMNIST

    def forward(self, x):
        # Pass input through first conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten tensor
        x = x.view(-1, 128 * 3 * 3)
        # Dropout and FC layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_data(partition_id: int, num_partitions: int):
    """
    Load and partition FEMNIST data for a specific client.

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
        # Define the partitioner
        partitioner = NaturalIdPartitioner(partition_by="writer_id")
        # Download and partition the dataset
        fds = FederatedDataset(dataset="flwrlabs/femnist", partitioners={"train": partitioner})

    # Let's get the partition and wrap it in a DataLoader
    # Note: FEMNIST is partitioned by writer_id, so partition_id (node_id) will map to a writer
    partition = fds.load_partition(partition_id, "train")
    # Apply transformation
    partition = partition.with_transform(batch_transform)

    # Split into train/test (80% train, 20% test)
    train_size = int(0.8 * len(partition))
    test_size = len(partition) - train_size
    train_partition, test_partition = random_split(
        partition, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Ensure reproducibility
    )

    # Wrap in DataLoaders
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)

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
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
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
    criterion = nn.CrossEntropyLoss().to(device)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss / len(testloader.dataset), accuracy


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
