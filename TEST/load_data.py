from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader

# Step 1: Load the IMDb dataset using the datasets module
imdb_dataset = load_dataset("imdb")

# Step 2: Separate the dataset into train and test sets
train_data = imdb_dataset['train']
test_data = imdb_dataset['test']

# Step 3: Function to partition the training and testing data for federated learning
def partition_data(data, num_clients):
    """Partition dataset into subsets for each client."""
    data_size = len(data)
    partition_size = data_size // num_clients
    partitions = []

    indices = np.random.permutation(data_size)  # Shuffle indices to create a random split
    for i in range(num_clients):
        # Calculate start and end indices for this client's partition
        start = i * partition_size
        end = start + partition_size

        # Get the partition for the current client
        partition_indices = indices[start:end]
        partition = data.select(partition_indices)
        partitions.append(partition)

    return partitions

# Step 4: Define the number of clients for federated learning
num_clients = 3  # Example: 3 clients

# Step 5: Partition the training and testing data among clients
client_train_data = partition_data(train_data, num_clients)
client_test_data = partition_data(test_data, num_clients)


