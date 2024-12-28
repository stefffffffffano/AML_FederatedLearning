from torch.utils.data import Subset
from copy import deepcopy
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def sharding(dataset, number_of_clients, number_of_classes=100):
    """
    Function that performs the sharding of the dataset given as input.
    dataset: dataset to be split;
    number_of_clients: the number of partitions we want to obtain;
    number_of_classes: (int) the number of classes inside each partition, or 100 for IID;
    """

    # Validation of input parameters
    if not (1 <= number_of_classes <= 100):
        raise ValueError("number_of_classes should be an integer between 1 and 100")

    # Shuffle dataset indices for randomness
    indices = np.random.permutation(len(dataset))

    # Compute basic partition sizes
    basic_partition_size = len(dataset) // number_of_clients
    remainder = len(dataset) % number_of_clients

    shards = []
    start_idx = 0

    if number_of_classes == 100:  # IID Case
        # Equally distribute indices among clients: we can just randomly assign to each client an equal amount of records
        for i in range(number_of_clients):
            end_idx = start_idx + basic_partition_size + (1 if i < remainder else 0)
            shards.append(Subset(dataset, indices[start_idx:end_idx]))
            start_idx = end_idx
    else:  # non-IID Case
        # Count of each class in the dataset
        from collections import Counter
        target_counts = Counter(target for _, target in dataset)

        # Calculate per client class allocation
        class_per_client = np.random.choice(list(target_counts.keys()), size=number_of_classes, replace=False)
        class_idx = {class_: np.where([target == class_ for _, target in dataset])[0] for class_ in class_per_client}

        # Assign class indices evenly to clients
        for i in range(number_of_clients):
            client_indices = np.array([], dtype=int)
            for class_ in class_per_client:
                n_samples = len(class_idx[class_]) // number_of_clients + (1 if i < remainder else 0)
                client_indices = np.concatenate((client_indices, class_idx[class_][:n_samples]))
                class_idx[class_] = np.delete(class_idx[class_], np.arange(n_samples))

            shards.append(Subset(dataset, indices=client_indices))

    return shards

def client_selection(number_of_clients, clients_fraction, dataset):
    print()

def client_update(model, client_id, client_data, criterion, optimizer, local_steps=4, detailed_print=False):
    """
    Trains a given client's local model on its dataset for a fixed number of steps (`local_steps`).

    Args:
        model (nn.Module): The local model to be updated.
        client_id (int): Identifier for the client (used for logging/debugging purposes).
        client_data (DataLoader): The data loader for the client's dataset.
        criterion (Loss): The loss function used for training (e.g., CrossEntropyLoss).
        optimizer (Optimizer): The optimizer used for updating model parameters (e.g., SGD).
        local_steps (int): Number of local epochs to train on the client's dataset.
        detailed_print (bool): If True, logs the final loss after training.

    Returns:
        dict: The state dictionary of the updated model.
    """
    model.train()  # Set the model to training mode
    for epoch in range(local_steps):
        for data, targets in client_data:
            # Move data and targets to the specified device (e.g., GPU or CPU)
            data, targets = data.to(DEVICE), targets.to(DEVICE)

            # Reset the gradients before backpropagation
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(data)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass: compute gradients and update weights
            loss.backward()
            optimizer.step()

    # Optionally, print the loss for the last epoch of training
    if detailed_print:
        print(f'Client {client_id} --> Final Loss (Epoch {epoch + 1}): {loss.item()}')

    # Return the updated model's state dictionary (weights)
    return model.state_dict()


def fedavg_aggregate(global_model, client_states, client_sizes):
    """
    Aggregates model updates from selected clients using the Federated Averaging (FedAvg) algorithm.
    The updates are weighted by the size of each client's dataset.

    Args:
        global_model (nn.Module): The global model whose structure is used for aggregation.
        client_states (list[dict]): A list of state dictionaries (model weights) from participating clients.
        client_sizes (list[int]): A list of dataset sizes for the participating clients.

    Returns:
        dict: The aggregated state dictionary with updated model parameters.
    """
    # Copy the global model's state dictionary for aggregation
    new_state = deepcopy(global_model.state_dict())

    # Calculate the total number of samples across all participating clients
    total_samples = sum(client_sizes)

    # Initialize all parameters in the new state to zero
    for key in new_state:
        new_state[key] = torch.zeros_like(new_state[key])

    # Perform a weighted average of client updates
    for state, size in zip(client_states, client_sizes):
        for key in new_state:
            # Add the weighted contribution of each client's parameters
            new_state[key] += (state[key] * size / total_samples)

    # Return the aggregated state dictionary with updated weights
    return new_state
