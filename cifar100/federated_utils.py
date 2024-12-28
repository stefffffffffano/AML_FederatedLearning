from torch.utils.data import Subset
from copy import deepcopy
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#TODO: ensure Identically distributed over classes!
def shard_dataset_iid(dataset, num_clients):
    """
    Splits the input dataset into `num_clients` equal-sized shards.
    Each shard corresponds to the training data for a single client

    Args:
        dataset (Dataset): The dataset to be split into shards.
        num_clients (int): The number of clients to divide the dataset among.

    Returns:
        list[Subset]: A list of dataset shards, one for each client.
    """
    # Calculate the number of items each client should receive
    num_items_per_client = len(dataset) // num_clients

    # Create shards as Subset objects, each containing a range of indices
    shards = [Subset(dataset, range(i * num_items_per_client, (i + 1) * num_items_per_client))
              for i in range(num_clients)]
    return shards


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
