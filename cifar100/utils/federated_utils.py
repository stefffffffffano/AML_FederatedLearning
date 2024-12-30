from torch.utils.data import Subset
from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from statistics import mean
import matplotlib.pyplot as plt
from torch.backends import cudnn
import time
import random


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.NLLLoss()  # our loss function for classification tasks on CIFAR-100

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
    remainder = len(dataset) % number_of_clients

    shards = []

    if number_of_classes == 100:  # IID Case
        # Count of each class in the dataset
        labels = np.array(dataset.targets)  
        sorted_indices = indices[np.argsort(labels)]

        # Split indices by class
        class_indices = [
            sorted_indices[labels[sorted_indices] == c] for c in range(100)
        ]

        # Distribute class indices equally among clients
        for client_id in range(number_of_clients):
            client_indices = []
            for class_indices_list in class_indices:
                num_samples_per_client = len(class_indices_list) // number_of_clients
                selected_indices = class_indices_list[:num_samples_per_client]
                client_indices.extend(selected_indices)
                class_indices_list = class_indices_list[num_samples_per_client:]

            shards.append(Subset(dataset, client_indices))
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

def client_selection(number_of_clients, clients_fraction, gamma=None):
    """
    Selects a subset of clients based on uniform or skewed distribution.
    
    Args:
    number_of_clients (int): Total number of clients.
    clients_fraction (float): Fraction of clients to be selected.
    uniform (bool): If True, selects clients uniformly. If False, selects clients based on a skewed distribution.
    gamma (float): Hyperparameter for the Dirichlet distribution controlling the skewness (only used if uniform=False).
    
    Returns:
    list: List of selected client indices.
    """
    num_clients_to_select = int(number_of_clients * clients_fraction)
    
    if gamma is None:
        # Uniformly select clients without replacement
        selected_clients = np.random.choice(number_of_clients, num_clients_to_select, replace=False)
    else:
        # Generate skewed probabilities using a Dirichlet distribution
        probabilities = np.random.dirichlet(np.ones(number_of_clients) * gamma)
        selected_clients = np.random.choice(number_of_clients, num_clients_to_select, replace=False, p=probabilities)
    
    return selected_clients
   

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


    cudnn.benchmark  # Calling this optimizes runtime

    model.train()  # Set the model to training mode
    step_count = 0
    while step_count < local_steps:
        for data, targets in client_data:
            # Move data and targets to the specified device (e.g., GPU or CPU)
            data, targets = data.to(DEVICE), targets.to(DEVICE)


            start_time = time.time()  # for testing-----------------------------

            # Reset the gradients before backpropagation
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(data)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass: compute gradients and update weights
            loss.backward()
            optimizer.step()

            # for testing ------------------------------------------------------
            if detailed_print:
              end_time = time.time()  # Record the end time
              elapsed_time = end_time - start_time  # Calculate the elapsed time
              print(f'Single step time taken: {elapsed_time:.4f} seconds')

            step_count +=1
            if step_count >= local_steps:
              break

    # Optionally, print the loss for the last epoch of training
    if detailed_print:
        print(f'Client {client_id} --> Final Loss (Step {step_count}/{local_steps}): {loss.item()}')


    # Return the updated model's state dictionary (weights)
    return model.state_dict()

def evaluate(model, dataloader):
    with torch.no_grad():
        model.train(False) # Set Network to evaluation mode
        running_corrects = 0
        losses = []
        for data, targets in dataloader:
            data = data.to(DEVICE)        # Move the data to the GPU
            targets = targets.to(DEVICE)  # Move the targets to the GPU
            # Forward Pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # Get predictions
            _, preds = torch.max(outputs.data, 1)
            # Update Corrects
            running_corrects += torch.sum(preds == targets.data).data.item()
            # Calculate Accuracy
            accuracy = running_corrects / float(len(dataloader.dataset))

    return accuracy, mean(losses)


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


# Federated Learning Training Loop
def train_federated(global_model, criterion, trainloader, validloader, num_clients, num_classes, rounds, lr, momentum, batchsize, wd, C=0.1, local_steps=4, log_freq=10, detailed_print=False):
    val_accuracies = []
    val_losses = []
    train_accuracies = []
    train_losses = []
    best_model_state = None  # The model with the best accuracy
    client_selection_count = [0] * num_clients #Count how many times a client has been selected
    best_val_acc = 0.0


    shards = sharding(trainloader.dataset, num_clients, num_classes) #each shard represent the training data for one client
    client_sizes = [len(shard) for shard in shards]

    global_model.to(DEVICE) #as alwayse, we move the global model to the specified device (CPU or GPU)

    # ********************* HOW IT WORKS ***************************************
    # The training runs for rounds iterations (GLOBAL_ROUNDS=2000)
    # Each round simulates one communication step in federated learning, including:
    # 1) client selection
    # 2) local training (of each client)
    # 3) central aggregation
    for round_num in range(rounds):
        if round_num % log_freq == 0 and detailed_print:
          print(f"------------------------------------- Round {round_num} ------------------------------------------------" )

        start_time = time.time()  # for testing-----------------------------

        # 1) client selection: In each round, a fraction C (e.g., 10%) of clients is randomly selected to participate.
        #     This reduces computation costs and mimics real-world scenarios where not all devices are active.
        selected_clients = random.sample(range(num_clients), int(C * num_clients))
        client_states = []
        for client_id in selected_clients:
            client_selection_count[client_id] += 1

        # 2) local training: for each client updates the model using the client's data for local_steps epochs
        for client_id in selected_clients:
            local_model = deepcopy(global_model) #it creates a local copy of the global model
            optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=wd) #same of the centralized version
            client_loader = DataLoader(shards[client_id], batch_size=batchsize, shuffle=True)

            local_state = client_update(local_model, client_id, client_loader, criterion, optimizer, local_steps, round_num % log_freq == 0 and detailed_print)
            client_states.append(local_state)

        # 3) central aggregation: aggregates participating client updates using fedavg_aggregate
        #    and replaces the current parameters of global_model with the returned ones.
        global_model.load_state_dict(fedavg_aggregate(global_model, client_states, [client_sizes[i] for i in selected_clients]))

        #Validation at the server
        #if round_num % log_freq:
        val_accuracy, val_loss = evaluate(global_model, validloader)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = deepcopy(global_model.state_dict())

        train_accuracy, train_loss = evaluate(global_model, trainloader)
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        if round_num % log_freq == 0 and detailed_print:
            print(f"------------------------------ Round {round_num} terminated: model updated -----------------------------" )
            print(f"accuracy: {val_accuracy}, best accuracy: {best_val_acc}")
            # for testing ------------------------------------------------------
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            print(f'Single round time taken: {elapsed_time:.4f} seconds\n\n')


    global_model.load_state_dict(best_model_state)

    return global_model, val_accuracies, val_losses, train_accuracies, train_losses, client_selection_count


def plot_client_selection(client_selection_count, file_name):
    """
    Bar plot to visualize the frequency of client selections in a federated learning simulation.
    
    Args:
        client_selection_count (list): list containing the number of times each client was selected.
        file_name (str): name of the file to save the plot.
    """
    # Fixed base directory
    directory = '../plots_federated/'
    # Ensure the base directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Complete path for the file
    file_path = os.path.join(directory, file_name)
    
    num_clients = len(client_selection_count)
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clients), client_selection_count, alpha=0.7, edgecolor='black')
    plt.xlabel("Client ID", fontsize=14)
    plt.ylabel("Selection Count", fontsize=14)
    plt.title("Client Selection Frequency", fontsize=16)
    plt.xticks(range(num_clients), fontsize=10, rotation=90 if num_clients > 20 else 0)
    plt.tight_layout()
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()
    
def test(global_model, test_loader):
    """
    Evaluate the global model on the test dataset.
    
    Args:
        global_model (nn.Module): The global model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        
    Returns:
        float: The accuracy of the model on the test dataset.
    """
    test_accuracy, _ = evaluate(global_model, test_loader)
    return test_accuracy

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, file_name):
    """
    Plot the training and validation metrics for a federated learning simulation.
    
    Args:
        train_losses (list): List of training losses.
        train_accuracies (list): List of training accuracies.
        val_losses (list): List of validation losses.
        val_accuracies (list): List of validation accuracies.
        file_name (str): Name of the file to save the plot.
    """
    # Fixed base directory
    directory = '../plots_federated/'
    # Ensure the base directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Complete path for the file
    file_path = os.path.join(directory, file_name)
    
    # Create a list of epochs for the x-axis
    epochs = list(range(1, len(train_losses) + 1))
    
    # Plot the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='x')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path.replace('.png', '_loss.png'), format='png', dpi=300)
    plt.close()
    
    # Plot the training and validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red', marker='x')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Training and Validation Accuracy', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path.replace('.png', '_accuracy.png'), format='png', dpi=300)
    plt.close()

def save_model(global_model, file_name):
    """
    Save the global model to a file.
    
    Args:
        global_model (nn.Module): The global model to be saved.
        file_name (str): Name of the file to save the model.
    """
    # Fixed base directory
    directory = '../models_federated/'
    # Ensure the base directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Complete path for the file
    file_path = os.path.join(directory, file_name)
    
    # Save the model to the specified file
    torch.save(global_model.state_dict(), file_path)

def load_model(model, file_name):
    """
    Load the model weights from a file.
    
    Args:
        model (nn.Module): The model to load the weights into.
        file_name (str): Name of the file to load the model from.
    """
    # Fixed base directory
    directory = '../models_federated/'
    # Complete path for the file
    file_path = os.path.join(directory, file_name)
    
    # Load the model weights from the specified file
    model.load_state_dict(torch.load(file_path))
    return model