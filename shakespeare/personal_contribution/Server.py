import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from Client import Client
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from torch.nn.utils.rnn import pad_sequence

log = logging.getLogger(__name__)

class Server:
    def __init__(self, global_model, device, char_to_idx):
        self.global_model = global_model
        self.device = device
        self.char_to_idx = char_to_idx
        # Ensure the checkpoint directory exists, creating it if necessary

    def fedavg_aggregate(self, client_states, client_sizes, client_avg_losses, client_avg_accuracies):
        """
        Aggregates model updates and client metrics from selected clients using the Federated Averaging (FedAvg) algorithm.
        The updates and metrics are weighted by the size of each client's dataset.

        Args:
            global_model (nn.Module): The global model whose structure is used for aggregation.
            client_states (list[dict]): A list of state dictionaries (model weights) from participating clients.
            client_sizes (list[int]): A list of dataset sizes for the participating clients.
            client_avg_losses (list[float]): A list of average losses for the participating clients.
            client_avg_accuracies (list[float]): A list of average accuracies for the participating clients.

        Returns:
            tuple: The aggregated state dictionary with updated model parameters, global average loss, and global average accuracy.
        """
        # Copy the global model's state dictionary for aggregation
        new_state = deepcopy(self.global_model.state_dict())

        # Calculate the total number of samples across all participating clients
        total_samples = sum(client_sizes)

        # Initialize all parameters in the new state to zero
        for key in new_state:
            new_state[key] = torch.zeros_like(new_state[key])

        # Initialize metrics
        total_loss = 0.0
        total_accuracy = 0.0

        # Perform a weighted average of client updates and metrics
        for state, size, avg_loss, avg_accuracy in zip(client_states, client_sizes, client_avg_losses, client_avg_accuracies):
            for key in new_state:
                # Add the weighted contribution of each client's parameters
                new_state[key] += (state[key] * size / total_samples)
            total_loss += avg_loss * size
            total_accuracy += avg_accuracy * size

        # Calculate global metrics
        global_avg_loss = total_loss / total_samples
        global_avg_accuracy = total_accuracy / total_samples

        # Return the aggregated state dictionary with updated weights and global metrics
        return new_state, global_avg_loss, global_avg_accuracy


    # Federated Learning Training Loop
    ### UPDATED
    #0. receives the shards (the result of sharding method)
    #1. receive the n clients to perform one round
    #2. averages the results coming from the model of the n clients
    #3. returns the model and the results averaged.
    def train_federated(self, criterion, lr, momentum, batchsize, wd, selected_clients, shards, local_steps=4):
      # selected_clients are objects of Individual class

        train_accuracies = []
        train_losses = []

        #shards = self.sharding(trainloader.dataset, num_clients, num_classes) #each shard represent the training data for one client
        client_sizes = [len(shard) for shard in shards]

        self.global_model.to(self.device) #as alwayse, we move the global model to the specified device (CPU or GPU)

        # we only mantain a round
        client_states = []
        client_avg_losses = []
        client_avg_accuracies = []

        # 2) local training: for each client updates the model using the client's data for local_steps epochs
        for client_id in selected_clients.genome:
            local_model = deepcopy(self.global_model) #it creates a local copy of the global model
            optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=wd) #same of the centralized version
            client_loader = DataLoader(shards[client_id], batch_size=batchsize, shuffle=True)

            client = Client(client_id, client_loader, local_model, self.device, self.char_to_idx)
            client_local_state, client_avg_loss, client_avg_accuracy  = client.client_update(client_loader, criterion, optimizer, local_steps)

            client_states.append(client_local_state)
            client_avg_losses.append(client_avg_loss)
            client_avg_accuracies.append(client_avg_accuracy)


        # 3) central aggregation: aggregates participating client updates using fedavg_aggregate
        #    and replaces the current parameters of global_model with the returned ones.
        aggregated_state, train_loss, train_accuracy = self.fedavg_aggregate(client_states, [client_sizes[i] for i in selected_clients.genome], client_avg_losses, client_avg_accuracies)
        # UPDATE THE FITNESS OF INDIVIDUAL:
        selected_clients.fitness = train_loss

        #self.global_model.load_state_dict(aggregated_state)

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)


        return aggregated_state, train_accuracy, train_loss

    def skewed_probabilities(self, number_of_clients, gamma=0.5):
            # Generate skewed probabilities using a Dirichlet distribution
            probabilities = np.random.dirichlet(np.ones(number_of_clients) * gamma)
            return probabilities

    def client_selection(self,number_of_clients, clients_fraction, probabilities=None):
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

        if probabilities is None:
            # Uniformly select clients without replacement
            selected_clients = np.random.choice(number_of_clients, num_clients_to_select, replace=False)
        else:
            selected_clients = np.random.choice(number_of_clients, num_clients_to_select, replace=False, p=probabilities)

        return selected_clients

    def char_to_tensor(self, characters):
        indices = [self.char_to_idx.get(char, self.char_to_idx['<pad>']) for char in characters] # Get the index for the character. If not found, use the index for padding.
        return torch.tensor(indices, dtype=torch.long)

    def sharding(self, data):
        """
        Prepares individual shards for each user, returning a Subset for each.

        Args:
        data (dict): Dataset containing user data.
        char_to_idx (dict): Character to index mapping dictionary for character conversion.

        Returns:
        list: List of Subsets, one for each user.
        """
        subsets = []

        for user in data['users']:
            input_tensors = []
            target_tensors = []

            for entry, target in zip(data['user_data'][user]['x'], data['user_data'][user]['y']):
              input_tensors.append(self.char_to_tensor(entry))  # Use the full sequence of x
              target_tensors.append(self.char_to_tensor(target))  # Directly use the corresponding y as target

            # Padding inputs to ensure all inputs in a batch have the same length
            padded_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=self.char_to_idx['<pad>'])
            targets = torch.cat(target_tensors)

            # Creating the TensorDataset for the user
            dataset = TensorDataset(padded_inputs, targets)

            # Since each user is treated as a separate "client", we create a Subset for each
            subsets.append(Subset(dataset, torch.arange(len(targets))))

        return subsets
