import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from copy import deepcopy
import numpy as np
import os
import random
import logging

from Client import Client
from utils.utils import evaluate
from utils.checkpointing_utils import save_checkpoint, load_checkpoint
from utils.plotting_utils import plot_local_data_distribution


log = logging.getLogger(__name__)

class Server:
    def __init__(self, global_model, device, CHECKPOINT_DIR):
        self.global_model = global_model
        self.device = device
        self.CHECKPOINT_DIR = CHECKPOINT_DIR
        # Ensure the checkpoint directory exists, creating it if necessary
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
    def train_federated(self, criterion, trainloader, validloader, num_clients, num_classes, rounds, lr, momentum, batchsize, wd, C=0.1, local_steps=4, log_freq=10, detailed_print=False,gamma=None):
        val_accuracies = []
        val_losses = []
        train_accuracies = []
        train_losses = []
        best_model_state = None  # The model with the best accuracy
        client_selection_count = [0] * num_clients #Count how many times a client has been selected
        best_val_acc = 0.0

        shards = self.sharding(trainloader.dataset, num_clients, num_classes) #each shard represent the training data for one client
        client_sizes = [len(shard) for shard in shards]

        self.global_model.to(self.device) #as alwayse, we move the global model to the specified device (CPU or GPU)

        #loading checkpoint if it exists
        checkpoint_start_step, data_to_load = load_checkpoint(model=self.global_model,optimizer=None,hyperparameters=f"LR{lr}_WD{wd}", subfolder="Federated/")
        if data_to_load is not None:
          val_accuracies = data_to_load['val_accuracies']
          val_losses = data_to_load['val_losses']
          train_accuracies = data_to_load['train_accuracies']
          train_losses = data_to_load['train_losses']
          client_selection_count = data_to_load['client_selection_count']
        probabilities = None
        if gamma is not None:
            probabilities = self.skewed_probabilities(num_clients, gamma)

        # ********************* HOW IT WORKS ***************************************
        # The training runs for rounds iterations (GLOBAL_ROUNDS=2000)
        # Each round simulates one communication step in federated learning, including:
        # 1) client selection
        # 2) local training (of each client)
        # 3) central aggregation
        for round_num in range(checkpoint_start_step, rounds):
            if (round_num+1) % log_freq == 0 and detailed_print:
              print(f"------------------------------------- Round {round_num+1} ------------------------------------------------" )

            # 1) client selection: In each round, a fraction C (e.g., 10%) of clients is randomly selected to participate.
            #     This reduces computation costs and mimics real-world scenarios where not all devices are active.
            selected_clients = self.client_selection(num_clients, C,probabilities)
            client_states = []
            client_avg_losses = []
            client_avg_accuracies = []
            for client_id in selected_clients:
                client_selection_count[client_id] += 1

            # 2) local training: for each client updates the model using the client's data for local_steps epochs
            for client_id in selected_clients:
                local_model = deepcopy(self.global_model) #it creates a local copy of the global model
                optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=wd) #same of the centralized version
                client_loader = DataLoader(shards[client_id], batch_size=batchsize, shuffle=True)

                print_log =  (round_num+1) % log_freq == 0 and detailed_print
                client = Client(client_id, client_loader, local_model, self.device)
                client_local_state, client_avg_loss, client_avg_accuracy  = client.client_update(client_loader, criterion, optimizer, local_steps, print_log)

                client_states.append(client_local_state)
                client_avg_losses.append(client_avg_loss)
                client_avg_accuracies.append(client_avg_accuracy)


            # 3) central aggregation: aggregates participating client updates using fedavg_aggregate
            #    and replaces the current parameters of global_model with the returned ones.
            aggregated_state, train_loss, train_accuracy = self.fedavg_aggregate(client_states, [client_sizes[i] for i in selected_clients], client_avg_losses, client_avg_accuracies)
        
            self.global_model.load_state_dict(aggregated_state)

            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            #Validation at the server
            val_accuracy, val_loss = evaluate(self.global_model, validloader)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_state = deepcopy(self.global_model.state_dict())

            if (round_num+1) % log_freq == 0:
                if detailed_print:
                    print(f"--> best validation accuracy: {best_val_acc:.2f}\n--> training accuracy: {train_accuracy:.2f}")
                    print(f"--> validation loss: {val_loss:.4f}\n--> training loss: {train_loss:.4f}")

                # checkpointing
                checkpoint_data = {
                    'val_accuracies': val_accuracies,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'train_losses': train_losses,
                    'client_selection_count': client_selection_count
                }
                save_checkpoint(self.global_model,optimizer=None, epoch=round_num, hyperparameters=f"LR{lr}_WD{wd}", subfolder="Federated/", checkpoint_data=checkpoint_data)
                if detailed_print:
                    print(f"------------------------------ Round {round_num+1} terminated: model updated -----------------------------\n\n" )

        self.global_model.load_state_dict(best_model_state)

        return self.global_model, val_accuracies, val_losses, train_accuracies, train_losses, client_selection_count

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



    def sharding(self, dataset, number_of_clients, number_of_classes=100):
        """
        Function that performs the sharding of the dataset given as input.
        dataset: dataset to be split (should be a PyTorch dataset or similar);
        number_of_clients: the number of partitions we want to obtain (e.g., 100 for 100 clients);
        number_of_classes: (int) the number of classes inside each partition, or 100 for IID (default to 100).
        """

        # Validate the number of classes input
        if not (1 <= number_of_classes <= 100):
            raise ValueError("number_of_classes should be an integer between 1 and 100")

        # Get labels for sorting
        labels = np.array([dataset[i][1] for i in range(len(dataset))]) 
        TOTAL_NUM_CLASSES = len(set(labels))

        shard_size = len(dataset) // (number_of_clients * number_of_classes)  # Shard size for each class per client
        #print("dataset len: ", len(dataset), ", shard size: ", shard_size, ", number of shards: ",(number_of_clients * number_of_classes))
        if shard_size == 0:
            raise ValueError("Shard size is too small; increase dataset size or reduce number of clients/classes.")


        # Divide the dataset into shards, each containing samples from one class
        shards = {}
        for i in range(TOTAL_NUM_CLASSES):  
            # Filter samples for the current class
            class_samples = [j for j in range(len(labels)) if labels[j] == i]
            shards_of_class_i = []
            # While there are enough samples to form a shard
            while len(class_samples) >= shard_size and len(shards_of_class_i) < number_of_clients*(number_of_classes/number_of_clients):
                # Take a shard of shard_size samples
                shards_of_class_i.append(class_samples[:shard_size])
                # Remove the shard_size samples from class_samples
                class_samples = class_samples[shard_size:]
            # Add the last shard (which might be smaller than shard_size)
            if len(class_samples) > 0 and len(shards_of_class_i) == number_of_clients*(number_of_classes/number_of_clients):
                # Distribute remaining samples among existing shards
                for sample in class_samples:
                    random_shard = random.choice(shards_of_class_i)
                    random_shard.append(sample)
            elif class_samples:
                shards_of_class_i.append(class_samples)
            # Store the class shards
            shards[i] = shards_of_class_i  # Store shards by class
        
        client_shards = []  # List to store the dataset for each client          
        for client_id in range(number_of_clients):
                
            client_labels = [label % TOTAL_NUM_CLASSES for label in range(client_id, client_id + number_of_classes)]
            #print(client_labels)

            # Collect the shards for the selected classes
            client_shard_indices = []
            for label in client_labels:
                shard = shards[label].pop(0)  # Pop the first shard from the class's shard list
                client_shard_indices.append(shard)

            # Flatten and combine the shard indices into one list
            client_indices = [idx for shard in client_shard_indices for idx in shard]

            #print(f"Client {client_id} has {len(client_indices)} samples divided in {len(client_shard_indices)} shards (classes).")
            # Create a Subset for the client
            client_dataset = Subset(dataset, client_indices)
            client_shards.append(client_dataset)
        
        return client_shards  # Return the list of dataset subsets (shards) for each client
        
    def plot_sharding_data_distribution(self, trainloader, num_clients, num_classes):
        shards = self.sharding(trainloader.dataset, num_clients, num_classes)
        for client_id in range(num_clients):
            client_dataset = shards[client_id]
            #print(f"Client {client_id} has {len(client_dataset)} samples.")
            plot_local_data_distribution(client_dataset,f"Nc{num_classes}", f"DataDistribution_client{client_id}.png")