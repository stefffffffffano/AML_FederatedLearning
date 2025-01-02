import sys
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
import numpy as np
from torch.utils.data import Subset
from statistics import mean
#from cifar100_loader import load_cifar100
#from models.model import LeNet5 #import the model
from Client import Client
import os
import json
from torch.utils.data import DataLoader, Subset

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
    def train_federated(self, criterion, trainloader, validloader, num_clients, num_classes, rounds, lr, momentum, batchsize, wd, C=0.1, local_steps=4, log_freq=10, detailed_print=False):
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
        checkpoint_start_step, data_to_load = self.load_checkpoint(optimizer=None,hyperparameters=f"LR{lr}_WD{wd}", subfolder="Federated/")
        if data_to_load is not None:
          val_accuracies = data_to_load['val_accuracies']
          val_losses = data_to_load['val_losses']
          train_accuracies = data_to_load['train_accuracies']
          train_losses = data_to_load['train_losses']
          client_selection_count = data_to_load['client_selection_count']


        # ********************* HOW IT WORKS ***************************************
        # The training runs for rounds iterations (GLOBAL_ROUNDS=2000)
        # Each round simulates one communication step in federated learning, including:
        # 1) client selection
        # 2) local training (of each client)
        # 3) central aggregation
        for round_num in range(checkpoint_start_step, rounds):
            if (round_num+1) % log_freq == 0:
              print(f"------------------------------------- Round {round_num+1} ------------------------------------------------" )

            #start_time = time.time()  # for testing-----------------------------

            # 1) client selection: In each round, a fraction C (e.g., 10%) of clients is randomly selected to participate.
            #     This reduces computation costs and mimics real-world scenarios where not all devices are active.
            selected_clients = random.sample(range(num_clients), int(C * num_clients))
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

                

                #local_state = client.client_update(client_loader, criterion, optimizer, local_steps, round_num % log_freq == 0 and detailed_print)
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
            #if round_num % log_freq:
            val_accuracy, val_loss = self.evaluate(validloader, criterion)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_state = deepcopy(self.global_model.state_dict())

            if (round_num+1) % log_freq == 0:

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
                self.save_checkpoint(optimizer=None, epoch=round_num, hyperparameters=f"LR{lr}_WD{wd}", subfolder="Federated/", checkpoint_data=checkpoint_data)

                print(f"------------------------------ Round {round_num} terminated: model updated -----------------------------\n\n" )


            # for testing ------------------------------------------------------
            #end_time = time.time()  # Record the end time
            #elapsed_time = end_time - start_time  # Calculate the elapsed time
            #print(f'Single round time taken: {elapsed_time:.4f} seconds\n\n')


        self.global_model.load_state_dict(best_model_state)

        return self.global_model, val_accuracies, val_losses, train_accuracies, train_losses, client_selection_count

    def sharding(self, dataset, number_of_clients, number_of_classes=100):
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
    

    def evaluate(self, dataloader, criterion):

        with torch.no_grad():
            self.global_model.train(False) # Set Network to evaluation mode
            running_corrects = 0
            losses = []

            for data, targets in dataloader:
                data = data.to(self.device)        # Move the data to the GPU
                targets = targets.to(self.device)  # Move the targets to the GPU

                # Forward Pass
                outputs = self.global_model(data)
                loss = criterion(outputs, targets)
                losses.append(loss.item())
                # Get predictions
                _, preds = torch.max(outputs.data, 1)
                # Update Corrects
                running_corrects += torch.sum(preds == targets.data).data.item()
                # Calculate Accuracy
                accuracy = running_corrects / float(len(dataloader.dataset))

        return accuracy, mean(losses)
    

    def save_checkpoint(self, optimizer, epoch, hyperparameters, subfolder="", checkpoint_data=None):
        """
        Saves the model checkpoint and removes the previous one if it exists.

        Arguments:
        model -- The model whose state is to be saved.
        optimizer -- The optimizer whose state is to be saved (can be None).
        epoch -- The current epoch of the training process.
        hyperparameters -- A string representing the model's hyperparameters for file naming.
        subfolder -- Optional subfolder within the checkpoint directory to save the checkpoint.
        checkpoint_data -- Data to save in a JSON file (e.g., training logs).
        """
        # Define the path for the subfolder where checkpoints will be stored
        subfolder_path = os.path.join(self.CHECKPOINT_DIR, subfolder)
        # Create the subfolder if it doesn't exist
        os.makedirs(subfolder_path, exist_ok=True)

        # Construct filenames for both the model checkpoint and the associated JSON file
        filename = f"model_epoch_{epoch}_params_{hyperparameters}.pth"
        filepath = os.path.join(subfolder_path, filename)
        filename_json = f"model_epoch_{epoch}_params_{hyperparameters}.json"
        filepath_json = os.path.join(subfolder_path, filename_json)

        # Define the filenames for the previous checkpoint files, to remove them if necessary
        previous_filepath = os.path.join(subfolder_path, f"model_epoch_{epoch - 1}_params_{hyperparameters}.pth")
        previous_filepath_json = os.path.join(subfolder_path, f"model_epoch_{epoch - 1}_params_{hyperparameters}.json")

        # Remove the previous checkpoint if it exists, but only for epochs greater than 1
        if epoch > 1 and os.path.exists(previous_filepath):
            os.remove(previous_filepath)
            os.remove(previous_filepath_json)

        # Prepare the checkpoint data dictionary
        checkpoint = {'model_state_dict': self.global_model.state_dict(), 'epoch': epoch}
        # If an optimizer is provided, save its state as well
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save the model and optimizer (if provided) state dictionary to the checkpoint file
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

        # If additional data (e.g., training logs) is provided, save it to a JSON file
        if checkpoint_data:
            with open(filepath_json, 'w') as json_file:
                json.dump(checkpoint_data, json_file, indent=4)


    def load_checkpoint(self, optimizer, hyperparameters, subfolder=""):
        """
        Loads the latest checkpoint available based on the specified hyperparameters.

        Arguments:
        model -- The model whose state will be updated from the checkpoint.
        optimizer -- The optimizer whose state will be updated from the checkpoint (can be None).
        hyperparameters -- A string representing the model's hyperparameters for file naming.
        subfolder -- Optional subfolder within the checkpoint directory to look for checkpoints.

        Returns:
        The next epoch to resume from and the associated JSON data if available.
        """
        # Define the path to the subfolder where checkpoints are stored
        subfolder_path = os.path.join(self.CHECKPOINT_DIR, subfolder)

        # If the subfolder doesn't exist, print a message and start from epoch 1
        if not os.path.exists(subfolder_path):
            print("No checkpoint found, starting from epoch 1.")
            return 1, None  # Epoch starts from 1

        # Search for checkpoint files in the subfolder that match the hyperparameters
        files = [f for f in os.listdir(subfolder_path) if f"params_{hyperparameters}" in f and f.endswith('.pth')]

        # If checkpoint files are found, load the one with the highest epoch number
        if files:
            latest_file = max(files, key=lambda x: int(x.split('_')[2]))  # Find the latest epoch file
            filepath = os.path.join(subfolder_path, latest_file)
            checkpoint = torch.load(filepath, weights_only=True)

            # Load the model state from the checkpoint
            self.global_model.load_state_dict(checkpoint['model_state_dict'])
            # If an optimizer is provided, load its state as well
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Try to load the associated JSON file if available
            json_filepath = os.path.join(subfolder_path, latest_file.replace('.pth', '.json'))
            json_data = None
            if os.path.exists(json_filepath):
                # If the JSON file exists, load its contents
                with open(json_filepath, 'r') as json_file:
                    json_data = json.load(json_file)
                print("Data loaded!")
            else:
                # If no JSON file exists, print a message
                print("No data found")

            # Print the epoch from which the model is resuming
            print(f"Checkpoint found: Resuming from epoch {checkpoint['epoch'] + 1}\n\n")
            return checkpoint['epoch'] + 1, json_data

        # If no checkpoint is found, print a message and start from epoch 1
        print("No checkpoint found, starting from epoch 1..\n\n")
        return 1, None  # Epoch starts from 1
    
    def delete_existing_checkpoints(self,subfolder=""):
        """
        Deletes all existing checkpoints in the specified subfolder.

        Arguments:
        subfolder -- Optional subfolder within the checkpoint directory to delete checkpoints from.
        """
        subfolder_path = os.path.join(self.CHECKPOINT_DIR, subfolder)
        if os.path.exists(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"All existing checkpoints in {subfolder_path} have been deleted.")
        else:
            print(f"No checkpoint folder found at {subfolder_path}.")