import torch
import torch.optim as optim
from copy import deepcopy
import numpy as np
from torch.utils.data import Subset
import os
from torch.utils.data import DataLoader, Subset
import logging
from torch.utils.data import DataLoader, TensorDataset
from Client import Client
from torch.nn.utils.rnn import pad_sequence
import json
log = logging.getLogger(__name__)




class Server:
    def __init__(self, global_model, device, char_to_idx):
        self.global_model = global_model
        self.device = device
        self.char_to_idx = char_to_idx

    def save_checkpoint(self, model, optimizer, epoch, hyperparameters, subfolder="", checkpoint_data=None):
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
        checkpoint = {'model_state_dict': model.state_dict(), 'epoch': epoch}
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

    def load_checkpoint(self, model, optimizer, hyperparameters, subfolder=""):
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
            model.load_state_dict(checkpoint['model_state_dict'])
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
    
    def delete_existing_checkpoints(self, subfolder=""):
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

    def char_to_tensor(self, characters):
        indices = [self.char_to_idx.get(char, self.char_to_idx['<pad>']) for char in characters] # Get the index for the character. If not found, use the index for padding.
        return torch.tensor(indices, dtype=torch.long)

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
    def train_federated(self, criterion, raw_data, num_clients, rounds, lr, momentum, batchsize, wd, C=0.1, local_steps=4, log_freq=10, detailed_print=True,gamma=None):
        # val_accuracies = []
        # val_losses = []
        train_accuracies = []
        train_losses = []
        best_model_state = None  # The model with the best accuracy
        client_selection_count = [0] * num_clients #Count how many times a client has been selected
        #best_val_acc = 0.0
        best_train_loss = float('inf')

        shards = self.sharding(raw_data) #each shard represent the training data for one client
        client_sizes = [len(shard) for shard in shards]

        self.global_model.to(self.device) #as alwayse, we move the global model to the specified device (CPU or GPU)

        #loading checkpoint if it exists
        # checkpoint_start_step, data_to_load = load_checkpoint(model=self.global_model,optimizer=None,hyperparameters=f"LR{lr}_WD{wd}", subfolder="Federated/")
        # if data_to_load is not None:
        #   train_accuracies = data_to_load['train_accuracies']
        #   train_losses = data_to_load['train_losses']
        #   client_selection_count = data_to_load['client_selection_count']
        probabilities = None
        if gamma is not None:
            probabilities = self.skewed_probabilities(num_clients, gamma)

        # ********************* HOW IT WORKS ***************************************
        # The training runs for rounds iterations (GLOBAL_ROUNDS=200)
        # Each round simulates one communication step in federated learning, including:
        # 1) client selection
        # 2) local training (of each client)
        # 3) central aggregation
        for round_num in range(rounds):
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
                client = Client(client_id, client_loader, local_model, self.device, self.char_to_idx)
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
            # #Validation at the server
            # val_accuracy, val_loss = evaluate(self.global_model, validloader)
            # val_accuracies.append(val_accuracy)
            # val_losses.append(val_loss)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_model_state = deepcopy(self.global_model.state_dict())

            if (round_num+1) % log_freq == 0:
                if detailed_print:
                    print(f"-->training accuracy: {train_accuracy:.2f}")
                    print(f"-->training loss: {train_loss:.4f}")

                # checkpointing
                checkpoint_data = {
                    'train_accuracies': train_accuracies,
                    'train_losses': train_losses,
                    'client_selection_count': client_selection_count
                }
                self.save_checkpoint(self.global_model,optimizer=None, epoch=round_num, hyperparameters=f"LR{lr}_WD{wd}", subfolder="Federated/", checkpoint_data=checkpoint_data)
                if detailed_print:
                    print(f"------------------------------ Round {round_num+1} terminated: model updated -----------------------------\n\n" )

        self.global_model.load_state_dict(best_model_state)

        return self.global_model, train_accuracies, train_losses, client_selection_count

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