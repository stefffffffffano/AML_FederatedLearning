from torch.backends import cudnn
import time


class Client:
    def __init__(self, client_id, data_loader, model, device):
        """
        Initializes a federated learning client.
        :param client_id: Unique identifier for the client.
        :param data_loader: Data loader specific to the client.
        :param model: The model class to be used by the client.
        :param device: The device (CPU/GPU) to perform computations.
        """
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model.to(device)
        self.device = device

    def client_update(self, client_data, criterion, optimizer, local_steps=4, detailed_print=False):
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

        self.model.train()  # Set the model to training mode
        step_count = 0
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        while step_count < local_steps:
            for data, targets in client_data:
                
                # Move data and targets to the specified device (e.g., GPU or CPU)
                data, targets = data.to(self.device), targets.to(self.device)

                # Reset the gradients before backpropagation
                optimizer.zero_grad()

                # Forward pass: compute model predictions
                outputs = self.model(data)

                # Compute the loss
                loss = criterion(outputs, targets)

                # Backward pass: compute gradients and update weights
                loss.backward()
                optimizer.step()

                #---------- Accumulate metrics
                #  Accumulates the weighted loss for the number of samples in the batch to account for any variation in 
                #  batch size due to, for example, the smaller final batch. A little too precise? :) 
                total_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                total_samples += data.size(0)

                step_count +=1
                if step_count >= local_steps:
                  break

        #---------- Compute averaged metrics
        avg_loss = total_loss / total_samples
        avg_accuracy = correct_predictions / total_samples * 100       

        # Optionally, print the loss for the last epoch of training
        if detailed_print:
          print(f'Client {self.client_id} --> Final Loss (Step {step_count}/{local_steps}): {loss.item()}')


        # Return the updated model's state dictionary (weights)
        return self.model.state_dict(), avg_loss, avg_accuracy