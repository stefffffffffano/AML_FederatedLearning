import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from .utils import evaluate


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

def save_data(global_model, val_accuracies, val_losses, train_accuracies, train_losses, file_name):
    """
    Save the global model, val_accuracies, val_losses, train_accuracies, and train_losses to a file.
    
    Args:
        global_model (nn.Module): The global model to be saved.
        val_accuracies (list): List of validation accuracies.
        val_losses (list): List of validation losses.
        train_accuracies (list): List of training accuracies.
        train_losses (list): List of training losses.
        file_name (str): Name of the file to save the data.
    """
    # Fixed base directory
    directory = './trained_models/'
    # Ensure the base directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Complete path for the file
    file_path = os.path.join(directory, file_name)
    
    # Save all data (model state and metrics) into a dictionary
    save_dict = {
        'model_state': global_model.state_dict(),
        'val_accuracies': val_accuracies,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'train_losses': train_losses
    }
    
    # Save the dictionary to the specified file
    torch.save(save_dict, file_path)
    print(f"Data saved successfully to {file_path}")

def load_data(model, file_name):
    """
    Load the model weights and metrics from a file.
    
    Args:
        model (nn.Module): The model to load the weights into.
        file_name (str): Name of the file to load the data from.
    
    Returns:
        tuple: A tuple containing the model, val_accuracies, val_losses, train_accuracies, and train_losses.
    """
    # Fixed base directory
    directory = './trained_models/'
    # Complete path for the file
    file_path = os.path.join(directory, file_name)
    
    # Load the saved data from the specified file
    save_dict = torch.load(file_path)
    
    # Load the model state
    model.load_state_dict(save_dict['model_state'])
    
    # Extract the metrics
    val_accuracies = save_dict['val_accuracies']
    val_losses = save_dict['val_losses']
    train_accuracies = save_dict['train_accuracies']
    train_losses = save_dict['train_losses']
    
    print(f"Data loaded successfully from {file_path}")
    
    return model, val_accuracies, val_losses, train_accuracies, train_losses