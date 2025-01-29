import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from .utils import evaluate


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

def plot_local_data_distribution(client_dataset, dir_name, file_name):
    """
    Plots the distribution of classes in a client's local dataset and ensures all 100 classes are displayed.
    
    Args:
        client_dataset: Dataset object containing (data, label) tuples.
        dir_name: Directory name for saving the plot.
        file_name: Name of the file to save the plot.
    """
    # Fixed base directory
    directory = './plots_federated/data_sharding_distributions/cifar100' + '_' + dir_name
    os.makedirs(directory, exist_ok=True)  # Ensure the base directory exists
    file_path = os.path.join(directory, file_name)  # Complete path for the file

    # Extract labels from the Subset dataset
    base_dataset = client_dataset.dataset
    indices = client_dataset.indices
    labels = np.array([base_dataset[idx][1] for idx in indices])

    # Count occurrences of each class
    class_labels, frequencies = np.unique(labels, return_counts=True)

    # Create an array for all 100 classes with zero counts
    total_classes = 100
    all_frequencies = np.zeros(total_classes, dtype=int)
    
    # Fill in the counts for existing classes
    all_frequencies[class_labels] = frequencies

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(total_classes), all_frequencies, alpha=0.7, color='C0', edgecolor='black')

    # Formatting
    plt.title('Class Distribution in Local Dataset', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(range(total_classes), rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()

    print(f"Plot saved to {file_path}")
    
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

def plot_metrics(train_accuracies, train_losses, val_accuracies,val_losses, file_name):
    """
    Plot the training and validation metrics for a federated learning simulation.
    
    Args:
        train_accuracies (list): List of training accuracies.
        train_losses (list): List of training losses.
        val_accuracies (list): List of validation accuracies.
        val_losses (list): List of validation losses.
        file_name (str): Name of the file to save the plot.
    """
    # Fixed base directory
    directory = './plots_federated/'
    # Ensure the base directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Complete path for the file
    file_path = os.path.join(directory, file_name)
    
    # Create a list of epochs for the x-axis
    epochs = list(range(1, len(train_losses) + 1))
    
    # Plot the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path.replace('.png', '_loss.png'), format='png', dpi=300)
    plt.close()
    
    # Plot the training and validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Training and Validation Accuracy', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path.replace('.png', '_accuracy.png'), format='png', dpi=300)
    plt.close()
    

def save_data(global_model, val_accuracies, val_losses, train_accuracies, train_losses,client_count, file_name):
    """
    Save the global model, val_accuracies, val_losses, train_accuracies,train_losses and client_count to a file.
    
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
        'train_losses': train_losses,
        'client_count': client_count
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
        tuple: A tuple containing the model, val_accuracies, val_losses, train_accuracies train_losses and client_count.
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
    #check if client_count is present in the dictionary
    if 'client_count' not in save_dict:
        client_count = 0
    else:
        client_count = save_dict['client_count']
    
    print(f"Data loaded successfully from {file_path}")
    
    return model, val_accuracies, val_losses, train_accuracies, train_losses,client_count