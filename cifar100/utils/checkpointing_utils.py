import os
import torch
import json

# Directory where checkpoints are stored
CHECKPOINT_DIR = '../checkpoints/'

# Ensure the checkpoint directory exists, creating it if necessary
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, hyperparameters, subfolder="", checkpoint_data=None):
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
    subfolder_path = os.path.join(CHECKPOINT_DIR, subfolder)
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

def load_checkpoint(model, optimizer, hyperparameters, subfolder=""):
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
    subfolder_path = os.path.join(CHECKPOINT_DIR, subfolder)
    
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
