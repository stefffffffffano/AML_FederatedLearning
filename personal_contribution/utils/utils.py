import torch
from statistics import mean

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, dataloader,criterion):
    """
    Computes the accuracy and the loss on the validation/test set depending on the dataloader passed
    """
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

    return accuracy*100, mean(losses)