import torch.nn as nn
import torch.nn.functional as F
"""
Model architecture for the CIFAR-100 dataset.
The model is based on the LeNet-5 architecture with some modifications.
Reference: Hsu et al., Federated Visual Classification with Real-World Data Distribution, ECCV 2020

CNN similar to LeNet5 which has two 5×5, 64-channel convolution layers, each precedes a 2×2
max-pooling layer, followed by two fully-connected layers with 384 and 192
channels respectively and finally a softmax linear classifier
"""


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 5 * 5, 384),  # Updated to be consistent with data augmentation
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 100)  # 100 classes for CIFAR-100
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the conv layers
        x = self.fc_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
