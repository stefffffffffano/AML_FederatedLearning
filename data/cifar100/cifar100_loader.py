import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

def load_cifar100(batch_size=32, validation_split=0.1, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the full training set
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=download, transform=transform)

    # Split the training set into training and validation sets
    num_train = len(full_trainset)
    num_valid = int(num_train * validation_split)
    num_train -= num_valid

    #Use the function used during lab 1 to split the dataset coherently
    trainset, validset = random_split(full_trainset, [num_train, num_valid])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)

    # Load the test dataset
    testset = datasets.CIFAR100(root='./data', train=False, download=download, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, validloader, testloader
