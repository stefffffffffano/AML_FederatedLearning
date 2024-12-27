import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_cifar100(batch_size=32, validation_split=0.1, download=True):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Load the entire training set
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=download, transform=transform)

    # Split the training set into training and validation mantaining the class distribution
    indexes = list(range(len(full_trainset)))  
    train_indexes, val_indexes = train_test_split(indexes, train_size=1-validation_split, test_size=validation_split, random_state=42, stratify=full_trainset.targets, shuffle=True)

    # Subset of the training set for training and validation
    train_dataset = Subset(full_trainset, train_indexes)
    val_dataset = Subset(full_trainset, val_indexes)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load the test set 
    testset = datasets.CIFAR100(root='./data', train=False, download=download, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


