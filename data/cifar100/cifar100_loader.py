import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_cifar100(batch_size=32, validation_split=0.1, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the full training set
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=download, transform=transform)

    indexes = range(0, len(full_trainset))
    splitting = train_test_split(indexes, train_size = 1-validation_split, random_state = 42, stratify = full_trainset.data["label"], shuffle = True)
    train_indexes = splitting[0]
    val_indexes = splitting[1]

    train_dataset = Subset(full_trainset, train_indexes)
    val_dataset = Subset(full_trainset, val_indexes)

    # Load the test dataset
    testset = datasets.CIFAR100(root='./data', train=False, download=download, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, testloader
