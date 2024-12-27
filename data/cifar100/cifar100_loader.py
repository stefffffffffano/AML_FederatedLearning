import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_cifar100(batch_size=32, validation_split=0.1, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Carica l'intero set di training
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=download, transform=transform)

    # Creare indici per il dataset e dividere in training e validation
    indexes = list(range(len(full_trainset)))  
    train_indexes, val_indexes = train_test_split(indexes, train_size=1-validation_split, test_size=validation_split, random_state=42, stratify=full_trainset.targets, shuffle=True)

    # Creare subset per training e validation
    train_dataset = Subset(full_trainset, train_indexes)
    val_dataset = Subset(full_trainset, val_indexes)

    # Creare DataLoader per i subset di training e validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Caricare il dataset di test
    testset = datasets.CIFAR100(root='./data', train=False, download=download, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


