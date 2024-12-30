import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_cifar100(batch_size=64, validation_split=0.1, download=True):
    # Transformations for training with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(24, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Transformations for validation/test without data augmentation
    test_transform = transforms.Compose([
        transforms.CenterCrop(24),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Load training dataset with appropriate transformations
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=download, transform=train_transform)

    # Split indices for training and validation
    indexes = list(range(len(full_trainset)))
    train_indexes, val_indexes = train_test_split(indexes, train_size=1-validation_split, test_size=validation_split, random_state=42, stratify=full_trainset.targets,shuffle=True)

    # Create subsets for training and validation
    train_dataset = Subset(full_trainset, train_indexes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Reload the training dataset with transformations for validation/test
    full_trainset_val = datasets.CIFAR100(root='./data', train=True, download=download, transform=test_transform)
    val_dataset = Subset(full_trainset_val, val_indexes)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load and transform the test dataset
    testset = datasets.CIFAR100(root='./data', train=False, download=download, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
