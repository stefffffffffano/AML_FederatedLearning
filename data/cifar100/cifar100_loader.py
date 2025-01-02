import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class CIFAR100DataLoader:
    def __init__(self, batch_size=64, validation_split=0.1, download=True, num_workers=4, pin_memory=True):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.download = download
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        # Load datasets
        self.train_loader, self.val_loader, self.test_loader = self._prepare_loaders()

    def _prepare_loaders(self):
        # Load the full training dataset
        full_trainset = datasets.CIFAR100(root='./data', train=True, download=self.download, transform=self.train_transform)

        # Split indices for training and validation
        indexes = list(range(len(full_trainset)))
        train_indexes, val_indexes = train_test_split(
            indexes,
            train_size=1 - self.validation_split,
            test_size=self.validation_split,
            random_state=42,
            stratify=full_trainset.targets,
            shuffle=True
        )

        # Create training and validation subsets
        train_dataset = Subset(full_trainset, train_indexes)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        full_trainset_val = datasets.CIFAR100(root='./data', train=True, download=self.download, transform=self.test_transform)
        val_dataset = Subset(full_trainset_val, val_indexes)
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        # Load the test dataset
        testset = datasets.CIFAR100(root='./data', train=False, download=self.download, transform=self.test_transform)
        test_loader = DataLoader(
            testset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        return train_loader, val_loader, test_loader

    def __iter__(self):
        """Allows iteration over all loaders for unified access."""
        return iter([self.train_loader, self.val_loader, self.test_loader])