import torch
from typing import Tuple
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_cifar10_data(
        batch_size: int = 32,
        transform: transforms = transforms.ToTensor(),
        val_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader,DataLoader]:

    # Download and load training data
    train_val_data = datasets.CIFAR10(
        root='data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Split into training and validation
    val_size = int(val_ratio * len(train_val_data))
    train_size = len(train_val_data) - val_size
    train_data, val_data = random_split(train_val_data, [train_size, val_size])
    
    # Download and load test data
    test_data = datasets.CIFAR10(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True
    )

    return train_loader,val_loader, test_loader