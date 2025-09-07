import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(batch_size=256, shuffle_test=True):
    """載入MNIST資料集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  # MNIST標準化
    ])

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=0, pin_memory=True)

    return train_loader, test_loader
