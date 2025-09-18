import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import PIL


def load_cifar10(batch_size=256, shuffle_test=True):
    """載入CIFAR-10資料集，使用數據增強，保持[0,1]範圍"""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=(-5, 5),
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))# to [1,-1]
    ])

    # 測試集不使用數據增強
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=0, pin_memory=True)

    return train_loader, test_loader
