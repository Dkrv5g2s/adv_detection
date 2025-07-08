import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(batch_size=32, num_workers=4):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize
    ])
    train_set = datasets.CIFAR10(root='./data', download=True,
                                 train=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root='./data', download=True,
                                 train=False, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
