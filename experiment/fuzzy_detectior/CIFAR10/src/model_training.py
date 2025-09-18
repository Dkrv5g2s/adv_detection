import torch
import torch.nn as nn
from tqdm import tqdm


def train_classifier(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    """訓練分類器"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,
                                                           min_lr=0)

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # compute training statistics
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y).sum().item()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader.dataset)
        avg_acc = correct / len(train_loader.dataset)

        scheduler.step(avg_loss)
        acc = eval_classifier(model, test_loader, device)
        print(f"[Epoch {ep + 1}] loss={avg_loss:.5f} accuracy={avg_acc:.4f} test_acc={acc:.4f}")

    return model


def eval_classifier(model, data_loader, device):
    """評估分類器"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
