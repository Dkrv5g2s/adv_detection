import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from datetime import datetime

# 設定 MLflow 追蹤 URI
mlflow.set_tracking_uri("http://localhost:5000")

# 設定實驗名稱
mlflow.set_experiment("Training_CNN_MNIST")

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置: {device}")

# 超參數設定
learning_rate = 0.0005
batch_size = 64
num_epochs = 10

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 載入 MNIST 資料集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# 分割訓練集為訓練和驗證集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 資料載入器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定義 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 訓練函數
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 驗證函數
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 測試函數
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    return accuracy

# 開始 MLflow 實驗
with mlflow.start_run(run_name=f"CNN_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list, train_accuracy_list = [], []
    val_loss_list, val_accuracy_list = [], []
    epochs = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        epochs.append(epoch + 1)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

    test_accuracy = test_model(model, test_loader, device)
    print(f"最終測試準確率: {test_accuracy:.2f}%")

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.pytorch.log_model(model,
                             artifact_path="model",
                             registered_model_name="CNN_MNIST_Model")

    # === Artifacts ===
    # 1. 訓練曲線圖
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss_list, 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_list, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accuracy_list, 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plot_path = "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(plot_path)

    # 2. 模型結構檔案
    model_structure_path = "model_structure.txt"
    with open(model_structure_path, "w", encoding='utf-8') as f:
        f.write("=== CNN Model Structure ===\n\n")
        f.write(str(model))
        f.write("\n\n=== Model Parameters ===\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")

    mlflow.log_artifact(model_structure_path)

    # 3. 結果摘要
    results_file = "final_results.txt"
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("=== Training Results Summary ===\n")
        f.write(f"Run ID: {run.info.run_id}\n")
        f.write(f"Experiment: CNN MNIST Classification\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n\n")

        f.write("=== Final Performance ===\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Best Validation Accuracy: {max(val_accuracy_list):.2f}%\n")
        f.write(f"Final Train Accuracy: {train_accuracy_list[-1]:.2f}%\n")
        f.write(f"Final Train Loss: {train_loss_list[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {val_loss_list[-1]:.4f}\n\n")

        f.write("=== Training Configuration ===\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Loss Function: CrossEntropyLoss\n")

    mlflow.log_artifact(results_file)

print("訓練完成！請在 MLflow UI 中查看結果。")
