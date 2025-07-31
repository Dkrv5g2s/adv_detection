import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import shap

# 自動偵測裝置：GPU (cuda) or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# Step 1: Load CIFAR-10 dataset
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = load_data()

# Step 2: Define target classifier
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

# Step 3: Train the classifier
def train_classifier(classifier, train_loader, epochs=10):
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            classifier.fit(images.cpu().numpy(), labels.cpu().numpy(), batch_size=128, nb_epochs=1)

train_classifier(classifier, train_loader)

# Step 4: Generate adversarial examples
def generate_adversarial_samples(classifier, test_loader):
    attack = FastGradientMethod(estimator=classifier, eps=0.1)
    adversarial_samples = []
    normal_samples = []
    for images, labels in test_loader:
        adversarial_images = attack.generate(x=images.cpu().numpy())
        adversarial_samples.append((adversarial_images, labels.numpy()))
        normal_samples.append((images.numpy(), labels.numpy()))
    return normal_samples, adversarial_samples

normal_samples, adversarial_samples = generate_adversarial_samples(classifier, test_loader)

# Step 5: Generate SHAP values
def generate_shap_signatures(classifier, samples):
    background = torch.tensor(samples[0][0][:50], dtype=torch.float32).to(device)

    # 使用 PyTorch 模型原始輸出
    model.eval()  # 確保 dropout, batchnorm 都停用
    explainer = shap.DeepExplainer(model, background)

    shap_values = []
    for images, labels in samples:
        images_tensor = torch.tensor(images, dtype=torch.float32).to(device)
        # 關鍵修正：關閉 additivity 檢查
        shap_batch = explainer.shap_values(images_tensor, check_additivity=False)
        shap_batch = np.array(shap_batch).mean(axis=0)
        flat_features = shap_batch.reshape(len(images), -1)
        shap_values.append((flat_features, labels))
    return shap_values

normal_shap_signatures = generate_shap_signatures(classifier, normal_samples)
adversarial_shap_signatures = generate_shap_signatures(classifier, adversarial_samples)

# Step 6: Train detector model
class DetectorModel(nn.Module):
    def __init__(self):
        super(DetectorModel, self).__init__()
        self.fc1 = nn.Linear(640, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

detector = DetectorModel().to(device)
detector_criterion = nn.BCELoss()
detector_optimizer = optim.Adam(detector.parameters(), lr=0.001)

def train_detector(detector, normal_signatures, adversarial_signatures, epochs=10):
    for epoch in range(epochs):
        for normal, adversarial in zip(normal_signatures, adversarial_signatures):
            normal_features, normal_labels = normal
            adversarial_features, adversarial_labels = adversarial

            features = torch.tensor(np.concatenate([normal_features, adversarial_features], axis=0), dtype=torch.float32).to(device)
            labels = torch.cat([
                torch.zeros(len(normal_labels)),
                torch.ones(len(adversarial_labels))
            ], dim=0).to(device)

            detector_optimizer.zero_grad()
            outputs = detector(features).squeeze()
            loss = detector_criterion(outputs, labels)
            loss.backward()
            detector_optimizer.step()

train_detector(detector, normal_shap_signatures, adversarial_shap_signatures)
