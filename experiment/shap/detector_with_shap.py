import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import shap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# Step 1: Load MNIST Dataset
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = load_data()

# Step 2: Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward_features(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return x  # last hidden layer

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train CNN Classifier
def train_classifier(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

train_classifier(model, train_loader, criterion, optimizer, device)

# Step 4: Generate Adversarial Samples
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

def generate_adversarial_samples(classifier, test_loader):
    attack = FastGradientMethod(estimator=classifier, eps=0.1)
    adversarial_samples = []
    normal_samples = []
    for images, labels in test_loader:
        images = images.to(device)
        adversarial_images = attack.generate(x=images.cpu().numpy())
        adversarial_samples.append((adversarial_images, labels.numpy()))
        normal_samples.append((images.cpu().numpy(), labels.numpy()))
    return normal_samples, adversarial_samples

normal_samples, adversarial_samples = generate_adversarial_samples(classifier, test_loader)

# Step 5: Generate SHAP Signatures
class FeatureModel(nn.Module):
    def __init__(self, base_model):
        super(FeatureModel, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model.forward_features(x)

feature_model = FeatureModel(model).to(device)
feature_model.eval()

def generate_shap_signatures(feature_model, samples, num_classes=10):
    feature_model.eval()
    background = torch.tensor(samples[0][0][:500], dtype=torch.float32).to(device)  # 使用 500 筆背景樣本
    explainer = shap.DeepExplainer(feature_model, background)

    shap_signatures = []

    for images, labels in samples:
        images_tensor = torch.tensor(images, dtype=torch.float32).to(device)
        shap_list = explainer.shap_values(images_tensor, check_additivity=False)
        print(shap_list.shap)
        concatenated = np.concatenate(shap_list, axis=1)  # shape: (batch, num_classes * feature_size)
        print(shap_list.shap)
        shap_signatures.append((concatenated, labels))
    return shap_signatures

normal_shap_signatures = generate_shap_signatures(feature_model, normal_samples)
adversarial_shap_signatures = generate_shap_signatures(feature_model, adversarial_samples)

# Step 6: Define and Train Detector Model
class DetectorModel(nn.Module):
    def __init__(self):
        super(DetectorModel, self).__init__()
        self.fc1 = nn.Linear(1280, 256)  # 10 * 128
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
    detector.train()
    for epoch in range(epochs):
        total_loss = 0
        for normal, adversarial in zip(normal_signatures, adversarial_signatures):
            normal_features, normal_labels = normal
            adversarial_features, adversarial_labels = adversarial

            features = torch.tensor(
                np.concatenate([normal_features, adversarial_features], axis=0),
                dtype=torch.float32
            ).to(device)
            labels = torch.cat([
                torch.zeros(len(normal_labels)),
                torch.ones(len(adversarial_labels))
            ], dim=0).to(device)

            detector_optimizer.zero_grad()
            outputs = detector(features).squeeze()
            loss = detector_criterion(outputs, labels)
            loss.backward()
            detector_optimizer.step()
            total_loss += loss.item()
        print(f"Detector Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

train_detector(detector, normal_shap_signatures, adversarial_shap_signatures)