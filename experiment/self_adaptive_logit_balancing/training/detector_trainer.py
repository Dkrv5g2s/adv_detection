"""
檢測器訓練器
training/detector_trainer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class DetectorTrainer:
    """訓練對抗樣本檢測器"""

    def __init__(self, detector, classifier_model, device, lr=0.001):
        """
        Args:
            detector: AdversarialDetectorMLP 模型
            classifier_model: 已訓練的分類模型（用於提取 log-softmax）
            device: 計算設備
            lr: 學習率
        """
        self.detector = detector
        self.classifier_model = classifier_model
        self.device = device

        self.optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # 確保分類模型在評估模式
        self.classifier_model.eval()

    def extract_log_softmax_features(self, images):
        """
        從圖像中提取 log-softmax 特徵

        Args:
            images: numpy array (N, 3, 32, 32) 或 torch tensor

        Returns:
            log_softmax_features: torch tensor (N, num_classes)
        """
        # 轉換為 tensor
        if isinstance(images, np.ndarray):
            images = torch.FloatTensor(images)

        images = images.to(self.device)

        # 提取特徵
        self.classifier_model.eval()
        with torch.no_grad():
            logits = self.classifier_model(images)
            log_softmax_features = F.log_softmax(logits, dim=1)

        return log_softmax_features

    def prepare_training_data(self, adversarial_data):
        """
        準備訓練數據

        Args:
            adversarial_data: dict，格式為
                {
                    'Clean': {'images': np.array, 'labels': np.array},
                    'PGD': {'images': np.array, 'labels': np.array},
                    ...
                }

        Returns:
            X: Log-softmax 特徵 (N, num_classes)
            y: 攻擊類型標籤 (N,)
        """
        print("\n[INFO] Preparing training data...")

        # 攻擊類型到標籤的映射
        attack_to_label = {
            'Clean': 0,
            'PGD': 1,
            'PGD-L2': 2,
            'APGD': 3,
            'APGDT': 4,
            'Square': 5,
            'FAB': 6,
            'CW': 7
        }

        X_list = []
        y_list = []

        for attack_name, data_info in adversarial_data.items():
            print(f"  Processing {attack_name}...")

            images = data_info['images']
            num_samples = len(images)

            # 提取 log-softmax 特徵（分批處理以節省記憶體）
            batch_size = 100
            features_list = []

            for i in range(0, num_samples, batch_size):
                batch_images = images[i:i+batch_size]
                batch_features = self.extract_log_softmax_features(batch_images)
                features_list.append(batch_features.cpu())

            features = torch.cat(features_list, dim=0)

            # 創建標籤
            attack_label = attack_to_label[attack_name]
            labels = torch.full((num_samples,), attack_label, dtype=torch.long)

            X_list.append(features)
            y_list.append(labels)

            print(f"    ✓ {attack_name}: {features.shape[0]} samples, label={attack_label}")

        # 合併所有數據
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        print(f"\n[INFO] Total training data: {X.shape[0]} samples")
        print(f"  Feature shape: {X.shape}")
        print(f"  Label shape: {y.shape}")
        print(f"  Label distribution:")
        for label in range(8):
            count = (y == label).sum().item()
            percentage = count / len(y) * 100
            print(f"    Label {label}: {count:5d} ({percentage:5.2f}%)")

        return X, y

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """
        訓練檢測器

        Args:
            X_train: 訓練特徵 (N_train, num_classes)
            y_train: 訓練標籤 (N_train,)
            X_val: 驗證特徵 (N_val, num_classes)
            y_val: 驗證標籤 (N_val,)
            epochs: 訓練輪數
            batch_size: 批次大小

        Returns:
            best_val_acc: 最佳驗證準確率
        """
        # 創建 DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        best_val_acc = 0.0

        print(f"\n[INFO] Training detector for {epochs} epochs...")

        for epoch in range(epochs):
            # 訓練階段
            self.detector.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向傳播
                self.optimizer.zero_grad()
                outputs = self.detector(batch_X)
                loss = self.criterion(outputs, batch_y)

                # 反向傳播
                loss.backward()
                self.optimizer.step()

                # 統計
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # 驗證階段
            val_acc = self.evaluate(val_loader)

            # 更新最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # 打印進度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1:3d}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Train Acc: {train_acc:6.2f}% | "
                      f"Val Acc: {val_acc:6.2f}% | "
                      f"Best: {best_val_acc:6.2f}%")

        print(f"\n[RESULT] Training completed!")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")

        return best_val_acc

    def evaluate(self, dataloader):
        """
        評估檢測器

        Args:
            dataloader: DataLoader

        Returns:
            accuracy: 準確率
        """
        self.detector.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.detector(batch_X)
                _, predicted = outputs.max(1)

                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        accuracy = 100. * correct / total
        return accuracy

    def predict(self, X):
        """
        預測攻擊類型

        Args:
            X: 特徵 (N, num_classes)

        Returns:
            predictions: 預測標籤 (N,)
        """
        self.detector.eval()

        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.detector(X)
            _, predictions = outputs.max(1)

        return predictions.cpu().numpy()
