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

    def __init__(self, detector, classifier_model, device, lr=0.001, weight_decay=1e-4, feature_batch_size=100):
        """
        Args:
            detector: AdversarialDetectorMLP 模型
            classifier_model: 已訓練的分類模型（用於提取 log-softmax）
            device: 計算設備
            lr: 學習率
            weight_decay: L2 正則化係數
            feature_batch_size: 用於特徵提取的批次大小
        """
        self.detector = detector
        self.classifier_model = classifier_model
        self.device = device
        self.feature_batch_size = feature_batch_size

        self.optimizer = torch.optim.Adam(
            detector.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=10,
            factor=0.5
        )

        # 確保分類模型在評估模式
        self.classifier_model.eval()

    def extract_log_softmax_features_batch_avg(self, images):
        """
        從圖像中提取 log-softmax 特徵（批次平均 + 排序）

        論文方法：
        1. 提取每個樣本的 log-softmax 值
        2. 對每個樣本的值進行排序（從小到大）
        3. 對所有樣本的「第 i 小」值求平均

        Args:
            images: numpy array (N, 3, 32, 32) 或 torch tensor

        Returns:
            log_softmax_features: torch tensor (N//batch_size, num_classes)
                每個特徵是 batch_size 個樣本排序後的平均
        """
        # 轉換為 tensor
        if isinstance(images, np.ndarray):
            images = torch.FloatTensor(images)

        images = images.to(self.device)
        num_samples = len(images)

        # 提取特徵
        self.classifier_model.eval()
        batch_features_list = []

        with torch.no_grad():
            # 按照 batch_size 分批處理
            for i in range(0, num_samples, self.feature_batch_size):
                batch_images = images[i:i + self.feature_batch_size]

                # 如果最後一批不足 batch_size，跳過
                if len(batch_images) < self.feature_batch_size:
                    continue

                # 提取 log-softmax
                logits = self.classifier_model(batch_images)
                log_softmax = - F.log_softmax(logits, dim=1)  # (batch_size, num_classes)

                # 對每個樣本的 log-softmax 值進行排序
                log_softmax_sorted, _ = torch.sort(log_softmax, dim=1)  # (batch_size, num_classes)
                # 排序後：每一行都是從小到大排列

                # 計算該批次的平均值（對每個位置求平均）
                batch_avg = log_softmax_sorted.mean(dim=0, keepdim=True)  # (1, num_classes)
                # batch_avg[0] = 所有樣本「最小值」的平均
                # batch_avg[1] = 所有樣本「第2小值」的平均
                # ...
                # batch_avg[K-1] = 所有樣本「最大值」的平均

                batch_features_list.append(batch_avg)

        # 合併所有批次的平均特徵
        if len(batch_features_list) > 0:
            batch_features = torch.cat(batch_features_list, dim=0)  # (num_batches, num_classes)
        else:
            batch_features = torch.empty(0, logits.shape[1]).to(self.device)

        return batch_features

    def prepare_training_data(self, adversarial_data):
        """
        準備訓練數據（使用批次平均）

        Args:
            adversarial_data: dict，格式為
                {
                    'Clean': {'images': np.array, 'labels': np.array},
                    'PGD': {'images': np.array, 'labels': np.array},
                    ...
                }

        Returns:
            X: Log-softmax 批次平均特徵 (N_batches, num_classes)
            y: 攻擊類型標籤 (N_batches,)
        """
        print(f"\n[INFO] Preparing training data (Batch size: {self.feature_batch_size})...")

        # 攻擊類型到標籤的映射
        attack_to_label = {
            'Clean': 0,
            'PGD-Linf': 1,
            'PGD-L2': 2,
            'APGD-Linf': 3,
            'APGDT-Linf': 4,
            'Square-Linf': 5,
            'FAB-Linf': 6,
            'CW-L2': 7
        }

        X_list = []
        y_list = []

        for attack_name, data_info in adversarial_data.items():
            print(f"  Processing {attack_name}...")

            images = data_info['images']
            num_samples = len(images)

            # 提取批次平均特徵
            batch_features = self.extract_log_softmax_features_batch_avg(images)
            num_batches = len(batch_features)

            # 創建標籤（每個批次一個標籤）
            attack_label = attack_to_label[attack_name]
            labels = torch.full((num_batches,), attack_label, dtype=torch.long)

            X_list.append(batch_features.cpu())
            y_list.append(labels)

            print(f"    ✓ {attack_name}: {num_samples} samples → {num_batches} batches (avg), label={attack_label}")

        # 合併所有數據
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        print(f"\n[INFO] Total training data: {X.shape[0]} batch averages")
        print(f"  Feature shape: {X.shape}")
        print(f"  Label shape: {y.shape}")
        print(f"  Label distribution:")
        for label in range(8):
            count = (y == label).sum().item()
            percentage = count / len(y) * 100
            print(f"    Label {label}: {count:5d} ({percentage:5.2f}%)")

        return X, y

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        訓練檢測器（加入 Early Stopping 和梯度裁剪）

        Args:
            X_train: 訓練特徵 (N_train, num_classes)
            y_train: 訓練標籤 (N_train,)
            X_val: 驗證特徵 (N_val, num_classes)
            y_val: 驗證標籤 (N_val,)
            epochs: 訓練輪數
            batch_size: 訓練批次大小

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
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50 # Early stopping patience

        print(f"\n[INFO] Training detector for {epochs} epochs...")
        print(f"[INFO] Batch size: {batch_size}, Learning rate: {self.optimizer.param_groups[0]['lr']}")

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

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.detector.parameters(), max_norm=1.0)

                self.optimizer.step()

                # 統計
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # 驗證階段
            val_loss, val_acc = self.evaluate_with_loss(val_loader)

            # 學習率調度
            self.scheduler.step(val_loss)

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0

                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.detector.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, 'best_adversarial_detector.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[INFO] Early stopping triggered at epoch {epoch + 1}")
                    break

            # 打印進度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch + 1:3d}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Train Acc: {train_acc:6.2f}% | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:6.2f}% | "
                      f"Best: {best_val_acc:6.2f}%")

        print(f"\n[RESULT] Training completed!")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")

        # 載入最佳模型
        checkpoint = torch.load('best_adversarial_detector.pth', weights_only=True)
        self.detector.load_state_dict(checkpoint['model_state_dict'])

        return best_val_acc

    def evaluate_with_loss(self, dataloader):
        """
        評估檢測器（返回 loss 和 accuracy）

        Args:
            dataloader: DataLoader

        Returns:
            loss: 平均損失
            accuracy: 準確率
        """
        self.detector.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.detector(batch_X)
                loss = self.criterion(outputs, batch_y)

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        avg_loss = running_loss / len(dataloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

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
