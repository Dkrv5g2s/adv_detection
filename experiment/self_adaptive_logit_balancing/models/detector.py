"""
對抗樣本檢測器
models/detector.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialDetectorMLP(nn.Module):
    """
    基於 Log-Softmax 模式的多分類對抗樣本檢測器

    輸入: WideResNet 的 log-softmax 輸出 (batch_size, num_classes)
    輸出: 攻擊類型預測 (batch_size, num_attack_types)

    攻擊類型標籤:
        0: Clean
        1: PGD-Linf
        2: PGD-L2
        3: APGD-Linf
        4: APGDT-Linf
        5: Square
        6: FAB-Linf
        7: CW-L2
    """

    def __init__(self, num_classes=10, num_attack_types=8, hidden_dim=128, dropout=0.3):
        """
        Args:
            num_classes: 分類模型的類別數（CIFAR-10 = 10）
            num_attack_types: 攻擊類型數量（包含 Clean）
            hidden_dim: 隱藏層維度
            dropout: Dropout 比例
        """
        super(AdversarialDetectorMLP, self).__init__()

        self.num_classes = num_classes
        self.num_attack_types = num_attack_types

        # 多層感知器架構
        self.fc1 = nn.Linear(num_classes, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, num_attack_types)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向傳播

        Args:
            x: Log-softmax 值 (batch_size, num_classes)

        Returns:
            logits: 攻擊類型的 logits (batch_size, num_attack_types)
        """
        # 確保輸入維度正確
        assert x.size(1) == self.num_classes, \
            f"Expected input size (*, {self.num_classes}), got {x.shape}"

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc4(x)

        return x

    def extract_features(self, classifier_model, images):
        """
        從分類模型中提取 log-softmax 特徵

        Args:
            classifier_model: 已訓練的分類模型（如 WideResNet）
            images: 輸入圖像 (batch_size, 3, 32, 32)

        Returns:
            log_softmax_values: Log-softmax 值 (batch_size, num_classes)
        """
        classifier_model.eval()
        with torch.no_grad():
            logits = classifier_model(images)
            log_softmax_values = F.log_softmax(logits, dim=1)
        return log_softmax_values

    def predict_attack_type(self, classifier_model, images):
        """
        預測攻擊類型（端到端）

        Args:
            classifier_model: 已訓練的分類模型
            images: 輸入圖像 (batch_size, 3, 32, 32)

        Returns:
            predictions: 預測的攻擊類型 (batch_size,)
            probabilities: 各類別的概率 (batch_size, num_attack_types)
        """
        # 提取特徵
        log_softmax_values = self.extract_features(classifier_model, images)

        # 預測
        self.eval()
        with torch.no_grad():
            logits = self.forward(log_softmax_values)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        return predictions, probabilities
