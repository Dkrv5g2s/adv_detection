"""
對抗樣本檢測器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialDetectorMLP(nn.Module):
    """
    基於 Log-Softmax 模式的多分類對抗樣本檢測器
    """

    def __init__(self, num_classes=10, num_attack_types=8, hidden_dim=128, dropout=0.3):
        super(AdversarialDetectorMLP, self).__init__()

        self.fc1 = nn.Linear(num_classes, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_attack_types)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
