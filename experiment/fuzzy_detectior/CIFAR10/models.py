import torch
import torch.nn as nn
import torch.nn.functional as F


class testCIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷積層保持不變
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_conv1 = nn.BatchNorm2d(128)
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.bn_conv3 = nn.BatchNorm2d(256)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.5)

        # 全連接層：1024 -> 512 -> 256 -> 64 -> 5 -> 10
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # 16384 -> 1024
        self.fc2 = nn.Linear(1024, 512)  # 1024 -> 512
        self.fc3 = nn.Linear(512, 256)  # 512 -> 256
        self.fc4 = nn.Linear(256, 64)  # 256 -> 64
        self.fc5 = nn.Linear(64, 5)  # 64 -> 5 (瓶頸層)
        self.fc6 = nn.Linear(5, 10)  # 5 -> 10 (最終輸出)

        # 批次正規化層
        self.bn_dense1 = nn.BatchNorm1d(1024)
        self.bn_dense2 = nn.BatchNorm1d(512)
        self.bn_dense3 = nn.BatchNorm1d(256)
        self.bn_dense4 = nn.BatchNorm1d(64)
        self.bn_dense5 = nn.BatchNorm1d(5)

    def conv_layers(self, x):
        out = F.relu(self.bn_conv1(self.conv1(x)))
        out = F.relu(self.bn_conv2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        out = F.relu(self.bn_conv3(self.conv3(out)))
        out = F.relu(self.bn_conv4(self.conv4(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        return out

    def dense_layers(self, x):
        # 1024 -> 512
        out = F.relu(self.bn_dense1(self.fc1(x)))
        out = self.dropout(out)

        # 512 -> 256
        out = F.relu(self.bn_dense2(self.fc2(out)))
        out = self.dropout(out)

        # 256 -> 64
        out = F.relu(self.bn_dense3(self.fc3(out)))
        out = self.dropout(out)

        # 64 -> 5 (瓶頸層)
        out = F.relu(self.bn_dense4(self.fc4(out)))
        out = self.dropout(out)

        # 5 -> 10 (最終輸出)
        out = F.relu(self.bn_dense5(self.fc5(out)))
        final_output = self.fc6(out)

        return final_output

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 256 * 8 * 8)
        logits = self.dense_layers(out)
        return logits


