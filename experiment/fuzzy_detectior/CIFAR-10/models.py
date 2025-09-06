import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 第一個卷積塊 - 增加深度和特徵圖數量
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x16x16

        # 第二個卷積塊
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128x8x8

        # 第三個卷積塊
        self.conv7 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 256x4x4

        # 第四個卷積塊
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)  # 512x2x2

        # 全連接層
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 第一個卷積塊
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)

        # 第二個卷積塊
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool2(x)

        # 第三個卷積塊
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.pool3(x)

        # 第四個卷積塊
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = self.pool4(x)

        # 全連接層
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        logits = self.fc3(x)
        return logits