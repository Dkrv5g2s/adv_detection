import torch
import torch.nn as nn
import torch.nn.functional as F


# class CIFAR10CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#
#         # 第一個卷積塊 - 增加更多特徵
#         self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)  # 32->64
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(2, 2)
#
#         # 第二個卷積塊
#         self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)  # 更早增加通道數
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.pool2 = nn.MaxPool2d(2, 2)
#
#         # 第三個卷積塊
#         self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)  # 增加到256
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.pool3 = nn.MaxPool2d(2, 2)
#
#         # 第四個卷積塊 - 新增
#         self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.bn8 = nn.BatchNorm2d(512)
#
#         # 全局平均池化 - 替代最後的MaxPool
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # 全連接層
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(512, 256)
#         self.dropout2 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(256, num_classes)
#
#     def forward(self, x):
#         # 第一個卷積塊
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool1(x)
#
#         # 第二個卷積塊
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool2(x)
#
#         # 第三個卷積塊
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = F.relu(self.bn6(self.conv6(x)))
#         x = self.pool3(x)
#
#         # 第四個卷積塊
#         x = F.relu(self.bn7(self.conv7(x)))
#         x = F.relu(self.bn8(self.conv8(x)))
#
#         # 全局平均池化
#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, 1)
#
#         # 全連接層
#         x = self.dropout1(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

class CIFAR10CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
    self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.bn_conv1 = nn.BatchNorm2d(128)
    self.bn_conv2 = nn.BatchNorm2d(128)
    self.bn_conv3 = nn.BatchNorm2d(256)
    self.bn_conv4 = nn.BatchNorm2d(256)
    self.bn_dense1 = nn.BatchNorm1d(1024)
    self.bn_dense2 = nn.BatchNorm1d(512)
    self.dropout_conv = nn.Dropout2d(p=0.25)
    self.dropout = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear(256 * 8 * 8, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 10)

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
    out = F.relu(self.bn_dense1(self.fc1(x)))
    out = self.dropout(out)
    out = F.relu(self.bn_dense2(self.fc2(out)))
    out = self.dropout(out)
    out = self.fc3(out)
    return out

  def forward(self, x):
    out = self.conv_layers(x)
    out = out.view(-1, 256 * 8 * 8)
    out = self.dense_layers(out)
    return out