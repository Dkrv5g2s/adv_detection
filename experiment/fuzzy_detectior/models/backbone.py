# models/backbone.py
import torch
import timm
from torch import nn

class EfficientNetFeat(nn.Module):
    """
    把 EfficientNetV2-XL 改成特徵抽取器，輸出 feature map (B, C, H, W)
    """
    def __init__(self, model_name='tf_efficientnetv2_l_in21k', pretrained=True):
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        # 最後一個 stage -> usually stride 32，feature dim = 640
        self.out_channels = self.net.feature_info[-1]['num_chs']

    def forward(self, x):
        feats = self.net(x)[-1]          # 取最後一層特徵
        return feats                     # (B, C, H, W)


# class EfficientNetFeat(nn.Module):
#     def __init__(self, model_name='efficientnetv2_xl', pretrained=True, train_backbone=False):
#         super().__init__()
#         self.net = timm.create_model(model_name, pretrained=pretrained, features_only=True)
#         self.out_channels = self.net.feature_info[-1]['num_chs']
#         if not train_backbone:          # <─ 新增
#             for p in self.parameters():
#                 p.requires_grad_(False)
#         self.train_backbone = train_backbone
#
#     def forward(self, x):
#         if self.train_backbone:
#             feats = self.net(x)[-1]
#         else:                           # <─ 推論模式 + no_grad
#             self.net.eval()
#             with torch.no_grad():
#                 feats = self.net(x)[-1]
#         return feats
