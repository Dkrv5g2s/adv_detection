# models/fuzzy_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Fuzzifier(nn.Module):
    """
    輸入兩張 feature maps -> 差值 -> 1x1 conv 產生 membership (low/ok/high)
    """
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.diff_cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1)        # 3 個模糊集合
        )

    def forward(self, Fc, Fa):
        diff = torch.abs(Fc - Fa)
        membership = self.diff_cnn(diff)     # (B, 3, H, W)
        membership = torch.mean(membership, dim=[2,3])  # GAP -> (B,3)
        μ = torch.sigmoid(membership)        # 對應論文 Fig.2 0~1
        return μ                             # (B,3)

class FuzzyRuleLayer(nn.Module):
    """
    模糊規則向量化：輸入 μ (B,3)，乘 learnable θ (3,) -> 分數
    """
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(3))

    def forward(self, μ):
        # 將模糊集合權重化並 softmax
        score = torch.sum(μ * self.theta, dim=1, keepdim=True)  # (B,1)
        return score

class Defuzzifier(nn.Module):
    """
    把 score -> 機率 y_hat，採用 Sigmoid
    """
    def forward(self, score):
        return torch.sigmoid(score)          # (B,1)

class FuzzyDetector(nn.Module):
    """
    backbone + fuzzifier + rule + defuzz
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fuzzifier = Fuzzifier(backbone.out_channels)
        self.rule      = FuzzyRuleLayer()
        self.defuzz    = Defuzzifier()

    def forward(self, x_clean, x_adv):
        Fc = self.backbone(x_clean)          # (B,C,H,W)
        Fa = self.backbone(x_adv)
        μ  = self.fuzzifier(Fc, Fa)          # (B,3)
        score = self.rule(μ)                 # (B,1)
        y_hat = self.defuzz(score)           # (B,1)  attacked prob
        return y_hat.squeeze(1), μ           # 回傳附加 μ 方便觀察
