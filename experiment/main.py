#!/usr/bin/env python
# =============================================================
# run_fuzzy_detector_fgsm.py
# 2023 SSCI〈Fuzzy Detectors Against Adversarial Attacks〉
#   • 5-trapezoidal MF (Very-Low ~ Very-High) + centroid defuzz
#   • FGSM (ART)  ε∈[0.01,0.04]
# =============================================================
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

# ------------------ 全域參數 --------------------------------
EPOCHS_DET = 5           # 示範用；論文最終 200
EPOCHS_CLS = 2           # 輔助分類器
BATCH_SIZE = 128         # 論文用 32
LR_DET     = 3e-4
LR_CLS     = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {DEVICE}')

# ------------------ 1. 輔助 CNN 分類器 ----------------------
class SimpleCIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

def train_classifier():
    tfm = transforms.ToTensor()
    ds  = datasets.CIFAR10('./data', True, tfm, download=True)
    ld  = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2)

    model = SimpleCIFARCNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_CLS)
    model.train()
    for ep in range(1, EPOCHS_CLS+1):
        loss_sum = tot = 0
        for x, y in ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0); tot += y.size(0)
        print(f'[CLS] Epoch {ep}/{EPOCHS_CLS}  loss {(loss_sum/tot):.4f}')
    return model

print('Pre-training classifier for FGSM …')
cls_model = train_classifier()

# ------------------ 2. ART FGSM ----------------------------
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

art_classifier = PyTorchClassifier(
    model        = cls_model,
    loss         = nn.CrossEntropyLoss(),
    input_shape  = (3,32,32),
    nb_classes   = 10,
    clip_values  = (0.0,1.0),
    device_type  = 'gpu' if DEVICE.type=='cuda' else 'cpu'
)
fgsm_attack = FastGradientMethod(estimator=art_classifier)   # ε 動態設定

# ------------------ 3. 梯形版 FuzzyDetector ----------------
class FuzzyDetector(nn.Module):
    """
    將 Fuzzifier / Rules / Defuzzifier 全整合 (梯形 MF)
    1. Fuzzifier : 5 條梯形 MF (a,b,c,d)  *可學*
    2. Rules     : consequent [0,0,0.5,1,1]
    3. Defuzz    : centroid
    4. Loss      : 論文 (3) 之 λ₁/λ₂ 雙段
    """
    def __init__(self, λ1: float = 10., λ2: float = 2.):
        super().__init__()

        # Fuzzifier ─ 5 條梯形 MF 的初始節點 (對應 Fig.2)
        init_a = [  0., 10., 30., 60., 80.]
        init_b = [  0., 20., 40., 70., 90.]
        init_c = [ 10., 30., 60., 80., 90.]
        init_d = [ 20., 40., 70., 90.,100.]

        self.register_buffer('scale', torch.tensor(100.))            # lf*100

        self.a = nn.Parameter(torch.tensor(init_a))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.c = nn.Parameter(torch.tensor(init_c))
        self.d = nn.Parameter(torch.tensor(init_d))

        # Rules 的 consequent
        self.register_buffer('conseq', torch.tensor([0., 0., 0.5, 1., 1.]))

        self.λ1, self.λ2 = λ1, λ2

    @staticmethod
    def _trapezoid_mf(x, a, b, c, d, eps: float = 1e-6):
        """
        梯形 membership function (可 broadcasting)：
               1
        ┌───────┐
        a b   c d
        """
        left  = (x - a) / (b - a + eps)
        right = (d - x) / (d - c + eps)
        plateau = torch.ones_like(x)
        μ = torch.min(torch.min(left, right), plateau)
        return torch.clamp(μ, 0., 1.)

    def forward(self, lf: torch.Tensor, label: torch.Tensor):
        # 1. Fuzzifier
        x  = (lf * self.scale).unsqueeze(1)                           # (B,1)
        μ  = self._trapezoid_mf(x, self.a, self.b, self.c, self.d)    # (B,5)

        # 2+3. 規則推論 + Centroid 去模糊化
        p_hat = (μ * self.conseq).sum(1) / (μ.sum(1) + 1e-8)          # (B,)

        # 4. 雙段式 λ₁/λ₂ 損失
        bce   = F.binary_cross_entropy(p_hat, label.float())          # L_F
        wrong = torch.any(label != p_hat.round())
        loss  = self.λ1 * bce if wrong else bce / self.λ2
        return p_hat, loss

# ------------------ 4. Backbone + Detector -----------------
class AttackDetectorNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))  # 去掉 fc
        self.detector = FuzzyDetector()
    def forward(self, img_c, img_a, label):
        fc = self.backbone(img_c).flatten(1)
        fa = self.backbone(img_a).flatten(1)
        lf = F.mse_loss(fa, fc, reduction='none').mean(1)                # 特徵差
        return self.detector(lf, label)

# ------------------ 5. FGSM 資料包裝 ------------------------
class AttackWrapperFGSM(Dataset):
    """
    每筆資料 50% clean、50% FGSM，ε 隨機 0.01~0.04
    回傳：(clean_img, adv/clean_img, detect_label)
    """
    def __init__(self, base_ds, attack):
        self.base_ds, self.attack = base_ds, attack
    def __len__(self): return len(self.base_ds)
    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        if random.random() < 0.5:
            return img, img.clone(), torch.tensor(0)

        eps = random.uniform(0.01, 0.04)
        self.attack.set_params(eps=eps, eps_step=eps)
        x_adv = self.attack.generate(x=img.unsqueeze(0).numpy(),
                                     y=np.array([label]))
        adv = torch.from_numpy(x_adv[0])
        return img, adv, torch.tensor(1)

# ------------------ 6. 訓練 / 評估 -------------------------
def run_epoch(model, loader, optim=None):
    model.train() if optim else model.eval()
    tot = correct = loss_sum = 0
    with torch.set_grad_enabled(optim is not None):
        for img_c, img_a, y in loader:
            img_c, img_a, y = img_c.to(DEVICE), img_a.to(DEVICE), y.to(DEVICE)
            p_hat, loss = model(img_c, img_a, y)
            if optim:
                optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item()*y.size(0)
            tot      += y.size(0)
            correct  += (p_hat.round() == y).sum().item()
    return loss_sum/tot, correct/tot

# ------------------ 7. 主程式 ------------------------------
def main():
    tfm = transforms.ToTensor()
    train_base = datasets.CIFAR10('./data', True,  tfm, download=True)
    test_base  = datasets.CIFAR10('./data', False, tfm, download=True)

    ds_train = AttackWrapperFGSM(train_base, fgsm_attack)
    ds_test  = AttackWrapperFGSM(test_base,  fgsm_attack)

    ld_tr = DataLoader(ds_train, batch_size=BATCH_SIZE,
                       shuffle=True,  num_workers=2, pin_memory=True)
    ld_te = DataLoader(ds_test,  batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=2, pin_memory=True)

    model = AttackDetectorNet().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR_DET)

    for ep in range(1, EPOCHS_DET+1):
        tr_loss, tr_acc = run_epoch(model, ld_tr, optim)
        te_loss, te_acc = run_epoch(model, ld_te)
        print(f'Epoch {ep:02d} | Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% '
              f'| Test loss {te_loss:.4f} acc {te_acc*100:.2f}%')

if __name__ == '__main__':
    main()
