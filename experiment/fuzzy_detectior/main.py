# main.py
import torch, random, numpy as np
from torch import nn, optim
from tqdm import tqdm
from multiprocessing import freeze_support

from datasets import get_loaders
from models.backbone import EfficientNetFeat
from models.fuzzy_net import FuzzyDetector
from attacks import get_art_classifier, fgsm_attack, pgd_attack
from utils import detection_rate, save_ckpt

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_batch_attack(img, labels, art_clf, method='fgsm'):
    if method=='fgsm':
        return fgsm_attack(art_clf, img, labels, eps=0.03).clamp(0,1)
    else:
        return pgd_attack(art_clf, img, labels, eps=0.03).clamp(0,1)

def train(detector, art_clf, loader, optimizer, criterion, epoch):
    detector.train()
    pbar = tqdm(loader, desc=f'Train {epoch}')
    λ1, λ2 = 1.0, 0.5
    for img, label in pbar:
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        img_adv = make_batch_attack(img, label, art_clf, 'fgsm')

        y_clean = torch.zeros(img.size(0), device=DEVICE)
        y_adv   = torch.ones(img.size(0),  device=DEVICE)

        y_hat_adv,_   = detector(img, img_adv)
        y_hat_clean,_ = detector(img, img)

        loss = λ1*criterion(y_hat_adv, y_adv) + λ2*criterion(y_hat_clean, y_clean)

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

@torch.no_grad()
def evaluate(detector, art_clf, loader, desc='Val'):
    detector.eval()
    tp=tn=fp=fn=0
    for img, label in tqdm(loader, desc=desc):
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        img_adv = make_batch_attack(img, label, art_clf, 'pgd')
        y_hat_adv,_   = detector(img, img_adv)
        y_hat_clean,_ = detector(img, img)

        pred_adv   = (y_hat_adv  >0.5).long()
        pred_clean = (y_hat_clean>0.5).long()
        tp += (pred_adv==1).sum().item()
        fn += (pred_adv==0).sum().item()
        tn += (pred_clean==0).sum().item()
        fp += (pred_clean==1).sum().item()
    dr = detection_rate(tp, tn, fp, fn)
    print(f'Detection Rate: {dr:.2f}%')

def main():
    # 1. Data  (→ 如仍報錯，可把 num_workers=0)
    train_loader, test_loader = get_loaders(batch_size=8, num_workers=2)

    # 2. 模型
    backbone = EfficientNetFeat().to(DEVICE)
    detector = FuzzyDetector(backbone).to(DEVICE)

    # 3. ART classifier（一次建立，之後重用）
    art_clf = get_art_classifier(backbone, DEVICE)

    # 4. opt / loss
    optimizer = optim.SGD(detector.parameters(), lr=8e-4,
                          momentum=0.9, weight_decay=1e-4)
    criterion = nn.BCELoss()

    EPOCHS = 10
    for ep in range(1, EPOCHS+1):
        train(detector, art_clf, train_loader, optimizer, criterion, ep)
        evaluate(detector, art_clf, test_loader, desc=f'Test@{ep}')

    save_ckpt({'model':detector.state_dict()}, 'ckpt/fuzzy_detector.pth')

if __name__ == '__main__':
    # Windows 需先呼叫 freeze_support，再跑 main
    freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
