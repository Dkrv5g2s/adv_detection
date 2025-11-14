import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchattacks import PGD, AutoAttack, APGDT
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 使用非互動式後端


# ==================== PreActResNet-18 架構 ====================
class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(PreActResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ==================== Energy Functions ====================
def compute_energy(model, x, y=None):
    """計算能量函數 E(x,y) = -log p(y|x)"""
    logits = model(x)
    if y is None:
        energy_x = -torch.logsumexp(logits, dim=1)
        return energy_x
    else:
        energy_xy = -logits.gather(1, y.view(-1, 1)).squeeze()
        energy_x = -torch.logsumexp(logits, dim=1)
        return energy_x, energy_xy


def compute_delta_energy(model, x, x_adv, y):
    """計算能量差異 ΔE"""
    energy_x, energy_xy = compute_energy(model, x, y)
    energy_x_adv, energy_xy_adv = compute_energy(model, x_adv, y)
    delta_ex = energy_x - energy_x_adv
    delta_exy = energy_xy - energy_xy_adv
    delta_energy = torch.sqrt(delta_ex ** 2 + delta_exy ** 2)
    return delta_energy, delta_ex, delta_exy


def der_loss(model, x, x_adv, y, gamma=0.0):
    """
    DER regularizer: eq. (7) in paper
    DER(x, x_adv, y) = max(0, ΔE(x, x_adv, y) - γ)
    """
    delta_energy, _, _ = compute_delta_energy(model, x, x_adv, y)
    der = torch.clamp(delta_energy - gamma, min=0.0)
    return der.mean()


# ==================== Logit Balancing Loss ====================
def logit_balancing_loss(outputs, targets, beta_lb=0.02):
    """
    Logit Balancing Loss

    針對不同樣本：
    - 預測正確: L_LB = β × SD(log(S(f(x)))_{i≠t}) × s_t
    - 預測錯誤: L_CE = -log(S(f(x))_t)

    Args:
        outputs: 模型輸出 logits [batch_size, num_classes]
        targets: 目標標籤 [batch_size]
        beta_lb: Logit Balancing 權重

    Returns:
        loss: 批次平均損失
        correct_mask: 預測正確的樣本 mask
    """
    batch_size = outputs.size(0)
    num_classes = outputs.size(1)

    # 計算 log-softmax 和 softmax
    log_softmax = F.log_softmax(outputs, dim=1)
    softmax = F.softmax(outputs, dim=1)

    # 判斷預測是否正確
    _, predicted = outputs.max(1)
    correct_mask = predicted.eq(targets)

    # 初始化損失
    loss = torch.zeros(batch_size, device=outputs.device)

    # ========== 預測正確：Logit Balancing Loss ==========
    if correct_mask.any():
        correct_indices = correct_mask.nonzero(as_tuple=True)[0]

        for idx in correct_indices:
            target_class = targets[idx].item()

            # 創建 mask 排除目標類別
            mask = torch.ones(num_classes, dtype=torch.bool, device=outputs.device)
            mask[target_class] = False

            # 獲取排除目標類別的 log-softmax 值
            log_softmax_without_target = -log_softmax[idx][mask]

            # 計算標準差（衡量分佈不平衡）
            std_dev = torch.std(log_softmax_without_target)

            # 自適應權重：目標類別的 softmax 值
            s_t = softmax[idx, target_class]

            # Logit Balancing Loss
            loss[idx] = beta_lb * std_dev * s_t

    # ========== 預測錯誤：Cross-Entropy Loss ==========
    if (~correct_mask).any():
        incorrect_indices = (~correct_mask).nonzero(as_tuple=True)[0]

        # 使用標準 Cross-Entropy Loss
        loss[incorrect_indices] = F.cross_entropy(
            outputs[incorrect_indices],
            targets[incorrect_indices],
            reduction='none'
        )

    return loss.mean(), correct_mask


# ==================== Attack Functions (論文設定) ====================
def rs_fgsm_attack(model, x, y, epsilon=8 / 255, alpha=None):

    if alpha is None:
        alpha = epsilon * 1.25  # Paper recommendation: α = 1.25ε

    # Step 1: Initialize δ = 0 or random
    delta = torch.zeros_like(x)

    # Random initialization: δ ~ U(-ε, ε) per channel
    for j in range(x.shape[1]):
        delta[:, j, :, :].uniform_(-epsilon, epsilon)

    # Clamp to valid image range [0,1]
    delta = torch.clamp(delta, 0 - x, 1 - x)
    delta.requires_grad = True

    # Step 2: Compute gradient on x + δ
    output = model(x + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()

    # Step 3: FGSM update: δ = δ + α · sign(∇L)
    grad = delta.grad.detach()
    delta = delta + alpha * torch.sign(grad)

    # Step 4: Project δ to ε-ball
    delta = torch.clamp(delta, -epsilon, epsilon)

    # Step 5: Clamp to valid image range [0,1]
    delta = torch.clamp(delta, 0 - x, 1 - x)

    # Step 6: Generate adversarial example
    x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv




def n_fgsm_attack(model, x, y, epsilon=8 / 255, alpha=None, unif=2.0, clip=-1):
    if alpha is None:
        alpha = epsilon

    # Step 1: 初始化隨機噪聲
    delta = torch.zeros_like(x)
    if unif > 0:
        for j in range(x.shape[1]):
            delta[:, j, :, :].uniform_(-unif * epsilon, unif * epsilon)

    delta = torch.clamp(delta, 0 - x, 1 - x)
    delta.requires_grad = True

    # Step 2: 計算梯度
    output = model(x + delta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, delta)[0].detach()

    # Step 3: FGSM 更新
    delta = delta + alpha * torch.sign(grad)

    # Step 4: 裁剪到 [0,1]
    delta = torch.clamp(delta, 0 - x, 1 - x)

    # Step 5: 條件性裁剪 (僅當 clip > 0)
    if clip > 0:
        clip_radius = clip * epsilon
        delta = torch.clamp(delta, -clip_radius, clip_radius)

    # Step 6: 生成對抗樣本
    x_adv = torch.clamp(x + delta, 0, 1).detach()
    return x_adv


def detect_aae(model, x, x_adv, y):
    """
    檢測 Abnormal Adversarial Examples (AAEs)
    AAE: loss_adv < loss_clean
    """
    model.eval()
    with torch.no_grad():
        loss_clean = F.cross_entropy(model(x), y, reduction='none')
        loss_adv = F.cross_entropy(model(x_adv), y, reduction='none')
        is_aae = (loss_adv < loss_clean)
    model.train()
    return is_aae


# ==================== Adversarial Examples Generation & Caching ====================
def generate_and_save_adversarial_examples(model, test_loader, device, model_name,
                                           epsilon=8 / 255, adv_dir='./adversarial_examples'):
    """生成並儲存對抗樣本 (PGD-20, APGDT 和 AutoAttack)"""
    os.makedirs(adv_dir, exist_ok=True)

    pgd_path = os.path.join(adv_dir, f'{model_name}_pgd20_eps{int(epsilon * 255)}.npz')
    apgdt_path = os.path.join(adv_dir, f'{model_name}_apgdt_eps{int(epsilon * 255)}.npz')
    aa_path = os.path.join(adv_dir, f'{model_name}_autoattack_eps{int(epsilon * 255)}.npz')

    if os.path.exists(pgd_path) and os.path.exists(apgdt_path) and os.path.exists(aa_path):
        print(f"✓ Adversarial examples already exist for {model_name}")
        pgd_size = os.path.getsize(pgd_path) / (1024 * 1024)
        apgdt_size = os.path.getsize(apgdt_path) / (1024 * 1024)
        aa_size = os.path.getsize(aa_path) / (1024 * 1024)
        print(f"  PGD-20: {pgd_size:.2f} MB")
        print(f"  APGDT: {apgdt_size:.2f} MB")
        print(f"  AutoAttack: {aa_size:.2f} MB")
        return pgd_path, apgdt_path, aa_path

    model.eval()

    print(f"\nCollecting test data...")
    all_data = []
    all_targets = []
    for data, target in tqdm(test_loader, desc="Loading test set"):
        all_data.append(data)
        all_targets.append(target)
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print(f"\n{'=' * 60}")
    print(f"Generating adversarial examples for {model_name}")
    print(f"Test set size: {len(all_data)}")
    print(f"{'=' * 60}")

    # ========== PGD-20 ==========
    if not os.path.exists(pgd_path):
        print(f"\nGenerating PGD-20 adversarial examples (α = ε/4)...")
        atk_pgd = PGD(model, eps=epsilon, alpha=epsilon / 4, steps=20, random_start=True)
        pgd_adv_data = []

        batch_size = 100
        num_batches = (len(all_data) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(all_data), batch_size),
                      total=num_batches,
                      desc="PGD-20",
                      unit="batch"):
            batch_data = all_data[i:i + batch_size].to(device)
            batch_target = all_targets[i:i + batch_size].to(device)
            batch_adv = atk_pgd(batch_data, batch_target)
            pgd_adv_data.append(batch_adv.detach().cpu())

        pgd_adv_data = torch.cat(pgd_adv_data, dim=0)

        np.savez_compressed(
            pgd_path,
            data=pgd_adv_data.numpy(),
            targets=all_targets.numpy()
        )
        file_size = os.path.getsize(pgd_path) / (1024 * 1024)
        print(f"✓ PGD-20 examples saved to {pgd_path} ({file_size:.2f} MB)")

    # ========== APGDT ==========
    if not os.path.exists(apgdt_path):
        print(f"\nGenerating APGDT adversarial examples...")
        atk_apgdt = APGDT(
            model,
            eps=epsilon,
            norm='Linf',
            seed=0,
            verbose=False,
            n_classes=10,
            n_restarts=1,
        )
        apgdt_adv_data = []

        batch_size = 100
        num_batches = (len(all_data) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(all_data), batch_size),
                      total=num_batches,
                      desc="APGDT",
                      unit="batch"):
            batch_data = all_data[i:i + batch_size].to(device)
            batch_target = all_targets[i:i + batch_size].to(device)
            batch_adv = atk_apgdt(batch_data, batch_target)
            apgdt_adv_data.append(batch_adv.detach().cpu())

        apgdt_adv_data = torch.cat(apgdt_adv_data, dim=0)

        np.savez_compressed(
            apgdt_path,
            data=apgdt_adv_data.numpy(),
            targets=all_targets.numpy()
        )
        file_size = os.path.getsize(apgdt_path) / (1024 * 1024)
        print(f"✓ APGDT examples saved to {apgdt_path} ({file_size:.2f} MB)")

    # ========== AutoAttack ==========
    if not os.path.exists(aa_path):
        print(f"\nGenerating AutoAttack adversarial examples...")
        atk_aa = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)
        aa_adv_data = []

        batch_size = 100
        num_batches = (len(all_data) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(all_data), batch_size),
                      total=num_batches,
                      desc="AutoAttack",
                      unit="batch"):
            batch_data = all_data[i:i + batch_size].to(device)
            batch_target = all_targets[i:i + batch_size].to(device)
            batch_adv = atk_aa(batch_data, batch_target)
            aa_adv_data.append(batch_adv.detach().cpu())

        aa_adv_data = torch.cat(aa_adv_data, dim=0)

        np.savez_compressed(
            aa_path,
            data=aa_adv_data.numpy(),
            targets=all_targets.numpy()
        )
        file_size = os.path.getsize(aa_path) / (1024 * 1024)
        print(f"✓ AutoAttack examples saved to {aa_path} ({file_size:.2f} MB)")

    return pgd_path, apgdt_path, aa_path


def load_adversarial_examples(adv_path):
    """載入對抗樣本"""
    if not os.path.exists(adv_path):
        raise FileNotFoundError(f"Adversarial examples not found: {adv_path}")

    print(f"✓ Loading adversarial examples from {adv_path}")
    adv_data = np.load(adv_path)

    data = torch.from_numpy(adv_data['data'])
    targets = torch.from_numpy(adv_data['targets'])

    return data, targets


# ==================== Loss History Tracking ====================
class LossHistory:
    """記錄訓練過程中的 loss"""

    def __init__(self):
        self.total_losses = []
        self.ce_losses = []
        self.lb_losses = []
        self.der_losses = []
        self.epochs = []

    def update(self, epoch, total_loss, ce_loss, lb_loss, der_loss):
        self.epochs.append(epoch)
        self.total_losses.append(total_loss)
        self.ce_losses.append(ce_loss)
        self.lb_losses.append(lb_loss)
        self.der_losses.append(der_loss)

    def plot(self, save_path, model_name):
        """繪製 loss 曲線"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Training Loss Curves', fontsize=16, fontweight='bold')

        # 1. Total Loss
        axes[0, 0].plot(self.epochs, self.total_losses, 'b-', linewidth=2, label='Total Loss')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=10)

        # 2. CE Loss
        axes[0, 1].plot(self.epochs, self.ce_losses, 'r-', linewidth=2, label='CE Loss')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].set_title('Cross-Entropy Loss', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=10)

        # 3. LB Loss
        axes[1, 0].plot(self.epochs, self.lb_losses, 'g-', linewidth=2, label='LB Loss')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Logit Balancing Loss', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=10)

        # 4. DER Loss
        axes[1, 1].plot(self.epochs, self.der_losses, 'm-', linewidth=2, label='DER Loss')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 1].set_title('DER Loss', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Individual loss curves saved to {save_path}")

    def plot_combined(self, save_path, model_name):
        """繪製所有 loss 在同一張圖"""
        plt.figure(figsize=(12, 8))

        plt.plot(self.epochs, self.total_losses, 'b-', linewidth=2, label='Total Loss', marker='o', markersize=4)
        plt.plot(self.epochs, self.ce_losses, 'r-', linewidth=2, label='CE Loss', marker='s', markersize=4)
        plt.plot(self.epochs, self.lb_losses, 'g-', linewidth=2, label='LB Loss', marker='^', markersize=4)
        plt.plot(self.epochs, self.der_losses, 'm-', linewidth=2, label='DER Loss', marker='d', markersize=4)

        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.title(f'{model_name} - Combined Training Loss Curves', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Combined loss curve saved to {save_path}")


# ==================== Training Functions (始終使用 LB + 高斯噪聲) ====================
def train_rs_der(model, train_loader, optimizer, scheduler, epoch, device,
                 epsilon=8 / 255, beta=0.5, gamma=0.0, beta_lb=0.02, sigma=24 / 255):
    """
    RS-DER Training with Logit Balancing Loss + Gaussian Noise (始終啟用)

    Args:
        sigma: 高斯噪聲標準差 (default: 24/255)
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_lb_loss = 0
    total_der_loss = 0
    correct = 0
    total = 0
    aae_count = 0
    lb_count = 0
    ce_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 生成對抗樣本
        x_adv = rs_fgsm_attack(model, data, target, epsilon=epsilon, alpha=epsilon * 1.25)

        # 檢測 AAE
        is_aae = detect_aae(model, data, x_adv, target)
        aae_count += is_aae.sum().item()

        optimizer.zero_grad()

        # ========== 關鍵：加入高斯噪聲 ==========
        # 每次都生成新的高斯噪聲（論文關鍵：每個 batch 不同噪聲）
        noise = torch.randn_like(x_adv) * sigma
        x_adv_noisy = torch.clamp(x_adv + noise, 0, 1)

        # 使用加噪聲後的對抗樣本計算輸出
        output = model(x_adv_noisy)

        # 計算 LB Loss（包含 CE 和 LB）
        base_loss, correct_mask = logit_balancing_loss(output, target, beta_lb=beta_lb)

        # 統計
        lb_count += correct_mask.sum().item()
        ce_count += (~correct_mask).sum().item()

        # 分別記錄 LB 和 CE 的貢獻
        with torch.no_grad():
            if correct_mask.any():
                correct_indices = correct_mask.nonzero(as_tuple=True)[0]
                lb_loss_only, _ = logit_balancing_loss(
                    output[correct_indices],
                    target[correct_indices],
                    beta_lb=beta_lb
                )
                total_lb_loss += lb_loss_only.item() * correct_indices.size(0)

            if (~correct_mask).any():
                incorrect_indices = (~correct_mask).nonzero(as_tuple=True)[0]
                ce_loss_only = F.cross_entropy(
                    output[incorrect_indices],
                    target[incorrect_indices]
                )
                total_ce_loss += ce_loss_only.item() * incorrect_indices.size(0)

        # 計算 DER Loss（只針對 AAE，使用原始對抗樣本）
        if is_aae.sum() > 0:
            aae_indices = is_aae.nonzero(as_tuple=True)[0]
            der = der_loss(model, data[aae_indices], x_adv[aae_indices],
                           target[aae_indices], gamma=gamma)
            total_loss_batch = base_loss + beta * der
            total_der_loss += der.item() * target.size(0)
        else:
            total_loss_batch = base_loss
            der = torch.tensor(0.0)

        total_loss_batch.backward()
        optimizer.step()
        scheduler.step()

        total_loss += total_loss_batch.item() * target.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']

            with torch.no_grad():
                _, correct_mask_display = logit_balancing_loss(output, target, beta_lb=beta_lb)

                if correct_mask_display.any():
                    correct_indices = correct_mask_display.nonzero(as_tuple=True)[0]
                    lb_display, _ = logit_balancing_loss(
                        output[correct_indices],
                        target[correct_indices],
                        beta_lb=beta_lb
                    )
                else:
                    lb_display = torch.tensor(0.0)

                if (~correct_mask_display).any():
                    incorrect_indices = (~correct_mask_display).nonzero(as_tuple=True)[0]
                    ce_display = F.cross_entropy(
                        output[incorrect_indices],
                        target[incorrect_indices]
                    )
                else:
                    ce_display = torch.tensor(0.0)

            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'LR: {current_lr:.6f} '
                  f'Loss: {total_loss_batch.item():.4f} '
                  f'CE: {ce_display.item():.4f} '
                  f'LB: {lb_display.item():.4f} '
                  f'DER: {der.item():.4f} '
                  f'Acc: {100. * correct / total:.2f}% '
                  f'AAE: {100. * aae_count / total:.2f}%')

    return (total_loss / total, total_ce_loss / total,
            total_lb_loss / total, total_der_loss / total, 100. * correct / total)


def train_n_der(model, train_loader, optimizer, scheduler, epoch, device,
                epsilon=8 / 255, beta=0.1, gamma=0.0, beta_lb=0.02, sigma=24 / 255):
    """
    N-DER Training with Logit Balancing Loss + Gaussian Noise (始終啟用)

    Args:
        sigma: 高斯噪聲標準差 (default: 24/255)
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_lb_loss = 0
    total_der_loss = 0
    correct = 0
    total = 0
    aae_count = 0
    lb_count = 0
    ce_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 生成對抗樣本
        x_adv = n_fgsm_attack(model, data, target, epsilon=epsilon)

        # 檢測 AAE
        is_aae = detect_aae(model, data, x_adv, target)
        aae_count += is_aae.sum().item()

        optimizer.zero_grad()

        # ========== 關鍵：加入高斯噪聲 ==========
        # 每次都生成新的高斯噪聲（論文關鍵：每個 batch 不同噪聲）
        noise = torch.randn_like(x_adv) * sigma
        x_adv_noisy = torch.clamp(x_adv + noise, 0, 1)

        # 使用加噪聲後的對抗樣本計算輸出
        output = model(x_adv_noisy)

        # 計算 LB Loss（包含 CE 和 LB）
        base_loss, correct_mask = logit_balancing_loss(output, target, beta_lb=beta_lb)

        # 統計
        lb_count += correct_mask.sum().item()
        ce_count += (~correct_mask).sum().item()

        # 分別記錄 LB 和 CE 的貢獻
        with torch.no_grad():
            if correct_mask.any():
                correct_indices = correct_mask.nonzero(as_tuple=True)[0]
                lb_loss_only, _ = logit_balancing_loss(
                    output[correct_indices],
                    target[correct_indices],
                    beta_lb=beta_lb
                )
                total_lb_loss += lb_loss_only.item() * correct_indices.size(0)

            if (~correct_mask).any():
                incorrect_indices = (~correct_mask).nonzero(as_tuple=True)[0]
                ce_loss_only = F.cross_entropy(
                    output[incorrect_indices],
                    target[incorrect_indices]
                )
                total_ce_loss += ce_loss_only.item() * incorrect_indices.size(0)

        # 計算 DER Loss（只針對 AAE，使用原始對抗樣本）
        if is_aae.sum() > 0:
            aae_indices = is_aae.nonzero(as_tuple=True)[0]
            der = der_loss(model, data[aae_indices], x_adv[aae_indices],
                           target[aae_indices], gamma=gamma)
            total_loss_batch = base_loss + beta * der
            total_der_loss += der.item() * target.size(0)
        else:
            total_loss_batch = base_loss
            der = torch.tensor(0.0)

        total_loss_batch.backward()
        optimizer.step()
        scheduler.step()

        total_loss += total_loss_batch.item() * target.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']

            with torch.no_grad():
                _, correct_mask_display = logit_balancing_loss(output, target, beta_lb=beta_lb)

                if correct_mask_display.any():
                    correct_indices = correct_mask_display.nonzero(as_tuple=True)[0]
                    lb_display, _ = logit_balancing_loss(
                        output[correct_indices],
                        target[correct_indices],
                        beta_lb=beta_lb
                    )
                else:
                    lb_display = torch.tensor(0.0)

                if (~correct_mask_display).any():
                    incorrect_indices = (~correct_mask_display).nonzero(as_tuple=True)[0]
                    ce_display = F.cross_entropy(
                        output[incorrect_indices],
                        target[incorrect_indices]
                    )
                else:
                    ce_display = torch.tensor(0.0)

            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'LR: {current_lr:.6f} '
                  f'Loss: {total_loss_batch.item():.4f} '
                  f'CE: {ce_display.item():.4f} '
                  f'LB: {lb_display.item():.4f} '
                  f'DER: {der.item():.4f} '
                  f'Acc: {100. * correct / total:.2f}% '
                  f'AAE: {100. * aae_count / total:.2f}%')

    return (total_loss / total, total_ce_loss / total,
            total_lb_loss / total, total_der_loss / total, 100. * correct / total)


# ==================== Evaluation Functions ====================
def evaluate_clean(model, test_loader, device):
    """評估 clean accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total


def evaluate_adversarial(model, adv_data, adv_targets, device, batch_size=100):
    """評估預先生成的對抗樣本"""
    model.eval()
    correct = 0
    total = 0

    num_batches = (len(adv_data) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(adv_data), batch_size),
                      total=num_batches,
                      desc="Evaluating",
                      unit="batch"):
            batch_data = adv_data[i:i + batch_size].to(device)
            batch_target = adv_targets[i:i + batch_size].to(device)
            output = model(batch_data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(batch_target.view_as(pred)).sum().item()
            total += batch_target.size(0)

    return 100. * correct / total


# ==================== Checkpoint Management ====================
def save_checkpoint(model, optimizer, epoch, filepath):
    """儲存 checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """載入 checkpoint"""
    if os.path.exists(filepath):
        print(f"✓ Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)
            epoch = 0

        print(f"  Loaded model from epoch {epoch}")
        return True, epoch
    else:
        print(f"✗ Checkpoint not found: {filepath}")
        return False, 0


def train_or_load_model(model_name, model, train_loader, test_loader, device,
                        epsilon=8 / 255, num_epochs=50, train_func=None,
                        beta=0.5, gamma=0.0, beta_lb=0.02, sigma=24 / 255,
                        checkpoint_dir='./checkpoints',
                        lr_schedule='cyclic', lr_min=0.0, lr_max=0.2,
                        momentum=0.9, weight_decay=5e-4,
                        plot_dir='./plots'):
    """
    訓練或載入模型（始終使用 LB Loss + 高斯噪聲）
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_cifar10.pth')

    optimizer = optim.SGD(model.parameters(), lr=lr_max,
                          momentum=momentum, weight_decay=weight_decay)

    lr_steps = num_epochs * len(train_loader)

    if lr_schedule == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=lr_steps / 2,
            step_size_down=lr_steps / 2,
            mode='triangular',
            cycle_momentum=False
        )
    elif lr_schedule == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(lr_steps * 0.5), int(lr_steps * 0.75)],
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    checkpoint_exists, loaded_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    if checkpoint_exists:
        print(f"✓ Model already trained! Skipping training for {model_name}.")
        return model

    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Beta_DER: {beta}")
    print(f"  Gamma: {gamma}")
    print(f"  Beta_LB: {beta_lb}")
    print(f"  Sigma (Gaussian Noise): {sigma:.4f}")
    print(f"  Epsilon: {epsilon}")
    print(f"{'=' * 60}")

    # 創建 loss history tracker
    loss_history = LossHistory()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_ce, train_lb, train_der, train_acc = train_func(
            model, train_loader, optimizer, scheduler, epoch, device,
            epsilon=epsilon, beta=beta, gamma=gamma, beta_lb=beta_lb, sigma=sigma
        )

        # 記錄 loss
        loss_history.update(epoch, train_loss, train_ce, train_lb, train_der)

        if epoch % 10 == 0 or epoch == num_epochs:
            clean_acc = evaluate_clean(model, test_loader, device)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\nEpoch {epoch} Summary:')
            print(f'  Current LR: {current_lr:.6f}')
            print(f'  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, LB: {train_lb:.4f}, DER: {train_der:.4f})')
            print(f'  Train Acc: {train_acc:.2f}%')
            print(f'  Clean Acc: {clean_acc:.2f}%')

    save_checkpoint(model, optimizer, num_epochs, checkpoint_path)
    print(f"✓ Training completed for {model_name}!")

    # 繪製並儲存 loss 曲線
    print(f"\nGenerating loss curves...")
    individual_plot_path = os.path.join(plot_dir, f'{model_name}_loss_curves_individual.png')
    combined_plot_path = os.path.join(plot_dir, f'{model_name}_loss_curves_combined.png')

    loss_history.plot(individual_plot_path, model_name)
    loss_history.plot_combined(combined_plot_path, model_name)

    return model


# ==================== Main ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = './checkpoints'
    adv_dir = './adversarial_examples'
    plot_dir = './plots'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Experimental Settings (Single-Step AT with LB Loss + Gaussian Noise)")
    print("=" * 60)
    print("Dataset: CIFAR-10")
    print("Architecture: PreActResNet-18")
    print("Epochs: 50")
    print("Epsilon: 8/255")
    print("Sigma (Gaussian Noise): 24/255")
    print("LR Schedule: CyclicLR (base_lr=0.0, max_lr=0.2)")
    print("Evaluation: Final checkpoint with PGD-20, APGDT and AutoAttack")
    print("RS-DER+LB+Noise: β_DER=1.0, β_LB=0.01, γ=0.0, σ=24/255")
    print("N-DER+LB+Noise: β_DER=0.4, β_LB=0.01, γ=0.0, σ=24/255")
    print("=" * 60)

    print("\nLoading CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    epsilon = 8 / 255
    sigma = 24 / 255
    num_epochs = 50

    # ========== RS-DER+LB+Noise ==========
    print("\n" + "=" * 60)
    print("RS-DER+LB+Noise Model (β_DER=1.0, β_LB=0.01, σ=24/255, γ=0.0)")
    print("=" * 60)

    model_rs = PreActResNet18(num_classes=10).to(device)
    model_rs = train_or_load_model(
        model_name='rs_der_lb_noise_tuned',
        model=model_rs,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epsilon=epsilon,
        num_epochs=num_epochs,
        train_func=train_rs_der,
        beta=1.0,
        gamma=0.0,
        beta_lb=0.01,
        sigma=sigma,
        checkpoint_dir=checkpoint_dir,
        lr_schedule='cyclic',
        lr_min=0.0,
        lr_max=0.2,
        momentum=0.9,
        weight_decay=5e-4,
        plot_dir=plot_dir
    )

    pgd_path_rs, apgdt_path_rs, aa_path_rs = generate_and_save_adversarial_examples(
        model_rs, test_loader, device, 'rs_der_lb_noise_tuned', epsilon=epsilon, adv_dir=adv_dir
    )

    print("\n" + "=" * 60)
    print("RS-DER+LB+Noise Evaluation (Tuned - Final Checkpoint)")
    print("=" * 60)

    clean_acc_rs = evaluate_clean(model_rs, test_loader, device)
    print(f'Clean Accuracy: {clean_acc_rs:.2f}%')

    pgd_adv_data, pgd_targets = load_adversarial_examples(pgd_path_rs)
    pgd_acc_rs = evaluate_adversarial(model_rs, pgd_adv_data, pgd_targets, device)
    print(f'PGD-20 Accuracy: {pgd_acc_rs:.2f}%')

    apgdt_adv_data, apgdt_targets = load_adversarial_examples(apgdt_path_rs)
    apgdt_acc_rs = evaluate_adversarial(model_rs, apgdt_adv_data, apgdt_targets, device)
    print(f'APGDT Accuracy: {apgdt_acc_rs:.2f}%')

    aa_adv_data, aa_targets = load_adversarial_examples(aa_path_rs)
    aa_acc_rs = evaluate_adversarial(model_rs, aa_adv_data, aa_targets, device)
    print(f'AutoAttack Accuracy: {aa_acc_rs:.2f}%')

    # ========== N-DER+LB+Noise ==========
    print("\n" + "=" * 60)
    print("N-DER+LB+Noise Model (β_DER=0.4, β_LB=0.01, σ=24/255, γ=0.0)")
    print("=" * 60)

    model_n = PreActResNet18(num_classes=10).to(device)
    model_n = train_or_load_model(
        model_name='n_der_lb_noise_tuned',
        model=model_n,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epsilon=epsilon,
        num_epochs=num_epochs,
        train_func=train_n_der,
        beta=0.4,
        gamma=0.0,
        beta_lb=0.01,
        sigma=sigma,
        checkpoint_dir=checkpoint_dir,
        lr_schedule='cyclic',
        lr_min=0.0,
        lr_max=0.2,
        momentum=0.9,
        weight_decay=5e-4,
        plot_dir=plot_dir
    )

    pgd_path_n, apgdt_path_n, aa_path_n = generate_and_save_adversarial_examples(
        model_n, test_loader, device, 'n_der_lb_noise_tuned', epsilon=epsilon, adv_dir=adv_dir
    )

    print("\n" + "=" * 60)
    print("N-DER+LB+Noise Evaluation (Tuned - Final Checkpoint)")
    print("=" * 60)

    clean_acc_n = evaluate_clean(model_n, test_loader, device)
    print(f'Clean Accuracy: {clean_acc_n:.2f}%')

    pgd_adv_data, pgd_targets = load_adversarial_examples(pgd_path_n)
    pgd_acc_n = evaluate_adversarial(model_n, pgd_adv_data, pgd_targets, device)
    print(f'PGD-20 Accuracy: {pgd_acc_n:.2f}%')

    apgdt_adv_data, apgdt_targets = load_adversarial_examples(apgdt_path_n)
    apgdt_acc_n = evaluate_adversarial(model_n, apgdt_adv_data, apgdt_targets, device)
    print(f'APGDT Accuracy: {apgdt_acc_n:.2f}%')

    aa_adv_data, aa_targets = load_adversarial_examples(aa_path_n)
    aa_acc_n = evaluate_adversarial(model_n, aa_adv_data, aa_targets, device)
    print(f'AutoAttack Accuracy: {aa_acc_n:.2f}%')

    # ========== 結果總結（只顯示 β_DER 和 β_LB）==========
    print("\n" + "=" * 60)
    print("Final Results Summary")
    print("=" * 60)
    print(f"\nCIFAR-10 (ε = 8/255, 50 epochs, Single-Step)")
    print(f"{'Method':<30} {'β_DER':<10} {'β_LB':<10} {'Clean':<10} {'PGD-20':<10} {'APGDT':<10} {'AA':<10}")
    print("-" * 100)
    print(
        f"{'RS-DER+LB+Noise':<30} {1.0:<10.1f} {0.01:<10.2f} {clean_acc_rs:>9.2f} {pgd_acc_rs:>9.2f} {apgdt_acc_rs:>9.2f} {aa_acc_rs:>9.2f}")
    print(
        f"{'N-DER+LB+Noise':<30} {0.4:<10.1f} {0.01:<10.2f} {clean_acc_n:>9.2f} {pgd_acc_n:>9.2f} {apgdt_acc_n:>9.2f} {aa_acc_n:>9.2f}")
    print("=" * 100)

    print(f"\n✓ All loss curves saved to {plot_dir}/")


if __name__ == '__main__':
    main()
