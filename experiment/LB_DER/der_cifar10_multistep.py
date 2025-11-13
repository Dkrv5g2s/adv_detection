import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchattacks import PGD, AutoAttack
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


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
    """計算能量函數"""
    logits = model(x)
    if y is None:
        energy_x = -torch.logsumexp(logits, dim=1)
        return energy_x
    else:
        energy_xy = -logits.gather(1, y.view(-1, 1)).squeeze()
        energy_x = -torch.logsumexp(logits, dim=1)
        return energy_x, energy_xy


def compute_delta_energy(model, x, x_adv, y):
    """計算能量差異"""
    energy_x, energy_xy = compute_energy(model, x, y)
    energy_x_adv, energy_xy_adv = compute_energy(model, x_adv, y)
    delta_ex = energy_x - energy_x_adv
    delta_exy = energy_xy - energy_xy_adv
    delta_energy = torch.sqrt(delta_ex ** 2 + delta_exy ** 2)
    return delta_energy, delta_ex, delta_exy


def der_loss(model, x, x_adv, y, gamma=0.2):
    """DER regularizer: eq. (7) in paper"""
    delta_energy, _, _ = compute_delta_energy(model, x, x_adv, y)
    der = torch.clamp(delta_energy - gamma, min=0.0)
    return der.mean()


# ==================== PGD Attack (論文設定) ====================
def pgd_attack(model, x, y, epsilon=8 / 255, alpha=None, num_steps=10, random_start=True):
    """
    PGD attack following paper settings:
    - alpha = epsilon / 4 (step size)
    - num_steps: 10 for training, 20 for evaluation
    """
    if alpha is None:
        alpha = epsilon / 4

    model.eval()

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x)

    delta.requires_grad = True

    for _ in range(num_steps):
        output = model(x + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = delta + alpha * grad.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(x + delta.data, 0, 1) - x
        delta.grad.zero_()

    model.train()
    return (x + delta).detach()


# ==================== Training Functions (嚴格按照論文) ====================
def train_sat(model, train_loader, optimizer, epoch, device, epsilon=8 / 255):
    """
    Standard Adversarial Training (SAT)
    論文: "we initially train the model using SAT and apply the regularizer
           during the final epochs"
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 生成對抗樣本 (PGD-10, alpha=epsilon/4)
        x_adv = pgd_attack(model, data, target, epsilon=epsilon,
                           alpha=epsilon / 4, num_steps=10, random_start=True)

        optimizer.zero_grad()
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')

    return total_loss / total, 100. * correct / total


def train_der_at(model, train_loader, optimizer, epoch, device,
                 epsilon=8 / 255, beta=6.0, gamma=0.2):
    """
    DER-AT: SAT + DER regularizer
    論文: Loss = CE(f(x_adv), y) + β * DER(x, x_adv, y)
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_der_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 生成對抗樣本 (PGD-10, alpha=epsilon/4)
        x_adv = pgd_attack(model, data, target, epsilon=epsilon,
                           alpha=epsilon / 4, num_steps=10, random_start=True)

        optimizer.zero_grad()

        # CE loss on adversarial examples
        output = model(x_adv)
        ce_loss = F.cross_entropy(output, target)

        # DER regularizer
        der = der_loss(model, data, x_adv, target, gamma=gamma)

        # Total loss
        total_loss_batch = ce_loss + beta * der

        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item() * target.size(0)
        total_ce_loss += ce_loss.item() * target.size(0)
        total_der_loss += der.item() * target.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {total_loss_batch.item():.4f} CE: {ce_loss.item():.4f} '
                  f'DER: {der.item():.4f} Acc: {100. * correct / total:.2f}%')

    return total_loss / total, total_ce_loss / total, total_der_loss / total, 100. * correct / total


# ==================== Validation Loss ====================
def compute_validation_loss(model, val_loader, device, epsilon=8 / 255,
                            use_der=False, beta=6.0, gamma=0.2):
    """計算驗證集損失"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_der_loss = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            # 生成對抗樣本
            x_adv = pgd_attack(model, data, target, epsilon=epsilon,
                               alpha=epsilon / 4, num_steps=10, random_start=True)

            output = model(x_adv)
            ce_loss = F.cross_entropy(output, target, reduction='sum')
            total_ce_loss += ce_loss.item()

            if use_der:
                delta_energy, _, _ = compute_delta_energy(model, data, x_adv, target)
                der = torch.clamp(delta_energy - gamma, min=0.0).sum()
                total_der_loss += der.item()
                total_loss += (ce_loss + beta * der).item()
            else:
                total_loss += ce_loss.item()

            total += target.size(0)

    model.train()
    return total_loss / total, total_ce_loss / total, total_der_loss / total if use_der else 0


# ==================== 生成並儲存對抗樣本 ====================
def generate_and_save_adversarial_examples(model, test_loader, device,
                                           epsilon=8 / 255, save_dir='./adv_examples'):
    """
    生成並儲存測試集的對抗樣本
    - PGD-20
    - AutoAttack
    """
    os.makedirs(save_dir, exist_ok=True)

    pgd_path = os.path.join(save_dir, f'pgd20_eps{int(epsilon * 255)}.npz')
    aa_path = os.path.join(save_dir, f'autoattack_eps{int(epsilon * 255)}.npz')

    # 檢查是否已存在
    if os.path.exists(pgd_path) and os.path.exists(aa_path):
        print(f"✓ 對抗樣本已存在，跳過生成")
        return pgd_path, aa_path

    model.eval()

    # 收集所有測試資料
    print("\n收集測試資料...")
    all_data = []
    all_targets = []
    for data, target in tqdm(test_loader, desc="載入測試集"):
        all_data.append(data)
        all_targets.append(target)

    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    print(f"測試集大小: {len(all_data)}")

    # ==================== 生成 PGD-20 ====================
    if not os.path.exists(pgd_path):
        print(f"\n生成 PGD-20 對抗樣本 (ε={epsilon})...")
        atk_pgd = PGD(model, eps=epsilon, alpha=epsilon / 4, steps=20, random_start=True)

        pgd_adv_data = []
        batch_size = 100

        for i in tqdm(range(0, len(all_data), batch_size), desc="PGD-20"):
            batch_data = all_data[i:i + batch_size].to(device)
            batch_target = all_targets[i:i + batch_size].to(device)

            batch_adv = atk_pgd(batch_data, batch_target).detach().cpu()
            pgd_adv_data.append(batch_adv)

        pgd_adv_data = torch.cat(pgd_adv_data, dim=0).numpy()

        # 儲存
        np.savez_compressed(
            pgd_path,
            data=pgd_adv_data,
            targets=all_targets.numpy(),
            epsilon=epsilon
        )
        print(f"✓ PGD-20 對抗樣本已儲存至: {pgd_path}")
    else:
        print(f"✓ PGD-20 對抗樣本已存在: {pgd_path}")

    # ==================== 生成 AutoAttack ====================
    if not os.path.exists(aa_path):
        print(f"\n生成 AutoAttack 對抗樣本 (ε={epsilon})...")
        atk_aa = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)

        aa_adv_data = []
        batch_size = 100

        for i in tqdm(range(0, len(all_data), batch_size), desc="AutoAttack"):
            batch_data = all_data[i:i + batch_size].to(device)
            batch_target = all_targets[i:i + batch_size].to(device)

            batch_adv = atk_aa(batch_data, batch_target).detach().cpu()
            aa_adv_data.append(batch_adv)

        aa_adv_data = torch.cat(aa_adv_data, dim=0).numpy()

        # 儲存
        np.savez_compressed(
            aa_path,
            data=aa_adv_data,
            targets=all_targets.numpy(),
            epsilon=epsilon
        )
        print(f"✓ AutoAttack 對抗樣本已儲存至: {aa_path}")
    else:
        print(f"✓ AutoAttack 對抗樣本已存在: {aa_path}")

    return pgd_path, aa_path


# ==================== 從儲存的對抗樣本評估 ====================
def evaluate_from_saved_adversarial(model, adv_path, device, batch_size=100):
    """從儲存的對抗樣本評估模型"""
    model.eval()

    # 載入對抗樣本
    print(f"\n載入對抗樣本: {adv_path}")
    adv_data_npz = np.load(adv_path)
    adv_data = torch.from_numpy(adv_data_npz['data'])
    targets = torch.from_numpy(adv_data_npz['targets'])

    print(f"對抗樣本數量: {len(adv_data)}")

    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(adv_data), batch_size), desc="評估"):
            batch_data = adv_data[i:i + batch_size].to(device)
            batch_target = targets[i:i + batch_size].to(device)

            output = model(batch_data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(batch_target.view_as(pred)).sum().item()
            total += batch_target.size(0)

    accuracy = 100. * correct / total
    return accuracy


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


# ==================== Plotting Functions ====================
def plot_training_curves(train_history, ro_start_epoch, save_dir='./checkpoints_der_at'):
    """繪製訓練曲線"""
    os.makedirs(save_dir, exist_ok=True)

    epochs = np.array(train_history['epoch'])
    train_loss = np.array(train_history['train_loss'])
    val_loss = np.array(train_history['val_loss'])
    train_acc = np.array(train_history['train_acc'])
    clean_acc = np.array(train_history['clean_acc'])
    pgd_acc = np.array(train_history['pgd_acc'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.axvline(x=ro_start_epoch, color='orange', linestyle='--', linewidth=2,
                label=f'DER Start (Epoch {ro_start_epoch})')
    ax1.axvspan(ro_start_epoch, epochs[-1], alpha=0.1, color='orange')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy Curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, clean_acc, 'g-', label='Test Clean Acc', linewidth=2, marker='^', markersize=3)
    ax2.plot(epochs, pgd_acc, 'r-', label='Test PGD-20 Acc', linewidth=2, marker='s', markersize=3)
    ax2.axvline(x=ro_start_epoch, color='orange', linestyle='--', linewidth=2,
                label=f'DER Start (Epoch {ro_start_epoch})')

    best_pgd_idx = np.argmax(pgd_acc)
    best_pgd_epoch = epochs[best_pgd_idx]
    ax2.axvline(x=best_pgd_epoch, color='purple', linestyle='--', linewidth=2,
                label=f'Best PGD (Epoch {best_pgd_epoch})')
    ax2.scatter([best_pgd_epoch], [pgd_acc[best_pgd_idx]], color='purple', s=100, zorder=5)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Loss Difference
    ax3 = axes[1, 0]
    loss_diff = train_loss - val_loss
    ax3.plot(epochs, loss_diff, 'b-', linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.axvline(x=ro_start_epoch, color='orange', linestyle='--', linewidth=2)
    ax3.fill_between(epochs, 0, loss_diff, where=(loss_diff > 0), color='red', alpha=0.3, label='Overfitting Region')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss Difference', fontsize=12)
    ax3.set_title('Loss Difference (Train - Val)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Overfitting Gap
    ax4 = axes[1, 1]
    gap = train_acc - pgd_acc
    ax4.plot(epochs, gap, 'r-', linewidth=2, marker='o', markersize=3, label='Overfitting Gap')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.axvline(x=ro_start_epoch, color='orange', linestyle='--', linewidth=2,
                label=f'DER Start (Epoch {ro_start_epoch})')
    ax4.axvspan(ro_start_epoch, epochs[-1], alpha=0.1, color='orange')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Gap (%)', fontsize=12)
    ax4.set_title('Overfitting Gap (Train Acc - PGD Acc)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 訓練曲線已儲存至: {os.path.join(save_dir, 'training_curves.png')}")


# ==================== Checkpoint Management ====================
def save_checkpoint(model, optimizer, epoch, save_path, **kwargs):
    """儲存 checkpoint（只儲存模型和優化器）"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """載入 checkpoint（只載入模型參數）"""
    if os.path.exists(checkpoint_path):
        print(f"✓ 載入 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        print(f"  已載入 epoch {epoch} 的模型")
        return True, epoch
    else:
        return False, 0


# ==================== Training with Checkpoints (論文設定) ====================
def train_with_checkpoints(model, train_loader, val_loader, test_loader,
                           optimizer, scheduler, device,
                           num_epochs=100, ro_start_epoch=80,
                           epsilon=8 / 255, beta=6.0, gamma=0.2,
                           save_dir='./checkpoints_der_at', resume=True):
    """
    論文訓練流程:
    1. 前 ro_start_epoch 個 epochs: 使用 SAT
    2. 後續 epochs: 使用 DER-AT (SAT + DER regularizer)
    3. 評估使用 PGD-20 和 AutoAttack
    """
    os.makedirs(save_dir, exist_ok=True)

    best_pgd_acc = 0
    best_epoch = 0
    start_epoch = 1

    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_ce_loss': [],
        'train_der_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_ce_loss': [],
        'val_der_loss': [],
        'clean_acc': [],
        'pgd_acc': []
    }

    # 嘗試載入 checkpoint（只載入模型和優化器）
    last_checkpoint_path = os.path.join(save_dir, 'last_checkpoint.pth')
    history_path = os.path.join(save_dir, 'train_history.json')

    if resume:
        checkpoint_loaded, loaded_epoch = load_checkpoint(
            model, optimizer, last_checkpoint_path, device
        )

        if checkpoint_loaded:
            start_epoch = loaded_epoch + 1

            # 載入訓練歷史（獨立載入）
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    train_history = json.load(f)
                print(f"✓ 訓練歷史已載入")

                # 從歷史中恢復 best_pgd_acc 和 best_epoch
                if train_history['pgd_acc']:
                    best_pgd_acc = max(train_history['pgd_acc'])
                    best_epoch = train_history['epoch'][train_history['pgd_acc'].index(best_pgd_acc)]

            if start_epoch > num_epochs:
                print(f"✓ 訓練已完成 (epoch {loaded_epoch}/{num_epochs})")
                return train_history, best_epoch

            print(f"✓ 從 epoch {start_epoch} 繼續訓練")

    # 訓練循環
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        if epoch < ro_start_epoch:
            # SAT 階段
            train_loss, train_acc = train_sat(
                model, train_loader, optimizer, epoch, device, epsilon=epsilon
            )
            train_ce_loss = train_loss
            train_der_loss = 0.0

            val_loss, val_ce_loss, val_der_loss = compute_validation_loss(
                model, val_loader, device, epsilon=epsilon, use_der=False
            )
        else:
            # DER-AT 階段
            train_loss, train_ce_loss, train_der_loss, train_acc = train_der_at(
                model, train_loader, optimizer, epoch, device,
                epsilon=epsilon, beta=beta, gamma=gamma
            )

            val_loss, val_ce_loss, val_der_loss = compute_validation_loss(
                model, val_loader, device, epsilon=epsilon,
                use_der=True, beta=beta, gamma=gamma
            )

        scheduler.step()

        # 評估
        clean_acc = evaluate_clean(model, test_loader, device)

        # 簡化的 PGD-20 評估（訓練時用快速版本）
        model.eval()
        correct = 0
        total = 0
        atk = PGD(model, eps=epsilon, alpha=epsilon / 4, steps=20)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            adv_data = atk(data, target)
            with torch.no_grad():
                output = model(adv_data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        pgd_acc = 100. * correct / total
        model.train()

        # 記錄歷史
        train_history['epoch'].append(epoch)
        train_history['train_loss'].append(train_loss)
        train_history['train_ce_loss'].append(train_ce_loss)
        train_history['train_der_loss'].append(train_der_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_ce_loss'].append(val_ce_loss)
        train_history['val_der_loss'].append(val_der_loss)
        train_history['clean_acc'].append(clean_acc)
        train_history['pgd_acc'].append(pgd_acc)

        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Test Clean Acc: {clean_acc:.2f}%, Test PGD-20 Acc: {pgd_acc:.2f}%')

        # 儲存 best checkpoint
        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(save_dir, 'best_checkpoint.pth'),
                clean_acc=clean_acc,
                pgd_acc=pgd_acc
            )
            print(f'  ✓ 新的 Best Checkpoint! PGD-20 Acc: {pgd_acc:.2f}%')

        # 儲存 last checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            last_checkpoint_path,
            clean_acc=clean_acc,
            pgd_acc=pgd_acc
        )

        # 儲存訓練歷史
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=4)

    return train_history, best_epoch


# ==================== Main Function (論文實驗設定) ====================
def main():
    """
    論文實驗設定 (Table III):
    - Dataset: CIFAR-10
    - Architecture: PreActResNet-18
    - Epochs: 100
    - Epsilon: 8/255
    - Beta: 6 (CIFAR-10)
    - Gamma: 0.2
    - Evaluation: PGD-20 and AutoAttack
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 論文參數設定
    epsilon = 8 / 255
    num_epochs = 100
    ro_start_epoch = 80
    beta = 6.0
    gamma = 0.2
    batch_size = 128

    save_dir = './checkpoints_der_at'
    adv_dir = './adv_examples'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    print(f"\n論文實驗參數 (Table III):")
    print(f"  Dataset: CIFAR-10")
    print(f"  Architecture: PreActResNet-18")
    print(f"  Epsilon: {epsilon} (8/255)")
    print(f"  Total Epochs: {num_epochs}")
    print(f"  DER Start Epoch: {ro_start_epoch}")
    print(f"  Beta: {beta}")
    print(f"  Gamma: {gamma}")

    # ==================== 載入資料集 ====================
    print("\n" + "=" * 60)
    print("載入資料集")
    print("=" * 60)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=100, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")
    print(f"測試集大小: {len(test_dataset)}")

    # ==================== 初始化模型 ====================
    print("\n" + "=" * 60)
    print("初始化模型")
    print("=" * 60)

    model = PreActResNet18(num_classes=10).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.1
    )

    # ==================== 訓練 ====================
    print("\n" + "=" * 60)
    print("開始訓練 (論文設定)")
    print("=" * 60)

    train_history, best_epoch = train_with_checkpoints(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        num_epochs=num_epochs,
        ro_start_epoch=ro_start_epoch,
        epsilon=epsilon,
        beta=beta,
        gamma=gamma,
        save_dir=save_dir,
        resume=True
    )

    # ==================== 繪製訓練曲線 ====================
    plot_training_curves(train_history, ro_start_epoch, save_dir=save_dir)

    # ==================== 生成對抗樣本 ====================
    print("\n" + "=" * 60)
    print("生成並儲存對抗樣本")
    print("=" * 60)

    # 載入 best checkpoint 生成對抗樣本
    best_checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth')
    load_checkpoint(model, None, best_checkpoint_path, device)

    pgd_path, aa_path = generate_and_save_adversarial_examples(
        model, test_loader, device, epsilon=epsilon, save_dir=adv_dir
    )

    # ==================== 評估 ====================
    print("\n" + "=" * 60)
    print("最終評估")
    print("=" * 60)

    # 評估 Best Checkpoint
    print("\n評估 Best Checkpoint:")
    print("-" * 60)
    load_checkpoint(model, None, best_checkpoint_path, device)

    clean_acc_best = evaluate_clean(model, test_loader, device)
    pgd_acc_best = evaluate_from_saved_adversarial(model, pgd_path, device)
    aa_acc_best = evaluate_from_saved_adversarial(model, aa_path, device)

    print(f"\nBest Checkpoint 結果:")
    print(f"  Clean Accuracy: {clean_acc_best:.2f}%")
    print(f"  PGD-20 Accuracy: {pgd_acc_best:.2f}%")
    print(f"  AutoAttack Accuracy: {aa_acc_best:.2f}%")

    # 評估 Last Checkpoint
    print("\n評估 Last Checkpoint (Final):")
    print("-" * 60)
    last_checkpoint_path = os.path.join(save_dir, 'last_checkpoint.pth')
    load_checkpoint(model, None, last_checkpoint_path, device)

    clean_acc_last = evaluate_clean(model, test_loader, device)
    pgd_acc_last = evaluate_from_saved_adversarial(model, pgd_path, device)
    aa_acc_last = evaluate_from_saved_adversarial(model, aa_path, device)

    print(f"\nLast Checkpoint 結果:")
    print(f"  Clean Accuracy: {clean_acc_last:.2f}%")
    print(f"  PGD-20 Accuracy: {pgd_acc_last:.2f}%")
    print(f"  AutoAttack Accuracy: {aa_acc_last:.2f}%")

    # ==================== 儲存最終結果 ====================
    final_results = {
        'settings': {
            'dataset': 'CIFAR-10',
            'architecture': 'PreActResNet-18',
            'epsilon': float(epsilon),
            'epochs': num_epochs,
            'ro_start_epoch': ro_start_epoch,
            'beta': beta,
            'gamma': gamma
        },
        'best': {
            'epoch': best_epoch,
            'clean': float(clean_acc_best),
            'pgd20': float(pgd_acc_best),
            'autoattack': float(aa_acc_best)
        },
        'last': {
            'epoch': num_epochs,
            'clean': float(clean_acc_last),
            'pgd20': float(pgd_acc_last),
            'autoattack': float(aa_acc_last)
        },
        'adversarial_examples': {
            'pgd20_path': pgd_path,
            'autoattack_path': aa_path
        }
    }

    with open(os.path.join(save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "=" * 60)
    print("論文 Table III 格式結果")
    print("=" * 60)
    print(f"\nDER-AT (Best)  | Clean: {clean_acc_best:.2f} | PGD-20: {pgd_acc_best:.2f} | AA: {aa_acc_best:.2f}")
    print(f"DER-AT (Final) | Clean: {clean_acc_last:.2f} | PGD-20: {pgd_acc_last:.2f} | AA: {aa_acc_last:.2f}")

    print(f"\n✓ 所有結果已儲存至 {save_dir}/")
    print(f"✓ 對抗樣本已儲存至 {adv_dir}/")


if __name__ == '__main__':
    main()
