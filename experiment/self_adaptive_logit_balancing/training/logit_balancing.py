"""
Logit Balancing 訓練實現 (Algorithm 1) - 修正版
根據論文完整實現，包含所有關鍵修正
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LogitBalancingTrainer:
    def __init__(self, model, device, beta=0.02, sigma=24 / 255, lr=0.02):
        """
        初始化 Logit Balancing 訓練器

        Args:
            model: 神經網路模型
            device: 訓練設備 (cuda/cpu)
            beta: Logit Balancing 權重 (論文建議: 0.02)
            sigma: 高斯噪聲標準差 (論文建議: 24/255)
            lr: 學習率 (論文建議: 0.02)
        """
        self.model = model
        self.device = device
        self.beta = beta
        self.sigma = sigma
        self.lr = lr

        # 使用 Adam
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr
        )

        # Cosine Annealing 學習率調度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def train_epoch(self, train_loader, epoch):
        """
        訓練一個 epoch

        Args:
            train_loader: 訓練數據加載器
            epoch: 當前 epoch 編號

        Returns:
            train_loss: 平均訓練損失
            train_acc: 訓練準確率
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 每次都生成新的高斯噪聲（論文關鍵：每個 epoch 不同噪聲）
            noise = torch.randn_like(inputs) * self.sigma
            inputs_noisy = torch.clamp(inputs + noise, 0, 1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs_noisy)

            # Logit Balancing Loss (Algorithm 1 + Eq. 7)
            loss = self._logit_balancing_loss(outputs, targets)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {train_loss / (batch_idx + 1):.3f}, '
                      f'Acc: {100. * correct / total:.2f}%')

        self.scheduler.step()

        return train_loss / len(train_loader), 100. * correct / total

    def _logit_balancing_loss(self, outputs, targets):
        """
        Logit Balancing Loss (Algorithm 1 + Equation 7)

        核心公式：
        - 預測正確: Loss_LB = β × SD(log(S(f(x)))_{i≠t}) × s_t
        - 預測錯誤: Loss_CE = -log(S(f(x))_t)

        其中：
        - β: logit balancing 權重
        - SD: 標準差
        - s_t: 目標類別的 softmax 值（自適應權重）
        - log(S(f(x)))_{i≠t}: 排除目標類別的 log-softmax 值

        Args:
            outputs: 模型輸出 logits [batch_size, num_classes]
            targets: 目標標籤 [batch_size]

        Returns:
            loss: 批次平均損失
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)

        # 計算 log-softmax 和 softmax
        log_softmax = F.log_softmax(outputs, dim=1)
        softmax = F.softmax(outputs, dim=1)

        # 判斷預測是否正確
        _, predicted = outputs.max(1)
        correct_mask = predicted.eq(targets)

        loss = torch.zeros(batch_size, device=outputs.device)

        # ========== 預測正確：Logit Balancing Loss (Eq. 7) ==========
        if correct_mask.any():
            correct_indices = correct_mask.nonzero(as_tuple=True)[0]

            for idx in correct_indices:
                target_class = targets[idx].item()

                # 創建 mask 排除目標類別
                mask = torch.ones(num_classes, dtype=torch.bool, device=outputs.device)
                mask[target_class] = False

                # 獲取排除目標類別的 log-softmax 值
                log_softmax_without_target = - log_softmax[idx][mask]

                # 計算標準差 (SD)
                std_dev = torch.std(log_softmax_without_target)

                # 自適應權重：目標類別的 softmax 值 (s_t)
                s_t = softmax[idx, target_class]

                # Logit Balancing Loss (Eq. 7)
                # Loss_LB = β × SD × s_t
                loss[idx] = self.beta * std_dev * s_t

        # ========== 預測錯誤：Cross-Entropy Loss (Eq. 5) ==========
        if (~correct_mask).any():
            incorrect_indices = (~correct_mask).nonzero(as_tuple=True)[0]

            # 使用標準 Cross-Entropy Loss
            # Loss_CE = -log(S(f(x))_t)
            loss[incorrect_indices] = F.cross_entropy(
                outputs[incorrect_indices],
                targets[incorrect_indices],
                reduction='none'
            )

        return loss.mean()

    def train(self, train_loader, epochs=100):
        """
        完整訓練流程

        Args:
            train_loader: 訓練數據加載器
            epochs: 訓練輪數 (論文建議: 100)

        Returns:
            model: 訓練完成的模型
        """
        print(f"\n{'=' * 70}")
        print("Training with Logit Balancing (Algorithm 1)")
        print(f"{'=' * 70}")
        print(f"Beta (β): {self.beta}")
        print(f"Sigma (σ): {self.sigma:.4f}")
        print(f"Learning Rate: {self.lr}")
        print(f"Optimizer: Adam")
        print(f"Epochs: {epochs}")
        print(f"{'=' * 70}\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)

            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            print(f"Epoch {epoch + 1} Summary: Loss={train_loss:.3f}, Acc={train_acc:.2f}%")

            # 每 10 個 epoch 顯示檢查點
            if (epoch + 1) % 10 == 0:
                print(f"\n[Checkpoint] Epoch {epoch + 1} completed")
                print(f"  Current LR: {self.scheduler.get_last_lr()[0]:.6f}")

        print(f"\n{'=' * 70}")
        print("Training Completed!")
        print(f"{'=' * 70}\n")

        return self.model