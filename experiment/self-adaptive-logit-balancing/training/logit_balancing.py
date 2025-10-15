"""
Logit Balancing 訓練實現 (Algorithm 1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LogitBalancingTrainer:
    def __init__(self, model, device, beta=0.02, sigma=24 / 255, lr=0.02):
        self.model = model
        self.device = device
        self.beta = beta
        self.sigma = sigma
        self.lr = lr

        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader, epoch):
        """訓練一個 epoch"""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 添加高斯噪聲
            noise = torch.randn_like(inputs) * self.sigma
            inputs_noisy = torch.clamp(inputs + noise, 0, 1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs_noisy)

            # Logit Balancing Loss (Algorithm 1)
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
        Logit Balancing Loss (Algorithm 1)
        - 如果預測正確：使用 Logit Balancing Loss
        - 如果預測錯誤：使用 Cross-Entropy Loss
        """
        batch_size = outputs.size(0)
        log_softmax = F.log_softmax(outputs, dim=1)
        softmax = F.softmax(outputs, dim=1)

        _, predicted = outputs.max(1)
        correct_mask = predicted.eq(targets)

        loss = torch.zeros(batch_size, device=outputs.device)

        # 預測正確：Logit Balancing Loss
        if correct_mask.any():
            correct_indices = correct_mask.nonzero(as_tuple=True)[0]
            for idx in correct_indices:
                target_class = targets[idx]

                # 排除目標類別的 log-softmax
                log_softmax_without_target = torch.cat([
                    log_softmax[idx, :target_class],
                    log_softmax[idx, target_class + 1:]
                ])

                # 計算標準差
                std_dev = torch.std(log_softmax_without_target)

                # Logit Balancing Loss
                loss[idx] = self.beta * std_dev * softmax[idx, target_class]

        # 預測錯誤：Cross-Entropy Loss
        if (~correct_mask).any():
            incorrect_indices = (~correct_mask).nonzero(as_tuple=True)[0]
            loss[incorrect_indices] = -log_softmax[incorrect_indices, targets[incorrect_indices]]

        return loss.mean()

    def train(self, train_loader, epochs=100):
        """完整訓練流程"""
        print(f"\n{'=' * 70}")
        print("Training with Logit Balancing (Algorithm 1)")
        print(f"{'=' * 70}")
        print(f"Beta (β): {self.beta}")
        print(f"Sigma (σ): {self.sigma:.4f}")
        print(f"Learning Rate: {self.lr}")
        print(f"Epochs: {epochs}")
        print(f"{'=' * 70}\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)

            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            print(f"Epoch {epoch + 1} Summary: Loss={train_loss:.3f}, Acc={train_acc:.2f}%")

            if (epoch + 1) % 10 == 0:
                print(f"\n[Checkpoint] Epoch {epoch + 1} completed")

        print(f"\n{'=' * 70}")
        print("Training Completed!")
        print(f"{'=' * 70}\n")

        return self.model
