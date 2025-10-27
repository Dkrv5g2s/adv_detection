import os
import torch
import numpy as np
import torchattacks
from self_adaptive_logit_balancing.utils.helpers import (
    save_adversarial_data,
    load_adversarial_data,
    check_cache_exists,
)
from self_adaptive_logit_balancing.config import Config

class AttackGenerator:
    def __init__(self, model, device, use_cache=True):
        """
        初始化攻擊生成器

        Args:
            model: 目標模型
            device: 計算設備
            use_cache: 是否使用緩存
        """
        self.model = model.to(device)
        self.device = device
        self.use_cache = use_cache

    def generate_clean(self, dataloader, num_samples=1000):
        """生成乾淨樣本"""
        images_list, labels_list = [], []

        for images, labels in dataloader:
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            if len(np.concatenate(images_list)) >= num_samples:
                break

        images = np.concatenate(images_list)[:num_samples]
        labels = np.concatenate(labels_list)[:num_samples]

        return {'images': images, 'labels': labels}

    def generate_pgd_linf(self, dataloader, num_samples=1000):
        """生成 PGD-L∞ 對抗樣本"""
        attack = torchattacks.PGD(
            self.model, eps=8 / 255, alpha=2 / 255, steps=20,random_start=True
        )
        return self._generate_with_torchattack(attack, dataloader, num_samples)

    def generate_pgd_l2(self, dataloader, num_samples=1000):
        """生成 PGD-L2 對抗樣本"""
        attack = torchattacks.PGDL2(
            self.model, eps=1.0, alpha=0.2, steps=40,random_start=True
        )
        return self._generate_with_torchattack(attack, dataloader, num_samples)

    def generate_apgd_linf(self, dataloader, num_samples=1000):
        """生成 APGD-L∞ 對抗樣本"""
        attack = torchattacks.APGD(
            self.model, eps=8 / 255, steps=40, loss='ce'
        )
        return self._generate_with_torchattack(attack, dataloader, num_samples)

    def generate_apgdt_linf(self, dataloader, num_samples=1000):
        """生成 APGDT-L∞ 對抗樣本"""
        attack = torchattacks.APGD(
            self.model, eps=8 / 255, steps=100,n_restarts=1,eot_iter=1,rho=0.75, loss='dlr'
        )
        return self._generate_with_torchattack(attack, dataloader, num_samples)

    def generate_square_linf(self, dataloader, num_samples=1000):
        """生成 Square Attack 對抗樣本"""
        attack = torchattacks.Square(
            self.model, eps=8 / 255, p_init=0.8, n_queries=5000, n_restarts=1
        )

        return self._generate_with_torchattack(attack, dataloader, num_samples)

    def generate_fab_linf(self, dataloader, num_samples=1000):
        """生成 FAB-L∞ 對抗樣本"""
        attack = torchattacks.FAB(
            self.model, eps=8 / 255, alpha_max=0.1, steps=100, n_restarts=1
        )
        return self._generate_with_torchattack(attack, dataloader, num_samples)

    def generate_cw_l2(self, dataloader, num_samples=1000):
        """生成 CW-L2 對抗樣本"""
        attack = torchattacks.CW(
            self.model, c=1.0,kappa=0, steps=1000, lr=0.01
        )
        return self._generate_with_torchattack(attack, dataloader, num_samples)

    def _generate_with_torchattack(self, attack, dataloader, num_samples):
        """使用 torchattacks 攻擊生成對抗樣本"""
        images_list, labels_list, adv_images_list = [], [], []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            # 生成對抗樣本
            adv_images = attack(images, labels)

            # 修正：使用 .detach() 分離張量
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            adv_images_list.append(adv_images.detach().cpu().numpy())

            if len(np.concatenate(images_list)) >= num_samples:
                break

        images = np.concatenate(images_list)[:num_samples]
        labels = np.concatenate(labels_list)[:num_samples]
        adv_images = np.concatenate(adv_images_list)[:num_samples]

        return {'images': adv_images, 'labels': labels}

    def generate_all_attacks(self, dataloader, num_samples=1000,
                             cache_dir=None, force_regenerate=False):
        """
        生成所有類型的對抗樣本（帶緩存功能）

        Args:
            dataloader: 數據加載器
            num_samples: 樣本數量
            cache_dir: 緩存目錄（如果為 None 則不使用緩存）
            force_regenerate: 是否強制重新生成（忽略緩存）

        Returns:
            adversarial_data: dict，包含所有攻擊類型的數據
        """
        attacks = {
            'Clean': self.generate_clean,
            'PGD-Linf': self.generate_pgd_linf,
            'PGD-L2': self.generate_pgd_l2,
            'APGD-Linf': self.generate_apgd_linf,
            'Square-Linf': self.generate_square_linf,
            'APGDT-Linf': self.generate_apgdt_linf,
            'FAB-Linf': self.generate_fab_linf,
            'CW-L2': self.generate_cw_l2
        }

        adversarial_data = {}
        use_cache = self.use_cache and cache_dir is not None and not force_regenerate
        model_name = os.path.splitext(os.path.basename(Config.SOURCE_MODEL_SAVE_PATH))[0]

        for attack_name, attack_func in attacks.items():
            print(f"\n[INFO] Processing {attack_name} samples...")

            # 獲取緩存路徑
            if use_cache:
                cache_path = Config.get_cache_path(attack_name, num_samples, model_name)

                # 嘗試從緩存載入
                if check_cache_exists(cache_path):
                    print(f"  → Cache found, attempting to load...")
                    cached_data = load_adversarial_data(cache_path, attack_name)

                    if cached_data is not None:
                        adversarial_data[attack_name] = cached_data
                        print(f"  ✓ {attack_name}: Loaded from cache")
                        continue
                    else:
                        print(f"  → Cache corrupted, regenerating...")

            # 生成新的對抗樣本
            print(f"  → Generating new samples...")
            adversarial_data[attack_name] = attack_func(dataloader, num_samples)
            adversarial_data[attack_name]['images'] = np.clip(
                adversarial_data[attack_name]['images'], 0.0, 1.0
            )
            print(f"  ✓ {attack_name}: {adversarial_data[attack_name]['images'].shape}")

            # 保存到緩存
            if use_cache:
                save_adversarial_data(
                    adversarial_data[attack_name],
                    cache_path,
                    attack_name
                )

        return adversarial_data