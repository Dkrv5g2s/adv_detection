"""
所有對抗攻擊生成器（帶緩存功能）
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn.functional as F
import numpy as np
from art.attacks.evasion import (
    ProjectedGradientDescent,
    AutoProjectedGradientDescent,
    SquareAttack,
    CarliniL2Method
)
from art.estimators.classification import PyTorchClassifier

from self_adaptive_logit_balancing.attacks.fab_attack import AutoAttackFABWrapper

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
        self.model = model
        self.device = device
        self.use_cache = use_cache

        # 創建 ART classifier
        self.art_classifier = PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(3, 32, 32),
            nb_classes=10,
            clip_values=(0, 1)
        )

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
        attack = ProjectedGradientDescent(
            estimator=self.art_classifier,
            eps=8 / 255,
            eps_step=2 / 255,
            max_iter=20,
            norm=np.inf
        )
        return self._generate_with_art(attack, dataloader, num_samples)

    def generate_pgd_l2(self, dataloader, num_samples=1000):
        """生成 PGD-L2 對抗樣本"""
        attack = ProjectedGradientDescent(
            estimator=self.art_classifier,
            eps=1.0,
            eps_step=0.2,
            max_iter=40,
            norm=2
        )
        return self._generate_with_art(attack, dataloader, num_samples)

    def generate_apgd_linf(self, dataloader, num_samples=1000):
        """
        生成 APGD-L∞ 對抗樣本 (Table 1 第二行)
        eps = 8/255, alpha = 2/255, steps = 40, sampling = 10
        """
        attack = AutoProjectedGradientDescent(
            estimator=self.art_classifier,
            norm=np.inf,
            eps=8 / 255,
            eps_step=2 / 255,
            max_iter=40,
            nb_random_init=10,
            loss_type='cross_entropy',
            verbose=False
        )
        return self._generate_with_art(attack, dataloader, num_samples)

    def generate_apgdt_linf(self, dataloader, num_samples=1000):
        """
        生成 APGDT-L∞ 對抗樣本 (Table 1 第三行)
        eps = 8/255, steps = 100, restarts = 1, eot_iter = 1, rho = 0.75
        注意: eot_iter 和 rho 在 ART 源碼中已實現
        """
        attack = AutoProjectedGradientDescent(
            estimator=self.art_classifier,
            norm=np.inf,
            eps=8 / 255,
            eps_step=2 / 255,
            max_iter=40,
            loss_type='difference_logits_ratio',
            verbose=False
        )
        return self._generate_with_art(attack, dataloader, num_samples)

    def generate_square_linf(self, dataloader, num_samples=1000):
        """生成 Square Attack 對抗樣本"""
        attack = SquareAttack(
            estimator=self.art_classifier,
            eps=8 / 255,
            max_iter=5000,
            nb_restarts=1,
            p_init=0.85,
            norm=np.inf
        )
        return self._generate_with_art(attack, dataloader, num_samples)

    def generate_fab_linf(self, dataloader, num_samples=1000):
        """
        生成 FAB-L∞ 對抗樣本 (Table 1 第五行)
        eps = 8/255, alpha = 0.1, steps = 100, restart = 1
        """
        attack = AutoAttackFABWrapper(
            model=self.model,
            device=self.device,
            eps=8 / 255,
            steps=100,
            n_restarts=1,
            alpha_max=0.1
        )
        return self._generate_with_custom(attack, dataloader, num_samples)

    def generate_cw_l2(self, dataloader, num_samples=1000):
        """生成 CW-L2 對抗樣本"""
        # attack = CarliniL2Method(
        #     classifier=self.art_classifier,
        #     initial_const=1.0,
        #     confidence=0.0,
        #     max_iter=1000,
        #     learning_rate=0.01
        # )
        attack = CarliniL2Method(
            classifier=self.art_classifier,
            initial_const=1,
            confidence=0.0,
            max_iter=100,  # 比論文少很多
            learning_rate=0.01,
            batch_size=16,
            binary_search_steps=3
        )
        return self._generate_with_art(attack, dataloader, num_samples)

    def _generate_with_art(self, attack, dataloader, num_samples):
        """使用 ART 攻擊生成對抗樣本"""
        images_list, labels_list = [], []

        for images, labels in dataloader:
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            if len(np.concatenate(images_list)) >= num_samples:
                break

        images = np.concatenate(images_list)[:num_samples]
        labels = np.concatenate(labels_list)[:num_samples]

        print(f"  Generating adversarial examples...")
        adv_images = attack.generate(x=images)

        return {'images': adv_images, 'labels': labels}

    def _generate_with_custom(self, attack, dataloader, num_samples):
        """使用自定義攻擊生成對抗樣本（與 ART 風格統一）"""
        images_list, labels_list, adv_images_list = [], [], []

        for images, labels in dataloader:
            images_np = images.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # 生成對抗樣本
            adv_images_np = attack.generate(x=images_np, y=labels_np)

            images_list.append(images_np)
            labels_list.append(labels_np)
            adv_images_list.append(adv_images_np)

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

        for attack_name, attack_func in attacks.items():
            print(f"\n[INFO] Processing {attack_name} samples...")

            # 獲取緩存路徑
            if use_cache:
                cache_path = Config.get_cache_path(attack_name, num_samples)

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
