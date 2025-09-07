import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from matplotlib import pyplot as plt
from tqdm import tqdm
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score

def build_art_classifier(model, device):
    """建立ART分類器"""
    loss_fn = nn.CrossEntropyLoss()
    art_model = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        input_shape=(1, 28, 28),
        nb_classes=10,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )
    return art_model

def generate_adversarial_samples(art_clf, data_loader, attack_types=['fgsm'], max_samples=1500):
    """生成對抗樣本"""
    # 先收集所有資料並隨機打亂
    all_data = []
    for batch_x, batch_y in data_loader:
        for i in range(len(batch_x)):
            all_data.append((batch_x[i:i + 1].numpy(), batch_y[i:i + 1].numpy()))

    # 隨機打亂並選取樣本
    np.random.shuffle(all_data)
    selected_data = all_data[:max_samples]

    # 重新組織成批次
    batch_size = 256
    all_batches = []
    for i in range(0, len(selected_data), batch_size):
        batch_data = selected_data[i:i + batch_size]
        x_batch = np.concatenate([x for x, y in batch_data], axis=0)
        y_batch = np.concatenate([y for x, y in batch_data], axis=0)
        all_batches.append((x_batch, y_batch))

    # 設定攻擊參數
    attacks = {}
    attack_params = {}

    if 'fgsm' in attack_types:
        eps = np.random.uniform(0.25, 0.45)
        # eps = 0.1
        attack_params['fgsm'] = {'eps': eps}
        attacks['fgsm'] = FastGradientMethod(estimator=art_clf, eps=eps)

    if 'pgd' in attack_types:
        eps = np.random.uniform(0.2, 0.4)
        max_iter = np.random.randint(80, 150)
        eps_step = eps / max_iter
        attack_params['pgd'] = {'eps': eps, 'max_iter': max_iter}
        attacks['pgd'] = ProjectedGradientDescent(
            estimator=art_clf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter
        )

    if 'deepfool' in attack_types:
        max_iter = np.random.randint(8, 15)
        epsilon = np.random.uniform(0.02, 0.08)
        attack_params['deepfool'] = {'max_iter': max_iter, 'epsilon': epsilon}
        attacks['deepfool'] = DeepFool(
            classifier=art_clf,
            max_iter=max_iter,
            epsilon=epsilon
        )

    results = {}

    # 儲存乾淨樣本
    results['clean'] = {'x': [], 'y': []}
    for x_np, y_np in all_batches:
        results['clean']['x'].append(x_np)
        results['clean']['y'].append(y_np)

    # 對每種攻擊類型產生對抗樣本
    for attack_name, attack in attacks.items():
        results[attack_name] = {'x': [], 'y': []}

        for i, (x_np, y_np) in enumerate(tqdm(all_batches, desc=f"{attack_name.upper()}", leave=False)):
            try:
                x_adv = attack.generate(x=x_np)
                results[attack_name]['x'].append(x_adv)
                results[attack_name]['y'].append(y_np)
            except Exception as e:
                results[attack_name]['x'].append(x_np)
                results[attack_name]['y'].append(y_np)

    # 合併所有批次
    for key in results:
        if results[key]['x']:
            results[key]['x'] = np.concatenate(results[key]['x'], axis=0)
            results[key]['y'] = np.concatenate(results[key]['y'], axis=0)

        # # 最後統一繪製所有攻擊的分布圖
        # try:
        #     plot_attack_distribution(results, attack_types, attack_params)
        # except Exception as e:
        #     print(f"Visualization error: {e}")

    return results, attack_params

def get_predictions(model, X, device, batch_size=256):
    """取得 softmax 機率向量"""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            logits = model(batch)
            p = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds, axis=0)

def evaluate_attack_effectiveness(results):
    """評估攻擊效果"""
    attack_effectiveness = {}

    # 獲取乾淨樣本的預測結果
    clean_predictions = results['clean']['predictions']
    clean_labels = results['clean']['labels']

    # 計算乾淨樣本的準確率
    clean_pred_classes = np.argmax(clean_predictions, axis=1)
    clean_accuracy = accuracy_score(clean_labels, clean_pred_classes)

    for attack_name, data in results.items():
        if attack_name == 'clean':
            continue

        # 獲取對抗樣本的預測結果
        adv_predictions = data['predictions']
        adv_labels = data['labels']

        # 計算對抗樣本的準確率
        adv_pred_classes = np.argmax(adv_predictions, axis=1)
        adv_accuracy = accuracy_score(adv_labels, adv_pred_classes)

        # 計算攻擊成功率（模型預測錯誤的比例）
        attack_success_rate = 1 - adv_accuracy

        attack_effectiveness[attack_name] = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': attack_success_rate
        }

    return attack_effectiveness


def plot_attack_distribution(results, attack_types, attack_params):
    """繪製所有攻擊與乾淨樣本的分布圖"""
    print(f"\nVisualizing all attacks vs Clean distribution...")

    # 準備數據
    np.random.seed(42)
    n_samples = 600  # 每類樣本數量

    all_images = []
    all_labels = []

    # 設定顏色和標記
    color_map = {
        'clean': 'blue',
        'fgsm': 'red',
        'pgd': 'green',
        'deepfool': 'orange'
    }

    marker_map = {
        'clean': 'o',
        'fgsm': '^',
        'pgd': 's',
        'deepfool': 'D'
    }

    # 收集乾淨樣本
    clean_images = results['clean']['x']
    if len(clean_images) > n_samples:
        clean_indices = np.random.choice(len(clean_images), n_samples, replace=False)
        clean_selected = clean_images[clean_indices]
    else:
        clean_selected = clean_images

    all_images.append(clean_selected)
    all_labels.extend(['Clean'] * len(clean_selected))

    # 收集攻擊樣本
    for attack_name in attack_types:
        if attack_name in results:
            attack_images = results[attack_name]['x']
            if len(attack_images) > n_samples:
                attack_indices = np.random.choice(len(attack_images), n_samples, replace=False)
                attack_selected = attack_images[attack_indices]
            else:
                attack_selected = attack_images

            all_images.append(attack_selected)
            all_labels.extend([attack_name.upper()] * len(attack_selected))

    # 合併所有圖像
    all_images = np.concatenate(all_images, axis=0)

    # 將圖像展平用於UMAP
    print("  Preparing data for UMAP...")
    flattened_images = all_images.reshape(len(all_images), -1)

    # 執行UMAP降維
    print("  Performing UMAP reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(flattened_images)

    # 繪製圖形
    plt.figure(figsize=(14, 10))

    # 繪製每類樣本
    unique_labels = list(set(all_labels))
    centers = {}

    for label in unique_labels:
        mask = np.array(all_labels) == label
        if np.any(mask):
            label_lower = label.lower()
            color = color_map.get(label_lower, 'black')
            marker = marker_map.get(label_lower, 'x')
            alpha = 0.6 if label == 'Clean' else 0.8
            size = 20 if label == 'Clean' else 25

            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                        c=color,
                        marker=marker,
                        label=label,
                        alpha=alpha,
                        s=size,
                        edgecolors='black' if label != 'Clean' else 'none',
                        linewidth=0.5)

            # 計算並標記中心點
            center = np.mean(embedding[mask], axis=0)
            centers[label] = center

            if label == 'Clean':
                plt.plot(center[0], center[1], 'bs', markersize=12,
                         markeredgecolor='black', markeredgewidth=2)
            else:
                plt.plot(center[0], center[1], '*', color=color, markersize=18,
                         markeredgecolor='black', markeredgewidth=1)

    # 格式化參數說明
    param_texts = []
    for attack_name in attack_types:
        if attack_name in attack_params:
            if attack_name == 'fgsm':
                param_texts.append(f"FGSM: ε = {attack_params[attack_name]['eps']:.3f}")
            elif attack_name == 'pgd':
                param_texts.append(
                    f"PGD: ε = {attack_params[attack_name]['eps']:.3f}, iter = {attack_params[attack_name]['max_iter']}")
            elif attack_name == 'deepfool':
                param_texts.append(
                    f"DeepFool: max_iter = {attack_params[attack_name]['max_iter']}, ε = {attack_params[attack_name]['epsilon']:.3f}")

    param_text = "\n".join(param_texts)

    plt.title(f'Adversarial Attacks vs Clean Samples Distribution (UMAP)\n{param_text}',
              fontsize=14, pad=20)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # 計算並顯示距離信息
    if 'Clean' in centers:
        clean_center = centers['Clean']
        distances = {}

        for label, center in centers.items():
            if label != 'Clean':
                distance = np.linalg.norm(clean_center - center)
                distances[label] = distance

                # 添加距離線
                plt.plot([clean_center[0], center[0]], [clean_center[1], center[1]],
                         '--', alpha=0.4, linewidth=1)

        # 在圖上添加距離信息
        distance_text = "Distances from Clean:\n"
        for label, dist in distances.items():
            distance_text += f"{label}: {dist:.2f}\n"

        plt.text(0.02, 0.98, distance_text, transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                 verticalalignment='top', fontsize=10)

    plt.tight_layout()

    # 保存圖片
    save_path = 'all_attacks_vs_clean_umap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"  Plot saved to: {save_path}")

