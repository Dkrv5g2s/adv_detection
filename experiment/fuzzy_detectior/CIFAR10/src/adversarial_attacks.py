import os

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
        input_shape=(3, 32, 32),
        nb_classes=10,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device_type='gpu' if device.type == 'cuda' else 'cpu'
    )
    return art_model


def generate_adversarial_samples(art_clf, data_loader, attack_types=['fgsm'], max_samples=1500, model=None,
                                 device=None):
    """生成對抗樣本並確保所有數據都在[-1,1]範圍內"""
    print("Generating adversarial samples...")

    # 先收集所有資料並隨機打亂
    all_data = []
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.cpu().numpy()
        batch_y = batch_y.cpu().numpy()

        for i in range(len(batch_x)):
            all_data.append((batch_x[i:i + 1], batch_y[i:i + 1]))

    # 隨機打亂並選取樣本
    np.random.shuffle(all_data)
    selected_data = all_data[:max_samples]

    # 重新組織成批次
    batch_size = 128
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
        eps = 0.08#np.random.uniform(0.03, 0.1)
        attack_params['fgsm'] = {'eps': eps}
        attacks['fgsm'] = FastGradientMethod(estimator=art_clf, eps=eps)
        print(f"FGSM epsilon: {eps}")

    if 'pgd' in attack_types:
        eps = 0.08#np.random.uniform(0.03, 0.1)
        max_iter = np.random.randint(10, 40)
        eps_step =0.03# np.random.uniform(0.01, 0.02)
        attack_params['pgd'] = {'eps': eps,'eps_step':eps_step, 'max_iter': max_iter}
        attacks['pgd'] = ProjectedGradientDescent(
            estimator=art_clf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter
        )
        print(f"PGD epsilon: {eps}, iterations: {max_iter}")

    if 'deepfool' in attack_types:
        max_iter = 100
        nb_grads = 5
        eps = 0.008

        attack_params['deepfool'] = {'max_iter': max_iter,'eps': eps,'nb_grads': nb_grads,}

        attacks['deepfool'] = DeepFool(
            classifier=art_clf,
            max_iter=max_iter,
            epsilon=eps,
            nb_grads=nb_grads,
        )

    results = {}

    # 儲存乾淨樣本
    results['clean'] = {'x': [], 'y': []}
    for x_np, y_np in all_batches:
        results['clean']['x'].append(x_np)
        results['clean']['y'].append(y_np)

    # 對每種攻擊類型產生對抗樣本
    for attack_name, attack in attacks.items():
        print(f"\n Generating {attack_name.upper()} attacks...")
        results[attack_name] = {'x': [], 'y': []}

        for i, (x_np, y_np) in enumerate(tqdm(all_batches, desc=f"{attack_name.upper()}", leave=False)):
            try:
                # 檢查原始預測
                original_preds = art_clf.predict(x_np)
                original_classes = np.argmax(original_preds, axis=1)

                # 生成對抗樣本
                x_adv = attack.generate(x=x_np)

                x_adv = np.clip(x_adv, -1, 1)


                # 檢查攻擊後的預測
                adv_preds = art_clf.predict(x_adv)
                adv_classes = np.argmax(adv_preds, axis=1)

                # 計算攻擊成功率
                success_rate = np.mean(original_classes != adv_classes)

                print(f"  Batch {i}: Attack success rate: {success_rate:.2%}")
                print(f"  Original classes: {original_classes[:5]}")
                print(f"  Adversarial classes: {adv_classes[:5]}")
                print(f"  Adversarial data range after clipping: [{x_adv.min():.3f}, {x_adv.max():.3f}]")

                results[attack_name]['x'].append(x_adv)
                results[attack_name]['y'].append(y_np)

            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                results[attack_name]['x'].append(x_np)
                results[attack_name]['y'].append(y_np)

    # 合併所有批次並確保數據一致性
    for key in results:
        if results[key]['x']:
            shapes = [arr.shape for arr in results[key]['x']]
            print(f"{key} batch shapes: {shapes[:3]}...")

            try:
                results[key]['x'] = np.concatenate(results[key]['x'], axis=0)
                results[key]['y'] = np.concatenate(results[key]['y'], axis=0)

                print(f"{key} final shape: {results[key]['x'].shape}")
                print(f"{key} final range: [{results[key]['x'].min():.3f}, {results[key]['x'].max():.3f}]")

            except Exception as e:
                print(f"Error concatenating {key}: {e}")
                fixed_x = []
                for arr in results[key]['x']:
                    if len(arr.shape) == 4:
                        fixed_x.append(arr)
                    else:
                        print(f"Skipping malformed array with shape: {arr.shape}")

                if fixed_x:
                    results[key]['x'] = np.concatenate(fixed_x, axis=0)
                    results[key]['y'] = np.concatenate(results[key]['y'], axis=0)



    # 繪製攻擊對比圖像
    try:
        save_attack_comparison_images(results, attack_types, save_dir='attack_comparison',
                                      model=model, device=device)
    except Exception as e:
        print(f"Image comparison error: {e}")
        print("Continuing without image comparison...")

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


def save_attack_comparison_images(results, attack_types, save_dir='attack_comparison', model=None, device=None):


    def prepare_cifar10_for_display(img):
        """準備CIFAR-10圖像用於matplotlib顯示"""
        # 轉換範圍：[-1, 1] → [0, 1]
        img_show = (img + 1) / 2

        # CIFAR-10維度轉換：(3, 32, 32) → (32, 32, 3)
        if len(img_show.shape) == 3 and img_show.shape[0] == 3:
            img_show = img_show.transpose(1, 2, 0)

        return img_show

    print("Saving attack comparison images...")
    os.makedirs(save_dir, exist_ok=True)

    # CIFAR-10類別名稱
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    for attack_type in attack_types:
        if attack_type not in results:
            print(f"Skipping {attack_type} - not found in results")
            continue

        try:
            print(f"Processing {attack_type}...")

            # 獲取數據
            orig_data = results['clean']['x']
            attack_data = results[attack_type]['x']
            labels = results['clean']['y']

            # 計算預測結果
            if model is not None and device is not None:
                print("Getting model predictions...")
                model.eval()

                with torch.no_grad():
                    n_samples = min(20, len(orig_data))

                    # 原始圖像預測
                    orig_tensor = torch.from_numpy(orig_data[:n_samples]).float().to(device)
                    orig_logits = model(orig_tensor)
                    orig_preds = torch.argmax(orig_logits, dim=1).cpu().numpy()

                    # 攻擊後圖像預測
                    attack_tensor = torch.from_numpy(attack_data[:n_samples]).float().to(device)
                    attack_logits = model(attack_tensor)
                    attack_preds = torch.argmax(attack_logits, dim=1).cpu().numpy()
            else:
                n_samples = min(20, len(orig_data))
                # 如果沒有模型，使用真實標籤作為預測
                if hasattr(labels, '__len__'):
                    orig_preds = labels[:n_samples]
                    attack_preds = labels[:n_samples]
                else:
                    orig_preds = [labels] * n_samples
                    attack_preds = [labels] * n_samples
                print("No model provided, using labels as predictions")

            # 找出攻擊成功的樣本（預測結果改變）
            successful_attacks = []
            for i in range(n_samples):
                if orig_preds[i] != attack_preds[i]:
                    successful_attacks.append(i)
                if len(successful_attacks) >= 5:  # 最多顯示5個成功案例
                    break

            # 如果沒有成功攻擊，就顯示前5個樣本
            if not successful_attacks:
                print(f"No successful attacks found for {attack_type}, showing first 5 samples")
                successful_attacks = list(range(min(5, n_samples)))

            # 準備顯示
            n_display = min(5, len(successful_attacks))
            selected_indices = successful_attacks[:n_display]

            # 創建子圖
            fig, axes = plt.subplots(2, n_display, figsize=(4 * n_display, 8))
            if n_display == 1:
                axes = axes.reshape(2, 1)

            for j, i in enumerate(selected_indices):
                # 準備圖像顯示
                orig_img_show = prepare_cifar10_for_display(orig_data[i])
                attack_img_show = prepare_cifar10_for_display(attack_data[i])

                # 獲取標籤和預測
                if hasattr(labels, '__getitem__') and len(labels) > i:
                    true_label = labels[i]
                else:
                    true_label = labels

                orig_pred = orig_preds[i]
                attack_pred = attack_preds[i]

                # 顯示原始圖像
                axes[0, j].imshow(orig_img_show, interpolation='nearest')
                axes[0, j].set_title(
                    f'Original\nTrue: {classes[true_label]}\nPred: {classes[orig_pred]}',
                    fontsize=11
                )
                axes[0, j].axis('off')

                # 顯示攻擊後圖像
                attack_status = "Success" if orig_pred != attack_pred else "Failed"
                axes[1, j].imshow(attack_img_show, interpolation='nearest')
                axes[1, j].set_title(
                    f'{attack_type.upper()} ({attack_status})\nPred: {classes[attack_pred]}',
                    fontsize=11
                )
                axes[1, j].axis('off')

                # 添加邊框顏色表示攻擊結果
                color = 'red' if orig_pred != attack_pred else 'green'
                for spine in axes[1, j].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)

            plt.tight_layout()

            # 保存圖像
            save_path = f'{save_dir}/{attack_type}_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved {attack_type} comparison to {save_path}")

        except Exception as e:
            print(f"Error processing {attack_type}: {e}")
            import traceback
            traceback.print_exc()





