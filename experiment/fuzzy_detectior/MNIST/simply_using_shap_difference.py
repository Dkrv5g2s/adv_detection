import os
import time
import random
import numpy as np
import torch
import warnings
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from fuzzy_detectior.MNIST.src.shap_signature import generate_shap_signatures
from fuzzy_detectior.MNIST.src.models import SimpleCNN
from fuzzy_detectior.MNIST.src.data_utils import load_mnist
from fuzzy_detectior.MNIST.src.model_training import train_classifier, eval_classifier
from fuzzy_detectior.MNIST.src.adversarial_attacks import (
    build_art_classifier,
    generate_adversarial_samples,
    get_predictions,
    evaluate_attack_effectiveness
)
from fuzzy_detectior.MNIST.src.feature_extraction import extract_feature_differences


warnings.filterwarnings('ignore')

ATTACK_TYPES = ['fgsm','pgd','deepfool']
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 500

MODEL_PATH = "src/simple_mnist_cnn.pth"
TRAINING_EPOCHS = 5
BATCH_SIZE = 256


def generate_seed():
    return int(time.time()) % 10000


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_detector_data(model, clean_data, adv_data, attack_type, device):
    """使用SHAP簽名準備偵測器資料"""
    print(f"[{attack_type}] Preparing SHAP detector data...")

    # 確保樣本數量一致
    min_samples = min(len(clean_data['images']), len(adv_data['images']))
    clean_images = clean_data['images'][:min_samples]
    adv_images = adv_data['images'][:min_samples]

    # 為clean圖片添加微小噪音
    # noise_std = np.random.uniform(0.01, 0.05)
    clean_images_noisy = clean_images #+ np.random.normal(0, noise_std, clean_images.shape)


    # 生成SHAP簽名
    print(f"[{attack_type}] Generating clean SHAP signatures...")
    clean_signatures = generate_shap_signatures(model, clean_images, device)

    print(f"[{attack_type}] Generating noisy clean SHAP signatures...")
    clean_signatures_noisy = generate_shap_signatures(model, clean_images_noisy, device)

    print(f"[{attack_type}] Generating adversarial SHAP signatures...")
    adv_signatures = generate_shap_signatures(model, adv_images, device)

    # 計算特徵差異
    clean_features_diff = extract_feature_differences(clean_signatures, clean_signatures_noisy)
    adv_features_diff = extract_feature_differences(clean_signatures, adv_signatures)

    # 調試資訊
    print(f"[{attack_type}] Clean SHAP diff stats - mean: {clean_features_diff.mean():.4f}, std: {clean_features_diff.std():.4f}")
    print(f"[{attack_type}] Adv SHAP diff stats - mean: {adv_features_diff.mean():.4f}, std: {adv_features_diff.std():.4f}")

    # 檢查差異是否合理
    ratio = adv_features_diff.mean() / (clean_features_diff.mean() + 1e-8)
    print(f"[{attack_type}] SHAP Adversarial/Clean ratio: {ratio:.2f}")

    if ratio < 2.0:
        print(f"[{attack_type}] Warning: SHAP adversarial differences may be too small!")

    return clean_features_diff, adv_features_diff


def find_optimal_threshold(clean_diffs, adv_diffs, attack_type):
    """找出最佳閾值和權重"""
    print(f"[{attack_type}] Finding optimal threshold...")

    # 嘗試不同的權重組合
    weight_methods = {
        '1': np.array([0.2, 0.3, 0.3, 0.2]),
        '2': np.array([0.25, 0.25, 0.25,0.25])
    }

    best_f1 = 0
    best_weights = None
    best_threshold = None
    best_method = None

    for method_name, weights in weight_methods.items():
        # 計算加權分數
        clean_scores = np.dot(clean_diffs, weights)
        adv_scores = np.dot(adv_diffs, weights)

        # 準備標籤
        all_scores = np.concatenate([clean_scores, adv_scores])
        all_labels = np.concatenate([
            np.zeros(len(clean_scores)),  # 0 = clean
            np.ones(len(adv_scores))  # 1 = adversarial
        ])

        # 測試不同閾值
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 100)

        for threshold in thresholds:
            predictions = (all_scores > threshold).astype(int)
            f1 = f1_score(all_labels, predictions)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights
                best_threshold = threshold
                best_method = method_name

    print(f"[{attack_type}] Best method: {best_method}")
    print(f"[{attack_type}] Best weights: {best_weights}")
    print(f"[{attack_type}] Best threshold: {best_threshold:.6f}")
    print(f"[{attack_type}] Training F1: {best_f1:.4f}")

    return best_weights, best_threshold, best_method


def simple_threshold_detection(test_clean_diff, test_adv_diff, weights, threshold, attack_type):
    """直接用閾值進行偵測"""
    print(f"[{attack_type}] Testing simple threshold detection...")

    # 計算測試分數
    clean_scores = np.dot(test_clean_diff, weights)
    adv_scores = np.dot(test_adv_diff, weights)

    # 準備測試資料
    all_scores = np.concatenate([clean_scores, adv_scores])
    all_labels = np.concatenate([
        np.zeros(len(clean_scores)),  # 0 = clean
        np.ones(len(adv_scores))  # 1 = adversarial
    ])

    # 預測
    predictions = (all_scores > threshold).astype(int)

    # 計算機率（距離閾值的距離）
    # sigmoid轉換
    # distances = all_scores - threshold
    # probabilities = 1 / (1 + np.exp(-distances * 5))  

    # min-max
    score_min, score_max = all_scores.min(), all_scores.max()
    if score_max > score_min:
        probabilities = (all_scores - score_min) / (score_max - score_min)
    else:
        probabilities = np.ones_like(all_scores) * 0.5

    # 計算指標
    accuracy = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, probabilities)

    # 混淆矩陣
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"[{attack_type}] Detection Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'threshold': threshold,
        'weights': weights,
        'confusion_matrix': cm,
        'clean_scores': clean_scores,
        'adv_scores': adv_scores
    }


def load_or_train_model(device):
    """載入或訓練分類器模型"""
    model = SimpleCNN()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        print("Loaded existing model.")
    else:
        print("Training new model...")
        train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, shuffle_test=True)
        model = train_classifier(model, train_loader, test_loader, device, epochs=TRAINING_EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)

    return model


def main():
    # 設定隨機種子
    seed = generate_seed()
    set_seed(seed)

    # 設定運算裝置
    device = get_device()
    print("Using device:", device)

    print("=== Direct Threshold-Based Adversarial Detection ===")
    print(f"Attack types: {ATTACK_TYPES}")
    print(f"Training samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}")
    print("Method: 5D features → weighted sum → threshold")

    # 載入資料
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, shuffle_test=True)

    # 訓練或載入分類器
    model = load_or_train_model(device)

    # 評估基礎模型
    base_acc = eval_classifier(model, test_loader, device)
    print(f"Base CNN Test Accuracy: {base_acc:.4f}")

    # 建立ART分類器
    art_clf = build_art_classifier(model, device)

    # === 第一階段：產生訓練用對抗樣本 ===
    print("\n=== Phase 1: Generating training adversarial samples ===")
    train_adv_samples, attack_params = generate_adversarial_samples(
        art_clf, train_loader,
        attack_types=ATTACK_TYPES,
        max_samples=TRAIN_SAMPLES,
        model=model,
        device=device
    )

    # 取得訓練用預測結果
    train_results = {}
    for attack_type, data in train_adv_samples.items():
        predictions = get_predictions(model, data['x'], device)
        train_results[attack_type] = {
            'predictions': predictions,
            'labels': data['y'],
            'images': data['x']
        }

    # 評估攻擊效果
    attack_effectiveness = evaluate_attack_effectiveness(train_results)

    # === 第二階段：找出最佳閾值 ===
    print("\n=== Phase 2: Finding optimal thresholds ===")
    thresholds = {}

    for attack_type in ATTACK_TYPES:
        if attack_type in train_results:
            clean_data = train_results['clean']
            adv_data = train_results[attack_type]

            # 準備訓練資料
            clean_features_diff, adv_features_diff = prepare_detector_data(
                model, clean_data, adv_data, attack_type, device
            )

            # 找出最佳閾值
            weights, threshold, method = find_optimal_threshold(
                clean_features_diff, adv_features_diff, attack_type
            )

            thresholds[attack_type] = {
                'weights': weights,
                'threshold': threshold,
                'method': method
            }

    # === 第三階段：測試偵測效果 ===
    print("\n=== Phase 3: Testing detection performance ===")

    # 產生新的測試資料
    test_adv_samples, test_attack_params= generate_adversarial_samples(
        art_clf, test_loader,
        attack_types=list(thresholds.keys()),
        max_samples=TEST_SAMPLES,
        model=model,
        device=device
    )
    test_results = {}
    for attack_type, data in test_adv_samples.items():
        predictions = get_predictions(model, data['x'], device)
        test_results[attack_type] = {
            'predictions': predictions,
            'labels': data['y'],
            'images': data['x']
        }

    # 測試每個攻擊類型
    detection_results = {}
    for attack_type in thresholds.keys():
        if attack_type in test_results:
            # 準備測試資料
            test_clean_diff, test_adv_diff = prepare_detector_data(
                model, test_results['clean'], test_results[attack_type], attack_type, device
            )

            # 直接用閾值偵測
            result = simple_threshold_detection(
                test_clean_diff, test_adv_diff,
                thresholds[attack_type]['weights'],
                thresholds[attack_type]['threshold'],
                attack_type
            )

            result['method'] = thresholds[attack_type]['method']
            detection_results[attack_type] = result

    # 最終結果
    print_results(seed, base_acc, detection_results, attack_effectiveness,test_attack_params)


def print_results(seed, base_acc, detection_results, attack_effectiveness, attack_params):
    """列印最終結果"""
    print("\n" + "=" * 100)
    print("DIRECT THRESHOLD DETECTION RESULTS SUMMARY")
    print("=" * 100)
    print(f"Random seed: {seed} | Base model accuracy: {base_acc:.4f}")
    print()

    # 表格標題
    header = f"{'Attack':<12} {'Detection':<10} {'F1-Score':<10} {'AUC':<8} {'Method':<12} {'Threshold':<12} {'Success Rate':<13} {'Params':<25}"
    print(header)
    print("-" * len(header))

    for attack_type in ATTACK_TYPES:
        if attack_type in detection_results:
            det_results = detection_results[attack_type]
            att_results = attack_effectiveness[attack_type]

            # 攻擊參數字串
            params_str = ""
            if attack_params and attack_type in attack_params:
                if attack_type == 'fgsm':
                    params_str = f"eps={attack_params[attack_type]['eps']:.3f}"
                elif attack_type == 'pgd':
                    params_str = f"eps={attack_params[attack_type]['eps']:.3f},step={attack_params[attack_type]['eps_step']:.3f},iter={attack_params[attack_type]['max_iter']}"
                elif attack_type == 'cw':
                    params_str = f"c={attack_params[attack_type]['confidence']:.1f}"
                elif attack_type == 'deepfool':
                    params_str = f"eps={attack_params[attack_type]['eps']:.3f},max_iter={attack_params[attack_type]['max_iter']},nb_grads={attack_params[attack_type]['nb_grads']}"

            print(f"{attack_type.upper():<12} "
                  f"{det_results['accuracy']:<10.4f} "
                  f"{det_results['f1']:<10.4f} "
                  f"{det_results['auc']:<8.4f} "
                  f"{det_results['method']:<12} "
                  f"{det_results['threshold']:<12.6f} "
                  f"{att_results['attack_success_rate']:<13.4f} "
                  f"{params_str:<25}")

    print("=" * 100)

    # 權重分析
    print("\nFEATURE WEIGHTS:")
    feature_names = ['MSE', 'MaxProb', 'Entropy', 'KL_Div', 'L1']
    for attack_type in ATTACK_TYPES:
        if attack_type in detection_results:
            weights = detection_results[attack_type]['weights']
            method = detection_results[attack_type]['method']
            print(
                f"{attack_type.upper()} ({method}): {' '.join([f'{name}={w:.3f}' for name, w in zip(feature_names, weights)])}")

    print("\nLIMITATIONS:")
    print("• Linear weighted sum cannot capture feature interactions")
    print("• Fixed weights cannot adapt to evolving attacks")
    print("• Single threshold assumes simple decision boundary")
    print("• No uncertainty handling or interpretable rules")
    print("→ Motivates evolved fuzzy inference systems!")


if __name__ == '__main__':
    main()
