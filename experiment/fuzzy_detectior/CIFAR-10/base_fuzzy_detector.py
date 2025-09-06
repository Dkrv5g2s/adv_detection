import os
import time
import random
import numpy as np
import torch
import warnings

from models import CIFAR10CNN
from data_utils import load_cifar10
from model_training import train_classifier, eval_classifier
from adversarial_attacks import (
    build_art_classifier,
    generate_adversarial_samples,
    get_predictions,
    evaluate_attack_effectiveness
)
from feature_extraction import extract_features, extract_feature_differences
from fuzzy_detector import train_fuzzy_detector, test_fuzzy_detector, TriangularFuzzySets

warnings.filterwarnings('ignore')


ATTACK_TYPES = ['fgsm', 'pgd']
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 500

MODEL_PATH = "./cifar10_cnn.pth"
TRAINING_EPOCHS = 32
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
    print(f"[{attack_type}] Preparing detector data...")

    # 確保樣本數量一致
    min_samples = min(len(clean_data['images']), len(adv_data['images']))
    clean_images = clean_data['images'][:min_samples]
    adv_images = adv_data['images'][:min_samples]

    # 為clean圖片添加微小noise
    noise_std = np.random.uniform(0.01, 0.05)  # 隨機噪聲強度
    clean_images_noisy = clean_images + np.random.normal(0, noise_std, clean_images.shape)
    clean_images_noisy = np.clip(clean_images_noisy, 0, 1)  # 確保像素值在合理範圍

    # 分別提取CNN特徵
    print(f"[{attack_type}] Extracting features...")
    clean_features = extract_features(model, clean_images, device)
    clean_features_noisy = extract_features(model, clean_images_noisy, device)
    adv_features = extract_features(model, adv_images, device)

    # 計算特徵差異
    clean_features_diff = extract_feature_differences(clean_features, clean_features_noisy)
    adv_features_diff = extract_feature_differences(clean_features, adv_features)

    # 調試資訊
    print(
        f"[{attack_type}] Clean diff stats - mean: {clean_features_diff.mean():.4f}, std: {clean_features_diff.std():.4f}")
    print(f"[{attack_type}] Adv diff stats - mean: {adv_features_diff.mean():.4f}, std: {adv_features_diff.std():.4f}")

    # 檢查差異是否合理
    ratio = adv_features_diff.mean() / (clean_features_diff.mean() + 1e-8)
    print(f"[{attack_type}] Adversarial/Clean ratio: {ratio:.2f}")

    if ratio < 2.0:
        print(f"[{attack_type}] Warning: Adversarial differences may be too small!")

    return clean_features_diff, adv_features_diff


def load_or_train_model(device):
    """載入或訓練分類器模型"""
    model = CIFAR10CNN()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        print("Loaded existing model.")
    else:
        print("Training new model...")
        train_loader, test_loader = load_cifar10(batch_size=BATCH_SIZE, shuffle_test=True)
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

    print("=== Fuzzy Adversarial Attack Detection ===")
    print(f"Attack types: {ATTACK_TYPES}")
    print(f"Training samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}")

    # 載入資料
    train_loader, test_loader = load_cifar10(batch_size=BATCH_SIZE, shuffle_test=True)

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
        attack_types=ATTACK_TYPES,  # 使用全域設定
        max_samples=TRAIN_SAMPLES
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

    # === 第二階段：訓練偵測器 ===
    print("\n=== Phase 2: Training detectors ===")
    detectors = {}

    for attack_type in ATTACK_TYPES:  # 使用全域設定
        if attack_type in train_results:
            clean_data = train_results['clean']
            adv_data = train_results[attack_type]

            # 準備偵測器資料
            clean_features_diff, adv_features_diff = prepare_detector_data(
                model, clean_data, adv_data, attack_type, device
            )

            # 訓練偵測器（純訓練，無測試）
            detector = train_fuzzy_detector(
                clean_features_diff, adv_features_diff, attack_type
            )

            detectors[attack_type] = detector

    # === 第三階段：產生測試用對抗樣本並測試偵測器 ===
    print("\n=== Phase 3: Generating test samples and testing detectors ===")

    # 產生新的測試資料
    test_adv_samples, _ = generate_adversarial_samples(
        art_clf, test_loader,
        attack_types=list(detectors.keys()),
        max_samples=TEST_SAMPLES
    )

    test_results = {}
    for attack_type, data in test_adv_samples.items():
        predictions = get_predictions(model, data['x'], device)
        test_results[attack_type] = {
            'predictions': predictions,
            'labels': data['y'],
            'images': data['x']
        }

    # 測試每個偵測器
    detection_results = {}
    for attack_type in detectors.keys():
        if attack_type in test_results:
            # 準備測試資料
            test_clean_diff, test_adv_diff = prepare_detector_data(
                model, test_results['clean'], test_results[attack_type], attack_type, device
            )

            # 測試偵測器
            test_result = test_fuzzy_detector(
                detectors[attack_type], test_clean_diff, test_adv_diff, attack_type
            )

            detection_results[attack_type] = test_result

    # 最終統整表格
    print_results(seed, base_acc, detection_results, attack_effectiveness, attack_params)


def print_results(seed, base_acc, detection_results, attack_effectiveness, attack_params):
    """列印最終結果"""
    print("\n" + "=" * 85)
    print("FINAL RESULTS SUMMARY")
    print("=" * 85)
    print(f"Random seed: {seed} | Base model accuracy: {base_acc:.4f}")
    print()

    # 表格標題
    header = f"{'Attack':<12} {'Detection':<10} {'F1-Score':<10} {'AUC':<8} {'Rules':<6} {'Success Rate':<13} {'Params':<25}"
    print(header)
    print("-" * len(header))

    for attack_type in ATTACK_TYPES:  # 使用全域設定
        if attack_type in detection_results:
            det_results = detection_results[attack_type]
            att_results = attack_effectiveness[attack_type]

            # 攻擊參數字串
            if attack_type == 'fgsm':
                params_str = f"eps={attack_params[attack_type]['eps']:.3f}"
            elif attack_type == 'pgd':
                params_str = f"eps={attack_params[attack_type]['eps']:.3f},iter={attack_params[attack_type]['max_iter']}"
            elif attack_type == 'cw':
                params_str = f"c={attack_params[attack_type]['confidence']:.1f}"
            elif attack_type == 'deepfool':
                params_str = f"overshoot={attack_params[attack_type]['overshoot']:.2f}"
            else:
                params_str = ""

            print(f"{attack_type.upper():<12} "
                  f"{det_results['accuracy']:<10.4f} "
                  f"{det_results['f1']:<10.4f} "
                  f"{det_results['auc']:<8.4f} "
                  f"{det_results['num_rules']:<6} "
                  f"{att_results['attack_success_rate']:<13.4f} "
                  f"{params_str:<25}")

    print("=" * 85)


if __name__ == '__main__':
    main()
