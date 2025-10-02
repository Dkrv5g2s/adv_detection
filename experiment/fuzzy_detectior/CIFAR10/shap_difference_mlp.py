import os
import time
import random
import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from fuzzy_detectior.CIFAR10.src.models import CIFAR10CNN
from fuzzy_detectior.CIFAR10.src.data_utils import load_cifar10
from fuzzy_detectior.CIFAR10.src.model_training import train_classifier, eval_classifier
from fuzzy_detectior.CIFAR10.src.adversarial_attacks import (
    build_art_classifier,
    generate_adversarial_samples,
    get_predictions,
    evaluate_attack_effectiveness
)
from fuzzy_detectior.CIFAR10.src.feature_extraction import extract_feature_differences
from fuzzy_detectior.CIFAR10.src.shap_signature import generate_shap_signatures

warnings.filterwarnings('ignore')

ATTACK_TYPES = ['fgsm', 'pgd', 'deepfool']
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 500

MODEL_PATH = "./src/cifar10_cnn.pth"
TRAINING_EPOCHS = 140
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


def prepare_detector_data(model, clean_data, adv_data, attack_type, device):
    """使用SHAP簽名準備偵測器資料"""
    print(f"[{attack_type}] Preparing SHAP detector data...")

    # 確保樣本數量一致
    min_samples = min(len(clean_data['images']), len(adv_data['images']))
    clean_images = clean_data['images'][:min_samples]
    adv_images = adv_data['images'][:min_samples]

    # 為clean圖片添加微小噪音
    # noise_std = np.random.uniform(0.01, 0.05)
    clean_images_noisy = clean_images  # + np.random.normal(0, noise_std, clean_images.shape)

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
    print(
        f"[{attack_type}] Clean SHAP diff stats - mean: {clean_features_diff.mean():.4f}, std: {clean_features_diff.std():.4f}")
    print(
        f"[{attack_type}] Adv SHAP diff stats - mean: {adv_features_diff.mean():.4f}, std: {adv_features_diff.std():.4f}")

    # 檢查差異是否合理
    ratio = adv_features_diff.mean() / (clean_features_diff.mean() + 1e-8)
    print(f"[{attack_type}] SHAP Adversarial/Clean ratio: {ratio:.2f}")

    if ratio < 2.0:
        print(f"[{attack_type}] Warning: SHAP adversarial differences may be too small!")

    # 轉換為 PyTorch 張量並移到指定設備
    clean_features_diff = torch.tensor(clean_features_diff, dtype=torch.float32, device=device)
    adv_features_diff = torch.tensor(adv_features_diff, dtype=torch.float32, device=device)

    # 準備標籤 (0: clean, 1: adversarial)
    clean_labels = torch.zeros(len(clean_features_diff), dtype=torch.long, device=device)
    adv_labels = torch.ones(len(adv_features_diff), dtype=torch.long, device=device)

    # 合併資料和標籤
    X = torch.cat([clean_features_diff, adv_features_diff], dim=0)
    y = torch.cat([clean_labels, adv_labels], dim=0)

    return X, y


class MLPDetector(nn.Module):
    def __init__(self, input_dim):
        super(MLPDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_mlp_detector(X_train, y_train, input_dim, epochs=50, batch_size=64, device=None):
    """訓練MLP偵測器"""
    detector = MLPDetector(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    detector.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = detector(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")

    return detector


def evaluate_mlp_detector(detector, X_test, y_test, device):
    """評估MLP偵測器並返回詳細指標"""
    detector.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        outputs = detector(X_test)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

        # 轉換為 numpy 以便計算指標
        y_true = y_test.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probabilities[:, 1].cpu().numpy()  # 對抗樣本的機率

        # 計算各種指標
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        # 計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 計算精確率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results = {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'method': 'MLP',
            'confusion_matrix': cm
        }

        print(f"MLP Detector Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

        return results


def main():
    # 設定隨機種子
    seed = generate_seed()
    set_seed(seed)

    # 設定運算裝置
    device = get_device()
    print("Using device:", device)

    print("=== Direct MLP-Based Adversarial Detection ===")
    print(f"Attack types: {ATTACK_TYPES}")
    print(f"Training samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}")
    print("Method: SHAP features → MLP classifier")

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

    attack_effectiveness = evaluate_attack_effectiveness(train_results)

    # === 第二階段：訓練MLP偵測器 ===
    print("\n=== Phase 2: Training MLP detector ===")
    all_X_train, all_y_train = [], []

    for attack_type in ATTACK_TYPES:
        if attack_type in train_results:
            clean_data = train_results['clean']
            adv_data = train_results[attack_type]

            # 準備訓練資料
            X_train, y_train = prepare_detector_data(model, clean_data, adv_data, attack_type, device)
            all_X_train.append(X_train)
            all_y_train.append(y_train)

    # 合併所有攻擊類型的資料
    X_train = torch.cat(all_X_train, dim=0)
    y_train = torch.cat(all_y_train, dim=0)

    # 訓練MLP偵測器
    input_dim = X_train.shape[1]
    mlp_detector = train_mlp_detector(X_train, y_train, input_dim, epochs=50, batch_size=64, device=device)

    # === 第三階段：測試偵測效果 ===
    print("\n=== Phase 3: Testing MLP detector ===")
    test_adv_samples, _ = generate_adversarial_samples(
        art_clf, test_loader,
        attack_types=ATTACK_TYPES,
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

    # 為每個攻擊類型單獨評估偵測器
    detection_results = {}

    for attack_type in ATTACK_TYPES:
        if attack_type in test_results:
            print(f"\n--- Evaluating {attack_type.upper()} detection ---")
            clean_data = test_results['clean']
            adv_data = test_results[attack_type]

            # 準備測試資料
            X_test, y_test = prepare_detector_data(model, clean_data, adv_data, attack_type, device)

            # 評估偵測器
            detection_results[attack_type] = evaluate_mlp_detector(mlp_detector, X_test, y_test, device)

    print_results(seed, base_acc, detection_results, attack_effectiveness, attack_params)


def print_results(seed, base_acc, detection_results, attack_effectiveness, attack_params):
    """列印最終結果"""
    print("\n" + "=" * 120)
    print("MLP-BASED DETECTION RESULTS SUMMARY")
    print("=" * 120)
    print(f"Random seed: {seed} | Base model accuracy: {base_acc:.4f}")
    print()

    # 表格標題
    header = f"{'Attack':<12} {'Detection':<10} {'F1-Score':<10} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'Method':<12} {'Success Rate':<13} {'Params':<25}"
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
                  f"{det_results['precision']:<10.4f} "
                  f"{det_results['recall']:<8.4f} "
                  f"{det_results['method']:<12} "
                  f"{att_results['attack_success_rate']:<13.4f} "
                  f"{params_str:<25}")

    print("=" * 120)


if __name__ == '__main__':
    main()
