import os
import time
import random
import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# MNIST 相關 imports
from fuzzy_detectior.MNIST.src.models import SimpleCNN
from fuzzy_detectior.MNIST.src.data_utils import load_mnist
from fuzzy_detectior.MNIST.src.adversarial_attacks import (
    build_art_classifier as mnist_build_art_classifier,
    generate_adversarial_samples as mnist_generate_adversarial_samples,
    get_predictions as mnist_get_predictions,
    evaluate_attack_effectiveness as mnist_evaluate_attack_effectiveness
)
from fuzzy_detectior.MNIST.src.feature_extraction import \
    extract_feature_differences
from fuzzy_detectior.MNIST.src.shap_signature import generate_shap_signatures

# CIFAR-10 相關 imports
from fuzzy_detectior.CIFAR10.src.models import CIFAR10CNN
from fuzzy_detectior.CIFAR10.src.data_utils import load_cifar10
from fuzzy_detectior.CIFAR10.src.adversarial_attacks import (
    build_art_classifier as cifar10_build_art_classifier,
    generate_adversarial_samples as cifar10_generate_adversarial_samples,
    get_predictions as cifar10_get_predictions,
    evaluate_attack_effectiveness as cifar10_evaluate_attack_effectiveness
)


warnings.filterwarnings('ignore')

# 訓練用攻擊類型 (MNIST)
TRAIN_ATTACK_TYPES = ['fgsm', 'pgd']
# 測試用攻擊類型 (CIFAR-10)
TEST_ATTACK_TYPES = ['deepfool']

TRAIN_SAMPLES = 1000
TEST_SAMPLES = 500
BATCH_SIZE = 256

# 模型路徑
MNIST_MODEL_PATH = "./MNIST/src/simple_mnist_cnn.pth"
CIFAR10_MODEL_PATH = "./CIFAR10/src/cifar10_cnn.pth"


def generate_seed():
    return int(time.time()) % 10000


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist_model(device):
    """載入訓練好的 MNIST 模型"""
    print("Loading MNIST SimpleCNN model...")
    model = SimpleCNN()

    if os.path.exists(MNIST_MODEL_PATH):
        model.load_state_dict(torch.load(MNIST_MODEL_PATH, map_location=device))
        print(f"MNIST model loaded from {MNIST_MODEL_PATH}")
    else:
        print(f"Warning: MNIST model file {MNIST_MODEL_PATH} not found. Using random weights.")

    model = model.to(device)
    model.eval()
    return model


def load_cifar10_model(device):
    """載入訓練好的 CIFAR-10 模型"""
    print("Loading CIFAR-10 CNN model...")
    model = CIFAR10CNN()

    if os.path.exists(CIFAR10_MODEL_PATH):
        model.load_state_dict(torch.load(CIFAR10_MODEL_PATH, map_location=device))
        print(f"CIFAR-10 model loaded from {CIFAR10_MODEL_PATH}")
    else:
        print(f"Warning: CIFAR-10 model file {CIFAR10_MODEL_PATH} not found. Using random weights.")

    model = model.to(device)
    model.eval()
    return model




def prepare_detector_data(model, clean_data, adv_data, attack_type, device):
    """使用SHAP簽名準備偵測器資料"""
    print(f"[{attack_type}] Preparing SHAP detector data...")

    # 確保樣本數量一致
    min_samples = min(len(clean_data['images']), len(adv_data['images']))
    clean_images = clean_data['images'][:min_samples]
    clean_labels = clean_data['labels'][:min_samples]
    adv_images = adv_data['images'][:min_samples]

    # 為每個clean圖片隨機選擇一個同類別的參考圖片
    clean_images_ref = []

    for i, (image, label) in enumerate(zip(clean_images, clean_labels)):
        # 找到同類別的其他樣本（排除自己）
        same_class_mask = (clean_labels == label) & (np.arange(len(clean_labels)) != i)
        same_class_indices = np.where(same_class_mask)[0]

        if len(same_class_indices) > 0:
            # 隨機選擇一個同類別樣本
            random_index = np.random.choice(same_class_indices)
            ref_image = clean_images[random_index]
        else:
            # 如果沒有同類別樣本，隨機選擇一個其他樣本
            other_indices = np.arange(len(clean_images))
            other_indices = other_indices[other_indices != i]  # 排除自己
            if len(other_indices) > 0:
                random_index = np.random.choice(other_indices)
                ref_image = clean_images[random_index]
            else:
                # 極端情況：只有一個樣本，使用自己
                ref_image = image

        clean_images_ref.append(ref_image)

    clean_images_ref = np.array(clean_images_ref)

    # 生成SHAP簽名
    print(f"[{attack_type}] Generating clean SHAP signatures...")
    clean_signatures = generate_shap_signatures(model, clean_images, device)

    print(f"[{attack_type}] Generating reference clean SHAP signatures...")
    clean_signatures_ref = generate_shap_signatures(model, clean_images_ref, device)

    print(f"[{attack_type}] Generating adversarial SHAP signatures...")
    adv_signatures = generate_shap_signatures(model, adv_images, device)

    # 計算特徵差異
    clean_features_diff = extract_feature_differences(clean_signatures, clean_signatures_ref)
    adv_features_diff = extract_feature_differences(clean_signatures, adv_signatures)

    # **轉換為 PyTorch tensors**
    clean_features_diff = torch.tensor(clean_features_diff, dtype=torch.float32, device=device)
    adv_features_diff = torch.tensor(adv_features_diff, dtype=torch.float32, device=device)

    # 創建標籤 (0 = clean, 1 = adversarial)
    clean_labels_tensor = torch.zeros(len(clean_features_diff), dtype=torch.long, device=device)
    adv_labels_tensor = torch.ones(len(adv_features_diff), dtype=torch.long, device=device)

    # 合併特徵和標籤
    X = torch.cat([clean_features_diff, adv_features_diff], dim=0)
    y = torch.cat([clean_labels_tensor, adv_labels_tensor], dim=0)

    # 調試資訊
    print(f"[{attack_type}] Clean SHAP diff stats - mean: {clean_features_diff.mean():.4f}, std: {clean_features_diff.std():.4f}")
    print(f"[{attack_type}] Adv SHAP diff stats - mean: {adv_features_diff.mean():.4f}, std: {adv_features_diff.std():.4f}")

    # 檢查差異是否合理
    ratio = adv_features_diff.mean() / (clean_features_diff.mean() + 1e-8)
    print(f"[{attack_type}] SHAP Adversarial/Clean ratio: {ratio:.2f}")

    if ratio < 2.0:
        print(f"[{attack_type}] Warning: SHAP adversarial differences may be too small!")

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

    print("=== Cross-Dataset Adversarial Detection ===")
    print(f"Training: MNIST + {TRAIN_ATTACK_TYPES}")
    print(f"Testing: CIFAR-10 + {TEST_ATTACK_TYPES}")
    print(f"Training samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}")
    print("Method: SHAP signatures → MLP classifier")

    # === 第一階段：在 MNIST 上訓練偵測器 ===
    print("\n=== Phase 1: Training detector on MNIST ===")

    # 載入 MNIST 模型和資料
    mnist_model = load_mnist_model(device)
    mnist_train_loader, _ = load_mnist(batch_size=BATCH_SIZE)

    # 建立 MNIST ART 分類器
    mnist_art_clf = mnist_build_art_classifier(mnist_model, device)

    # 生成 MNIST 訓練用對抗樣本
    print("Generating MNIST adversarial samples...")
    train_adv_samples, train_attack_params = mnist_generate_adversarial_samples(
        mnist_art_clf, mnist_train_loader,
        attack_types=TRAIN_ATTACK_TYPES,
        max_samples=TRAIN_SAMPLES,
        model=mnist_model,
        device=device
    )

    # 取得 MNIST 訓練用預測結果
    train_results = {}
    for attack_type, data in train_adv_samples.items():
        predictions = mnist_get_predictions(mnist_model, data['x'], device)
        train_results[attack_type] = {
            'predictions': predictions,
            'labels': data['y'],
            'images': data['x']
        }

    # 評估 MNIST 攻擊效果
    train_attack_effectiveness = mnist_evaluate_attack_effectiveness(train_results)

    # 準備偵測器訓練資料
    all_X_train, all_y_train = [], []

    for attack_type in TRAIN_ATTACK_TYPES:
        if attack_type in train_results:
            clean_data = train_results['clean']
            adv_data = train_results[attack_type]

            # 準備 MNIST 訓練資料
            X_train, y_train = prepare_detector_data(
                mnist_model, clean_data, adv_data, attack_type, device
            )
            all_X_train.append(X_train)
            all_y_train.append(y_train)

    # 合併所有攻擊類型的資料
    X_train = torch.cat(all_X_train, dim=0)
    y_train = torch.cat(all_y_train, dim=0)

    # 訓練MLP偵測器
    input_dim = X_train.shape[1]
    print(f"Training MLP detector with input dimension: {input_dim}")
    mlp_detector = train_mlp_detector(X_train, y_train, input_dim, epochs=50, batch_size=64, device=device)

    # === 第二階段：在 CIFAR-10 上測試偵測器 ===
    print("\n=== Phase 2: Testing detector on CIFAR-10 ===")

    # 載入 CIFAR-10 模型和資料
    cifar10_model = load_cifar10_model(device)
    _, cifar10_test_loader = load_cifar10(batch_size=BATCH_SIZE, shuffle_test=True)

    # 建立 CIFAR-10 ART 分類器
    cifar10_art_clf = cifar10_build_art_classifier(cifar10_model, device)

    # 生成 CIFAR-10 測試用對抗樣本
    print("Generating CIFAR-10 adversarial samples...")
    test_adv_samples, test_attack_params = cifar10_generate_adversarial_samples(
        cifar10_art_clf, cifar10_test_loader,
        attack_types=TEST_ATTACK_TYPES,
        max_samples=TEST_SAMPLES,
        model=cifar10_model,
        device=device
    )

    # 取得 CIFAR-10 測試用預測結果
    test_results = {}
    for attack_type, data in test_adv_samples.items():
        predictions = cifar10_get_predictions(cifar10_model, data['x'], device)
        test_results[attack_type] = {
            'predictions': predictions,
            'labels': data['y'],
            'images': data['x']
        }

    # 評估 CIFAR-10 攻擊效果
    test_attack_effectiveness = cifar10_evaluate_attack_effectiveness(test_results)

    # 為每個攻擊類型單獨評估偵測器
    detection_results = {}

    for attack_type in TEST_ATTACK_TYPES:
        if attack_type in test_results:
            print(f"\n--- Evaluating {attack_type.upper()} detection ---")
            clean_data = test_results['clean']
            adv_data = test_results[attack_type]

            # 準備 CIFAR-10 測試資料
            X_test, y_test = prepare_detector_data(
                cifar10_model, clean_data, adv_data, attack_type, device
            )

            # 檢查特徵維度是否匹配
            if X_test.shape[1] != input_dim:
                print(f"Warning: Feature dimension mismatch! Train: {input_dim}, Test: {X_test.shape[1]}")
                # 如果維度不匹配，需要調整
                if X_test.shape[1] > input_dim:
                    X_test = X_test[:, :input_dim]  # 截斷
                else:
                    # 填充零
                    padding = torch.zeros(X_test.shape[0], input_dim - X_test.shape[1], device=device)
                    X_test = torch.cat([X_test, padding], dim=1)

            # 評估偵測器
            detection_results[attack_type] = evaluate_mlp_detector(mlp_detector, X_test, y_test, device)

    print_results(seed, detection_results, train_attack_effectiveness, test_attack_effectiveness,
                  train_attack_params, test_attack_params)


def print_results(seed, detection_results, train_attack_effectiveness, test_attack_effectiveness,
                  train_attack_params, test_attack_params):
    """列印最終結果"""
    print("\n" + "=" * 120)
    print("CROSS-DATASET DETECTION RESULTS SUMMARY")
    print("=" * 120)
    print(f"Random seed: {seed}")
    print("Training: MNIST + FGSM/PGD → Testing: CIFAR-10 + FGSM/PGD/DeepFool")
    print("Method: SHAP signatures → MLP classifier")
    print()

    # 訓練階段攻擊效果 (MNIST)
    print("Training Phase Attack Effectiveness (MNIST):")
    for attack_type in TRAIN_ATTACK_TYPES:
        if attack_type in train_attack_effectiveness:
            eff = train_attack_effectiveness[attack_type]
            params_str = ""
            if train_attack_params and attack_type in train_attack_params:
                if attack_type == 'fgsm':
                    params_str = f"eps={train_attack_params[attack_type]['eps']:.3f}"
                elif attack_type == 'pgd':
                    params_str = f"eps={train_attack_params[attack_type]['eps']:.3f}, iter={train_attack_params[attack_type]['max_iter']}"

            print(f"  {attack_type.upper()}: Success Rate = {eff['attack_success_rate']:.4f}, {params_str}")

    print()

    # 測試階段攻擊效果 (CIFAR-10)
    print("Testing Phase Attack Effectiveness (CIFAR-10):")
    for attack_type in TEST_ATTACK_TYPES:
        if attack_type in test_attack_effectiveness:
            eff = test_attack_effectiveness[attack_type]
            params_str = ""
            if test_attack_params and attack_type in test_attack_params:
                if attack_type == 'fgsm':
                    params_str = f"eps={test_attack_params[attack_type]['eps']:.3f}"
                elif attack_type == 'pgd':
                    params_str = f"eps={test_attack_params[attack_type]['eps']:.3f}, iter={test_attack_params[attack_type]['max_iter']}"
                elif attack_type == 'deepfool':
                    params_str = f"max_iter={test_attack_params[attack_type]['max_iter']}, eps={test_attack_params[attack_type]['eps']:.3f}"

            print(f"  {attack_type.upper()}: Success Rate = {eff['attack_success_rate']:.4f}, {params_str}")

    print()

    # 偵測結果表格
    header = f"{'Attack':<12} {'Dataset':<10} {'Accuracy':<10} {'F1-Score':<10} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'Method':<12}"
    print(header)
    print("-" * len(header))

    for attack_type in TEST_ATTACK_TYPES:
        if attack_type in detection_results:
            det_results = detection_results[attack_type]

            print(f"{attack_type.upper():<12} "
                  f"{'CIFAR-10':<10} "
                  f"{det_results['accuracy']:<10.4f} "
                  f"{det_results['f1']:<10.4f} "
                  f"{det_results['auc']:<8.4f} "
                  f"{det_results['precision']:<10.4f} "
                  f"{det_results['recall']:<8.4f} "
                  f"{det_results['method']:<12}")

    print("=" * 120)
    print("Note: Detector trained on MNIST SHAP features but tested on CIFAR-10 SHAP features")
    print("This demonstrates cross-dataset generalization capability")


if __name__ == '__main__':
    main()
