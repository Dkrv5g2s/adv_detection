# 導入所需套件和設定
import os
import random
import time
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, CarliniL2Method
from art.estimators.classification import PyTorchClassifier
import warnings

warnings.filterwarnings('ignore')


# 設定隨機種子以確保結果可重現
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 定義CNN模型架構
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特徵提取器
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)  # 32x28x28
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # 64x28x28
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)  # 64x14x14

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)  # 128x14x14
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)  # 256x14x14
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)  # 256x7x7

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


# 資料載入函數
def load_mnist(batch_size=256, shuffle_test=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST標準化
    ])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=0, pin_memory=True)
    return train_loader, test_loader


# 模型訓練函數
def train_classifier(model, train_loader, test_loader, epochs=5, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for ep in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        acc = eval_classifier(model, test_loader)
        print(f"[Epoch {ep + 1}] loss={avg_loss:.4f} test_acc={acc:.4f}")
    return model


# 模型評估函數
def eval_classifier(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# 建立ART分類器
def build_art_classifier(model):
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

    return results, attack_params


# 添加缺少的 get_logits 函數
def get_logits(model, X, batch_size=256):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            batch_logits = model(batch)
            logits.append(batch_logits.cpu().numpy())
    return np.concatenate(logits, axis=0)


# 取得 softmax 機率向量
def get_predictions(model, X, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            logits = model(batch)
            p = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds, axis=0)


# 評估攻擊效果的函數
def evaluate_attack_effectiveness(results):
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


# 模糊集合定義
class TriangularFuzzySets:
    def __init__(self, centers=None, width=None):
        if centers is None:
            centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        if width is None:
            width = np.array([0.3, 0.25, 0.25, 0.25, 0.3])

        self.centers = centers
        self.widths = width
        self.K = len(centers)
        self.labels = ["Very Low", "Low", "OK", "High", "Very High"]

    def membership(self, x):
        """計算隸屬度函數"""
        x_expand = np.expand_dims(x, axis=-1)
        c = self.centers.reshape((1,) * (x_expand.ndim - 1) + (self.K,))
        w = self.widths.reshape((1,) * (x_expand.ndim - 1) + (self.K,))

        # 三角隸屬函數
        mu = np.maximum(0, 1 - np.abs(x_expand - c) / (w + 1e-12))
        return mu


def extract_feature_differences(logits_clean, logits_adv):
    # 將 logits 轉換為機率分佈 (softmax)
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 數值穩定性
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    prob_clean = softmax(logits_clean)
    prob_adv = softmax(logits_adv)

    # 1. 均方誤差 (MSE)
    mse_diff = np.mean((logits_adv - logits_clean) ** 2, axis=1)

    # 2. 最大支持度差異 (Maximum Support Difference)
    # 支持度定義為最大機率值
    max_support_clean = np.max(prob_clean, axis=1)
    max_support_adv = np.max(prob_adv, axis=1)
    max_support_diff = np.abs(max_support_clean - max_support_adv)

    # 3. 熵差異 (Entropy Difference)
    def entropy(p):
        # 避免 log(0)，加入小常數
        p_safe = np.clip(p, 1e-12, 1.0)
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    entropy_clean = entropy(prob_clean)
    entropy_adv = entropy(prob_adv)
    entropy_diff = np.abs(entropy_clean - entropy_adv)

    # 4. Kullback-Leibler 散度 (KL Divergence)
    def kl_divergence(p, q):
        # KL(P||Q) = Σ p_i * log(p_i / q_i)
        p_safe = np.clip(p, 1e-12, 1.0)
        q_safe = np.clip(q, 1e-12, 1.0)
        return np.sum(p_safe * np.log(p_safe / q_safe), axis=1)

    # 計算雙向 KL 散度的平均值以確保對稱性
    kl_clean_to_adv = kl_divergence(prob_clean, prob_adv)
    kl_adv_to_clean = kl_divergence(prob_adv, prob_clean)
    kl_diff = (kl_clean_to_adv + kl_adv_to_clean) / 2

    # 5. L1差異
    l1_diff = np.mean(np.abs(logits_adv - logits_clean), axis=1)

    # 組合所有差異指標
    all_diffs = [mse_diff, max_support_diff, entropy_diff, kl_diff, l1_diff]

    # 正規化每個特徵到 [0,1]
    normalized_diffs = []
    for i, diff in enumerate(all_diffs):
        if diff.max() > diff.min():
            norm_diff = (diff - diff.min()) / (diff.max() - diff.min())
        else:
            norm_diff = np.zeros_like(diff)
        normalized_diffs.append(norm_diff)

    return np.column_stack(normalized_diffs)


# 模糊規則偵測器
@dataclass
class FuzzyRule:
    prototype: np.ndarray
    spread: np.ndarray
    output: float
    support: float
    confidence: float
    potential: float
    hits: int
    attack_type: str = None


class FuzzyDetector:
    def __init__(self,
                 init_spread=0.15,
                 learning_rate=0.08,
                 add_threshold=0.2,
                 fire_threshold=0.12,
                 max_rules=150,
                 alpha=0.92,
                 attack_type=None):
        self.rules = []
        self.init_spread = init_spread
        self.learning_rate = learning_rate
        self.add_threshold = add_threshold
        self.fire_threshold = fire_threshold
        self.max_rules = max_rules
        self.alpha = alpha
        self.attack_type = attack_type

        # 初始化模糊集合系統
        self.fuzzy_sets = TriangularFuzzySets()

    def fuzzify(self, normalized_diffs):
        """將正規化的差異特徵轉換為模糊特徵"""
        # 轉換為模糊特徵
        all_fuzzy_features = []
        for i in range(normalized_diffs.shape[1]):
            fuzzy_feat = self.fuzzy_sets.membership(normalized_diffs[:, i])
            all_fuzzy_features.append(fuzzy_feat)

        # 合併所有模糊特徵
        combined_features = np.concatenate(all_fuzzy_features, axis=1)
        return combined_features

    def _rule_activation(self, x, rule):
        diff = (x - rule.prototype) ** 2
        spread_sq = rule.spread ** 2 + 1e-12
        activation = np.exp(-np.sum(diff / spread_sq))
        return activation

    def _support_function(self, rule, class_samples):
        if len(class_samples) == 0:
            return 0.0

        total_activation = 0
        for sample in class_samples:
            activation = self._rule_activation(sample, rule)
            if activation > self.fire_threshold:
                total_activation += activation

        return total_activation / len(class_samples)

    def _confidence_function(self, rule, class_samples, total_samples):
        if len(total_samples) == 0:
            return 0.0

        correct_activation = 0
        total_activation = 0

        for sample in class_samples:
            activation = self._rule_activation(sample, rule)
            if activation > self.fire_threshold:
                correct_activation += activation

        for sample in total_samples:
            activation = self._rule_activation(sample, rule)
            if activation > self.fire_threshold:
                total_activation += activation

        return correct_activation / (total_activation + 1e-12)

    def predict_proba(self, x):
        if not self.rules:
            return 0.5

        numerator = 0
        denominator = 0

        for rule in self.rules:
            activation = self._rule_activation(x, rule)
            if activation > self.fire_threshold:
                numerator += activation * rule.output
                denominator += activation

        if denominator == 0:
            max_activation = 0
            best_output = 0.5
            for rule in self.rules:
                activation = self._rule_activation(x, rule)
                if activation > max_activation:
                    max_activation = activation
                    best_output = rule.output
            return best_output

        result = numerator / denominator
        return np.clip(result, 0.0, 1.0)

    def update(self, x, label, class_samples=None, total_samples=None):
        # online training
        if class_samples is None:
            class_samples = [x]
        if total_samples is None:
            total_samples = [x]

        # 1.計算激活度並決定是否新增規則
        activations = []
        for rule in self.rules:
            activation = self._rule_activation(x, rule)
            activations.append(activation)

        max_activation = max(activations) if activations else 0

        if max_activation < self.add_threshold and len(self.rules) < self.max_rules:
            new_rule = FuzzyRule(
                prototype=x.copy(),
                spread=np.full_like(x, self.init_spread),
                output=float(label),
                support=0.0,
                confidence=0.0,
                potential=1.0,
                hits=1,
                attack_type=self.attack_type
            )
            self.rules.append(new_rule)

        # 2.更新被觸發的規則
        for i, rule in enumerate(self.rules):
            activation = activations[i] if i < len(activations) else self._rule_activation(x, rule)

            if activation > self.fire_threshold:
                # 自適應學習率
                adaptive_lr = self.learning_rate * activation

                # 更新原型
                rule.prototype += adaptive_lr * (x - rule.prototype)

                # 更新擴散
                error = np.abs(x - rule.prototype)
                rule.spread = rule.spread * (1 - adaptive_lr) + adaptive_lr * error

                # 更新輸出
                rule.output += adaptive_lr * 0.3 * (label - rule.output)

                # 更新統計
                rule.hits += 1
                rule.potential = self.alpha * rule.potential + (1 - self.alpha) * activation

                # 更新支持度和信心度
                rule.support = self._support_function(rule, class_samples)
                rule.confidence = self._confidence_function(rule, class_samples, total_samples)
            else:
                # 衰減潛力值
                rule.potential *= self.alpha

        # 3.規則裁剪
        min_potential = 0.005
        self.rules = [rule for rule in self.rules if rule.potential > min_potential]


def train_fuzzy_detector(model, clean_data, adv_data, attack_type, test_ratio=0.3):
    # 確保樣本數量一致
    min_samples = min(len(clean_data['images']), len(adv_data['images']))
    clean_images = clean_data['images'][:min_samples]
    clean_labels = clean_data['labels'][:min_samples]
    adv_images = adv_data['images'][:min_samples]
    adv_labels = adv_data['labels'][:min_samples]

    # 獲取乾淨樣本和對抗樣本的預測結果
    clean_predictions = get_predictions(model, clean_images)
    adv_predictions = get_predictions(model, adv_images)

    # 計算攻擊成功的樣本索引
    clean_pred_classes = np.argmax(clean_predictions, axis=1)
    adv_pred_classes = np.argmax(adv_predictions, axis=1)

    # 只保留原本預測正確的樣本
    originally_correct = (clean_pred_classes == clean_labels)

    print(
        f"[{attack_type}] 原本正確預測: {originally_correct.sum()}/{len(originally_correct)} ({originally_correct.mean():.3f})")

    # 篩選出原本預測正確的樣本
    correct_indices = np.where(originally_correct)[0]

    # 只使用原本預測正確的樣本
    clean_images_correct = clean_images[correct_indices]
    clean_labels_correct = clean_labels[correct_indices]
    adv_images_correct = adv_images[correct_indices]
    adv_predictions_correct = adv_predictions[correct_indices]

    # 在原本正確的基礎上，判斷攻擊是否成功
    adv_pred_classes_correct = np.argmax(adv_predictions_correct, axis=1)
    attack_successful = (adv_pred_classes_correct != clean_labels_correct)

    print(
        f"[{attack_type}] 在正確預測樣本中，攻擊成功: {attack_successful.sum()}/{len(attack_successful)} ({attack_successful.mean():.3f})")

    # 獲取乾淨樣本和對抗樣本的logits（只針對原本正確的樣本）
    clean_logits = get_logits(model, clean_images_correct)
    adv_logits = get_logits(model, adv_images_correct)

    # 計算logits差異特徵
    logits_diff = extract_feature_differences(clean_logits, adv_logits)

    # 初始化偵測器
    detector = FuzzyDetector(attack_type=attack_type)

    # 模糊化特徵
    features_fuzz = detector.fuzzify(logits_diff)

    # 建立標籤：攻擊成功=1，攻擊失敗=0（只針對原本正確預測的樣本）
    labels = attack_successful.astype(int)

    X = features_fuzz
    y = labels

    print(f"[{attack_type}] 訓練資料 - 攻擊失敗(標籤0): {np.sum(y == 0)}, 攻擊成功(標籤1): {np.sum(y == 1)}")

    # 檢查是否有足夠的正樣本
    if np.sum(y == 1) < 10:
        print(f"[{attack_type}] 警告: 成功攻擊樣本太少 ({np.sum(y == 1)})，可能影響訓練效果")

    # 分割訓練/測試
    n_train = int(len(X) * (1 - test_ratio))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"[{attack_type}] 訓練集: {len(X_train)}, 測試集: {len(X_test)}")
    print(f"[{attack_type}] 訓練集標籤分布 - 0: {np.sum(y_train == 0)}, 1: {np.sum(y_train == 1)}")
    print(f"[{attack_type}] 測試集標籤分布 - 0: {np.sum(y_test == 0)}, 1: {np.sum(y_test == 1)}")

    # 隨機打亂訓練順序
    train_indices = np.random.permutation(len(X_train))

    for idx in tqdm(train_indices, desc=f"Training {attack_type}", leave=False):
        i = train_indices[idx] if idx < len(train_indices) else idx
        current_class_samples = X_train[y_train == y_train[i]][:20]
        detector.update(X_train[i], y_train[i],
                        current_class_samples, X_train[:min(i + 1, 100)])

    # 測試
    y_pred_proba = []
    y_pred_binary = []

    for i in range(len(X_test)):
        prob = detector.predict_proba(X_test[i])
        y_pred_proba.append(prob)
        y_pred_binary.append(1 if prob > 0.5 else 0)

    # 計算指標
    accuracy = accuracy_score(y_test, y_pred_binary)

    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_binary, average='binary', zero_division=0
        )
        auc = roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        print(f"[{attack_type}] Metric calculation error: {e}")
        precision, recall, f1 = 0.0, 0.0, 0.0
        auc = 0.5

    print(f"[{attack_type}] Results - Acc: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}, Rules: {len(detector.rules)}")

    results = {
        'detector': detector,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'y_pred_binary': y_pred_binary,
        'num_rules': len(detector.rules),
        'successful_attacks': attack_successful.sum(),
        'total_attacks': len(attack_successful),
        'attack_success_rate': attack_successful.mean(),
        'originally_correct_count': len(correct_indices),
        'total_samples': min_samples,
        'originally_correct_and_attack_successful': attack_successful.sum(),
    }

    return results


def main():
    seed = int(time.time()) % 10000
    set_seed(seed)

    # 設定運算裝置
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("=== Fuzzy Adversarial Attack Detection ===")

    train_loader, test_loader = load_mnist(batch_size=256, shuffle_test=True)

    # 訓練或載入分類器
    model_path = "./simple_mnist_cnn.pth"
    model = SimpleCNN()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("Loaded existing model.")
    else:
        print("Training new model...")
        model = train_classifier(model, train_loader, test_loader, epochs=5)
        torch.save(model.state_dict(), model_path)

    # 評估基礎模型
    base_acc = eval_classifier(model, test_loader)
    print(f"Base CNN Test Accuracy: {base_acc:.4f}")

    # 建立ART分類器
    art_clf = build_art_classifier(model)

    # 產生對抗樣本
    print("\nGenerating adversarial samples...")
    adv_samples, attack_params = generate_adversarial_samples(
        art_clf, test_loader,
        attack_types=['fgsm', 'pgd'],
        max_samples=1000
    )

    # 取得預測結果
    print("Getting predictions...")
    results = {}
    for attack_type, data in adv_samples.items():
        predictions = get_predictions(model, data['x'])
        results[attack_type] = {
            'predictions': predictions,
            'labels': data['y'],
            'images': data['x']
        }

    # 評估攻擊效果
    attack_effectiveness = evaluate_attack_effectiveness(results)

    # 對每種攻擊類型訓練偵測器
    detection_results = {}

    print("Training detectors...")
    for attack_type in ['fgsm', 'pgd']:
        if attack_type in results:
            clean_data = results['clean']
            adv_data = results[attack_type]

            detector_results = train_fuzzy_detector(model, clean_data, adv_data, attack_type)
            detection_results[attack_type] = detector_results

    # 最終統整表格
    print("\n" + "=" * 120)
    print("FINAL RESULTS SUMMARY")
    print("=" * 120)
    print(f"Random seed: {seed} | Base model accuracy: {base_acc:.4f}")
    print()
    print("訓練策略：僅使用原本正確預測的樣本，攻擊成功標記為1，攻擊失敗標記為0")
    print()

    # 表格標題
    header = f"{'Attack':<12} {'Detection':<10} {'F1-Score':<10} {'AUC':<8} {'Rules':<6} {'Success Rate':<13} {'originally_correct_&attack_successful/Total':<40} {'Params':<25}"
    print(header)
    print("-" * len(header))

    for attack_type in ['fgsm', 'pgd']:
        if attack_type in detection_results:
            det_results = detection_results[attack_type]
            att_results = attack_effectiveness[attack_type]

            # 攻擊參數字串
            if attack_type == 'fgsm':
                params_str = f"eps={attack_params[attack_type]['eps']:.3f}"
            elif attack_type == 'pgd':
                params_str = f"eps={attack_params[attack_type]['eps']:.3f},iter={attack_params[attack_type]['max_iter']}"
            elif attack_type == 'deepfool':
                params_str = f"iter={attack_params[attack_type]['max_iter']},ε={attack_params[attack_type]['epsilon']:.3f}"
            else:
                params_str = ""

            # originally_correct_&attack_successful/Total 統計
            originally_correct_and_attack_successful = det_results['originally_correct_and_attack_successful']
            total_samples = det_results['total_samples']
            success_total = f"{originally_correct_and_attack_successful}/{total_samples}"

            print(f"{attack_type.upper():<12} "
                  f"{det_results['accuracy']:<10.4f} "
                  f"{det_results['f1']:<10.4f} "
                  f"{det_results['auc']:<8.4f} "
                  f"{det_results['num_rules']:<6} "
                  f"{att_results['attack_success_rate']:<13.4f} "
                  f"{success_total:<40} "
                  f"{params_str:<25}")

    print("=" * 120)
    print("註解說明：")
    print("- Detection: 偵測器準確率（區分攻擊成功vs攻擊失敗的能力）")
    print("- Success Rate: 攻擊成功率（對抗樣本成功欺騙模型的比例）")
    print("- originally_correct_&attack_successful/Total: 原本正確且攻擊成功的樣本數/總樣本數")
    print("- 標籤策略: 攻擊成功=1, 攻擊失敗=0")
    print("- 僅使用原本預測正確的樣本進行訓練和測試")


if __name__ == '__main__':
    main()

