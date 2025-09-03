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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

    def forward_features(self, x):
        """提取深層特徵（在最後分類層之前）"""
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
        features = F.relu(self.fc2(x))  # 128維特徵向量
        return features

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

    # 設定攻擊參數（適中強度）
    attacks = {}
    attack_params = {}

    if 'fgsm' in attack_types:
        eps = np.random.uniform(0.35, 0.45)
        attack_params['fgsm'] = {'eps': eps}
        attacks['fgsm'] = FastGradientMethod(estimator=art_clf, eps=eps)

    if 'pgd' in attack_types:
        eps = np.random.uniform(0.20, 0.30)
        max_iter = np.random.randint(40, 80)
        eps_step = eps / max_iter
        attack_params['pgd'] = {'eps': eps, 'max_iter': max_iter}
        attacks['pgd'] = ProjectedGradientDescent(
            estimator=art_clf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter
        )

    if 'deepfool' in attack_types:
        max_iter = np.random.randint(5, 10)
        epsilon = np.random.uniform(0.01, 0.03)
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


# 🔧 新增：提取深層特徵函數
def extract_deep_features(model, images, batch_size=256):
    """
    提取模型倒數第二層的深層特徵（128維）
    """
    model.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[i:i + batch_size]).float().to(device)
            # 使用 forward_features 方法提取128維特徵
            deep_features = model.forward_features(batch)  # (batch_size, 128)
            features.append(deep_features.cpu().numpy())

    return np.concatenate(features, axis=0)


# 🔧 新增：計算深層特徵差異
def compute_feature_differences(features1, features2):
    """
    計算兩組深層特徵之間的各種差異指標
    features1, features2: (N, 128) 深層特徵向量
    """

    # 1. 歐幾里得距離
    euclidean_dist = np.linalg.norm(features1 - features2, axis=1)

    # 2. 餘弦距離
    def cosine_distance(a, b):
        dot_product = np.sum(a * b, axis=1)
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        cosine_sim = dot_product / (norm_a * norm_b + 1e-12)
        return 1 - cosine_sim

    cosine_dist = cosine_distance(features1, features2)

    # 3. 曼哈頓距離（L1距離）
    manhattan_dist = np.sum(np.abs(features1 - features2), axis=1)

    # 4. 特徵向量的統計差異
    # 均值差異
    mean_diff = np.abs(np.mean(features1, axis=1) - np.mean(features2, axis=1))

    # 標準差差異
    std_diff = np.abs(np.std(features1, axis=1) - np.std(features2, axis=1))

    # 5. 特徵激活模式差異
    # 計算每個特徵維度的相對變化
    relative_change = np.mean(np.abs((features1 - features2) / (features1 + 1e-12)), axis=1)

    # 6. 特徵能量差異
    energy1 = np.sum(features1 ** 2, axis=1)
    energy2 = np.sum(features2 ** 2, axis=1)
    energy_diff = np.abs(energy1 - energy2)

    # 組合所有差異指標
    all_diffs = [euclidean_dist, cosine_dist, manhattan_dist,
                 mean_diff, std_diff, relative_change, energy_diff]

    # 正規化每個特徵到 [0,1]
    normalized_diffs = []
    for diff in all_diffs:
        if diff.max() > diff.min():
            norm_diff = (diff - diff.min()) / (diff.max() - diff.min())
        else:
            norm_diff = np.zeros_like(diff)
        normalized_diffs.append(norm_diff)

    return np.column_stack(normalized_diffs)


# 取得 softmax 機率向量（用於攻擊效果評估）
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


# 改進模糊偵測器類別
class FuzzyDetector:
    def __init__(self,
                 init_spread=0.2,
                 learning_rate=0.05,
                 add_threshold=0.3,
                 fire_threshold=0.15,
                 max_rules=100,
                 alpha=0.9,
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
        noise = np.random.normal(0, 0.01)
        result = np.clip(result + noise, 0.0, 1.0)
        return result

    def update(self, x, label, class_samples=None, total_samples=None):
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
                rule.output += adaptive_lr * 0.2 * (label - rule.output)

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
        min_potential = 0.01
        self.rules = [rule for rule in self.rules if rule.potential > min_potential]


# 🔧 完全重寫的訓練函數（使用深層特徵）
def train_fuzzy_detector(clean_data, adv_data, fuzzy_sets, attack_type, test_ratio=0.3):
    """
    修正版：使用深層特徵的訓練函數
    clean_data 和 adv_data 包含 'images', 'labels', 'predictions'
    """

    # 🎯 關鍵修正：使用圖像數據提取深層特徵
    clean_images = clean_data['images']  # (N, 1, 28, 28)
    adv_images = adv_data['images']  # (N, 1, 28, 28)

    print(f"Extracting deep features for {attack_type}...")

    # 提取深層特徵（128維）
    clean_features = extract_deep_features(model, clean_images)  # (N, 128)
    adv_features = extract_deep_features(model, adv_images)  # (N, 128)

    print(f"Clean features shape: {clean_features.shape}")
    print(f"Adversarial features shape: {adv_features.shape}")

    # 🔧 避免資料洩漏的方法：
    # 將乾淨樣本分成兩組，使用一組作為參考基準
    n_clean = len(clean_features)
    mid_point = n_clean // 2

    clean_ref = clean_features[:mid_point]  # 前半作為參考
    clean_test = clean_features[mid_point:]  # 後半作為測試

    # 使用前半乾淨樣本的平均作為基準
    clean_baseline = np.mean(clean_ref, axis=0, keepdims=True)  # (1, 128)

    # 計算特徵差異（相對於基準）
    clean_features_diff = compute_feature_differences(clean_test, clean_baseline)
    adv_features_diff = compute_feature_differences(adv_features, clean_baseline)

    # 標籤
    clean_labels = np.zeros(len(clean_features_diff))  # 乾淨樣本標籤為0
    adv_labels = np.ones(len(adv_features_diff))  # 對抗樣本標籤為1

    # 初始化偵測器
    detector = FuzzyDetector(attack_type=attack_type)

    # 模糊化特徵
    clean_fuzzy_features = detector.fuzzify(clean_features_diff)
    adv_fuzzy_features = detector.fuzzify(adv_features_diff)

    # 建立訓練資料
    X = np.vstack([clean_fuzzy_features, adv_fuzzy_features])
    y = np.hstack([clean_labels, adv_labels])

    # 資料分割
    n_train = int(len(X) * (1 - test_ratio))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 🔍 特徵分析
    print(f"\n🔍 Deep Feature Analysis for {attack_type}:")
    clean_train = X_train[y_train == 0]
    adv_train = X_train[y_train == 1]

    if len(clean_train) > 0 and len(adv_train) > 0:
        print(f"Clean samples - Mean: {np.mean(clean_train):.4f}, Std: {np.std(clean_train):.4f}")
        print(f"Adv samples - Mean: {np.mean(adv_train):.4f}, Std: {np.std(adv_train):.4f}")

        # 檢查特徵分離度
        clean_mean = np.mean(clean_train, axis=0)
        adv_mean = np.mean(adv_train, axis=0)
        separation = np.mean(np.abs(clean_mean - adv_mean))
        print(f"Feature separation: {separation:.4f}")

    # 訓練偵測器
    train_indices = np.random.permutation(len(X_train))

    for idx in tqdm(train_indices, desc=f"Training {attack_type}", leave=False):
        i = train_indices[idx] if idx < len(train_indices) else idx
        current_class_samples = X_train[y_train == y_train[i]][:15]
        detector.update(X_train[i], y_train[i],
                        current_class_samples, X_train[:min(i + 1, 80)])

    # 測試
    y_pred_proba = []
    y_pred_binary = []

    for i in range(len(X_test)):
        prob = detector.predict_proba(X_test[i])
        y_pred_proba.append(prob)
        y_pred_binary.append(1 if prob > 0.5 else 0)

    # 計算指標
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_binary, average='binary')

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.5

    # 交叉驗證檢查
    try:
        rf_clf = RandomForestClassifier(random_state=42, n_estimators=50)
        cv_scores = cross_val_score(rf_clf, X, y, cv=3)
        print(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        if cv_scores.mean() > 0.98:
            print("⚠️  WARNING: Possible overfitting detected!")
    except:
        pass

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
        'clean_baseline': clean_baseline  # 保存基準用於後續測試
    }

    return results


def main():
    seed = int(time.time()) % 10000
    set_seed(seed)

    # 設定運算裝置
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("=== Fuzzy Adversarial Attack Detection (Deep Features Version) ===")

    train_loader, test_loader = load_mnist(batch_size=256, shuffle_test=True)

    # 訓練或載入分類器
    model_path = "./simple_mnist_cnn.pth"
    global model
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

    # 取得預測結果（用於攻擊效果評估）
    print("Getting predictions for attack effectiveness evaluation...")
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

    print("\n🎯 Attack Effectiveness Analysis:")
    for attack_type, effectiveness in attack_effectiveness.items():
        print(f"{attack_type.upper():12} - Success Rate: {effectiveness['attack_success_rate']:.4f}, "
              f"Adv Accuracy: {effectiveness['adversarial_accuracy']:.4f}")

    fuzzy_sets = TriangularFuzzySets()

    # 對每種攻擊類型訓練偵測器
    detection_results = {}

    print("\nTraining detectors using deep features...")
    for attack_type in ['fgsm', 'pgd']:
        if attack_type in results:
            clean_data = results['clean']
            adv_data = results[attack_type]

            detector_results = train_fuzzy_detector(clean_data, adv_data, fuzzy_sets, attack_type)
            detection_results[attack_type] = detector_results

    # 🔍 額外的驗證測試
    print("\n🔬 Additional Validation Tests:")

    # 測試1: 噪音樣本測試
    print("Testing with noisy clean samples...")
    noise_levels = [0.01, 0.05, 0.1]
    for noise_level in noise_levels:
        noisy_clean = results['clean']['images'] + np.random.normal(0, noise_level,
                                                                    results['clean']['images'].shape)

        # 🔧 修正：使用深層特徵
        noisy_features = extract_deep_features(model, noisy_clean)

        # 測試每個偵測器對噪音的敏感度
        for attack_type, det_results in detection_results.items():
            detector = det_results['detector']
            clean_baseline = det_results['clean_baseline']  # 使用保存的基準

            # 計算噪音樣本與基準的差異
            noisy_features_diff = compute_feature_differences(noisy_features, clean_baseline)
            noisy_fuzzy_features = detector.fuzzify(noisy_features_diff)

            # 計算誤報率
            false_positives = 0
            for i in range(min(100, len(noisy_fuzzy_features))):
                prob = detector.predict_proba(noisy_fuzzy_features[i])
                if prob > 0.5:
                    false_positives += 1

            false_positive_rate = false_positives / min(100, len(noisy_fuzzy_features))
            print(f"  {attack_type} detector - Noise {noise_level:.2f}: FPR = {false_positive_rate:.4f}")

    # 測試2: 跨攻擊檢測能力
    print("\nCross-attack detection capability:")
    for detector_type in detection_results.keys():
        detector = detection_results[detector_type]['detector']
        clean_baseline = detection_results[detector_type]['clean_baseline']

        for test_attack_type in detection_results.keys():
            if detector_type != test_attack_type:
                test_images = results[test_attack_type]['images']

                # 🔧 修正：使用深層特徵
                test_features = extract_deep_features(model, test_images)
                test_features_diff = compute_feature_differences(test_features, clean_baseline)
                test_fuzzy_features = detector.fuzzify(test_features_diff)

                # 測試檢測率
                detections = 0
                for i in range(min(100, len(test_fuzzy_features))):
                    prob = detector.predict_proba(test_fuzzy_features[i])
                    if prob > 0.5:
                        detections += 1

                cross_detection_rate = detections / min(100, len(test_fuzzy_features))
                print(f"  {detector_type} → {test_attack_type}: {cross_detection_rate:.4f}")

    # 🔧 修正特徵重要性分析
    print("\nFeature importance analysis:")
    for attack_type, det_results in detection_results.items():
        detector = det_results['detector']

        if len(detector.rules) > 0:
            # 計算規則的平均原型
            avg_prototype = np.mean([rule.prototype for rule in detector.rules], axis=0)

            # 找出最重要的特徵維度
            feature_importance = np.abs(avg_prototype - 0.5)  # 距離中性值的距離
            top_features = np.argsort(feature_importance)[-5:]  # 前5個重要特徵

            # 🔧 修正格式化問題
            importance_values = feature_importance[top_features]
            importance_str = ', '.join([f"{val:.4f}" for val in importance_values])

            print(f"  {attack_type} - Top features: {top_features.tolist()}, Importance: [{importance_str}]")
        else:
            print(f"  {attack_type} - No rules generated")

    # 最終統整表格
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY (DEEP FEATURES VERSION)")
    print("=" * 100)
    print(f"Random seed: {seed} | Base model accuracy: {base_acc:.4f}")
    print()

    # 表格標題
    header = f"{'Attack':<12} {'Detection':<10} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'AUC':<8} {'Rules':<6} {'Success Rate':<13} {'Params':<30}"
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

            print(f"{attack_type.upper():<12} "
                  f"{det_results['accuracy']:<10.4f} "
                  f"{det_results['precision']:<10.4f} "
                  f"{det_results['recall']:<8.4f} "
                  f"{det_results['f1']:<10.4f} "
                  f"{det_results['auc']:<8.4f} "
                  f"{det_results['num_rules']:<6} "
                  f"{att_results['attack_success_rate']:<13.4f} "
                  f"{params_str:<30}")

    print("=" * 100)

    # 🎯 總結和建議
    print("\n📊 ANALYSIS SUMMARY:")
    avg_detection = np.mean([det_results['accuracy'] for det_results in detection_results.values()])
    avg_f1 = np.mean([det_results['f1'] for det_results in detection_results.values()])
    avg_precision = np.mean([det_results['precision'] for det_results in detection_results.values()])
    avg_recall = np.mean([det_results['recall'] for det_results in detection_results.values()])

    print(f"Average Detection Accuracy: {avg_detection:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    # 性能評估
    if avg_detection > 0.95:
        print("⚠️  Still showing high accuracy - consider further validation")
        print("   Recommendation: Check for data leakage or overfitting")
    elif avg_detection > 0.85:
        print("✅ Reasonable detection performance")
        print("   The fuzzy detector shows good discriminative ability")
    elif avg_detection > 0.70:
        print("📈 Moderate detection performance")
        print("   Consider tuning hyperparameters or feature engineering")
    else:
        print("❌ Detection performance needs significant improvement")
        print("   Consider different feature extraction or model architecture")

    # 平衡性評估
    if abs(avg_precision - avg_recall) < 0.1:
        print("⚖️  Good precision-recall balance")
    elif avg_precision > avg_recall + 0.1:
        print("🎯 High precision, lower recall - conservative detector")
    else:
        print("🔍 High recall, lower precision - aggressive detector")

    # 規則複雜度分析
    total_rules = sum([det_results['num_rules'] for det_results in detection_results.values()])
    avg_rules = total_rules / len(detection_results) if len(detection_results) > 0 else 0
    print(f"Average rules per detector: {avg_rules:.1f}")

    if avg_rules < 20:
        print("🎯 Compact rule base - good interpretability")
    elif avg_rules < 50:
        print("📊 Moderate rule base - acceptable complexity")
    else:
        print("📈 Large rule base - may need pruning")

    # 🔍 額外統計分析
    print("\n🔍 DETAILED ANALYSIS:")

    # 攻擊強度分析
    print("Attack Strength Analysis:")
    for attack_type, params in attack_params.items():
        effectiveness = attack_effectiveness[attack_type]['attack_success_rate']
        if attack_type == 'fgsm':
            print(f"  FGSM (ε={params['eps']:.3f}): {effectiveness:.4f} success rate")
        elif attack_type == 'pgd':
            print(f"  PGD (ε={params['eps']:.3f}, iter={params['max_iter']}): {effectiveness:.4f} success rate")
        elif attack_type == 'deepfool':
            print(
                f"  DeepFool (iter={params['max_iter']}, ε={params['epsilon']:.3f}): {effectiveness:.4f} success rate")

    # 檢測器穩定性分析
    print("\nDetector Stability Analysis:")
    for attack_type, det_results in detection_results.items():
        precision = det_results['precision']
        recall = det_results['recall']
        f1 = det_results['f1']

        stability_score = 1 - abs(precision - recall)  # 精確度和召回率的平衡性
        print(f"  {attack_type.upper()} detector stability: {stability_score:.4f}")

    # 深層特徵分析
    print("\nDeep Feature Analysis:")
    for attack_type, det_results in detection_results.items():
        detector = det_results['detector']
        if len(detector.rules) > 0:
            # 分析規則的分佈
            rule_outputs = [rule.output for rule in detector.rules]
            rule_potentials = [rule.potential for rule in detector.rules]

            print(f"  {attack_type.upper()}:")
            print(f"    Rule outputs - Mean: {np.mean(rule_outputs):.4f}, Std: {np.std(rule_outputs):.4f}")
            print(f"    Rule potentials - Mean: {np.mean(rule_potentials):.4f}, Std: {np.std(rule_potentials):.4f}")
            print(f"    Active rules: {len([r for r in detector.rules if r.potential > 0.1])}")

    # 保存結果
    results_summary = {
        'seed': seed,
        'base_accuracy': base_acc,
        'attack_params': attack_params,
        'attack_effectiveness': attack_effectiveness,
        'detection_results': {k: {
            'accuracy': v['accuracy'],
            'precision': v['precision'],
            'recall': v['recall'],
            'f1': v['f1'],
            'auc': v['auc'],
            'num_rules': v['num_rules']
        } for k, v in detection_results.items()},
        'summary_metrics': {
            'avg_detection_accuracy': avg_detection,
            'avg_f1_score': avg_f1,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_rules_per_detector': avg_rules
        }
    }

    # 可選：保存到文件
    import json
    try:
        with open(f'deep_features_detection_results_{seed}.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n💾 Results saved to: deep_features_detection_results_{seed}.json")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

    # 🔍 最終驗證建議
    print("\n🔬 VALIDATION RECOMMENDATIONS:")
    print("1. Run multiple times with different seeds to check consistency")
    print("2. Test on different datasets (CIFAR-10, Fashion-MNIST)")
    print("3. Evaluate against stronger attacks (C&W, AutoAttack)")
    print("4. Compare with traditional ML detectors (SVM, Random Forest)")
    print("5. Analyze computational efficiency and real-time performance")
    print("6. Visualize deep feature distributions using t-SNE or UMAP")

    # 🎯 實驗建議
    print("\n🎯 DEEP FEATURES EXPERIMENT SUGGESTIONS:")
    if avg_detection < 0.8:
        print("• Consider using different layers for feature extraction")
        print("• Try feature selection or dimensionality reduction")
        print("• Experiment with different distance metrics in feature space")

    if avg_rules > 80:
        print("• Consider rule pruning strategies to reduce complexity")
        print("• Implement rule merging for similar prototypes")
        print("• Use feature clustering before rule generation")

    if abs(avg_precision - avg_recall) > 0.2:
        print("• Adjust detection threshold for better precision-recall balance")
        print("• Consider cost-sensitive learning approaches")
        print("• Use ensemble methods to improve stability")

    # 深層特徵特定建議
    print("\n🧠 DEEP FEATURES SPECIFIC RECOMMENDATIONS:")
    print("• Analyze which feature dimensions are most discriminative")
    print("• Consider using attention mechanisms to weight important features")
    print("• Experiment with different CNN architectures (ResNet, DenseNet)")
    print("• Try using features from multiple layers simultaneously")
    print("• Investigate feature visualization techniques")

    print("\n✅ Deep Features Experiment completed successfully!")
    print("=" * 100)


if __name__ == '__main__':
    main()

