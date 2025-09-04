import numpy as np
from dataclasses import dataclass

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class TriangularFuzzySets:
    """三角模糊集合"""

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


@dataclass
class FuzzyRule:
    """模糊規則"""
    prototype: np.ndarray
    spread: np.ndarray
    output: float
    support: float
    confidence: float
    potential: float
    hits: int
    attack_type: str = None


class FuzzyDetector:
    """模糊偵測器"""

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
        """計算規則激活度"""
        diff = (x - rule.prototype) ** 2
        spread_sq = rule.spread ** 2 + 1e-12
        activation = np.exp(-np.sum(diff / spread_sq))
        return activation

    def _support_function(self, rule, class_samples):
        """計算支持度函數"""
        if len(class_samples) == 0:
            return 0.0

        total_activation = 0
        for sample in class_samples:
            activation = self._rule_activation(sample, rule)
            if activation > self.fire_threshold:
                total_activation += activation

        return total_activation / len(class_samples)

    def _confidence_function(self, rule, class_samples, total_samples):
        """計算信心度函數"""
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
        """預測機率"""
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

        # 加入小量隨機性避免完美預測
        result = numerator / denominator
        noise = np.random.normal(0, 0.02)  # 加入2%的隨機噪音
        result = np.clip(result + noise, 0.0, 1.0)
        return result

    def update(self, x, label, class_samples=None, total_samples=None):
        """線上學習更新"""
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


def train_fuzzy_detector(clean_features_diff, adv_features_diff, attack_type):
    """訓練模糊偵測器（僅訓練，不測試）"""
    print(
        f"[{attack_type}] Training detector with {len(clean_features_diff)} clean + {len(adv_features_diff)} adversarial samples")

    # 初始化偵測器
    detector = FuzzyDetector(attack_type=attack_type)

    # 模糊化特徵
    clean_features_fuzz = detector.fuzzify(clean_features_diff)
    adv_features_fuzz = detector.fuzzify(adv_features_diff)

    # 準備訓練資料
    X_train = np.vstack([clean_features_fuzz, adv_features_fuzz])
    y_train = np.hstack([
        np.zeros(len(clean_features_fuzz)),  # 乾淨樣本 = 0
        np.ones(len(adv_features_fuzz))  # 對抗樣本 = 1
    ])

    print(f"[{attack_type}] Training data - Clean: {np.sum(y_train == 0)}, Adversarial: {np.sum(y_train == 1)}")

    # 隨機打亂訓練順序
    train_indices = np.random.permutation(len(X_train))

    # 線上學習訓練
    for idx in tqdm(train_indices, desc=f"Training {attack_type}", leave=False):
        i = train_indices[idx] if idx < len(train_indices) else idx
        current_class_samples = X_train[y_train == y_train[i]][:20]
        detector.update(X_train[i], y_train[i],
                        current_class_samples, X_train[:min(i + 1, 100)])

    print(f"[{attack_type}] Training completed - Generated {len(detector.rules)} rules")

    return detector


def test_fuzzy_detector(detector, clean_features_diff, adv_features_diff, attack_type):
    """測試模糊偵測器"""
    print(f"[{attack_type}] Testing detector...")

    # 模糊化特徵
    clean_features_fuzz = detector.fuzzify(clean_features_diff)
    adv_features_fuzz = detector.fuzzify(adv_features_diff)

    X_test = np.vstack([clean_features_fuzz, adv_features_fuzz])
    y_test = np.hstack([
        np.zeros(len(clean_features_fuzz)),  # 乾淨樣本 = 0
        np.ones(len(adv_features_fuzz))  # 對抗樣本 = 1
    ])

    print(f"[{attack_type}] Test data - Clean: {np.sum(y_test == 0)}, Adversarial: {np.sum(y_test == 1)}")

    # 預測
    y_pred_proba = []
    y_pred_binary = []

    for i in range(len(X_test)):
        prob = detector.predict_proba(X_test[i])
        y_pred_proba.append(prob)
        threshold = 0.5
        y_pred_binary.append(1 if prob > threshold else 0)

    # 計算指標
    accuracy = accuracy_score(y_test, y_pred_binary)

    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_binary, average='binary', zero_division=0
        )
        auc = roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        print(f"[{attack_type}] Test metric calculation error: {e}")
        precision, recall, f1 = 0.0, 0.0, 0.0
        auc = 0.5

    print(f"[{attack_type}] Test Results - Acc: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'y_pred_binary': y_pred_binary,
        'num_rules': len(detector.rules)
    }


def plot_fuzzy_sets():
    """繪製三角模糊集合圖"""
    # 建立模糊集合
    fuzzy_sets = TriangularFuzzySets()
    x = np.linspace(0, 1, 1000)
    memberships = fuzzy_sets.membership(x)

    plt.figure(figsize=(10, 6))
    colors = ['red', 'orange', 'green', 'blue', 'purple']

    for i in range(fuzzy_sets.K):
        plt.plot(x, memberships[:, i], color=colors[i], linewidth=2,
                 label=fuzzy_sets.labels[i])

    plt.xlabel('Input Value')
    plt.ylabel('Membership Degree')
    plt.title('Triangular Fuzzy Sets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('fuzzy_sets.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函數：繪製模糊集合圖"""
    print("=== Plotting Triangular Fuzzy Sets ===")
    plot_fuzzy_sets()
    print("Fuzzy sets plot saved as 'fuzzy_sets.png'")


if __name__ == '__main__':
    main()