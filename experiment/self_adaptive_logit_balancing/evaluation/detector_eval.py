### 8. `evaluation/detector_eval.py` - 檢測器評估


"""
檢測器性能評估
"""
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class DetectorEvaluator:
    def __init__(self, attack_types):
        self.attack_types = attack_types

    def evaluate(self, y_true, y_pred):
        """評估檢測器性能"""
        cm = confusion_matrix(y_true, y_pred)

        # 計算每類的準確率
        class_accuracies = {}
        for i, attack_name in enumerate(self.attack_types):
            if cm[i, :].sum() > 0:
                acc = cm[i, i] / cm[i, :].sum() * 100
            else:
                acc = 0
            class_accuracies[attack_name] = acc

        # 計算總體準確率
        overall_acc = np.trace(cm) / cm.sum() * 100

        # 計算 Precision, Recall, F1
        report = classification_report(
            y_true, y_pred,
            target_names=self.attack_types,
            output_dict=True,
            zero_division=0
        )

        return {
            'confusion_matrix': cm,
            'class_accuracies': class_accuracies,
            'overall_accuracy': overall_acc,
            'classification_report': report
        }

    def print_results(self, results):
        """打印評估結果（與附圖格式相同）"""
        cm = results['confusion_matrix']
        class_accs = results['class_accuracies']
        overall_acc = results['overall_accuracy']

        print(f"\n{'=' * 90}")
        print("Detection Results (Confusion Matrix)")
        print(f"{'=' * 90}\n")

        # 打印混淆矩陣（百分比格式）
        print("Predicted →")
        print(f"{'True ↓':<12}", end="")
        for attack_name in self.attack_types:
            print(f"{attack_name:>10}", end="")
        print()
        print("-" * 90)

        for i, true_attack in enumerate(self.attack_types):
            print(f"{true_attack:<12}", end="")
            row_sum = cm[i, :].sum()
            for j in range(len(self.attack_types)):
                if row_sum > 0:
                    percentage = cm[i, j] / row_sum * 100
                    print(f"{percentage:>9.1f}%", end="")
                else:
                    print(f"{'0.0%':>10}", end="")
            print()

        print("-" * 90)

        # 打印每類準確率
        print(f"\n{'=' * 90}")
        print("Per-Class Detection Accuracy")
        print(f"{'=' * 90}\n")

        for attack_name, acc in class_accs.items():
            print(f"  {attack_name:<15}: {acc:>6.2f}%")

        print(f"\n  {'Overall':<15}: {overall_acc:>6.2f}%")
        print(f"{'=' * 90}\n")

        return cm

    def print_detailed_metrics(self, results):
        """打印詳細指標"""
        report = results['classification_report']

        print(f"\n{'=' * 90}")
        print("Detailed Classification Metrics")
        print(f"{'=' * 90}\n")

        print(f"{'Attack Type':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>10}")
        print("-" * 90)

        for attack_name in self.attack_types:
            metrics = report[attack_name]
            print(f"{attack_name:<15} | "
                  f"{metrics['precision'] * 100:>9.2f}% | "
                  f"{metrics['recall'] * 100:>9.2f}% | "
                  f"{metrics['f1-score'] * 100:>9.2f}% | "
                  f"{int(metrics['support']):>10}")

        print("-" * 90)

        # Macro 和 Weighted 平均
        macro = report['macro avg']
        weighted = report['weighted avg']

        print(f"{'Macro Avg':<15} | "
              f"{macro['precision'] * 100:>9.2f}% | "
              f"{macro['recall'] * 100:>9.2f}% | "
              f"{macro['f1-score'] * 100:>9.2f}% | "
              f"{int(macro['support']):>10}")

        print(f"{'Weighted Avg':<15} | "
              f"{weighted['precision'] * 100:>9.2f}% | "
              f"{weighted['recall'] * 100:>9.2f}% | "
              f"{weighted['f1-score'] * 100:>9.2f}% | "
              f"{int(weighted['support']):>10}")

        print("-" * 90)
