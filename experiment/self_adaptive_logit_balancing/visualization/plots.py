"""
所有可視化函數
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    def __init__(self, attack_types):
        self.attack_types = attack_types
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        cm = cm.T

        cm_percentage = np.zeros_like(cm, dtype=float)
        for i in range(cm.shape[0]):
            col_sum = cm[:, i].sum()
            if col_sum > 0:
                cm_percentage[:, i] = cm[:, i] / col_sum

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(cm_percentage, annot=True, fmt='.1%', cmap='Greens',
                    xticklabels=self.attack_types,
                    yticklabels=self.attack_types,
                    cbar_kws={'label': 'Detection Rate'},
                    vmin=0, vmax=1,
                    linewidths=0.5, linecolor='gray',
                    ax=ax)

        ax.set_xlabel('True Attack Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Attack Type', fontsize=12, fontweight='bold')
        ax.set_title('Detection Results of Batch Size = 60',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Confusion matrix saved to {save_path}")
        plt.close()

    def plot_robustness_comparison(self, robustness_results, save_path='robustness_comparison.png'):
        """繪製魯棒性比較圖"""
        fig, ax = plt.subplots(figsize=(12, 6))

        attacks = list(robustness_results.keys())
        accuracies = list(robustness_results.values())

        colors = ['green' if attack == 'Clean' else 'red' for attack in attacks]
        bars = ax.bar(attacks, accuracies, color=colors, alpha=0.7, edgecolor='black')

        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Robustness Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Robustness comparison saved to {save_path}")
        plt.close()

    def plot_detection_performance(self, results, save_path='detection_performance.png'):
        """繪製檢測性能分析"""
        report = results['classification_report']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左圖：每類檢測準確率
        class_accs = results['class_accuracies']
        attacks = list(class_accs.keys())
        accs = list(class_accs.values())

        colors = ['green' if acc > 90 else 'orange' if acc > 70 else 'red' for acc in accs]
        bars1 = ax1.barh(attacks, accs, color=colors, alpha=0.7, edgecolor='black')

        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height() / 2.,
                     f'{width:.1f}%',
                     ha='left', va='center', fontsize=10, fontweight='bold')

        ax1.set_xlabel('Detection Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Per-Class Detection Accuracy', fontsize=13, fontweight='bold')
        ax1.set_xlim(0, 110)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')

        # 右圖：Precision, Recall, F1-Score
        metrics_names = ['Precision', 'Recall', 'F1-Score']
        x = np.arange(len(self.attack_types))
        width = 0.25

        precisions = [report[attack]['precision'] * 100 for attack in self.attack_types]
        recalls = [report[attack]['recall'] * 100 for attack in self.attack_types]
        f1_scores = [report[attack]['f1-score'] * 100 for attack in self.attack_types]

        ax2.bar(x - width, precisions, width, label='Precision', alpha=0.8, edgecolor='black')
        ax2.bar(x, recalls, width, label='Recall', alpha=0.8, edgecolor='black')
        ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, edgecolor='black')

        ax2.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Detection Metrics Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.attack_types, rotation=45, ha='right')
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 110)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Detection performance saved to {save_path}")
        plt.close()

    def plot_log_softmax_distribution(self, distribution_stats, save_path='log_softmax_distribution.png'):
        """繪製 Log-Softmax 分布（與附圖格式相同：Min vs Max 散點圖）"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # 定義顏色
        colors = {
            'Clean': 'black',
            'PGD-Linf': 'blue',
            'APGD-Linf': 'darkorange',
            'APGDT-Linf': 'dimgray',
            'Square-Linf': 'gold',
            'FAB-Linf': 'purple',
            'PGD-L2': 'green',
            'CW-L2': 'deepskyblue'
        }

        # 定義標記形狀
        markers = {
            'Clean': 'o',  # 圓形
            'PGD-Linf': 'D',  # 菱形
            'APGD-Linf': 's',  # 正方形
            'APGDT-Linf': '^',  # 上三角
            'Square-Linf': 'v',  # 下三角
            'FAB-Linf': '*',  # 星形
            'PGD-L2': 'p',  # 五角形
            'CW-L2': 'h'  # 六角形
        }

        # 定義點大小
        sizes = {
            'Clean': 80,
            'PGD-Linf': 70,
            'APGD-Linf': 70,
            'APGDT-Linf': 80,
            'Square-Linf': 80,
            'FAB-Linf': 120,  # 星形需要大一點
            'PGD-L2': 80,
            'CW-L2': 80
        }

        # 繪製每個攻擊類型的散點
        for attack_name, stats in distribution_stats.items():
            batch_mins = stats.get('batch_mins', [])
            batch_maxs = stats.get('batch_maxs', [])

            if len(batch_mins) > 0 and len(batch_maxs) > 0:
                ax.scatter(
                    batch_mins, batch_maxs,
                    c=colors.get(attack_name, 'black'),
                    marker=markers.get(attack_name, 'o'),
                    s=sizes.get(attack_name, 80),
                    label=attack_name,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.8
                )

        ax.set_xlabel('Min Log-Softmax', fontsize=12, fontweight='bold')
        ax.set_ylabel('Max Log-Softmax', fontsize=12, fontweight='bold')
        ax.set_title('Log-Softmax Distribution (Batch Average, Batch Size=60)',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Log-Softmax distribution saved to {save_path}")
        plt.close()


