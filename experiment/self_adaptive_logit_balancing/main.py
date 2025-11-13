"""
主程序 - 完整實驗流程
"""
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.wide_resnet import WideResNet
from models.detector import AdversarialDetectorMLP
from attacks.attack_generator import AttackGenerator
from training.logit_balancing import LogitBalancingTrainer
from training.detector_trainer import DetectorTrainer
from evaluation.detector_eval import DetectorEvaluator
from visualization.plots import Visualizer
from utils.helpers import (
    get_device_info,
    save_model,
    load_model,
    split_data,
    evaluate_model_accuracy,
    compute_log_softmax_stats
)

def main():
    print("\n" + "="*70)
    print("Self-Adaptive Logit Balancing Training and Adversarial Detection")
    print("="*70 + "\n")

    # ==================== 初始化 ====================
    device = get_device_info()

    # 創建結果目錄
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # ==================== 步驟 1: 準備數據集 ====================
    print(f"\n{'='*70}")
    print("STEP 1: Loading CIFAR-10 Dataset")
    print(f"{'='*70}\n")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=Config.DATA_ROOT, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=Config.DATA_ROOT, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        trainset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        testset, batch_size=Config.TEST_BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    # 創建小測試集用於生成對抗樣本
    test_subset_indices = list(range(Config.ADV_NUM_SAMPLES))
    test_subset = Subset(testset, test_subset_indices)
    test_subset_loader = DataLoader(
        test_subset, batch_size=Config.TEST_BATCH_SIZE, shuffle=False
    )

    print(f"[INFO] Training samples: {len(trainset)}")
    print(f"[INFO] Test samples: {len(testset)}")
    print(f"[INFO] Adversarial generation samples: {Config.ADV_NUM_SAMPLES}")

    # ==================== 步驟 2: 訓練 Logit Balancing 模型 ====================
    print(f"\n{'=' * 70}")
    print("STEP 2: Training Logit Balancing Model")
    print(f"{'=' * 70}\n")

    model = WideResNet(
        depth=Config.MODEL_DEPTH,
        widen_factor=Config.MODEL_WIDEN_FACTOR,
        dropout_rate=Config.MODEL_DROPOUT,
        num_classes=Config.NUM_CLASSES
    ).to(device)

    print(f"[INFO] Model: WideResNet-{Config.MODEL_DEPTH}-{Config.MODEL_WIDEN_FACTOR}")
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 嘗試載入已訓練的模型，若失敗則重新訓練
    if os.path.exists(Config.LB_MODEL_SAVE_PATH):
        print(f"[INFO] Loading existing model from {Config.LB_MODEL_SAVE_PATH}...")
        try:
            model = load_model(model, Config.LB_MODEL_SAVE_PATH, device)
            print(f"[SUCCESS] Model loaded successfully!")
        except Exception as e:
            print(f"[WARNING] Failed to load model: {e}")
            print(f"[INFO] Training new model...")
            trainer = LogitBalancingTrainer(
                model=model,
                device=device,
                beta=Config.LB_BETA,
                sigma=Config.LB_SIGMA,
                lr=Config.LB_LR
            )
            model = trainer.train(train_loader, epochs=Config.LB_EPOCHS)
            save_model(model, Config.LB_MODEL_SAVE_PATH)
    else:
        print(f"[INFO] No existing model found. Training new model...")
        trainer = LogitBalancingTrainer(
            model=model,
            device=device,
            beta=Config.LB_BETA,
            sigma=Config.LB_SIGMA,
            lr=Config.LB_LR
        )
        model = trainer.train(train_loader, epochs=Config.LB_EPOCHS)
        save_model(model, Config.LB_MODEL_SAVE_PATH)

    # 評估乾淨數據準確率
    clean_acc = evaluate_model_accuracy(model, test_loader, device)
    print(f"\n[RESULT] Clean Test Accuracy: {clean_acc:.2f}%\n")

    # ==================== 步驟 3: 生成對抗樣本 ====================
    print(f"\n{'=' * 70}")
    print("STEP 3: Generating Adversarial Examples")
    print(f"{'=' * 70}\n")

    # 打印緩存信息
    from utils.helpers import print_cache_summary
    print_cache_summary(Config.ADVERSARIAL_CACHE_DIR)

    # 從 Config.SOURCE_MODEL_SAVE_PATH 加載預訓練模型
    source_model = WideResNet(
        depth=Config.MODEL_DEPTH,
        widen_factor=Config.MODEL_WIDEN_FACTOR,
        dropout_rate=Config.MODEL_DROPOUT,
        num_classes=Config.NUM_CLASSES
    ).to(device)

    print(f"[INFO] Loading source model from {Config.SOURCE_MODEL_SAVE_PATH}...")
    try:
        source_model = load_model(source_model, Config.SOURCE_MODEL_SAVE_PATH, device)
        print(f"[SUCCESS] Source model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load source model: {e}")
        sys.exit(1)  # 如果無法加載模型，直接退出程序，避免後續執行錯誤

    # 初始化攻擊生成器（啟用緩存）
    attack_gen = AttackGenerator(
        model=source_model,
        device=device,
        use_cache=Config.USE_CACHE
    )

    # 生成所有攻擊（自動使用緩存）
    adversarial_data = attack_gen.generate_all_attacks(
        dataloader=test_subset_loader,
        num_samples=Config.ADV_NUM_SAMPLES,
        cache_dir=Config.ADVERSARIAL_CACHE_DIR,
        force_regenerate=Config.FORCE_REGENERATE
    )

    # 打印生成結果摘要
    print(f"\n{'=' * 70}")
    print("Adversarial Sample Generation Summary")
    print(f"{'=' * 70}")
    for attack_name, data_info in adversarial_data.items():
        print(f"  {attack_name:<15}: {data_info['images'].shape[0]} samples")
    print(f"{'=' * 70}\n")

    # ==================== 步驟 4: 評估魯棒性 ====================
    print(f"\n{'=' * 70}")
    print("STEP 4: Evaluating Model Robustness")
    print(f"{'=' * 70}\n")

    robustness_results = {}
    gaussian_std = 16 / 255

    for attack_name, data_info in adversarial_data.items():
        images = torch.FloatTensor(data_info['images']).to(device)
        labels = torch.LongTensor(data_info['labels']).to(device)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(images), Config.TEST_BATCH_SIZE):
                batch_images = images[i:i + Config.TEST_BATCH_SIZE]
                batch_labels = labels[i:i + Config.TEST_BATCH_SIZE]

                # 加入高斯噪聲
                noise = torch.randn_like(batch_images) * gaussian_std
                batch_images = batch_images + noise
                batch_images = torch.clamp(batch_images, 0, 1)

                outputs = model(batch_images)
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        acc = 100. * correct / total
        robustness_results[attack_name] = acc
        print(f"  {attack_name:<15}: {acc:>6.2f}%")

    # ==================== 步驟 5: 分析 Log-Softmax 分布 ====================
    print(f"\n{'=' * 70}")
    print("STEP 5: Analyzing Log-Softmax Distribution (Batch Average)")
    print(f"{'=' * 70}\n")

    distribution_stats = {}
    for attack_name, data_info in adversarial_data.items():

        stats = compute_log_softmax_stats(
            model, data_info['images'], device, feature_batch_size=Config.DETECTOR_FEATURE_BATCH_SIZE
        )
        distribution_stats[attack_name] = stats
        print(f"  {attack_name:<15}: Min={stats['avg_min']:.3f}, "
              f"Max={stats['avg_max']:.3f}, Mean={stats['avg_mean']:.3f}, "
              f"Batches={len(stats['batch_mins'])}")

    # ==================== 步驟 6: 訓練檢測器 ====================
    print(f"\n{'=' * 70}")
    print("STEP 6: Training Adversarial Detector(Batch Average Features)")
    print(f"{'=' * 70}\n")

    # 初始化檢測器
    detector = AdversarialDetectorMLP(
        num_classes=Config.NUM_CLASSES,
        num_attack_types=len(Config.ATTACK_TYPES),
        hidden_dims=Config.DETECTOR_HIDDEN_DIMS,  # 使用列表
        dropout=Config.DETECTOR_DROPOUT
    ).to(device)

    print(f"[INFO] Detector Architecture:")
    print(f"  Input: Log-softmax values ({Config.NUM_CLASSES} dimensions)")
    print(f"  Hidden layers: {Config.DETECTOR_HIDDEN_DIMS}")
    print(f"  Output: {len(Config.ATTACK_TYPES)} attack types")
    print(f"  Dropout: {Config.DETECTOR_DROPOUT}")
    print(f"  Parameters: {sum(p.numel() for p in detector.parameters()):,}")

    # 初始化訓練器
    detector_trainer = DetectorTrainer(
        detector=detector,
        classifier_model=model,
        device=device,
        lr=Config.DETECTOR_LR,
        weight_decay=Config.DETECTOR_WEIGHT_DECAY,
        feature_batch_size=Config.DETECTOR_FEATURE_BATCH_SIZE
    )

    # 準備訓練數據（自動提取 log-softmax 特徵）
    X, y = detector_trainer.prepare_training_data(adversarial_data)

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    print(f"\n[INFO] Data split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # 訓練檢測器
    # 訓練時使用更大的 batch size
    best_acc = detector_trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=Config.DETECTOR_EPOCHS,
        batch_size=32
    )

    # 保存檢測器
    # save_model(detector, Config.DETECTOR_SAVE_PATH)
    # print(f"[INFO] Detector saved to {Config.DETECTOR_SAVE_PATH}")

    # ==================== 步驟 7: 評估檢測器 ====================
    print(f"\n{'='*70}")
    print("STEP 7: Evaluating Detector Performance")
    print(f"{'='*70}\n")

    evaluator = DetectorEvaluator(Config.ATTACK_TYPES)

    # 在測試集上評估
    y_pred = detector_trainer.predict(X_test)
    results = evaluator.evaluate(y_test, y_pred)

    # 打印結果（與附圖格式相同）
    evaluator.print_results(results)
    evaluator.print_detailed_metrics(results)

    # ==================== 步驟 8: 可視化結果 ====================
    print(f"\n{'='*70}")
    print("STEP 8: Generating Visualizations")
    print(f"{'='*70}\n")

    visualizer = Visualizer(Config.ATTACK_TYPES)

    # 1. 混淆矩陣（與附圖相同格式）
    visualizer.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
    )

    # 2. 魯棒性比較
    visualizer.plot_robustness_comparison(
        robustness_results,
        save_path=os.path.join(Config.RESULTS_DIR, 'robustness_comparison.png')
    )

    # 3. 檢測性能分析
    visualizer.plot_detection_performance(
        results,
        save_path=os.path.join(Config.RESULTS_DIR, 'detection_performance.png')
    )

    # 4. Log-Softmax 分布
    visualizer.plot_log_softmax_distribution(
        distribution_stats,
        save_path=os.path.join(Config.RESULTS_DIR, 'log_softmax_distribution.png')
    )

    # ==================== 步驟 9: 生成報告 ====================
    print(f"\n{'='*70}")
    print("STEP 9: Generating Final Report")
    print(f"{'='*70}\n")

    report_path = os.path.join(Config.RESULTS_DIR, 'experiment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Self-Adaptive Logit Balancing Training and Adversarial Detection\n")
        f.write("Experiment Report\n")
        f.write("="*70 + "\n\n")

        # 配置信息
        f.write("1. Configuration\n")
        f.write("-"*70 + "\n")
        f.write(f"Model: WideResNet-{Config.MODEL_DEPTH}-{Config.MODEL_WIDEN_FACTOR}\n")
        f.write(f"Dataset: CIFAR-10\n")
        f.write(f"Training Epochs: {Config.LB_EPOCHS}\n")
        f.write(f"Beta (β): {Config.LB_BETA}\n")
        f.write(f"Sigma (σ): {Config.LB_SIGMA:.4f}\n")
        f.write(f"Detector Hidden Dim: {Config.DETECTOR_HIDDEN_DIMS}\n")
        f.write(f"Detector Epochs: {Config.DETECTOR_EPOCHS}\n\n")

        # 魯棒性結果
        f.write("2. Model Robustness Results\n")
        f.write("-"*70 + "\n")
        for attack_name, acc in robustness_results.items():
            f.write(f"{attack_name:<15}: {acc:>6.2f}%\n")
        f.write("\n")

        # 檢測器性能
        f.write("3. Detector Performance\n")
        f.write("-"*70 + "\n")
        f.write(f"Overall Accuracy: {results['overall_accuracy']:.2f}%\n\n")

        f.write("Per-Class Accuracy:\n")
        for attack_name, acc in results['class_accuracies'].items():
            f.write(f"  {attack_name:<15}: {acc:>6.2f}%\n")
        f.write("\n")

        # 混淆矩陣
        f.write("4. Confusion Matrix (Percentages)\n")
        f.write("-"*70 + "\n")
        cm = results['confusion_matrix']
        f.write(f"{'True \\ Pred':<12}")
        for attack_name in Config.ATTACK_TYPES:
            f.write(f"{attack_name:>10}")
        f.write("\n")

        for i, true_attack in enumerate(Config.ATTACK_TYPES):
            f.write(f"{true_attack:<12}")
            row_sum = cm[i, :].sum()
            for j in range(len(Config.ATTACK_TYPES)):
                if row_sum > 0:
                    percentage = cm[i, j] / row_sum * 100
                    f.write(f"{percentage:>9.1f}%")
                else:
                    f.write(f"{'0.0%':>10}")
            f.write("\n")
        f.write("\n")

        # Log-Softmax 統計
        f.write("5. Log-Softmax Distribution Statistics\n")
        f.write("-"*70 + "\n")
        for attack_name, stats in distribution_stats.items():
            f.write(f"{attack_name}:\n")
            f.write(f"  Min: {stats['avg_min']:.4f}, Max: {stats['avg_max']:.4f}, "
                   f"Mean: {stats['avg_mean']:.4f}, Std: {stats['avg_std']:.4f}\n")
        f.write("\n")

        f.write("="*70 + "\n")
        f.write("Experiment Completed Successfully!\n")
        f.write("="*70 + "\n")

    print(f"[INFO] Report saved to {report_path}")

    # ==================== 完成 ====================
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")

    print("Summary:")
    print(f"  • Clean Accuracy: {clean_acc:.2f}%")
    print(f"  • Detector Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"  • Best Attack Detection: {max(results['class_accuracies'].items(), key=lambda x: x[1])}")
    print(f"  • Worst Attack Detection: {min(results['class_accuracies'].items(), key=lambda x: x[1])}")
    print(f"\nAll results saved to: {Config.RESULTS_DIR}/")
    print(f"  • confusion_matrix.png")
    print(f"  • robustness_comparison.png")
    print(f"  • detection_performance.png")
    print(f"  • log_softmax_distribution.png")
    print(f"  • experiment_report.txt")
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()
