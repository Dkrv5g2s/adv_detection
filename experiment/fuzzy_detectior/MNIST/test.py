import os
import time
import random
import numpy as np
import torch
import warnings
from torch.utils.data import TensorDataset, DataLoader

from fuzzy_detectior.MNIST.src.models import SimpleCNN
from fuzzy_detectior.MNIST.src.data_utils import load_mnist
from fuzzy_detectior.MNIST.src.model_training import train_classifier, eval_classifier

warnings.filterwarnings('ignore')

MODEL_PATH = "src/simple_mnist_cnn.pth"
TRAINING_EPOCHS = 5
BATCH_SIZE = 64
TEST_SAMPLES = 1000


def generate_seed():
    return int(time.time()) % 10000


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_noise_accuracy(model, original_images, original_labels, device):
    """測試不同噪聲強度對準確度的影響"""

    # 確保圖片格式正確
    if len(original_images.shape) == 3:
        original_images = np.expand_dims(original_images, axis=1)

    # 原始圖片準確率
    original_dataset = TensorDataset(
        torch.tensor(original_images, dtype=torch.float32),
        torch.tensor(original_labels, dtype=torch.long)
    )
    original_loader = DataLoader(original_dataset, batch_size=BATCH_SIZE, shuffle=False)
    original_acc = eval_classifier(model, original_loader, device)

    print(f"Original accuracy: {original_acc:.4f}")

    # 測試不同噪聲強度
    noise_std_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]

    results = []

    for noise_std in noise_std_values:
        # 加噪聲（你原始的方式）
        clean_images_noisy = original_images + np.random.normal(0, noise_std, original_images.shape)
        clean_images_noisy = np.clip(clean_images_noisy, 0, 1)

        # 測試噪聲圖片準確率
        noisy_dataset = TensorDataset(
            torch.tensor(clean_images_noisy, dtype=torch.float32),
            torch.tensor(original_labels, dtype=torch.long)
        )
        noisy_loader = DataLoader(noisy_dataset, batch_size=BATCH_SIZE, shuffle=False)
        noisy_acc = eval_classifier(model, noisy_loader, device)

        # 計算準確率下降
        acc_drop = original_acc - noisy_acc
        acc_drop_percent = (acc_drop / original_acc) * 100 if original_acc > 0 else 0

        results.append({
            'noise_std': noise_std,
            'original_acc': original_acc,
            'noisy_acc': noisy_acc,
            'acc_drop': acc_drop,
            'acc_drop_percent': acc_drop_percent
        })

        print(
            f"Noise std: {noise_std:.3f} | Noisy acc: {noisy_acc:.4f} | Drop: {acc_drop:.4f} ({acc_drop_percent:.2f}%)")

    return results


def load_or_train_model(device):
    """載入或訓練分類器模型"""
    model = SimpleCNN()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        print("✅ Loaded existing model from", MODEL_PATH)
    else:
        print("🔄 Training new model...")
        train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, shuffle_test=True)
        model = train_classifier(model, train_loader, test_loader, device, epochs=TRAINING_EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Model trained and saved to", MODEL_PATH)

    return model


def print_results(results):
    """列印結果"""
    print("\n" + "=" * 80)
    print("🔊 NOISE ACCURACY TEST RESULTS")
    print("=" * 80)

    print(f"{'Noise Std':<12} {'Original Acc':<12} {'Noisy Acc':<10} {'Drop':<8} {'Drop %':<8} {'Status':<10}")
    print("-" * 70)

    for result in results:
        drop_percent = result['acc_drop_percent']

        # 判斷狀態
        if drop_percent < 2.0:
            status = "MINIMAL"
        elif drop_percent < 5.0:
            status = "SMALL"
        elif drop_percent < 10.0:
            status = "MODERATE"
        elif drop_percent < 20.0:
            status = "LARGE"
        else:
            status = "SEVERE"

        print(f"{result['noise_std']:<12.3f} {result['original_acc']:<12.4f} "
              f"{result['noisy_acc']:<10.4f} {result['acc_drop']:<8.4f} "
              f"{drop_percent:<8.2f} {status:<10}")

    # 找出推薦的噪聲參數
    print("\n🎯 RECOMMENDED NOISE PARAMETERS:")
    print("-" * 40)

    recommended = [r for r in results if 2.0 <= r['acc_drop_percent'] <= 8.0]

    if recommended:
        for result in recommended:
            print(f"✅ noise_std = {result['noise_std']:.3f} (drop: {result['acc_drop_percent']:.2f}%)")
    else:
        print("❌ No noise parameters found in the recommended range (2-8% drop).")

    print(f"\n📊 Average accuracy drop: {np.mean([r['acc_drop_percent'] for r in results]):.2f}%")


def main():
    # 設定隨機種子
    seed = generate_seed()
    set_seed(seed)

    # 設定運算裝置
    device = get_device()
    print("🖥️  Using device:", device)

    print("\n" + "=" * 80)
    print("🔊 NOISE ACCURACY TESTING")
    print("=" * 80)
    print(f"🎲 Random seed: {seed}")
    print(f"📊 Test samples: {TEST_SAMPLES}")

    # 載入資料
    print("\n🔄 Loading MNIST data...")
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, shuffle_test=True)

    # 訓練或載入分類器
    print("\n🔄 Loading/Training classifier...")
    model = load_or_train_model(device)

    # 評估基礎模型
    print("\n📊 Evaluating base model...")
    base_acc = eval_classifier(model, test_loader, device)
    print(f"🎯 Base CNN Test Accuracy: {base_acc:.4f}")

    # 準備測試資料
    print(f"\n🔄 Preparing {TEST_SAMPLES} test samples...")
    test_images = []
    test_labels = []

    for batch_idx, (data, target) in enumerate(test_loader):
        test_images.append(data.numpy())
        test_labels.append(target.numpy())

        if len(test_images) * BATCH_SIZE >= TEST_SAMPLES:
            break

    test_images = np.concatenate(test_images, axis=0)[:TEST_SAMPLES]
    test_labels = np.concatenate(test_labels, axis=0)[:TEST_SAMPLES]

    print(f"Test data shape: {test_images.shape}")
    print(f"Test data range: [{test_images.min():.4f}, {test_images.max():.4f}]")

    # 測試噪聲對準確度的影響
    print("\n🔄 Testing noise effects on accuracy...")
    results = test_noise_accuracy(model, test_images, test_labels, device)

    # 列印結果
    print_results(results)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n🏁 Program finished.")
