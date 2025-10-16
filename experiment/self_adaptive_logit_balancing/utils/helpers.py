"""
輔助函數
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
from datetime import datetime


def get_device_info():
    """獲取設備信息"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[INFO] Using device: {device}")
        print(f"[INFO] GPU: {gpu_name}")
        print(f"[INFO] GPU Memory: {gpu_memory:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"[INFO] Using device: {device}")
    return device


def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def load_model(model, path, device):
    """
    載入模型權重

    Args:
        model: PyTorch 模型
        path: 模型權重路徑
        device: 設備 (cpu/cuda)

    Returns:
        載入權重後的模型
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # 處理不同的保存格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def split_data(X, y, test_size=0.2, random_state=42):
    """分割數據集"""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def evaluate_model_accuracy(model, dataloader, device):
    """評估模型準確率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def compute_log_softmax_stats(model, images, device, batch_size=60):
    """
    計算 log-softmax 統計數據（批次平均 + 排序版本）

    Args:
        model: 分類模型
        images: numpy array (N, 3, 32, 32)
        device: 計算設備
        batch_size: 批次大小（預設 60）

    Returns:
        stats: dict，包含 min, max, mean, std（基於批次平均）
    """
    if isinstance(images, np.ndarray):
        images = torch.FloatTensor(images)

    images = images.to(device)
    num_samples = len(images)

    model.eval()

    # 儲存每個批次的統計值
    batch_mins = []
    batch_maxs = []
    batch_means = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_images = images[i:i + batch_size]

            # 如果最後一批不足 batch_size，跳過
            if len(batch_images) < batch_size:
                continue

            # 提取 log-softmax
            logits = model(batch_images)
            log_softmax = - F.log_softmax(logits, dim=1)

            # 對每個樣本排序
            log_softmax_sorted, _ = torch.sort(log_softmax, dim=1)

            # 計算該批次的平均值（排序後再平均）
            batch_avg = log_softmax_sorted.mean(dim=0)  # (num_classes,)

            # 記錄該批次平均的統計值
            batch_mins.append(batch_avg[0].item())  # 最小值（排序後第一個）
            batch_maxs.append(batch_avg[-1].item())  # 最大值（排序後最後一個）
            batch_means.append(batch_avg.mean().item())

    # 計算所有批次的統計
    if len(batch_mins) > 0:
        stats = {
            'avg_min': np.mean(batch_mins),
            'avg_max': np.mean(batch_maxs),
            'avg_mean': np.mean(batch_means),
            'avg_std': np.std(batch_means),
            'batch_mins': batch_mins,  # 用於繪圖
            'batch_maxs': batch_maxs  # 用於繪圖
        }
    else:
        stats = {
            'avg_min': 0.0,
            'avg_max': 0.0,
            'avg_mean': 0.0,
            'avg_std': 0.0,
            'batch_mins': [],
            'batch_maxs': []
        }

    return stats


# ========== 對抗樣本緩存相關函數 ==========

def save_adversarial_data(data, cache_path, attack_name):
    """
    保存對抗樣本數據到緩存

    Args:
        data: dict，包含 'images' 和 'labels'
        cache_path: 緩存文件路徑
        attack_name: 攻擊名稱
    """
    try:
        # 確保目錄存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 保存數據（使用壓縮）
        np.savez_compressed(
            cache_path,
            images=data['images'],
            labels=data['labels'],
            attack_name=attack_name,
            timestamp=datetime.now().isoformat()
        )

        file_size = os.path.getsize(cache_path) / (1024 ** 2)  # MB
        print(f"  ✓ Cached to: {cache_path} ({file_size:.2f} MB)")

    except Exception as e:
        print(f"  ✗ Failed to cache {attack_name}: {e}")


def load_adversarial_data(cache_path, attack_name):
    """
    從緩存載入對抗樣本數據

    Args:
        cache_path: 緩存文件路徑
        attack_name: 攻擊名稱

    Returns:
        data: dict，包含 'images' 和 'labels'，如果載入失敗則返回 None
    """
    try:
        if not os.path.exists(cache_path):
            return None

        # 載入數據
        cached = np.load(cache_path, allow_pickle=True)

        data = {
            'images': cached['images'],
            'labels': cached['labels']
        }

        # 驗證數據
        if data['images'].shape[0] != data['labels'].shape[0]:
            print(f"  ✗ Cache corrupted for {attack_name}: shape mismatch")
            return None

        file_size = os.path.getsize(cache_path) / (1024 ** 2)  # MB
        timestamp = cached['timestamp'].item() if 'timestamp' in cached else 'Unknown'

        print(f"  ✓ Loaded from cache: {cache_path}")
        print(f"    - Size: {file_size:.2f} MB")
        print(f"    - Samples: {data['images'].shape[0]}")
        print(f"    - Cached at: {timestamp}")

        return data

    except Exception as e:
        print(f"  ✗ Failed to load cache for {attack_name}: {e}")
        return None


def check_cache_exists(cache_path):
    """
    檢查緩存文件是否存在

    Args:
        cache_path: 緩存文件路徑

    Returns:
        bool: 是否存在
    """
    return os.path.exists(cache_path)


def clear_adversarial_cache(cache_dir):
    """
    清空對抗樣本緩存目錄

    Args:
        cache_dir: 緩存目錄路徑
    """
    if not os.path.exists(cache_dir):
        print(f"[INFO] Cache directory does not exist: {cache_dir}")
        return

    import shutil
    try:
        shutil.rmtree(cache_dir)
        print(f"[INFO] Cleared cache directory: {cache_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to clear cache: {e}")


def print_cache_summary(cache_dir):
    """
    打印緩存摘要信息

    Args:
        cache_dir: 緩存目錄路徑
    """
    if not os.path.exists(cache_dir):
        print(f"[INFO] No cache directory found: {cache_dir}")
        return

    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]

    if not cache_files:
        print(f"[INFO] Cache directory is empty: {cache_dir}")
        return

    print(f"\n{'='*70}")
    print("Adversarial Sample Cache Summary")
    print(f"{'='*70}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Total Files: {len(cache_files)}\n")

    total_size = 0
    for filename in sorted(cache_files):
        filepath = os.path.join(cache_dir, filename)
        file_size = os.path.getsize(filepath) / (1024 ** 2)  # MB
        total_size += file_size

        # 嘗試讀取樣本數量
        try:
            cached = np.load(filepath, allow_pickle=True)
            num_samples = cached['images'].shape[0]
            timestamp = cached['timestamp'].item() if 'timestamp' in cached else 'Unknown'
            print(f"  • {filename}")
            print(f"    - Size: {file_size:.2f} MB")
            print(f"    - Samples: {num_samples}")
            print(f"    - Cached at: {timestamp}")
        except:
            print(f"  • {filename}")
            print(f"    - Size: {file_size:.2f} MB")
            print(f"    - Status: Corrupted or unreadable")
        print()

    print(f"Total Cache Size: {total_size:.2f} MB")
    print(f"{'='*70}\n")
