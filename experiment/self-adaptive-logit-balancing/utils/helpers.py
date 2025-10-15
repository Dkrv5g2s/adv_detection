"""
輔助函數
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split


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


def compute_log_softmax_stats(model, images, device, batch_size=100):
    """計算 Log-Softmax 統計信息"""
    model.eval()

    images_tensor = torch.FloatTensor(images).to(device)
    num_batches = (len(images_tensor) + batch_size - 1) // batch_size

    all_log_softmax = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images_tensor))
        batch_images = images_tensor[start_idx:end_idx]

        with torch.no_grad():
            outputs = model(batch_images)
            log_softmax = torch.log_softmax(outputs, dim=1)
            all_log_softmax.append(log_softmax.cpu().numpy())

    all_log_softmax = np.vstack(all_log_softmax)

    stats = {
        'avg_min': np.mean(np.min(all_log_softmax, axis=1)),
        'avg_max': np.mean(np.max(all_log_softmax, axis=1)),
        'avg_mean': np.mean(all_log_softmax),
        'avg_std': np.mean(np.std(all_log_softmax, axis=1)),
        'std_min': np.std(np.min(all_log_softmax, axis=1)),
        'std_max': np.std(np.max(all_log_softmax, axis=1))
    }

    return stats
