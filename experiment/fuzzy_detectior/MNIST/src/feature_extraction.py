import numpy as np
import torch
import torch.nn.functional as F

def extract_features(model, images, device, batch_size=256):
    """從模型中提取logit特徵"""
    model.eval()
    logits = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            # 取得當前批次
            batch = images[i:i + batch_size]

            # 轉換為tensor並移到正確設備
            if isinstance(batch, np.ndarray):
                batch_tensor = torch.from_numpy(batch).float().to(device)
            else:
                batch_tensor = batch.float().to(device)

            # 確保輸入維度正確 (batch_size, 1, 28, 28)
            if len(batch_tensor.shape) == 3:
                batch_tensor = batch_tensor.unsqueeze(1)

            # 前向傳播獲取logits
            batch_logits = model(batch_tensor)

            # 轉回CPU並添加到結果列表
            logits.append(batch_logits.cpu().numpy())

    # 合併所有批次的結果
    return np.concatenate(logits, axis=0)

def extract_feature_differences(p_clean, p_adv):
    """提取特徵差異"""
    # 1. MSE差異
    mse_diff = np.mean((p_adv - p_clean) ** 2, axis=1)

    # 2. 最大機率差異
    max_clean = np.max(p_clean, axis=1)
    max_adv = np.max(p_adv, axis=1)
    max_diff = np.abs(max_clean - max_adv)

    # 3. 熵差異
    def entropy(p):
        p_safe = p + 1e-12
        return -np.sum(p_safe * np.log(p_safe), axis=1)

    entropy_clean = entropy(p_clean)
    entropy_adv = entropy(p_adv)
    entropy_diff = np.abs(entropy_adv - entropy_clean)

    # 4. KL散度
    def kl_divergence(p, q):
        p_safe = p + 1e-12
        q_safe = q + 1e-12
        return np.sum(p_safe * np.log(p_safe / q_safe), axis=1)

    kl_diff = kl_divergence(p_clean, p_adv)

    # 5. L1差異
    l1_diff = np.mean(np.abs(p_adv - p_clean), axis=1)

    # 組合所有差異指標
    all_diffs = [mse_diff, max_diff, kl_diff, l1_diff]

    # 正規化每個特徵到 [0,1]
    normalized_diffs = []
    for diff in all_diffs:
        if diff.max() > diff.min():
            norm_diff = (diff - diff.min()) / (diff.max() - diff.min())
        else:
            norm_diff = np.zeros_like(diff)
        normalized_diffs.append(norm_diff)

    return np.column_stack(normalized_diffs)
