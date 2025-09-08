import numpy as np
import torch
import torch.nn as nn
import shap
from tqdm import tqdm


class LogitToSoftmax(nn.Module):
    """將logits轉換為softmax機率的包裝器"""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, logits):
        return self.softmax(logits)


def extract_logits(model, images, device, batch_size=16):
    """提取模型的logits（最後一層，進入softmax前）"""
    model.eval()
    all_logits = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = torch.tensor(images[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(batch_images)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def compute_shap_values_batch(logit_classifier, explainer, logit_data, device, batch_size=32):
    """分批計算SHAP值"""
    shap_values = []

    for i in tqdm(range(0, len(logit_data), batch_size), desc="Computing SHAP", leave=False):
        batch_data = torch.tensor(logit_data[i:i + batch_size], dtype=torch.float32).to(device)
        batch_shap = explainer.shap_values(batch_data)
        shap_values.extend(batch_shap)

    return np.array(shap_values)


def extract_shap_signature(shap_values):
    """從SHAP值提取簽名特徵"""
    # 直接重塑形狀：將每個樣本的(10, 10)展平為(100,)
    n_samples = shap_values.shape[0]
    signatures = shap_values.reshape(n_samples, -1)  # (n_samples, 100)
    return signatures


def generate_shap_signatures(model, images, device, batch_size=16):
    """
    生成SHAP簽名，輸出為10*10=100維

    Args:
        model: 訓練好的CNN模型
        images: 輸入圖像 (numpy array)
        device: 計算設備
        batch_size: 批次大小

    Returns:
        signatures: SHAP簽名特徵 (n_samples, 100)
    """
    print("Extracting logits...")
    logits = extract_logits(model, images, device, batch_size)
    print(f"Logits shape: {logits.shape}")

    # 建立logit到softmax的分類器
    logit_classifier = LogitToSoftmax().to(device)

    # 建立背景樣本（隨機選擇100個樣本作為背景）
    background_indices = np.random.choice(len(logits), min(100, len(logits)), replace=False)
    background_tensor = torch.tensor(logits[background_indices], dtype=torch.float32).to(device)

    # 建立SHAP解釋器
    explainer = shap.DeepExplainer(logit_classifier, background_tensor)

    # 分批計算SHAP值
    print("Computing SHAP values...")
    shap_values_all = compute_shap_values_batch(logit_classifier, explainer, logits, device)
    print(f"SHAP values shape: {shap_values_all.shape}")

    # 提取簽名特徵
    signatures = extract_shap_signature(shap_values_all)
    print(f"Final signatures shape: {signatures.shape}")

    return signatures
