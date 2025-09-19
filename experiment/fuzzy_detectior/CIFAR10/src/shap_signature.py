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

##############################################################################################################################################################
# for top 5 logits to calculation
class TopFiveLogitToSoftmax(nn.Module):
    """取logits中前五大的值，其餘補0，再做softmax"""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, logits):
        # 獲取每個樣本前5大的值和索引
        top5_values, top5_indices = torch.topk(logits, k=5, dim=1)

        # 創建與原始logits相同形狀的零張量
        masked_logits = torch.zeros_like(logits)

        # 將前5大的值填回對應位置
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1).expand(-1, 5)
        masked_logits[batch_indices, top5_indices] = top5_values

        # 對處理後的logits做softmax
        return self.softmax(masked_logits)


def extract_top5_shap_signature(shap_values):
    """從SHAP值提取Top5簽名特徵"""
    # 對於Top5版本，形狀為(n_samples, 5, 10)，展平為(n_samples, 50)
    n_samples = shap_values.shape[0]
    signatures = shap_values.reshape(n_samples, -1)  # (n_samples, 50)
    return signatures




def generate_top5_shap_signatures(model, images, device, batch_size=16):
    """
    生成SHAP簽名

    Args:
        model: 訓練好的CNN模型
        images: 輸入圖像 (numpy array)
        device: 計算設備
        batch_size: 批次大小
        use_top5: 是否使用Top5版本 (True: 輸出50維, False: 輸出100維)

    Returns:
        signatures: SHAP簽名特徵 (n_samples, 50 or 100)
    """
    print("Extracting logits...")
    logits = extract_logits(model, images, device, batch_size)
    print(f"Logits shape: {logits.shape}")


    logit_classifier = TopFiveLogitToSoftmax().to(device)
    print("Using Top5 Logit to Softmax classifier")


    # 建立背景樣本（隨機選擇100個樣本作為背景）
    background_indices = np.random.choice(len(logits), min(100, len(logits)), replace=False)
    background_tensor = torch.tensor(logits[background_indices], dtype=torch.float32).to(device)

    # 建立SHAP解釋器
    explainer = shap.DeepExplainer(logit_classifier, background_tensor)

    # 分批計算SHAP值
    print("Computing SHAP values...")
    shap_values_all = compute_shap_values_batch_no_check(logit_classifier, explainer, logits, device)
    print(f"SHAP values shape: {shap_values_all.shape}")


    signatures = extract_top5_shap_signature(shap_values_all)
    print(f"Final Top5 signatures shape: {signatures.shape}")


    return signatures


def compute_shap_values_batch_no_check(model, explainer, logits, device, batch_size=16):
    """計算SHAP值，關閉可加性檢查"""
    all_shap_values = []

    for i in range(0, len(logits), batch_size):
        batch_data = torch.tensor(logits[i:i + batch_size], dtype=torch.float32).to(device)

        # 關閉可加性檢查
        batch_shap = explainer.shap_values(batch_data, check_additivity=False)

        if isinstance(batch_shap, list):
            batch_shap = np.array(batch_shap).transpose(1, 0, 2)
            batch_shap = batch_shap.reshape(batch_shap.shape[0], -1)

        all_shap_values.append(batch_shap)

    return np.vstack(all_shap_values)
##########################################################################################################################################################################
# for extra linear to calculation
class LinearExplainModel(nn.Module):
    """5→10的線性層+softmax模型，用於SHAP分析"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch_size, 5) - 前5大logits
        logits = self.linear(x)
        return self.softmax(logits)


def extract_top5_logits(model, images, device, batch_size=16):
    """提取原始模型logits的前5大值"""
    model.eval()
    all_top5_logits = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = torch.tensor(images[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(batch_images)  # (batch_size, 10)

            # 取前5大logits
            top5_values, _ = torch.topk(logits, k=5, dim=1)  # (batch_size, 5)
            all_top5_logits.append(top5_values.cpu().numpy())

    return np.concatenate(all_top5_logits, axis=0)


def compute_linear_shap_values_batch(linear_classifier, explainer, top5_data, device, batch_size=32):
    """分批計算線性模型的SHAP值"""
    shap_values = []

    for i in tqdm(range(0, len(top5_data), batch_size), desc="Computing Linear SHAP", leave=False):
        batch_data = torch.tensor(top5_data[i:i + batch_size], dtype=torch.float32).to(device)
        batch_shap = explainer.shap_values(batch_data)
        shap_values.extend(batch_shap)

    return np.array(shap_values)


def extract_linear_shap_signature(shap_values):
    """從線性SHAP值提取簽名特徵"""
    # 直接重塑形狀：將每個樣本的(5, 10)展平為(50,)
    n_samples = shap_values.shape[0]
    signatures = shap_values.reshape(n_samples, -1)  # (n_samples, 50)
    return signatures


def generate_linear_shap_signatures(model, images, device, batch_size=16):
    """
    生成線性SHAP簽名，輸出為5*10=50維

    Args:
        model: 訓練好的CNN模型
        images: 輸入圖像 (numpy array)
        device: 計算設備
        batch_size: 批次大小

    Returns:
        signatures: 線性SHAP簽名特徵 (n_samples, 50)
    """
    print("Extracting top5 logits...")
    top5_logits = extract_top5_logits(model, images, device, batch_size)
    print(f"Top5 logits shape: {top5_logits.shape}")

    # 建立5→10線性分類器
    linear_classifier = LinearExplainModel().to(device)

    # 建立背景樣本（隨機選擇100個樣本作為背景）
    background_indices = np.random.choice(len(top5_logits), min(100, len(top5_logits)), replace=False)
    background_tensor = torch.tensor(top5_logits[background_indices], dtype=torch.float32).to(device)

    # 建立SHAP解釋器
    explainer = shap.DeepExplainer(linear_classifier, background_tensor)

    # 分批計算SHAP值
    print("Computing linear SHAP values...")
    shap_values_all = compute_linear_shap_values_batch(linear_classifier, explainer, top5_logits, device)
    print(f"Linear SHAP values shape: {shap_values_all.shape}")

    # 提取簽名特徵
    signatures = extract_linear_shap_signature(shap_values_all)
    print(f"Final linear signatures shape: {signatures.shape}")

    return signatures


##########################################################################################################################################################################
# for dense five to calculation

class Dense5ToSoftmax(nn.Module):
    """將5維dense5特徵轉換為softmax機率的包裝器"""

    def __init__(self, fc6_layer):
        super().__init__()
        self.fc6 = fc6_layer  # 5 -> 10 的最後一層
        self.softmax = nn.Softmax(dim=1)

    def forward(self, dense5_features):
        logits = self.fc6(dense5_features)
        return self.softmax(logits)


def extract_dense5(model, images, device, batch_size=16):
    """提取倒數第二層的5維特徵（fc5 + bn + relu的輸出）"""
    model.eval()
    all_features = []

    # Hook函數來捕獲bn_dense5的輸出
    features_hook = []

    def hook_fn(module, input, output):
        features_hook.append(output.detach())

    # 在bn_dense5層註冊hook
    hook = model.bn_dense5.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = torch.tensor(images[i:i + batch_size], dtype=torch.float32).to(device)
                features_hook.clear()

                # 前向傳播
                _ = model(batch_images)

                # 獲取ReLU後的特徵
                if features_hook:
                    relu_features = torch.relu(features_hook[0])  # 應用ReLU
                    all_features.append(relu_features.cpu().numpy())
    finally:
        hook.remove()

    return np.concatenate(all_features, axis=0)


def compute_shap_values_batch_dens5(dense5_classifier, explainer, dense5_data, device, batch_size=32):
    """分批計算SHAP值"""
    shap_values = []

    for i in tqdm(range(0, len(dense5_data), batch_size), desc="Computing SHAP", leave=False):
        batch_data = torch.tensor(dense5_data[i:i + batch_size], dtype=torch.float32).to(device)
        batch_shap = explainer.shap_values(batch_data)
        shap_values.extend(batch_shap)

    return np.array(shap_values)


def extract_shap_signature_dens5(shap_values):
    """從SHAP值提取簽名特徵"""
    # 直接重塑形狀：將每個樣本的(10, 5)展平為(50,)
    n_samples = shap_values.shape[0]
    signatures = shap_values.reshape(n_samples, -1)  # (n_samples, 50)
    return signatures


def generate_shap_signatures_dens5(model, images, device, batch_size=16):
    """
    生成SHAP簽名，基於5維dense5特徵，輸出為5*10=50維

    Args:
        model: 訓練好的CNN模型
        images: 輸入圖像 (numpy array)
        device: 計算設備
        batch_size: 批次大小

    Returns:
        signatures: SHAP簽名特徵 (n_samples, 50)
    """
    print("Extracting dense5 features...")
    dense5_features = extract_dense5(model, images, device, batch_size)
    print(f"Dense5 features shape: {dense5_features.shape}")

    # 建立dense5到softmax的分類器
    dense5_classifier = Dense5ToSoftmax(model.fc6).to(device)

    # 建立背景樣本（隨機選擇100個樣本作為背景）
    background_indices = np.random.choice(len(dense5_features), min(100, len(dense5_features)), replace=False)
    background_tensor = torch.tensor(dense5_features[background_indices], dtype=torch.float32).to(device)

    # 建立SHAP解釋器
    explainer = shap.DeepExplainer(dense5_classifier, background_tensor)

    # 分批計算SHAP值
    print("Computing SHAP values...")
    shap_values_all = compute_shap_values_batch(dense5_classifier, explainer, dense5_features, device)
    print(f"SHAP values shape: {shap_values_all.shape}")

    # 提取簽名特徵
    signatures = extract_shap_signature(shap_values_all)
    print(f"Final signatures shape: {signatures.shape}")

    return signatures