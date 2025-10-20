"""
配置文件 - 所有超參數和設置
"""
import torch
import os

class Config:
    # 設備配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 數據集配置
    DATA_ROOT = './data'
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 100
    NUM_WORKERS = 4

    # 模型配置
    MODEL_DEPTH = 34
    MODEL_WIDEN_FACTOR = 10
    MODEL_DROPOUT = 0.0

    # Logit Balancing 訓練配置
    LB_EPOCHS = 100
    LB_BETA = 0.02
    LB_SIGMA = 24 / 255
    LB_LR = 0.02
    LB_MOMENTUM = 0.9
    LB_WEIGHT_DECAY = 5e-4

    # 對抗樣本生成配置
    ADV_NUM_SAMPLES = 3000
    PGD_EPS = 8 / 255
    PGD_ALPHA = 2 / 255
    PGD_STEPS = 10

    # 攻擊類型（依序）
    ATTACK_TYPES = [
        'Clean',  # 0
        'PGD-Linf',  # 1
        'PGD-L2',  # 2
        'APGD-Linf',  # 3
        'APGDT-Linf',  # 4
        'Square-Linf',  # 5
        'FAB-Linf',  # 6
        'CW-L2'  # 7
    ]

    # 檢測器配置
    DETECTOR_HIDDEN_DIM = 128
    DETECTOR_DROPOUT = 0.0
    DETECTOR_LR = 0.001
    DETECTOR_EPOCHS = 120
    DETECTOR_FEATURE_BATCH_SIZE = 120


    # 保存路徑
    MODEL_SAVE_PATH = 'logit_balancing_model(82.65).pth'
    DETECTOR_SAVE_PATH = 'adversarial_detector.pth'
    RESULTS_DIR = './results'



    # ========== 對抗樣本緩存配置 ==========
    ADVERSARIAL_CACHE_DIR = './adversarial_cache'  # 緩存目錄
    USE_CACHE = True  # 是否使用緩存
    FORCE_REGENERATE = False  # 是否強制重新生成（忽略緩存）

    @classmethod
    def get_cache_path(cls, attack_name, num_samples, model_name=None):
        """
        獲取特定攻擊的緩存文件路徑，加入模型名稱

        Args:
            attack_name: 攻擊名稱
            num_samples: 樣本數量
            model_name: 模型名稱

        Returns:
            緩存文件路徑
        """
        os.makedirs(cls.ADVERSARIAL_CACHE_DIR, exist_ok=True)
        model_suffix = f"_{model_name}" if model_name else ""
        filename = f"{attack_name}_{num_samples}samples{model_suffix}.npz"
        return os.path.join(cls.ADVERSARIAL_CACHE_DIR, filename)

    @classmethod
    def get_all_cache_info(cls):
        """
        獲取所有緩存文件的信息

        Returns:
            dict: 緩存信息
        """
        if not os.path.exists(cls.ADVERSARIAL_CACHE_DIR):
            return {}

        cache_info = {}
        for filename in os.listdir(cls.ADVERSARIAL_CACHE_DIR):
            if filename.endswith('.npz'):
                filepath = os.path.join(cls.ADVERSARIAL_CACHE_DIR, filename)
                file_size = os.path.getsize(filepath) / (1024 ** 2)  # MB
                cache_info[filename] = {
                    'path': filepath,
                    'size_mb': file_size
                }
        return cache_info
