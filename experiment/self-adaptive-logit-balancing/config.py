"""
配置文件 - 所有超參數和設置
"""
import torch


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
    ADV_NUM_SAMPLES = 1000
    PGD_EPS = 8 / 255
    PGD_ALPHA = 2 / 255
    PGD_STEPS = 10

    # 檢測器配置
    DETECTOR_HIDDEN_DIM = 128
    DETECTOR_EPOCHS = 50
    DETECTOR_BATCH_SIZE = 32
    DETECTOR_LR = 0.001
    DETECTOR_DROPOUT = 0.3

    # 攻擊類型
    ATTACK_TYPES = ['Clean', 'PGD', 'PGD-L2', 'APGD', 'Square', 'APGDT', 'FAB', 'CW']

    # 保存路徑
    MODEL_SAVE_PATH = 'logit_balancing_model.pth'
    DETECTOR_SAVE_PATH = 'adversarial_detector.pth'
    RESULTS_DIR = './results'
