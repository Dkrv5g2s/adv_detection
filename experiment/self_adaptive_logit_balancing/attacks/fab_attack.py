from autoattack import AutoAttack
import torch

class AutoAttackFABWrapper:
    """
    AutoAttack FAB 包裝器，提供與 ART 兼容的接口
    """

    def __init__(self, model, device, eps=8/255, steps=100,
                 n_restarts=1, alpha_max=0.1):
        self.model = model
        self.device = device
        self.eps = eps
        self.steps = steps
        self.n_restarts = n_restarts
        self.alpha_max = alpha_max

    def generate(self, x, y=None):
        """
        生成對抗樣本（與 ART 接口兼容）

        Args:
            x: 輸入圖像 numpy array [batch_size, C, H, W]
            y: 標籤 numpy array [batch_size] (可選)

        Returns:
            adv_x: 對抗樣本 numpy array
        """
        # 轉換為 torch tensor
        x_tensor = torch.from_numpy(x).to(self.device)

        if y is not None:
            y_tensor = torch.from_numpy(y).to(self.device)
        else:
            # 如果沒有提供標籤，使用模型預測
            with torch.no_grad():
                outputs = self.model(x_tensor)
                y_tensor = outputs.argmax(dim=1)

        # 創建 AutoAttack 實例
        adversary = AutoAttack(
            self.model,
            norm='Linf',
            eps=self.eps,
            version='custom',
            verbose=False,
            device=self.device
        )

        # 只運行 FAB 攻擊
        adversary.attacks_to_run = ['fab']

        # 設置 FAB 參數
        adversary.fab.n_restarts = self.n_restarts
        adversary.fab.n_iter = self.steps
        adversary.fab.alpha_max = self.alpha_max

        # 生成對抗樣本
        try:
            adv_tensor = adversary.run_standard_evaluation(
                x_tensor, y_tensor, bs=len(x_tensor)
            )
        except Exception as e:
            print(f"Warning: FAB attack failed, using original images. Error: {e}")
            adv_tensor = x_tensor

        # 轉回 numpy
        return adv_tensor.cpu().numpy()
