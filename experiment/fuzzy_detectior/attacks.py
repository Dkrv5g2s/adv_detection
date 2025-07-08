# attacks.py
import torch, numpy as np
from torch import nn, optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

def get_art_classifier(backbone, device, nb_classes=10):
    """
    取 Image -> logits，用於產生對抗樣本
    """
    model = nn.Sequential(
        backbone,                     # (B,3,224,224) -> (B,C,H,W)
        nn.AdaptiveAvgPool2d(1),      # (B,C,1,1)
        nn.Flatten(),                 # (B,C)
        nn.Linear(backbone.out_channels, nb_classes)
    ).to(device)                      # 重要：整個模型搬到 GPU/CPU 同一裝置

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 只是 ART 要求，之後不會真的用到

    clf = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        device_type='gpu' if 'cuda' in device else 'cpu',
        channels_first=True
    )
    return clf


def fgsm_attack(classifier, images, labels, eps=8/255):
    """
    images : torch.Tensor (B,3,224,224) on device
    labels : torch.LongTensor (B,)       on device
    """
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    np_imgs   = images.detach().cpu().numpy()
    np_labels = labels.detach().cpu().numpy()
    adv_np    = attack.generate(x=np_imgs, y=np_labels)
    return torch.tensor(adv_np, device=images.device)


def pgd_attack(classifier, images, labels, eps=8/255, eps_step=2/255, max_iter=40):
    attack = ProjectedGradientDescent(classifier, eps=eps,
                                      eps_step=eps_step, max_iter=max_iter)
    adv_np = attack.generate(images.detach().cpu().numpy(),
                             labels.detach().cpu().numpy())
    return torch.tensor(adv_np, device=images.device)
