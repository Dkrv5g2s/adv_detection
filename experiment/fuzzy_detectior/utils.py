# utils.py
import torch, os

def detection_rate(tp, tn, fp, fn):
    return 100. * (tp + tn) / (tp + tn + fp + fn)

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
