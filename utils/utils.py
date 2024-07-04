from torchvision.transforms.functional import normalize
import torch
import torch.nn as nn
import numpy as np
import os 

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def make_scoremap(mask, label, target_cls, bg_label=0, ignore_index=255):
    pred_prob_mask = torch.softmax(mask, dim=1)
    pred_score_mask = pred_prob_mask[:, 0]
    pred_score_mask = torch.clamp(pred_score_mask, 0, 1)
    
    labels_unique = torch.unique(label)
    for c in labels_unique:
        if c == 0:
            continue
        mask = label == c
        if c in target_cls:
            pred_score_mask[mask] = bg_label
        elif c == ignore_index:
            pred_score_mask[mask] = bg_label
        else:
            pred_score_mask[mask] = 1
    
    return pred_score_mask