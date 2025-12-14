import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS


class ClassificationLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
      
        self.loss_module = MODELS.build(dict(type=type))

    def forward(self, logits, labels, **batch):
        return self.loss_module(logits, labels)